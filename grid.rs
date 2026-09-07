//! Directory thumbnail grid: square center-crop thumbnails that fill the
//! whole window (unlike nsxiv's fixed thumbnail sizes), with a white border
//! on the selection.
//!
//! Decoding happens on a background worker thread; the main thread only
//! drains finished decodes and uploads textures (which needs the GL
//! context). This keeps frame times short, so key taps are never swallowed
//! by multi-second decode stalls.

use std::{
    path::{Path, PathBuf},
    sync::mpsc::{Receiver, Sender, channel},
};

use anyhow::{Context, Result};
use raylib::{color::Color, prelude::*};

const MARGIN: f32 = 8.0;
const GAP: f32 = 8.0;
/// Maximum number of decodes in flight at once.
const MAX_INFLIGHT: usize = 3;

/// A finished background decode, matched to an entry by its unique id
/// (indices shift when failed entries are spliced out; ids never do).
struct DecodeResult {
    id: u64,
    res: Result<(Vec<u8>, u32, u32), String>,
}

/// One grid cell: the image file plus its uploaded full-resolution texture
/// (thumbs are drawn by cropping a center square, so a resize never needs a
/// re-decode; the GPU scales it down every frame).
pub struct GridEntry {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub texture: Option<Texture2D>,
    /// Unique, stable id used to match async decode results.
    id: u64,
    /// A decode job for this entry is queued or in flight.
    queued: bool,
}

/// What the user asked the grid to do this frame.
#[derive(Clone, Copy, PartialEq)]
pub enum GridAction {
    None,
    /// Open the grid entry at this index in image mode.
    Open(usize),
    Quit,
}

pub struct Grid {
    pub entries: Vec<GridEntry>,
    pub selected: usize,
    job_tx: Option<Sender<(u64, PathBuf)>>,
    result_rx: Receiver<DecodeResult>,
    inflight: usize,
}

impl Grid {
    /// List the images in a directory (sorted by file name) and spawn the
    /// background decode worker.
    pub fn from_dir(dir: &Path) -> Result<Grid> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("failed to read directory {dir:?}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_image_path(p))
            .collect();
        paths.sort_by_key(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()));

        let (job_tx, job_rx) = channel::<(u64, PathBuf)>();
        let (res_tx, res_rx) = channel::<DecodeResult>();
        // Worker: decode one job at a time until the grid (and its sender)
        // is dropped, which closes the channel and ends the loop.
        std::thread::spawn(move || {
            while let Ok((id, path)) = job_rx.recv() {
                let res = crate::decode_image(&path)
                    .map(|d| (d.rgba, d.width, d.height))
                    .map_err(|e| format!("{e:#}"));
                if res_tx.send(DecodeResult { id, res }).is_err() {
                    break; // grid gone
                }
            }
        });

        Ok(Grid {
            entries: paths
                .into_iter()
                .enumerate()
                .map(|(i, path)| GridEntry {
                    path,
                    width: 0,
                    height: 0,
                    texture: None,
                    id: i as u64,
                    queued: false,
                })
                .collect(),
            selected: 0,
            job_tx: Some(job_tx),
            result_rx: res_rx,
            inflight: 0,
        })
    }

    /// Drain finished background decodes (uploading textures, which needs
    /// the main thread) and hand out new jobs, selection-first. Runs every
    /// frame; never blocks.
    pub fn load_pending(&mut self, rl: &mut RaylibHandle, thread: &RaylibThread) {
        // 1. Apply finished decodes. Collect failures for removal so entry
        // indices stay valid until we splice.
        let mut to_remove: Vec<usize> = Vec::new();
        while let Ok(res) = self.result_rx.try_recv() {
            self.inflight -= 1;
            let Some(i) = self.entries.iter().position(|e| e.id == res.id) else {
                continue; // entry was spliced out meanwhile; drop the result
            };
            match res.res {
                Ok((rgba, w, h)) if self.entries[i].texture.is_none() => {
                    match crate::upload_rgba(rl, thread, &rgba, w, h) {
                        Ok(t) => {
                            let e = &mut self.entries[i];
                            e.width = w;
                            e.height = h;
                            e.texture = Some(t);
                        }
                        Err(err) => {
                            eprintln!("vv: {}: {err:#}", self.entries[i].path.display());
                            to_remove.push(i);
                        }
                    }
                }
                Ok(_) => {} // texture already loaded (opened synchronously)
                Err(err) => {
                    eprintln!(
                        "vv: {}: decode failed: {err}",
                        self.entries[i].path.display()
                    );
                    to_remove.push(i);
                }
            }
        }
        // Splice out failed entries (reverse order keeps indices valid).
        for &i in to_remove.iter().rev() {
            self.entries.remove(i);
            if i < self.selected {
                self.selected -= 1;
            }
        }
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));

        // 2. Dispatch new jobs, starting near the selection so what you look
        // at appears first.
        let n = self.entries.len();
        while self.inflight < MAX_INFLIGHT {
            let mut scan = (self.selected..n).chain(0..self.selected);
            let Some(i) =
                scan.find(|&i| self.entries[i].texture.is_none() && !self.entries[i].queued)
            else {
                break;
            };
            let e = &mut self.entries[i];
            e.queued = true;
            if let Some(tx) = &self.job_tx {
                if tx.send((e.id, e.path.clone())).is_err() {
                    break; // worker gone
                }
            }
            self.inflight += 1;
        }
    }

    /// Make sure the entry at `i` has a texture (decoding synchronously if
    /// the worker hasn't delivered it yet — the user explicitly asked to
    /// open it, so a short block is fine). On failure the entry is removed
    /// and false returned.
    pub fn ensure_loaded(
        &mut self,
        rl: &mut RaylibHandle,
        thread: &RaylibThread,
        i: usize,
    ) -> bool {
        if i >= self.entries.len() {
            return false;
        }
        if self.entries[i].texture.is_some() {
            return true;
        }
        match crate::load_image_texture(rl, thread, &self.entries[i].path) {
            Ok((t, w, h)) => {
                self.entries[i].width = w;
                self.entries[i].height = h;
                self.entries[i].texture = Some(t);
                true
            }
            Err(err) => {
                eprintln!("vv: {}: {err:#}", self.entries[i].path.display());
                self.entries.remove(i);
                self.selected = self.selected.min(self.entries.len().saturating_sub(1));
                false
            }
        }
    }

    /// Grid navigation: h/j/k/l + arrows move the selection, Enter opens the
    /// selected image, q/ESC quit.
    pub fn handle_input(&mut self, rl: &mut RaylibHandle, win_w: f32, win_h: f32) -> GridAction {
        if self.entries.is_empty() {
            return GridAction::None;
        }
        // Drain the character queue. Some input setups (IMEs, unusual X11
        // input methods) deliver Enter as a character event ('\n'/'\r')
        // instead of (or in addition to) a key-press event; accept both.
        // Grid mode otherwise never drains the queue, so unmatched chars
        // would just pile up.
        let mut enter_char = false;
        while let Some(c) = rl.get_char_pressed() {
            if c == '\n' || c == '\r' {
                enter_char = true;
            }
        }
        let (cols, ..) = grid_layout(self.entries.len(), win_w, win_h);
        let n = self.entries.len();
        let sel = self.selected;
        if rl.is_key_pressed(KeyboardKey::KEY_H) || rl.is_key_pressed(KeyboardKey::KEY_LEFT) {
            if sel % cols != 0 {
                self.selected -= 1;
            }
        }
        if rl.is_key_pressed(KeyboardKey::KEY_L) || rl.is_key_pressed(KeyboardKey::KEY_RIGHT) {
            if sel % cols != cols - 1 && sel + 1 < n {
                self.selected += 1;
            }
        }
        if rl.is_key_pressed(KeyboardKey::KEY_K) || rl.is_key_pressed(KeyboardKey::KEY_UP) {
            if sel >= cols {
                self.selected -= cols;
            }
        }
        if rl.is_key_pressed(KeyboardKey::KEY_J) || rl.is_key_pressed(KeyboardKey::KEY_DOWN) {
            if sel + cols < n {
                self.selected += cols;
            }
        }
        if enter_char
            || rl.is_key_pressed(KeyboardKey::KEY_ENTER)
            || rl.is_key_pressed(KeyboardKey::KEY_KP_ENTER)
        {
            return GridAction::Open(self.selected);
        }
        if rl.is_key_pressed(KeyboardKey::KEY_Q) || rl.is_key_pressed(KeyboardKey::KEY_ESCAPE) {
            return GridAction::Quit;
        }
        GridAction::None
    }

    pub fn draw(&self, d: &mut RaylibDrawHandle, win_w: f32, win_h: f32) {
        if self.entries.is_empty() {
            let msg = "no images found in directory";
            let tw = d.measure_text(msg, 20);
            d.draw_text(
                msg,
                (win_w as i32 - tw) / 2,
                win_h as i32 / 2 - 10,
                20,
                Color::GRAY,
            );
            return;
        }
        let (cols, cw, ch, side) = grid_layout(self.entries.len(), win_w, win_h);
        for (i, e) in self.entries.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let cx = MARGIN + col as f32 * (cw + GAP);
            let cy = MARGIN + row as f32 * (ch + GAP);
            let thumb = Rectangle {
                x: cx + (cw - side) / 2.0,
                y: cy + (ch - side) / 2.0,
                width: side,
                height: side,
            };
            if let Some(tex) = &e.texture {
                // Square thumbnail: crop the center square out of the full
                // texture and scale it into the cell (GPU does the downscale,
                // so resizing never re-decodes).
                let s = e.width.min(e.height) as f32;
                let src = Rectangle {
                    x: (e.width as f32 - s) / 2.0,
                    y: (e.height as f32 - s) / 2.0,
                    width: s,
                    height: s,
                };
                d.draw_texture_pro(tex, src, thumb, Vector2::ZERO, 0.0, Color::WHITE);
            } else {
                // Still decoding: placeholder square.
                d.draw_rectangle_rec(thumb, Color::new(28, 28, 28, 255));
            }
            if i == self.selected {
                d.draw_rectangle_lines_ex(thumb, 4.0, Color::WHITE);
            }
        }
    }
}

/// Is this path likely an image we can decode? (Grid directory filter.)
fn is_image_path(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "jpg" | "jpeg" | "png" | "jxl" | "gif" | "webp" | "bmp" | "tif" | "tiff" | "tga"
        )
    })
}

/// Compute the grid layout: how many columns, and the cell size, so square
/// thumbnails fill the whole window. The thumbnail size is chosen fresh
/// every frame (unlike nsxiv's fixed sizes) by trying every column count
/// and keeping the one with the largest square side; ties prefer the
/// column count whose cells are closest to square (least empty space).
/// Returns (columns, cell_w, cell_h, square side).
fn grid_layout(n: usize, win_w: f32, win_h: f32) -> (usize, f32, f32, f32) {
    let n = n.max(1);
    let aw = (win_w - 2.0 * MARGIN).max(1.0);
    let ah = (win_h - 2.0 * MARGIN).max(1.0);

    let mut best = (1usize, aw, ah);
    // No layout chosen yet: -inf so the first candidate always wins.
    let mut best_score = (f32::NEG_INFINITY, 0.0f32);
    for cols in 1..=n {
        let rows = (n + cols - 1) / cols;
        let cw = ((aw - (cols - 1) as f32 * GAP) / cols as f32).max(1.0);
        let ch = ((ah - (rows - 1) as f32 * GAP) / rows as f32).max(1.0);
        let score = (cw.min(ch), -cw.max(ch));
        if score > best_score {
            best_score = score;
            best = (cols, cw, ch);
        }
    }
    let (cols, cw, ch) = best;
    (cols, cw, ch, cw.min(ch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_layout_single_image_fills_smaller_side() {
        // One image, 100x100 window, 8px margin -> 84x84 cell and thumb.
        let (cols, cw, ch, side) = grid_layout(1, 100.0, 100.0);
        assert_eq!(cols, 1);
        assert_eq!(cw, 84.0);
        assert_eq!(ch, 84.0);
        assert_eq!(side, 84.0);
    }

    #[test]
    fn grid_layout_spans_full_window() {
        // Cells are sized to divide the available space, so the grid must
        // span the whole window in both dimensions (grid takes all space).
        for (n, win_w, win_h) in [(1, 100.0, 100.0), (7, 1200.0, 700.0), (100, 1600.0, 900.0)] {
            let (cols, cw, ch, side) = grid_layout(n, win_w, win_h);
            let rows = (n + cols - 1) / cols;
            let span_w = 2.0 * MARGIN + cols as f32 * cw + (cols - 1) as f32 * GAP;
            let span_h = 2.0 * MARGIN + rows as f32 * ch + (rows - 1) as f32 * GAP;
            assert!((span_w - win_w).abs() < 0.01, "n={n}: span_w={span_w}");
            assert!((span_h - win_h).abs() < 0.01, "n={n}: span_h={span_h}");
            assert_eq!(side, cw.min(ch));
        }
    }

    #[test]
    fn grid_layout_never_crashes_on_degenerate_input() {
        let (cols, _, _, side) = grid_layout(3, 50.0, 800.0);
        assert!(cols >= 1 && side > 0.0);
        let _ = grid_layout(0, 0.0, 0.0);
    }

    #[test]
    fn grid_layout_side_is_square_of_cell_minimum() {
        let (_, cw, ch, side) = grid_layout(7, 1200.0, 700.0);
        assert_eq!(side, cw.min(ch));
    }

    #[test]
    fn image_extension_filter() {
        assert!(is_image_path(Path::new("foo.png")));
        assert!(is_image_path(Path::new("foo.PNG")));
        assert!(is_image_path(Path::new("a/b/c.jpeg")));
        assert!(is_image_path(Path::new("photo.JXL")));
        assert!(!is_image_path(Path::new("notes.txt")));
        assert!(!is_image_path(Path::new("archive.tar.gz")));
        assert!(!is_image_path(Path::new("noext")));
    }

    #[test]
    fn from_dir_lists_images_sorted_ignores_others() {
        let dir = std::env::temp_dir().join(format!("vv-test-grid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Tiny valid PNG (1x1 red) via the image crate.
        let png = image::DynamicImage::new_rgb8(1, 1);
        png.save(dir.join("b.png")).unwrap();
        std::fs::write(dir.join("a.txt"), "not an image").unwrap();
        std::fs::write(dir.join("c.png"), "invalid png content").unwrap(); // listed, decode fails later

        let grid = Grid::from_dir(&dir).unwrap();
        let names: Vec<String> = grid
            .entries
            .iter()
            .map(|e| e.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["b.png", "c.png"]);

        std::fs::remove_dir_all(&dir).ok();
    }
}
