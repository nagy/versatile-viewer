//! Directory thumbnail grid: square center-crop thumbnails that fill the
//! whole window (unlike nsxiv's fixed thumbnail sizes), with a white border
//! on the selection.
//!
//! Decoding runs on the shared rayon pool (several files decode in
//! parallel); the main thread only drains finished decodes and uploads
//! textures (which needs the GL context). Jobs are dispatched
//! priority-first: the neighbors of whatever is on screen (grid selection,
//! or the open image in image view) decode before the rest. This keeps
//! frame times short, so key taps are never swallowed by decode stalls, and
//! makes the neighbors ready to open instantly.

use std::{
    path::{Path, PathBuf},
    sync::mpsc::{Receiver, Sender, channel},
};

use anyhow::{Context, Result};
use raylib::{color::Color, prelude::*};

const MARGIN: f32 = 8.0;
const GAP: f32 = 8.0;
/// Maximum number of decodes in flight at once (jobs run in parallel on the
/// shared rayon pool; jxl-oxide parallelizes each decode further inside).
const MAX_INFLIGHT: usize = 6;

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
    pub id: u64,
    /// A decode job for this entry is queued or in flight.
    pub queued: bool,
    /// Texture currently held by the image view (taken out of the grid); it
    /// is put back when the view is done, and never re-dispatched meanwhile.
    pub viewing: bool,
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
    result_tx: Sender<DecodeResult>,
    result_rx: Receiver<DecodeResult>,
    inflight: usize,
    /// Auto-repeat state for the four direction keys (h/j/k/l + arrows),
    /// indexed [left, right, up, down] (xset r rate values).
    rep_dir: [crate::keyrepeat::RepeatState; 4],
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

        let (res_tx, res_rx) = channel::<DecodeResult>();

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
                    viewing: false,
                })
                .collect(),
            selected: 0,
            result_tx: res_tx,
            result_rx: res_rx,
            inflight: 0,
            rep_dir: Default::default(),
        })
    }

    /// Drain finished background decodes (uploading textures, which needs
    /// the main thread) and hand out new jobs: `priority` indices first
    /// (neighbors of what is on screen), then a wraparound scan from
    /// `scan_start`. Runs every frame; never blocks. Entries whose texture
    /// is currently held by the image view are skipped.
    pub fn load_pending(
        &mut self,
        rl: &mut RaylibHandle,
        thread: &RaylibThread,
        priority: &[usize],
        scan_start: usize,
    ) {
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

        // 2. Dispatch new jobs: `priority` indices first (the selection's
        // or the open image's neighbors), then a wraparound scan from
        // `scan_start`. Jobs run on the shared rayon pool, so several
        // decodes proceed in parallel; each sends its result back over the
        // channel, and the main thread uploads the texture above.
        let n = self.entries.len();
        let mut order: Vec<usize> = Vec::new();
        for &i in priority {
            if i < n && !order.contains(&i) {
                order.push(i);
            }
        }
        let start = scan_start.min(n);
        for i in (start..n).chain(0..start) {
            if !order.contains(&i) {
                order.push(i);
            }
        }
        for i in order {
            if self.inflight >= MAX_INFLIGHT {
                break;
            }
            let e = &mut self.entries[i];
            if e.texture.is_some() || e.queued || e.viewing {
                continue;
            }
            e.queued = true;
            self.inflight += 1;
            let id = e.id;
            let path = e.path.clone();
            let tx = self.result_tx.clone();
            rayon::spawn(move || {
                let res = crate::decode_image(&path)
                    .map(|d| (d.rgba, d.width, d.height))
                    .map_err(|e| format!("{e:#}"));
                // Receiver gone (grid dropped): result is discarded and the
                // job simply ends.
                let _ = tx.send(DecodeResult { id, res });
            });
        }
    }

    /// Grid index of the entry with this stable id (ids never shift when
    /// failed entries are spliced out; indices do).
    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.entries.iter().position(|e| e.id == id)
    }

    /// Remove the entry with this stable id, if present.
    pub fn remove_entry_by_id(&mut self, id: u64) {
        if let Some(i) = self.index_of(id) {
            self.remove_entry(i);
        }
    }

    /// Indices of the grid neighbors of `sel` — left, right, up, down, in
    /// that order — as prefetch priority. Needs the window size for the
    /// column count. Duplicates and out-of-range indices are skipped.
    pub fn prefetch_neighbors(&self, sel: usize, win_w: f32, win_h: f32) -> Vec<usize> {
        let n = self.entries.len();
        if n == 0 {
            return Vec::new();
        }
        let (cols, ..) = grid_layout(n, win_w, win_h);
        let mut v = Vec::with_capacity(4);
        for i in [
            sel.checked_sub(1),
            Some(sel + 1),
            sel.checked_sub(cols),
            Some(sel + cols),
        ] {
            if let Some(i) = i {
                if i < n && !v.contains(&i) {
                    v.push(i);
                }
            }
        }
        v
    }

    /// Remove a failed entry so it disappears from the grid. Indices after
    /// `i` shift; the selection is clamped.
    pub fn remove_entry(&mut self, i: usize) {
        if i < self.entries.len() {
            self.entries.remove(i);
            self.selected = self.selected.min(self.entries.len().saturating_sub(1));
        }
    }

    /// Grid navigation: h/j/k/l + arrows move the selection (auto-repeat
    /// while held, at the X server's rate — xset r rate), g/G jump to the
    /// first/last image, Enter opens the selected image, q quits. ESC is
    /// inert here (grid is the home view).
    pub fn handle_input(&mut self, rl: &mut RaylibHandle, win_w: f32, win_h: f32) -> GridAction {
        if self.entries.is_empty() {
            return GridAction::None;
        }
        // Drain the raw key queue instead of is_key_pressed(). A press whose
        // release lands within the same frame is invisible to is_key_pressed
        // (raylib snapshots current->previous key state once per frame, so a
        // press+release pair between two polls nets out to 0->0), which is
        // easy to hit here: frames stall on full-res texture uploads, and a
        // crisp Enter tap fits inside one. GetKeyPressed()/GetCharPressed()
        // queue every press event during the poll, so nothing is lost.
        let mut enter = false;
        let mut quit = false;
        let mut left = false;
        let mut right = false;
        let mut up = false;
        let mut down = false;
        let mut jump_first = false;
        let mut jump_last = false;
        while let Some(k) = rl.get_key_pressed() {
            match k {
                KeyboardKey::KEY_ENTER | KeyboardKey::KEY_KP_ENTER => enter = true,
                KeyboardKey::KEY_Q => quit = true,
                KeyboardKey::KEY_H | KeyboardKey::KEY_LEFT => left = true,
                KeyboardKey::KEY_L | KeyboardKey::KEY_RIGHT => right = true,
                KeyboardKey::KEY_K | KeyboardKey::KEY_UP => up = true,
                KeyboardKey::KEY_J | KeyboardKey::KEY_DOWN => down = true,
                KeyboardKey::KEY_G => {
                    if rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
                        || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT)
                    {
                        jump_last = true;
                    } else {
                        jump_first = true;
                    }
                }
                _ => {}
            }
        }
        // Some input setups (IMEs, unusual X11 input methods) deliver Enter
        // as a character event ('\n'/'\r') instead of (or in addition to) a
        // key-press event; accept both. The queue is drained every frame, so
        // unmatched chars never pile up.
        while let Some(c) = rl.get_char_pressed() {
            if c == '\n' || c == '\r' {
                enter = true;
            }
        }
        // Auto-repeat for the direction keys: the initial press fires
        // immediately (edge from the queue above); holding fires at the X
        // server's repeat rate after its delay (xset r rate values).
        let now = rl.get_time();
        let (delay, rate) = crate::keyrepeat::settings();
        let down_left = rl.is_key_down(KeyboardKey::KEY_H) || rl.is_key_down(KeyboardKey::KEY_LEFT);
        let down_right =
            rl.is_key_down(KeyboardKey::KEY_L) || rl.is_key_down(KeyboardKey::KEY_RIGHT);
        let down_up = rl.is_key_down(KeyboardKey::KEY_K) || rl.is_key_down(KeyboardKey::KEY_UP);
        let down_down = rl.is_key_down(KeyboardKey::KEY_J) || rl.is_key_down(KeyboardKey::KEY_DOWN);
        let rep = &mut self.rep_dir;
        let left = rep[0].tick(left, down_left, now, delay, rate);
        let right = rep[1].tick(right, down_right, now, delay, rate);
        let up = rep[2].tick(up, down_up, now, delay, rate);
        let down = rep[3].tick(down, down_down, now, delay, rate);
        let (cols, ..) = grid_layout(self.entries.len(), win_w, win_h);
        let n = self.entries.len();
        let mut sel = self.selected;
        if left && sel % cols != 0 {
            sel -= 1;
        }
        if right && sel % cols != cols - 1 && sel + 1 < n {
            sel += 1;
        }
        if up && sel >= cols {
            sel -= cols;
        }
        if down && sel + cols < n {
            sel += cols;
        }
        // g/G: jump to the first/last image (overrides held direction keys).
        if jump_first {
            sel = 0;
        }
        if jump_last {
            sel = n - 1;
        }
        self.selected = sel;
        if enter {
            return GridAction::Open(self.selected);
        }
        if quit {
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
    fn prefetch_neighbors_left_right_up_down() {
        let dir = std::env::temp_dir().join(format!("vv-test-pf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..9 {
            std::fs::write(dir.join(format!("{i:02}.png")), b"").unwrap();
        }
        let grid = Grid::from_dir(&dir).unwrap();
        // Square window: 9 images lay out as a 3x3 grid, so index 4's
        // neighbors are 3 (left), 5 (right), 1 (up), 7 (down).
        assert_eq!(grid.prefetch_neighbors(4, 640.0, 640.0), vec![3, 5, 1, 7]);
        // Top-left corner: only right (1) and down (3) exist.
        assert_eq!(grid.prefetch_neighbors(0, 640.0, 640.0), vec![1, 3]);
        std::fs::remove_dir_all(&dir).ok();
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
