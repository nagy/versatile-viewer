//! Directory thumbnail grid: square center-crop thumbnails that fill the
//! whole window (unlike nsxiv's fixed thumbnail sizes), with a white border
//! on the selection.
//!
//! +/- zoom the thumbnails (same 25% steps as the image-view free zoom).
//! The default zoom fills the window exactly (largest possible square
//! thumbs); zooming out re-fits with more, smaller thumbnails — still an
//! exact fill; zooming in grows the thumbs past that maximum, so the grid
//! overflows vertically and scrolls (mouse wheel; the selection is kept on
//! screen while navigating).
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
    sync::{
        Arc,
        mpsc::{Receiver, Sender, channel},
    },
};

use anyhow::{Context, Result};
use raylib::{color::Color, prelude::*};

use crate::document::{Document, open_document};

/// What a grid cell shows: an image/PDF file (directory grid) or one page of
/// an open document (page grid).
#[derive(Clone)]
pub enum EntrySource {
    /// A file on disk. Decoded lazily: `open_document` at decode time.
    File(PathBuf),
    /// Page `page` of an already-open document (page grid).
    DocPage { doc: Arc<dyn Document>, page: usize },
}

const MARGIN: f32 = 8.0;
const GAP: f32 = 8.0;
/// Grid zoom step for +/- (same 25% steps as the image-view free zoom).
const GRID_ZOOM_STEP: f32 = 1.25;
/// Lower bound for the thumbnail side when zooming the grid out.
const GRID_MIN_SIDE: f32 = 32.0;
/// Maximum number of decodes in flight at once (jobs run in parallel on the
/// shared rayon pool; jxl-oxide parallelizes each decode further inside).
const MAX_INFLIGHT: usize = 6;

/// A finished background decode, matched to an entry by its unique id
/// (indices shift when failed entries are spliced out; ids never do).
struct DecodeResult {
    id: u64,
    res: Result<(Vec<u8>, u32, u32), String>,
}

/// One grid cell: a source (file or document page) plus its uploaded
/// full-resolution texture (thumbs are drawn by cropping a center square, or
/// for document pages by aspect-fitting the whole page, so a resize never
/// needs a re-decode; the GPU scales it down every frame).
pub struct GridEntry {
    pub source: EntrySource,
    /// Natural dimensions (pixels, or PDF points for document pages).
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
    /// Leave this grid (ESC on a page grid; back to the previous grid).
    Back,
    Quit,
}

pub struct Grid {
    pub entries: Vec<GridEntry>,
    pub selected: usize,
    /// True for the directory grid (the home view; ESC does nothing there),
    /// false for page grids pushed on top of it (ESC goes back).
    pub root: bool,
    result_tx: Sender<DecodeResult>,
    result_rx: Receiver<DecodeResult>,
    inflight: usize,
    /// Auto-repeat state for the four direction keys (h/j/k/l + arrows),
    /// indexed [left, right, up, down] (xset r rate values).
    rep_dir: [crate::keyrepeat::RepeatState; 4],
    /// Thumbnail size zoom (1.0 = exact-fill default; +/- steps it).
    zoom: f32,
    /// Vertical scroll offset (only used when zoomed in past the exact fill).
    scroll: f32,
}

impl Grid {
    /// List the images (and PDFs) in a directory (sorted by file name) and
    /// spawn the background decode worker.
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
                    source: EntrySource::File(path),
                    width: 0,
                    height: 0,
                    texture: None,
                    id: i as u64,
                    queued: false,
                    viewing: false,
                })
                .collect(),
            selected: 0,
            root: true,
            result_tx: res_tx,
            result_rx: res_rx,
            inflight: 0,
            rep_dir: Default::default(),
            zoom: 1.0,
            scroll: 0.0,
        })
    }

    /// Build a page-overview grid for an open document: one entry per page.
    /// Entries start texture-less; decoding dispatches like any other grid
    /// (rayon pool, priority = selection neighbors). ESC pops back out.
    pub fn from_document(doc: Arc<dyn Document>) -> Grid {
        let entries = (0..doc.page_count())
            .map(|page| {
                // Natural size known up front (PDF points); failures fall
                // back to 0x0 and the decode job reports the real error.
                let (width, height) = doc
                    .page_info(page)
                    .map(|i| (i.width, i.height))
                    .unwrap_or((0, 0));
                GridEntry {
                    source: EntrySource::DocPage {
                        doc: doc.clone(),
                        page,
                    },
                    width,
                    height,
                    texture: None,
                    id: page as u64,
                    queued: false,
                    viewing: false,
                }
            })
            .collect();
        let (res_tx, res_rx) = channel::<DecodeResult>();
        Grid {
            entries,
            selected: 0,
            root: false,
            result_tx: res_tx,
            result_rx: res_rx,
            inflight: 0,
            rep_dir: Default::default(),
            zoom: 1.0,
            scroll: 0.0,
        }
    }

    /// Human-readable label for log lines: file path or "page N of M".
    fn label(&self, i: usize) -> String {
        match &self.entries[i].source {
            EntrySource::File(p) => p.display().to_string(),
            EntrySource::DocPage { page, .. } => {
                format!("page {} of {}", page + 1, self.entries.len())
            }
        }
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
                            eprintln!("vv: {}: {err:#}", self.label(i));
                            to_remove.push(i);
                        }
                    }
                }
                Ok(_) => {} // texture already loaded (opened synchronously)
                Err(err) => {
                    eprintln!("vv: {}: decode failed: {err}", self.label(i));
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
            let source = e.source.clone();
            let tx = self.result_tx.clone();
            rayon::spawn(move || {
                // Image files: open (sniff) + render page 0 at native size.
                // Document pages: render at 1.0 = natural size (PDF points,
                // plenty for thumbnails; the GPU downscales per frame).
                let res = match &source {
                    EntrySource::File(path) => open_document(path).and_then(|d| d.render(0, 1.0)),
                    EntrySource::DocPage { doc, page } => doc.render(*page, 1.0),
                }
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
        let (cols, ..) = grid_layout_at(n, win_w, win_h, self.zoom);
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

    /// Scroll the selection into view. Used when the selection was set (or
    /// the layout changed) from outside the navigation code — e.g. when
    /// returning to the grid from image view, or after a +/- zoom step.
    pub fn ensure_visible(&mut self, win_w: f32, win_h: f32) {
        if self.entries.is_empty() {
            return;
        }
        let n = self.entries.len();
        let (cols, _cw, ch, _side, content_h) = grid_layout_at(n, win_w, win_h, self.zoom);
        let scroll_max = (content_h - win_h).max(0.0);
        let row = self.selected / cols;
        let top = MARGIN + row as f32 * (ch + GAP);
        let bottom = top + ch;
        if top - self.scroll < MARGIN {
            self.scroll = top - MARGIN;
        }
        if bottom - self.scroll > win_h - MARGIN {
            self.scroll = bottom - (win_h - MARGIN);
        }
        self.scroll = self.scroll.clamp(0.0, scroll_max);
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
        let mut back = false;
        let mut left = false;
        let mut right = false;
        let mut up = false;
        let mut down = false;
        let mut jump_first = false;
        let mut jump_last = false;
        let mut zoom_in = false;
        let mut zoom_out = false;
        while let Some(k) = rl.get_key_pressed() {
            match k {
                KeyboardKey::KEY_ENTER | KeyboardKey::KEY_KP_ENTER => enter = true,
                KeyboardKey::KEY_Q => quit = true,
                // ESC: on a page grid, leave it (back to the directory grid).
                // On the root grid ESC is inert — it never quits the program.
                KeyboardKey::KEY_ESCAPE => back = true,
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
                KeyboardKey::KEY_EQUAL | KeyboardKey::KEY_KP_ADD => zoom_in = true,
                KeyboardKey::KEY_MINUS | KeyboardKey::KEY_KP_SUBTRACT => zoom_out = true,
                _ => {}
            }
        }
        // Some input setups (IMEs, unusual X11 input methods) deliver Enter
        // as a character event ('\n'/'\r') instead of (or in addition to) a
        // key-press event; accept both. This drain also covers the +/- zoom
        // characters, so nothing piles up. The queue is drained every frame,
        // so unmatched chars never accumulate.
        while let Some(c) = rl.get_char_pressed() {
            match c {
                '\n' | '\r' => enter = true,
                '+' => zoom_in = true,
                '-' => zoom_out = true,
                _ => {}
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
        // +/- zoom the thumbnails (same 25% steps and key detection as the
        // image-view free zoom: the US-layout =/- keycodes, numpad included,
        // plus the typed character for non-US layouts). The default zoom is
        // the exact-fill layout; zooming out re-fits with more, smaller
        // thumbnails (still an exact fill); zooming in overflows the window
        // vertically and enables scrolling.
        let n = self.entries.len();
        let mut follow = false;
        if zoom_in || zoom_out {
            let aw = (win_w - 2.0 * MARGIN).max(1.0);
            let ah = (win_h - 2.0 * MARGIN).max(1.0);
            let base = best_fill_side(n, aw, ah);
            if zoom_in {
                // Largest useful thumb side: one thumb fills the window.
                let max_zoom = (aw.min(ah) / base).max(1.0);
                self.zoom = (self.zoom * GRID_ZOOM_STEP).min(max_zoom);
            } else {
                let min_zoom = (GRID_MIN_SIDE / base).min(1.0);
                self.zoom = (self.zoom / GRID_ZOOM_STEP).max(min_zoom);
            }
            follow = true;
        }
        let (cols, _cw, _ch, side, content_h) = grid_layout_at(n, win_w, win_h, self.zoom);
        let scroll_max = (content_h - win_h).max(0.0);
        // Mouse wheel scrolls the grid (only meaningful once zoomed in past
        // the exact-fill layout, where nothing overflows).
        let wheel = rl.get_mouse_wheel_move();
        if wheel != 0.0 && scroll_max > 0.0 {
            self.scroll = (self.scroll - wheel * (side + GAP) * 3.0).clamp(0.0, scroll_max);
        }
        let old_sel = self.selected;
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
        // Keep the selection on screen after it moved or the layout changed
        // under it (+/- zoom); manual wheel scrolling is left untouched.
        if self.selected != old_sel || follow {
            self.ensure_visible(win_w, win_h);
        }
        if back && !self.root {
            return GridAction::Back;
        }
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
        let (cols, cw, ch, side, _) = grid_layout_at(self.entries.len(), win_w, win_h, self.zoom);
        for (i, e) in self.entries.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let cx = MARGIN + col as f32 * (cw + GAP);
            let cy = MARGIN + row as f32 * (ch + GAP) - self.scroll;
            // Offscreen (scrolled out) cells: skip the draw work.
            if cy + ch < 0.0 || cy > win_h {
                continue;
            }
            // Thumbnails: image files crop a center square out of the full
            // texture (GPU scales it down); document pages aspect-fit the
            // whole page into the cell — cropping a text page would cut
            // content away, and the point-dimensions give the right shape.
            let (thumb, src) = if let EntrySource::DocPage { .. } = &e.source {
                let (ew, eh) = (e.width.max(1) as f32, e.height.max(1) as f32);
                let s = (side / ew).min(side / eh);
                let fw = ew * s;
                let fh = eh * s;
                (
                    Rectangle {
                        x: cx + (cw - fw) / 2.0,
                        y: cy + (ch - fh) / 2.0,
                        width: fw,
                        height: fh,
                    },
                    Rectangle {
                        x: 0.0,
                        y: 0.0,
                        width: e.width as f32,
                        height: e.height as f32,
                    },
                )
            } else {
                let thumb = Rectangle {
                    x: cx + (cw - side) / 2.0,
                    y: cy + (ch - side) / 2.0,
                    width: side,
                    height: side,
                };
                let s = e.width.min(e.height) as f32;
                let src = Rectangle {
                    x: (e.width as f32 - s) / 2.0,
                    y: (e.height as f32 - s) / 2.0,
                    width: s,
                    height: s,
                };
                (thumb, src)
            };
            if let Some(tex) = &e.texture {
                d.draw_texture_pro(tex, src, thumb, Vector2::ZERO, 0.0, Color::WHITE);
            } else {
                // Still decoding: placeholder.
                d.draw_rectangle_rec(thumb, Color::new(28, 28, 28, 255));
            }
            if i == self.selected {
                d.draw_rectangle_lines_ex(thumb, 4.0, Color::WHITE);
            }
        }
    }
}

/// Is this path likely something we can decode? (Grid directory filter.)
/// PDFs included: they open a page-overview grid instead of an image view.
fn is_image_path(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "jpg"
                | "jpeg"
                | "png"
                | "jxl"
                | "gif"
                | "webp"
                | "bmp"
                | "tif"
                | "tiff"
                | "tga"
                | "pdf"
        )
    })
}

/// Largest exact-fill square thumbnail side for `n` entries in an
/// `aw` x `ah` available area: the best over all column counts (ties prefer
/// the column count whose cells are closest to square, i.e. least empty
/// space). This is also the zoom-1.0 reference size for +/- zooming.
fn best_fill_side(n: usize, aw: f32, ah: f32) -> f32 {
    let n = n.max(1);
    let mut best_score = (f32::NEG_INFINITY, 0.0f32);
    let mut best_side = 1.0f32;
    for cols in 1..=n {
        let rows = (n + cols - 1) / cols;
        let cw = ((aw - (cols - 1) as f32 * GAP) / cols as f32).max(1.0);
        let ch = ((ah - (rows - 1) as f32 * GAP) / rows as f32).max(1.0);
        let score = (cw.min(ch), -cw.max(ch));
        if score > best_score {
            best_score = score;
            best_side = cw.min(ch);
        }
    }
    best_side
}

/// Compute the grid layout for `n` entries at zoom `zoom` (1.0 = default):
/// how many columns, the cell size, the square thumbnail side, and the total
/// content height (which may exceed the window when zoomed in; the caller
/// scrolls by the excess).
///
/// At the default zoom the layout is the exact fill described in
/// [`best_fill_side`]: cells divide the available space, so the grid spans
/// the whole window in both dimensions.
///
/// Zooming out shrinks the target thumbnail side and picks the column count
/// whose fill-derived side lands closest to it — the grid still spans the
/// window exactly, now with more (smaller) thumbnails.
///
/// Zooming in grows the target side past the largest exact-fill size, so the
/// rows no longer fit vertically: cells stay square at the target side
/// (columns stretch a little so the grid still spans the width), rows
/// overflow, and the caller scrolls.
fn grid_layout_at(n: usize, win_w: f32, win_h: f32, zoom: f32) -> (usize, f32, f32, f32, f32) {
    let n = n.max(1);
    let aw = (win_w - 2.0 * MARGIN).max(1.0);
    let ah = (win_h - 2.0 * MARGIN).max(1.0);

    let base = best_fill_side(n, aw, ah);
    let target = (base * zoom).max(GRID_MIN_SIDE);
    if target <= base {
        // Exact fill: pick the column count whose fill-derived side is
        // closest to the target (at zoom 1.0 this reproduces the exact-fill
        // optimum; ties prefer the least empty space, i.e. the smaller
        // max(cw, ch)).
        let mut best = (1usize, aw, ah);
        let mut best_key = (f32::INFINITY, f32::INFINITY);
        for cols in 1..=n {
            let rows = (n + cols - 1) / cols;
            let cw = ((aw - (cols - 1) as f32 * GAP) / cols as f32).max(1.0);
            let ch = ((ah - (rows - 1) as f32 * GAP) / rows as f32).max(1.0);
            let key = ((cw.min(ch) - target).abs(), cw.max(ch));
            if key < best_key {
                best_key = key;
                best = (cols, cw, ch);
            }
        }
        let (cols, cw, ch) = best;
        let rows = (n + cols - 1) / cols;
        let content_h = 2.0 * MARGIN + rows as f32 * ch + (rows - 1) as f32 * GAP;
        (cols, cw, ch, cw.min(ch), content_h)
    } else {
        // Overflow: square cells at the target side, as many columns as fit
        // the width; the rows scroll vertically.
        let target = target.min(aw.min(ah));
        let cols = (((aw + GAP) / (target + GAP)).floor() as usize).clamp(1, n);
        let cw = ((aw - (cols - 1) as f32 * GAP) / cols as f32).max(target);
        let rows = (n + cols - 1) / cols;
        let content_h = 2.0 * MARGIN + rows as f32 * target + (rows - 1) as f32 * GAP;
        (cols, cw, target, target, content_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_layout_default_single_image_fills_smaller_side() {
        // One image, 100x100 window, 8px margin -> 84x84 cell and thumb.
        let (cols, cw, ch, side, content_h) = grid_layout_at(1, 100.0, 100.0, 1.0);
        assert_eq!(cols, 1);
        assert_eq!(cw, 84.0);
        assert_eq!(ch, 84.0);
        assert_eq!(side, 84.0);
        assert_eq!(content_h, 100.0);
    }

    #[test]
    fn grid_layout_default_spans_full_window() {
        // At the default zoom, cells are sized to divide the available
        // space, so the grid must span the whole window in both dimensions
        // and the content height must equal the window height.
        for (n, win_w, win_h) in [(1, 100.0, 100.0), (7, 1200.0, 700.0), (100, 1600.0, 900.0)] {
            let (cols, cw, ch, side, content_h) = grid_layout_at(n, win_w, win_h, 1.0);
            let rows = (n + cols - 1) / cols;
            let span_w = 2.0 * MARGIN + cols as f32 * cw + (cols - 1) as f32 * GAP;
            let span_h = 2.0 * MARGIN + rows as f32 * ch + (rows - 1) as f32 * GAP;
            assert!((span_w - win_w).abs() < 0.01, "n={n}: span_w={span_w}");
            assert!((span_h - win_h).abs() < 0.01, "n={n}: span_h={span_h}");
            assert_eq!(content_h, win_h);
            assert_eq!(side, cw.min(ch));
        }
    }

    #[test]
    fn grid_layout_never_crashes_on_degenerate_input() {
        let (cols, _, _, side, _) = grid_layout_at(3, 50.0, 800.0, 1.0);
        assert!(cols >= 1 && side > 0.0);
        let _ = grid_layout_at(0, 0.0, 0.0, 1.0);
        let _ = grid_layout_at(5, 800.0, 600.0, 0.0);
        let _ = grid_layout_at(5, 800.0, 600.0, 1000.0);
    }

    #[test]
    fn grid_layout_zoom_out_keeps_exact_fill_with_smaller_thumbs() {
        // 9 images in a 640x640 window: default is 3x3 with ~208px thumbs;
        // zooming out must shrink the thumbs, add columns, and still span
        // the window exactly (no scrolling).
        let (d_cols, _, _, d_side, _) = grid_layout_at(9, 640.0, 640.0, 1.0);
        let (cols, cw, ch, side, content_h) = grid_layout_at(9, 640.0, 640.0, 0.8);
        assert!(cols > d_cols, "cols={cols}, default={d_cols}");
        assert!(side < d_side);
        assert_eq!(side, cw.min(ch));
        let rows = (9 + cols - 1) / cols;
        let span_w = 2.0 * MARGIN + cols as f32 * cw + (cols - 1) as f32 * GAP;
        let span_h = 2.0 * MARGIN + rows as f32 * ch + (rows - 1) as f32 * GAP;
        assert!((span_w - 640.0).abs() < 0.01, "span_w={span_w}");
        assert!((span_h - 640.0).abs() < 0.01, "span_h={span_h}");
        assert_eq!(content_h, 640.0);
    }

    #[test]
    fn grid_layout_zoom_in_overflows_and_grows_thumbs() {
        // Zooming in grows thumbs past the largest exact-fill size: fewer
        // columns, square thumbs at the zoomed size, content taller than
        // the window (scrollable).
        let (d_cols, _, _, d_side, _) = grid_layout_at(9, 640.0, 640.0, 1.0);
        let (cols, cw, ch, side, content_h) = grid_layout_at(9, 640.0, 640.0, 2.0);
        assert!(cols < d_cols, "cols={cols}, default={d_cols}");
        assert!(side > d_side);
        assert_eq!(side, ch);
        assert!(cw >= side);
        assert!(content_h > 640.0, "content_h={content_h}");
    }

    #[test]
    fn grid_layout_side_is_square_of_cell_minimum() {
        let (_, cw, ch, side, _) = grid_layout_at(7, 1200.0, 700.0, 1.0);
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
            .map(|e| match &e.source {
                EntrySource::File(p) => p.file_name().unwrap().to_string_lossy().to_string(),
                EntrySource::DocPage { .. } => unreachable!("dir grid only lists files"),
            })
            .collect();
        assert_eq!(names, vec!["b.png", "c.png"]);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn from_document_builds_one_entry_per_page() {
        // Multi-page document: one entry per page with point dimensions,
        // non-root (ESC pops), textures decoded lazily like any other grid.
        let doc =
            crate::pdf::PdfDocument::from_data(crate::pdf::tests::MINIMAL_PDF.to_vec()).unwrap();
        let grid = Grid::from_document(Arc::new(doc) as Arc<dyn Document>);
        assert_eq!(grid.entries.len(), 1);
        assert!(!grid.root);
        assert_eq!(grid.entries[0].width, 612);
        assert_eq!(grid.entries[0].height, 792);
        assert!(matches!(
            &grid.entries[0].source,
            EntrySource::DocPage { page: 0, .. }
        ));
    }
}
