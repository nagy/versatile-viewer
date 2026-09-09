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
    cell::RefCell,
    collections::HashSet,
    path::{Path, PathBuf},
    sync::mpsc::{Receiver, Sender, channel},
};

use anyhow::{Context, Result};
use raylib::{color::Color, prelude::*};

use crate::blurbg::{self, BlurData};

const MARGIN: f32 = 8.0;
const GAP: f32 = 8.0;
/// Grid zoom step for +/- (same 25% steps as the image-view free zoom).
const GRID_ZOOM_STEP: f32 = 1.25;
/// Lower bound for the thumbnail side when zooming the grid out.
const GRID_MIN_SIDE: f32 = 32.0;
/// Maximum number of decodes in flight at once (jobs run in parallel on the
/// shared rayon pool; jxl-oxide parallelizes each decode further inside).
const MAX_INFLIGHT: usize = 6;

/// Long side of the thumbnail texture stored per grid entry: the whole
/// image, aspect preserved (the square crop for grid cells happens at
/// draw time). The full-resolution decode is downscaled to this on the
/// decode worker and the full-size texture kept only for a small keep-set
/// (the selection or open entry plus its prefetched neighbors), so VRAM
/// stays bounded on large directories. The image view also shows the
/// thumb as a full-frame placeholder while a decode catches up, so it
/// must cover the whole image, not just its center square.
const THUMB_LONG_SIDE: u32 = 1024;

/// A finished background decode, matched to an entry by its unique id
/// (indices shift when entries come and go; ids never do).
/// `blur` is the tiny blurred copy for the `VV_BLUR_BG` gimmick (None when
/// the gimmick is off), computed here on the worker so the main thread
/// never touches full-res pixels for it.
struct DecodeResult {
    id: u64,
    res: anyhow::Result<DecodeOk>,
}

/// Successful decode payload: the thumbnail RGBA (whole image, aspect
/// preserved, long side `THUMB_LONG_SIDE`) plus the untouched
/// full-resolution RGBA. The main thread uploads the thumb for every
/// entry but uploads the full texture only for keep-set entries
/// (selection/open + prefetched neighbors), dropping the rest. The
/// buffers live only until the next frame drains them, bounded by
/// `MAX_INFLIGHT`.
type DecodeOk = (Vec<u8>, u32, u32, Option<BlurData>, (Vec<u8>, u32, u32));

/// Grid layout: (`cols`, `cell_w`, `cell_h`, `side`, `content_h`).
type Layout = (usize, f32, f32, f32, f32);
/// Cache key: entry count + bit patterns of window size and zoom.
type LayoutKey = (usize, u32, u32, u32);

/// One grid cell: the image file plus its uploaded full-resolution texture
/// (thumbs are drawn by cropping a center square, so a resize never needs a
/// re-decode; the GPU scales it down every frame).
pub struct GridEntry {
    pub path: PathBuf,
    /// Full-resolution image dimensions (as decoded; not the thumb's).
    pub width: u32,
    pub height: u32,
    /// Thumbnail texture (whole image, aspect preserved, long side
    /// `THUMB_LONG_SIDE`) — what the grid cell draws (center-cropped at
    /// draw time) and what the image view shows as a placeholder while a
    /// decode catches up. Full-resolution textures live in `full` only
    /// for a small keep-set, so VRAM stays bounded on large directories.
    pub texture: Option<Texture2D>,
    /// Full-resolution texture; kept only for the selection/open entry and
    /// its prefetched neighbors (see `load_pending`), so opening them is
    /// instant. Dropped as soon as the entry leaves that set.
    pub full: Option<Texture2D>,
    /// Unique, stable id used to match async decode results.
    pub id: u64,
    /// A decode job for this entry is queued or in flight.
    pub queued: bool,
    /// Set when the decode (or texture upload) failed; the cell stays
    /// visible as a dimmed error square instead of silently vanishing.
    pub failed: Option<String>,
    /// Texture currently held by the image view (taken out of the grid); it
    /// is put back when the view is done, and never re-dispatched meanwhile.
    pub viewing: bool,
    /// Tiny blurred copy for the `VV_BLUR_BG` background (kept when the
    /// texture is out in the image view; the view clones it).
    pub blur: Option<BlurData>,
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
    /// Thumbnail size zoom (1.0 = exact-fill default; +/- steps it).
    zoom: f32,
    /// Vertical scroll offset (only used when zoomed in past the exact fill).
    scroll: f32,
    /// Cached layout, keyed by (entry count, window size, zoom) — recomputed
    /// only when any of those change, since the layout math is O(n). Borrowed
    /// interiorly because `draw` takes &self.
    layout_cache: RefCell<Option<(LayoutKey, Layout)>>,
    /// `VV_BLUR_BG` gimmick on? Read once; decode workers compute the tiny
    /// `VV_BLUR_BG` gimmick settings, read once: decode workers compute the
    /// tiny blurred copy only when enabled, at this texture long side.
    blur_enabled: bool,
    blur_px: u32,
}

impl Grid {
    /// List the images in a directory (sorted by file name) and spawn the
    /// background decode worker.
    pub fn from_dir(dir: &Path) -> Result<Grid> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .with_context(|| format!("failed to read directory {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    // Dotfiles are hidden, like nsxiv and file managers do.
                    && !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.'))
                    && is_image_path(p)
            })
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
                    full: None,
                    id: i as u64,
                    queued: false,
                    failed: None,
                    viewing: false,
                    blur: None,
                })
                .collect(),
            selected: 0,
            result_tx: res_tx,
            result_rx: res_rx,
            inflight: 0,
            layout_cache: RefCell::new(None),
            rep_dir: Default::default(),
            zoom: 1.0,
            scroll: 0.0,
            blur_enabled: blurbg::enabled(),
            blur_px: blurbg::blur_px(),
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
        // Ids that should hold a full-resolution texture: the entry the
        // scan starts at (the selection or the open image) plus the
        // prefetch priorities (its neighbors). Everything else stores only
        // the square thumbnail.
        let keep: HashSet<u64> = priority
            .iter()
            .filter_map(|&i| self.entries.get(i).map(|e| e.id))
            .chain(
                self.entries
                    .get(scan_start.min(self.entries.len().saturating_sub(1)))
                    .map(|e| e.id),
            )
            .collect();

        // 1. Apply finished decodes. Failed decodes mark their entry as
        // failed (the cell stays visible, dimmed, with an error glyph)
        // instead of splicing it out, so the grid count never lies.
        while let Ok(res) = self.result_rx.try_recv() {
            self.inflight -= 1;
            let Some(i) = self.entries.iter().position(|e| e.id == res.id) else {
                continue; // entry was spliced out meanwhile; drop the result
            };
            match res.res {
                Ok((thumb, tw, th, blur, full)) if self.entries[i].texture.is_none() => {
                    match crate::upload_rgba(rl, thread, &thumb, tw, th) {
                        Ok(t) => {
                            let (fw, fh) = (full.1, full.2);
                            let e = &mut self.entries[i];
                            e.width = fw;
                            e.height = fh;
                            e.texture = Some(t);
                            e.blur = blur;
                            e.queued = false;
                            if keep.contains(&e.id) {
                                match crate::upload_rgba(rl, thread, &full.0, fw, fh) {
                                    Ok(ft) => e.full = Some(ft),
                                    Err(err) => {
                                        // Non-fatal: the thumb still shows;
                                        // opening falls back to streaming.
                                        eprintln!(
                                            "vv: {}: {err:#} (full-res texture skipped)",
                                            e.path.display()
                                        );
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            eprintln!("vv: {}: {err:#}", self.entries[i].path.display());
                            let e = &mut self.entries[i];
                            e.queued = false;
                            e.failed = Some(format!("{err:#}"));
                        }
                    }
                }
                Ok((_, _, _, _, (full, fw, fh))) => {
                    // Re-decode of a thumb-only entry (keep-set refill):
                    // the thumb already exists, so only the full-res
                    // texture is wanted — and only if the entry is still
                    // in the keep set (it may have been navigated away
                    // from meanwhile).
                    self.entries[i].queued = false;
                    if keep.contains(&res.id)
                        && let Some(e) = self.entries.get_mut(i)
                        && e.full.is_none()
                    {
                        match crate::upload_rgba(rl, thread, &full, fw, fh) {
                            Ok(ft) => e.full = Some(ft),
                            Err(err) => {
                                eprintln!("vv: {}: {err:#}", e.path.display());
                                e.failed = Some(format!("{err:#}"));
                            }
                        }
                    }
                }
                Err(err) => {
                    eprintln!(
                        "vv: {}: decode failed: {err}",
                        self.entries[i].path.display()
                    );
                    let e = &mut self.entries[i];
                    e.queued = false;
                    e.failed = Some(format!("decode failed: {err}"));
                }
            }
        }
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));

        // Evict full-res textures from entries that left the keep set (the
        // selection/open entry and its neighbors moved on). Thumb textures
        // stay; the GPU context lives on this thread so unloading is safe.
        for e in &mut self.entries {
            if e.full.is_some() && !keep.contains(&e.id) {
                drop(e.full.take());
            }
        }

        // 2. Dispatch new jobs: `priority` indices first (the selection's
        // or the open image's neighbors), then a wraparound scan from
        // `scan_start`. Jobs run on the shared rayon pool, so several
        // decodes proceed in parallel; each sends its result back over the
        // channel, and the main thread uploads the texture above. The
        // per-entry checks below (texture present, queued, viewing) make
        // revisiting a priority index in the wraparound harmless, so no
        // extra dedup bookkeeping is needed — O(1) per entry.
        let n = self.entries.len();
        let start = scan_start.min(n);
        let order = priority
            .iter()
            .copied()
            .filter(|&i| i < n)
            .chain((start..n).chain(0..start));
        for i in order {
            if self.inflight >= MAX_INFLIGHT {
                break;
            }
            let e = &mut self.entries[i];
            if e.queued || e.viewing || e.failed.is_some() {
                continue;
            }
            // Decode when there is no thumb yet. Re-decode a thumb-only
            // entry when it is in the keep set but its full-res texture is
            // missing (its decode drained while outside the keep set, so
            // the full texture was dropped then). Thumb-only entries
            // outside the keep set are never re-decoded.
            if e.texture.is_some() && !(keep.contains(&e.id) && e.full.is_none()) {
                continue;
            }
            e.queued = true;
            self.inflight += 1;
            let id = e.id;
            let path = e.path.clone();
            let tx = self.result_tx.clone();
            let blur_enabled = self.blur_enabled;
            let blur_px = self.blur_px;
            rayon::spawn(move || {
                let res = crate::decode_image(&path).map(|d| {
                    let blur = blur_enabled
                        .then(|| blurbg::small_blur(&d.rgba, d.width, d.height, blur_px));
                    let (thumb, tw, th) =
                        crate::downscale_rgba(d.rgba.clone(), d.width, d.height, THUMB_LONG_SIDE);
                    (thumb, tw, th, blur, (d.rgba, d.width, d.height))
                });
                // Receiver gone (grid dropped): result is discarded and the
                // job simply ends.
                let _ = tx.send(DecodeResult { id, res });
            });
        }
    }

    /// Layout for the current state, via the cache (see `layout_cache`).
    fn layout(&self, win_w: f32, win_h: f32) -> Layout {
        let n = self.entries.len();
        let key = (n, win_w.to_bits(), win_h.to_bits(), self.zoom.to_bits());
        let mut cache = self.layout_cache.borrow_mut();
        if cache.as_ref().is_none_or(|(k, _)| *k != key) {
            *cache = Some((key, grid_layout_at(n, win_w, win_h, self.zoom)));
        }
        cache.as_ref().expect("just cached").1
    }

    /// Grid index of the entry with this stable id (ids stay stable for
    /// the grid's lifetime; indices do not).
    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.entries.iter().position(|e| e.id == id)
    }

    /// Mark the entry with this stable id as failed (e.g. a streaming load
    /// that errored mid-way). The cell stays, dimmed, with an error glyph.
    pub fn mark_failed(&mut self, id: u64, err: String) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.queued = false;
            e.failed = Some(err);
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
        let (cols, ..) = self.layout(win_w, win_h);
        let mut v = Vec::with_capacity(4);
        for i in [
            sel.checked_sub(1),
            Some(sel + 1),
            sel.checked_sub(cols),
            Some(sel + cols),
        ] {
            if let Some(i) = i
                && i < n
                && !v.contains(&i)
            {
                v.push(i);
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
        let (cols, _cw, ch, _side, content_h) = self.layout(win_w, win_h);
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

    /// Grid navigation: h/j/k/l + arrows move the selection (auto-repeat
    /// while held, at the X server's rate — xset r rate), g/G jump to the
    /// first/last image, Enter opens the selected image, q quits. ESC is
    /// inert here (grid is the home view).
    // One frame's input pipeline (queue drain, repeat, zoom, scroll,
    // navigation, mouse) is intentionally linear; see main() too.
    #[allow(clippy::too_many_lines)]
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
        let mut zoom_in = false;
        let mut zoom_out = false;
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
        let (cols, _cw, _ch, side, content_h) = self.layout(win_w, win_h);
        let scroll_max = (content_h - win_h).max(0.0);
        // Mouse wheel scrolls the grid (only meaningful once zoomed in past
        // the exact-fill layout, where nothing overflows).
        let wheel = rl.get_mouse_wheel_move();
        if wheel != 0.0 && scroll_max > 0.0 {
            self.scroll = (self.scroll - wheel * (side + GAP) * 3.0).clamp(0.0, scroll_max);
        }
        let old_sel = self.selected;
        let mut sel = self.selected;
        if left && !sel.is_multiple_of(cols) {
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
        if enter {
            return GridAction::Open(self.selected);
        }
        if quit {
            return GridAction::Quit;
        }
        // Mouse: click selects a cell; clicking the already-selected cell
        // opens it (first click selects, second opens — nsxiv-style).
        if rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT) {
            let m = rl.get_mouse_position();
            let (cols, cw, ch, _side, _content_h) = self.layout(win_w, win_h);
            // Grid coordinates: content is drawn at MARGIN + col*(cw+GAP)
            // minus scroll, so add scroll back to the cursor position.
            let (mx, my) = (m.x - MARGIN, m.y + self.scroll - MARGIN);
            let col = (mx / (cw + GAP)).floor();
            let row = (my / (ch + GAP)).floor();
            if col >= 0.0
                && row >= 0.0
                && col < cols as f32
                && mx - col * (cw + GAP) <= cw
                && my - row * (ch + GAP) <= ch
            {
                let idx = row as usize * cols + col as usize;
                if idx < self.entries.len() {
                    if idx == self.selected {
                        return GridAction::Open(idx);
                    }
                    self.selected = idx;
                    self.ensure_visible(win_w, win_h);
                }
            }
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
        let (cols, cw, ch, side, _) = self.layout(win_w, win_h);
        for (i, e) in self.entries.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let cx = MARGIN + col as f32 * (cw + GAP);
            let cy = MARGIN + row as f32 * (ch + GAP) - self.scroll;
            // Offscreen (scrolled out) cells: skip the draw work.
            if cy + ch < 0.0 || cy > win_h {
                continue;
            }
            let thumb = Rectangle {
                x: cx + (cw - side) / 2.0,
                y: cy + (ch - side) / 2.0,
                width: side,
                height: side,
            };
            if let Some(tex) = &e.texture {
                // The thumb texture keeps the image's aspect ratio; the
                // square cell shows its center crop (draw-time crop, so a
                // resize never needs a re-decode).
                let tw = tex.width() as f32;
                let th = tex.height() as f32;
                let side = tw.min(th);
                let src = Rectangle {
                    x: (tw - side) / 2.0,
                    y: (th - side) / 2.0,
                    width: side,
                    height: side,
                };
                d.draw_texture_pro(tex, src, thumb, Vector2::ZERO, 0.0, Color::WHITE);
            } else if e.failed.is_some() {
                // Decode failed: dim red placeholder with an error glyph —
                // the file stays visible instead of silently vanishing.
                d.draw_rectangle_rec(thumb, Color::new(72, 26, 26, 255));
                let msg = "!";
                let tw = d.measure_text(msg, 24);
                d.draw_text(
                    msg,
                    (thumb.x + (thumb.width - tw as f32) / 2.0) as i32,
                    (thumb.y + (thumb.height - 24.0) / 2.0) as i32,
                    24,
                    Color::new(214, 92, 92, 255),
                );
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

/// Largest exact-fill square thumbnail side for `n` entries in an
/// `aw` x `ah` available area: the best over all column counts (ties prefer
/// the column count whose cells are closest to square, i.e. least empty
/// space). This is also the zoom-1.0 reference size for +/- zooming.
fn best_fill_side(n: usize, aw: f32, ah: f32) -> f32 {
    let n = n.max(1);
    let mut best_score = (f32::NEG_INFINITY, 0.0f32);
    let mut best_side = 1.0f32;
    for cols in 1..=n {
        let rows = n.div_ceil(cols);
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
            let rows = n.div_ceil(cols);
            let cw = ((aw - (cols - 1) as f32 * GAP) / cols as f32).max(1.0);
            let ch = ((ah - (rows - 1) as f32 * GAP) / rows as f32).max(1.0);
            let key = ((cw.min(ch) - target).abs(), cw.max(ch));
            if key < best_key {
                best_key = key;
                best = (cols, cw, ch);
            }
        }
        let (cols, cw, ch) = best;
        let rows = n.div_ceil(cols);
        let content_h = 2.0 * MARGIN + rows as f32 * ch + (rows - 1) as f32 * GAP;
        (cols, cw, ch, cw.min(ch), content_h)
    } else {
        // Overflow: square cells at the target side, as many columns as fit
        // the width; the rows scroll vertically.
        let target = target.min(aw.min(ah));
        let cols = (((aw + GAP) / (target + GAP)).floor() as usize).clamp(1, n);
        let cw = ((aw - (cols - 1) as f32 * GAP) / cols as f32).max(target);
        let rows = n.div_ceil(cols);
        let content_h = 2.0 * MARGIN + rows as f32 * target + (rows - 1) as f32 * GAP;
        (cols, cw, target, target, content_h)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the asserted values are exact grid arithmetic
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn mark_failed_keeps_the_entry() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let png = image::DynamicImage::new_rgb8(1, 1);
        png.save(dir.join("a.png")).unwrap();
        let mut grid = Grid::from_dir(dir).unwrap();
        assert_eq!(grid.entries.len(), 1);
        let id = grid.entries[0].id;
        grid.mark_failed(id, "boom".to_string());
        assert_eq!(grid.entries.len(), 1, "failed entry stays in the grid");
        assert_eq!(grid.entries[0].failed.as_deref(), Some("boom"));
        assert!(!grid.entries[0].queued);
    }

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
            let rows = n.div_ceil(cols);
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
        let rows = 9_usize.div_ceil(cols);
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
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        for i in 0..9 {
            std::fs::write(dir.join(format!("{i:02}.png")), b"").unwrap();
        }
        let grid = Grid::from_dir(dir).unwrap();
        // Square window: 9 images lay out as a 3x3 grid, so index 4's
        // neighbors are 3 (left), 5 (right), 1 (up), 7 (down).
        assert_eq!(grid.prefetch_neighbors(4, 640.0, 640.0), vec![3, 5, 1, 7]);
        // Top-left corner: only right (1) and down (3) exist.
        assert_eq!(grid.prefetch_neighbors(0, 640.0, 640.0), vec![1, 3]);
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
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        // Tiny valid PNG (1x1 red) via the image crate.
        let png = image::DynamicImage::new_rgb8(1, 1);
        png.save(dir.join("b.png")).unwrap();
        std::fs::write(dir.join("a.txt"), "not an image").unwrap();
        std::fs::write(dir.join("c.png"), "invalid png content").unwrap(); // listed, decode fails later
        std::fs::write(dir.join(".hidden.png"), "dotfile").unwrap(); // skipped

        let grid = Grid::from_dir(dir).unwrap();
        let names: Vec<String> = grid
            .entries
            .iter()
            .map(|e| e.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["b.png", "c.png"]);
    }
}
