//! Directory thumbnail grid.
//! Square center-crop thumbnails fill the whole window, unlike the fixed
//! thumbnail sizes of nsxiv.
//! A white border marks the selection.
//!
//! +/- zoom the thumbnails (the same 25% steps as the image-view free zoom).
//! The default zoom fills the window exactly with the largest possible square
//! thumbs.
//! Zooming out re-fits with more, smaller thumbnails and still fills exactly.
//! Zooming in grows the thumbs past that maximum, so the grid overflows
//! vertically and scrolls with the mouse wheel.
//! The code keeps the selection on screen while navigating.
//!
//! Decoding runs on the shared rayon pool, so several files decode in
//! parallel.
//! The main thread only drains finished decodes and uploads textures.
//! Texture uploads need the GL context.
//! The code dispatches jobs priority-first.
//! The neighbors of whatever is on screen (grid selection, or the open image
//! in image view) decode before the rest.
//! While the open image still decodes in image view, the code pauses
//! everything else, so the open image gets all the cores.
//! This keeps frame times short.
//! Decode stalls then never swallow key taps.
//! The neighbors are ready to open at once.

use std::{
    cell::RefCell,
    collections::HashSet,
    path::{Path, PathBuf},
    sync::mpsc::{Receiver, Sender, channel},
};

use anyhow::{Context, Result};
use raylib::{color::Color, prelude::*};

use crate::{
    DecodedImage,
    blurbg::{self},
};

const MARGIN: f32 = 8.0;
const GAP: f32 = 8.0;
/// Grid zoom step for +/- (the same 25% steps as the image-view free zoom).
const GRID_ZOOM_STEP: f32 = 1.25;
/// Lower bound for the thumbnail side when the grid zooms out.
const GRID_MIN_SIDE: f32 = 32.0;
/// The maximum number of decodes in flight at once.
/// Jobs run in parallel on the shared rayon pool.
/// jxl-oxide parallelizes each decode further inside.
/// The code allows one job per logical core, at least 2.
/// JPEG and PNG decodes through the `image` crate use one thread per file.
/// The cap therefore decides how many cores work on a large directory.
fn max_inflight() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(2, std::num::NonZeroUsize::get)
            .max(2)
    })
}

/// Long side of the thumbnail texture stored per grid entry.
/// The thumbnail holds the whole image with the aspect preserved.
/// The square crop for grid cells happens at draw time.
/// The decode worker downscales the full-resolution decode to this size.
/// The code keeps the full-size texture only for a small keep-set: the
/// selection or open entry plus its prefetched neighbors.
/// VRAM therefore stays bounded on large directories.
/// The image view also shows the thumb as a full-frame placeholder while a
/// decode catches up.
/// The thumbnail must cover the whole image, not only its center square.
const THUMB_LONG_SIDE: u32 = 1024;

/// A finished background decode, matched to an entry by its unique id.
/// Indices shift when entries come and go.
/// Ids never shift.
struct DecodeResult {
    id: u64,
    res: anyhow::Result<DecodeOk>,
}

/// Successful decode payload: the thumbnail and the untouched
/// full-resolution image.
/// The main thread uploads the thumb for every entry.
/// It uploads the full texture only for keep-set entries (selection or open
/// plus prefetched neighbors) and drops the rest.
/// The buffers live only until the next frame drains them, bounded by
/// `max_inflight()`.
struct DecodeOk {
    /// The whole image with the aspect preserved, long side
    /// [`THUMB_LONG_SIDE`].
    thumb: DecodedImage,
    /// The tiny blurred copy for the `VV_BLUR_BG` gimmick (None when off).
    blur: Option<DecodedImage>,
    /// The untouched full-resolution image.
    full: DecodedImage,
}

/// Grid layout, in window pixels.
#[derive(Clone, Copy)]
struct Layout {
    cols: usize,
    cell_w: f32,
    cell_h: f32,
    /// The square thumbnail side drawn inside each cell.
    side: f32,
    /// The total content height (rows + gaps + margins), for scrolling.
    content_h: f32,
}

/// Cache key for [`Grid::layout`]: entry count and bit patterns of the window
/// size and zoom.
type LayoutKey = (usize, u32, u32, u32);

/// One grid cell: the image file and its uploaded full-resolution texture.
/// The code draws thumbs by cropping a center square, so a resize never
/// needs a re-decode.
/// The GPU scales the texture down every frame.
pub struct GridEntry {
    pub path: PathBuf,
    /// Full-resolution image dimensions (as decoded, not the thumb's).
    pub width: u32,
    pub height: u32,
    /// The thumbnail texture (the whole image with the aspect preserved,
    /// long side `THUMB_LONG_SIDE`).
    /// The grid cell draws this texture center-cropped at draw time.
    /// The image view shows it as a placeholder while a decode catches up.
    /// Full-resolution textures live in `full` only for a small keep-set,
    /// so VRAM stays bounded on large directories.
    pub texture: Option<Texture2D>,
    /// The full-resolution texture.
    /// The code keeps it only for the selection or open entry and its
    /// prefetched neighbors (see `load_pending`), so opening them is instant.
    /// It drops as soon as the entry leaves that set.
    pub full: Option<Texture2D>,
    /// A unique, stable id used to match async decode results.
    pub id: u64,
    /// A decode job for this entry is queued or in flight.
    pub queued: bool,
    /// Set when the decode (or texture upload) failed.
    /// The cell stays visible as a dimmed error square instead of vanishing.
    pub failed: Option<String>,
    /// The texture the image view holds now (taken out of the grid).
    /// The grid puts it back when the view ends and never re-dispatches it
    /// in between.
    pub viewing: bool,
    /// The tiny blurred copy for the `VV_BLUR_BG` background.
    /// The grid keeps it while the texture is out in the image view.
    /// The view clones it.
    pub blur: Option<DecodedImage>,
}

/// What the user asked the grid to do this frame.
#[derive(Clone, Copy, PartialEq)]
pub enum GridAction {
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
    /// indexed [left, right, up, down], from the xset r rate values.
    rep_dir: [crate::keyrepeat::RepeatState; 4],
    /// Thumbnail size zoom (1.0 = exact-fill default, and +/- steps it).
    zoom: f32,
    /// The vertical scroll offset, used only when zoomed in past the exact
    /// fill.
    scroll: f32,
    /// The cached layout, keyed by (entry count, window size, zoom).
    /// The code recomputes it only when one of those changes, because the
    /// layout math is O(n).
    /// The field uses interior mutability because `draw` takes &self.
    layout_cache: RefCell<Option<(LayoutKey, Layout)>>,
    /// The `VV_BLUR_BG` gimmick settings, read once.
    /// Decode workers compute the tiny blurred copy only when enabled, at
    /// this texture long side.
    blur_enabled: bool,
    blur_px: u32,
}

impl Grid {
    /// List the images in a directory (sorted by file name) and start the
    /// background decode worker.
    ///
    /// # Errors
    ///
    /// The function errors when it cannot read the directory.
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

    /// Drain finished background decodes, which upload textures on the main
    /// thread, and hand out new jobs.
    /// The code dispatches `priority` indices first (neighbors of what is on
    /// screen), then a wraparound scan from `scan_start`.
    /// With `scan` false it dispatches only `priority` entries and pauses the
    /// rest of the queue.
    /// The image view sets this while the open image still decodes, so the
    /// visible image gets all the cores.
    /// The function runs every frame and never blocks.
    /// It skips entries whose texture the image view holds now.
    pub fn load_pending(
        &mut self,
        rl: &mut RaylibHandle,
        thread: &RaylibThread,
        priority: &[usize],
        scan_start: usize,
        scan: bool,
    ) {
        // Ids that hold a full-resolution texture: the entry the
        // scan starts at (the selection or the open image) plus the
        // prefetch priorities (its neighbors).
        // Everything else stores only the square thumbnail.
        let keep: HashSet<u64> = priority
            .iter()
            .filter_map(|&i| self.entries.get(i).map(|e| e.id))
            .chain(
                self.entries
                    .get(scan_start.min(self.entries.len().saturating_sub(1)))
                    .map(|e| e.id),
            )
            .collect();

        self.drain_results(rl, thread, &keep);
        self.dispatch(priority, scan_start, scan, &keep);
    }

    /// Step 1 of [`Self::load_pending`]: apply finished decodes.
    /// A failed decode marks its entry as failed (the cell stays visible,
    /// dimmed, with an error glyph) instead of deleting it, so the grid count
    /// never lies.
    /// Textures upload here because that needs the main thread.
    fn drain_results(&mut self, rl: &mut RaylibHandle, thread: &RaylibThread, keep: &HashSet<u64>) {
        while let Ok(res) = self.result_rx.try_recv() {
            self.inflight -= 1;
            let Some(i) = self.entries.iter().position(|e| e.id == res.id) else {
                continue; // entry was spliced out meanwhile; drop the result
            };
            match res.res {
                Ok(ok) if self.entries[i].texture.is_none() => {
                    let DecodeOk { thumb, blur, full } = ok;
                    match crate::upload_rgba(rl, thread, &thumb.data, thumb.width, thumb.height) {
                        Ok(t) => {
                            let e = &mut self.entries[i];
                            e.width = full.width;
                            e.height = full.height;
                            e.texture = Some(t);
                            e.blur = blur;
                            e.queued = false;
                            if keep.contains(&e.id) {
                                match crate::upload_rgba(
                                    rl,
                                    thread,
                                    &full.data,
                                    full.width,
                                    full.height,
                                ) {
                                    Ok(ft) => e.full = Some(ft),
                                    Err(err) => {
                                        // Non-fatal: the thumb still shows.
                                        // Opening falls back to streaming.
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
                Ok(DecodeOk { full, .. }) => {
                    // Re-decode of a thumb-only entry (keep-set refill).
                    // The thumb already exists, so the code wants only the
                    // full-res texture, and only if the entry still stays
                    // in the keep set (the selection possibly moved on
                    // meanwhile).
                    self.entries[i].queued = false;
                    if keep.contains(&res.id)
                        && let Some(e) = self.entries.get_mut(i)
                        && e.full.is_none()
                    {
                        match crate::upload_rgba(rl, thread, &full.data, full.width, full.height) {
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
        // selection or open entry and its neighbors moved on).
        // Thumb textures stay.
        // The GPU context lives on this thread, so unloading is safe.
        for e in &mut self.entries {
            if e.full.is_some() && !keep.contains(&e.id) {
                drop(e.full.take());
            }
        }
    }

    /// Step 2 of [`Self::load_pending`]: dispatch new jobs.
    /// The code dispatches `priority` indices first (the selection or the
    /// open image neighbors), then a wraparound scan from `scan_start`.
    /// Jobs run on the shared rayon pool, so several decodes proceed in
    /// parallel.
    /// Each job sends its result back over the channel for `drain_results` to
    /// upload.
    /// The per-entry checks below (texture present, queued, viewing) make
    /// revisiting a priority index in the wraparound harmless.
    /// The code needs no extra dedup bookkeeping: O(1) per entry.
    fn dispatch(&mut self, priority: &[usize], scan_start: usize, scan: bool, keep: &HashSet<u64>) {
        let n = self.entries.len();
        let start = scan_start.min(n);
        let order = priority.iter().copied().filter(|&i| i < n).chain(
            scan.then(|| (start..n).chain(0..start))
                .into_iter()
                .flatten(),
        );

        for i in order {
            if self.inflight >= max_inflight() {
                break;
            }
            let e = &mut self.entries[i];
            if e.queued || e.viewing || e.failed.is_some() {
                continue;
            }
            // Decode when there is no thumb yet.
            // Re-decode a thumb-only entry when it stays in the keep set but
            // its full-res texture is missing (its decode drained outside
            // the keep set, so the code dropped the full texture then).
            // The code never re-decodes thumb-only entries outside the keep
            // set.
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
                let res = crate::decode_image(&path).map(|full| {
                    let blur = blur_enabled.then(|| blurbg::small_blur(&full, blur_px));
                    let thumb = crate::downscale_rgba(full.clone(), THUMB_LONG_SIDE);
                    DecodeOk { thumb, blur, full }
                });
                // Receiver gone (grid dropped): the code discards the
                // result and the job simply ends.
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

    /// The grid index of the entry with this stable id.
    /// The ids stay stable for the grid lifetime, and the indices do not.
    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.entries.iter().position(|e| e.id == id)
    }

    /// Mark the entry with this stable id as failed (e.g. a streaming load
    /// that errored in the middle).
    /// The cell stays, dimmed, with an error glyph.
    pub fn mark_failed(&mut self, id: u64, err: String) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.queued = false;
            e.failed = Some(err);
        }
    }

    /// The indices of the grid neighbors of `sel` (left, right, up, down, in
    /// that order), as prefetch priority.
    /// The function needs the window size for the column count.
    /// It skips duplicates and out-of-range indices.
    pub fn prefetch_neighbors(&self, sel: usize, win_w: f32, win_h: f32) -> Vec<usize> {
        let n = self.entries.len();
        if n == 0 {
            return Vec::new();
        }
        let cols = self.layout(win_w, win_h).cols;
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

    /// Scroll the selection into view.
    /// The code uses this when it sets the selection (or the layout changed)
    /// from outside the navigation code, for example when returning to the
    /// grid from image view, or after a +/- zoom step.
    pub fn ensure_visible(&mut self, win_w: f32, win_h: f32) {
        if self.entries.is_empty() {
            return;
        }
        let l = self.layout(win_w, win_h);
        let scroll_max = (l.content_h - win_h).max(0.0);
        let row = self.selected / l.cols;
        let top = MARGIN + row as f32 * (l.cell_h + GAP);
        let bottom = top + l.cell_h;
        if top - self.scroll < MARGIN {
            self.scroll = top - MARGIN;
        }
        if bottom - self.scroll > win_h - MARGIN {
            self.scroll = bottom - (win_h - MARGIN);
        }
        self.scroll = self.scroll.clamp(0.0, scroll_max);
    }

    /// Grid navigation.
    /// h/j/k/l and the arrows move the selection, with auto-repeat while
    /// held at the X server rate (xset r rate).
    /// g/G jump to the first or last image.
    /// Enter opens the selected image and q quits.
    /// ESC does nothing here, because the grid is the home view.
    ///
    /// `None` means no action this frame.
    // One frame input pipeline (queue drain, repeat, zoom, scroll,
    // navigation, mouse) is intentionally linear.
    // See main() too.
    #[allow(clippy::too_many_lines)]
    pub fn handle_input(
        &mut self,
        rl: &mut RaylibHandle,
        win_w: f32,
        win_h: f32,
    ) -> Option<GridAction> {
        if self.entries.is_empty() {
            return None;
        }
        // Drain the raw key queue instead of is_key_pressed().
        // A press whose release lands within the same frame is invisible to
        // is_key_pressed (raylib snapshots the current and previous key
        // state once per frame, so a press+release pair between two polls
        // nets out to 0->0).
        // This case is easy to hit here: frames stall on full-res texture
        // uploads, and a crisp Enter tap fits inside one.
        // GetKeyPressed() and GetCharPressed() queue every press event
        // during the poll, so nothing is lost.
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
        let mut open_at_cursor = false;
        while let Some(k) = rl.get_key_pressed() {
            match k {
                KeyboardKey::KEY_ENTER | KeyboardKey::KEY_KP_ENTER => enter = true,
                KeyboardKey::KEY_Q => quit = true,
                KeyboardKey::KEY_F => open_at_cursor = true,
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
        // as a character event ('\n'/ '\r') instead of, or in addition to, a
        // key-press event.
        // The code accepts both.
        // This drain also covers the +/- zoom characters, so nothing piles
        // up.
        // The code drains the queue every frame, so unmatched chars never
        // accumulate.
        while let Some(c) = rl.get_char_pressed() {
            match c {
                '\n' | '\r' => enter = true,
                '+' => zoom_in = true,
                '-' => zoom_out = true,
                _ => {}
            }
        }
        // Auto-repeat for the direction keys: the initial press fires
        // at once (edge from the queue above).
        // Holding fires at the X server repeat rate after its delay (xset r
        // rate values).
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
        // +/- zoom the thumbnails (the same 25% steps and key detection as
        // the image-view free zoom: the US-layout =/- keycodes, numpad
        // included, plus the typed character for non-US layouts).
        // The default zoom is the exact-fill layout.
        // Zooming out re-fits with more, smaller thumbnails (still an exact
        // fill).
        // Zooming in overflows the window vertically and enables scrolling.
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
        let l = self.layout(win_w, win_h);
        let (cols, side) = (l.cols, l.side);
        let scroll_max = (l.content_h - win_h).max(0.0);
        // Mouse wheel scrolls the grid (only meaningful once zoomed in past
        // the exact-fill layout, where nothing overflows).
        let wheel = rl.get_mouse_wheel_move();
        if wheel != 0.0 && scroll_max > 0.0 {
            self.scroll = (self.scroll - wheel * (side + GAP) * 3.0).clamp(0.0, scroll_max);
        }
        let old_sel = self.selected;
        let mut sel = self.selected;
        // h/l wrap between rows: l on a row's rightmost element moves to the
        // next row's first element, h on the leftmost moves back to the end of
        // the previous row (never past the first or last element).
        if left {
            if sel.is_multiple_of(cols) {
                if sel > 0 {
                    sel = sel.saturating_sub(1); // first element of a row -> last of the previous
                }
            } else {
                sel -= 1;
            }
        }
        if right {
            if sel % cols == cols - 1 {
                let next = sel + (cols - sel % cols); // first element of the next row
                if next < n {
                    sel = next;
                }
            } else if sel + 1 < n {
                sel += 1;
            }
        }
        if up && sel >= cols {
            sel -= cols;
        }
        if down && sel + cols < n {
            sel += cols;
        }
        // g/G: jump to the first or last image (overrides held direction
        // keys).
        if jump_first {
            sel = 0;
        }
        if jump_last {
            sel = n - 1;
        }
        self.selected = sel;
        // Keep the selection on screen after it moved or the layout changed
        // under it (+/- zoom).
        // The code leaves manual wheel scrolling untouched.
        if self.selected != old_sel || follow {
            self.ensure_visible(win_w, win_h);
        }
        if enter {
            return Some(GridAction::Open(self.selected));
        }
        if quit {
            return Some(GridAction::Quit);
        }
        // Mouse: a click selects a cell.
        // Clicking the already-selected cell opens it (first click selects,
        // second opens, in the nsxiv style).
        // `f` opens the cell under the cursor directly (a double-click
        // without the clicking: select and open in one step).
        if rl.is_mouse_button_pressed(MouseButton::MOUSE_BUTTON_LEFT)
            && let Some(idx) = self.cell_at(rl.get_mouse_position(), win_w, win_h)
        {
            if idx == self.selected {
                return Some(GridAction::Open(idx));
            }
            self.selected = idx;
            self.ensure_visible(win_w, win_h);
        }
        if open_at_cursor && let Some(idx) = self.cell_at(rl.get_mouse_position(), win_w, win_h) {
            self.selected = idx;
            self.ensure_visible(win_w, win_h);
            return Some(GridAction::Open(idx));
        }
        None
    }

    /// The grid cell under the window-space point, if any.
    fn cell_at(&self, m: Vector2, win_w: f32, win_h: f32) -> Option<usize> {
        let l = self.layout(win_w, win_h);
        let (cw, ch) = (l.cell_w, l.cell_h);
        // Grid coordinates: the code draws content at MARGIN + col*(cw+GAP)
        // minus scroll, so add scroll back to the cursor position.
        let (mx, my) = (m.x - MARGIN, m.y + self.scroll - MARGIN);
        let col = (mx / (cw + GAP)).floor();
        let row = (my / (ch + GAP)).floor();
        if col >= 0.0
            && row >= 0.0
            && col < l.cols as f32
            && mx - col * (cw + GAP) <= cw
            && my - row * (ch + GAP) <= ch
        {
            let idx = row as usize * l.cols + col as usize;
            if idx < self.entries.len() {
                return Some(idx);
            }
        }
        None
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
        let l = self.layout(win_w, win_h);
        let (cols, cw, ch, side) = (l.cols, l.cell_w, l.cell_h, l.side);
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
                // The thumb texture keeps the aspect ratio of the image.
                // The square cell shows its center crop (draw-time crop, so a
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
                // Decode failed: dim red placeholder with an error glyph.
                // The file stays visible instead of vanishing silently.
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

/// Do the code most likely decode this path as an image? (Grid directory
/// filter.)
fn is_image_path(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "jpg" | "jpeg" | "png" | "jxl" | "gif" | "webp" | "bmp" | "tif" | "tiff" | "tga"
        )
    })
}

/// The largest exact-fill square thumbnail side for `n` entries in an
/// `aw` x `ah` available area.
/// The code takes the best over all column counts.
/// Ties prefer the column count whose cells stay closest to square, that is,
/// the least empty space.
/// This is also the zoom-1.0 reference size for +/- zooming.
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
/// content height.
/// The content height can exceed the window when zoomed in, and the caller
/// scrolls by the excess.
///
/// At the default zoom the layout is the exact fill described in
/// [`best_fill_side`].
/// Cells divide the available space, so the grid spans the whole window in
/// both dimensions.
///
/// Zooming out shrinks the target thumbnail side and picks the column count
/// whose fill-derived side lands closest to it.
/// The grid still spans the window exactly, now with more smaller thumbnails.
///
/// Zooming in grows the target side past the largest exact-fill size, so the
/// rows no longer fit vertically.
/// Cells stay square at the target side and columns stretch a little so the
/// grid still spans the width.
/// Rows overflow and the caller scrolls.
fn grid_layout_at(n: usize, win_w: f32, win_h: f32, zoom: f32) -> Layout {
    let n = n.max(1);
    let aw = (win_w - 2.0 * MARGIN).max(1.0);
    let ah = (win_h - 2.0 * MARGIN).max(1.0);

    let base = best_fill_side(n, aw, ah);
    let target = (base * zoom).max(GRID_MIN_SIDE);
    if target <= base {
        // Exact fill: pick the column count whose fill-derived side is
        // closest to the target (at zoom 1.0 this reproduces the exact-fill
        // optimum).
        // Ties prefer the least empty space, that is, the smaller
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
        Layout {
            cols,
            cell_w: cw,
            cell_h: ch,
            side: cw.min(ch),
            content_h,
        }
    } else {
        // Overflow: square cells at the target side, as many columns as fit
        // the width.
        // The rows scroll vertically.
        let target = target.min(aw.min(ah));
        let cols = (((aw + GAP) / (target + GAP)).floor() as usize).clamp(1, n);
        let cw = ((aw - (cols - 1) as f32 * GAP) / cols as f32).max(target);
        let rows = n.div_ceil(cols);
        let content_h = 2.0 * MARGIN + rows as f32 * target + (rows - 1) as f32 * GAP;
        Layout {
            cols,
            cell_w: cw,
            cell_h: target,
            side: target,
            content_h,
        }
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
        let l = grid_layout_at(1, 100.0, 100.0, 1.0);
        assert_eq!(l.cols, 1);
        assert_eq!(l.cell_w, 84.0);
        assert_eq!(l.cell_h, 84.0);
        assert_eq!(l.side, 84.0);
        assert_eq!(l.content_h, 100.0);
    }

    #[test]
    fn grid_layout_default_spans_full_window() {
        // At the default zoom, the code sizes cells to divide the available
        // space, so the grid must span the whole window in both dimensions
        // and the content height must equal the window height.
        for (n, win_w, win_h) in [(1, 100.0, 100.0), (7, 1200.0, 700.0), (100, 1600.0, 900.0)] {
            let l = grid_layout_at(n, win_w, win_h, 1.0);
            let rows = n.div_ceil(l.cols);
            let span_w = 2.0 * MARGIN + l.cols as f32 * l.cell_w + (l.cols - 1) as f32 * GAP;
            let span_h = 2.0 * MARGIN + rows as f32 * l.cell_h + (rows - 1) as f32 * GAP;
            assert!((span_w - win_w).abs() < 0.01, "n={n}: span_w={span_w}");
            assert!((span_h - win_h).abs() < 0.01, "n={n}: span_h={span_h}");
            assert_eq!(l.content_h, win_h);
            assert_eq!(l.side, l.cell_w.min(l.cell_h));
        }
    }

    #[test]
    fn grid_layout_never_crashes_on_degenerate_input() {
        let l = grid_layout_at(3, 50.0, 800.0, 1.0);
        assert!(l.cols >= 1 && l.side > 0.0);
        let _ = grid_layout_at(0, 0.0, 0.0, 1.0);
        let _ = grid_layout_at(5, 800.0, 600.0, 0.0);
        let _ = grid_layout_at(5, 800.0, 600.0, 1000.0);
    }

    #[test]
    fn grid_layout_zoom_out_keeps_exact_fill_with_smaller_thumbs() {
        // 9 images in a 640x640 window: default is 3x3 with ~208px thumbs.
        // Zooming out must shrink the thumbs, add columns, and still span
        // the window exactly (no scrolling).
        let default = grid_layout_at(9, 640.0, 640.0, 1.0);
        let l = grid_layout_at(9, 640.0, 640.0, 0.8);
        assert!(
            l.cols > default.cols,
            "cols={}, default={}",
            l.cols,
            default.cols
        );
        assert!(l.side < default.side);
        assert_eq!(l.side, l.cell_w.min(l.cell_h));
        let rows = 9_usize.div_ceil(l.cols);
        let span_w = 2.0 * MARGIN + l.cols as f32 * l.cell_w + (l.cols - 1) as f32 * GAP;
        let span_h = 2.0 * MARGIN + rows as f32 * l.cell_h + (rows - 1) as f32 * GAP;
        assert!((span_w - 640.0).abs() < 0.01, "span_w={span_w}");
        assert!((span_h - 640.0).abs() < 0.01, "span_h={span_h}");
        assert_eq!(l.content_h, 640.0);
    }

    #[test]
    fn grid_layout_zoom_in_overflows_and_grows_thumbs() {
        // Zooming in grows thumbs past the largest exact-fill size: fewer
        // columns, square thumbs at the zoomed size, content taller than
        // the window (scrollable).
        let default = grid_layout_at(9, 640.0, 640.0, 1.0);
        let l = grid_layout_at(9, 640.0, 640.0, 2.0);
        assert!(
            l.cols < default.cols,
            "cols={}, default={}",
            l.cols,
            default.cols
        );
        assert!(l.side > default.side);
        assert_eq!(l.side, l.cell_h);
        assert!(l.cell_w >= l.side);
        assert!(l.content_h > 640.0, "content_h={}", l.content_h);
    }

    #[test]
    fn grid_layout_side_is_square_of_cell_minimum() {
        let l = grid_layout_at(7, 1200.0, 700.0, 1.0);
        assert_eq!(l.side, l.cell_w.min(l.cell_h));
    }

    #[test]
    fn prefetch_neighbors_left_right_up_down() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        for i in 0..9 {
            std::fs::write(dir.join(format!("{i:02}.png")), b"").unwrap();
        }
        let grid = Grid::from_dir(dir).unwrap();
        // Square window: 9 images lay out as a 3x3 grid, so index 4 has
        // neighbors 3 (left), 5 (right), 1 (up), 7 (down).
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
        // Tiny valid PNG (1x1 red) through the image crate.
        let png = image::DynamicImage::new_rgb8(1, 1);
        png.save(dir.join("b.png")).unwrap();
        std::fs::write(dir.join("a.txt"), "not an image").unwrap();
        std::fs::write(dir.join("c.png"), "invalid png content").unwrap(); // listed, the decode fails later
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
