//! Directory thumbnail grid.
//! Square center-crop thumbnails fill the whole window, unlike the fixed
//! thumbnail sizes of nsxiv.
//! A white border marks the selection.
//!
//! The grid has a Vim-style visual selection (see [`Visual`]).
//! `v` selects charwise, `V` linewise, and `Ctrl+v` blockwise.
//! The same key toggles its mode off, `o` swaps the anchor and the cursor,
//! and ESC clears the selection.
//! Every selected cell draws the same white border as the cursor.
//! While a visual selection is active, every cell outside it draws dimmed
//! with the `DIM_TINT` grey, so the selected set stands out.
//! The plain cursor marking dims nothing.
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
    sync::{
        Arc,
        mpsc::{Receiver, Sender, channel},
    },
};

use derive_more::{Display, Error, From};
use raylib::{color::Color, prelude::*};

use crate::{
    DecodedImage,
    blurbg::{self},
    dim::{ImgH, ImgW, LongSide, Rect, WinH, WinW},
    error::DecodeError,
    thumbcache::Cache,
};

const MARGIN: f32 = 8.0;
const GAP: f32 = 8.0;
/// Grid zoom step for +/- (the same 25% steps as the image-view free zoom).
const GRID_ZOOM_STEP: f32 = 1.25;
/// Lower bound for the thumbnail side when the grid zooms out.
const GRID_MIN_SIDE: f32 = 32.0;
/// The tint of a cell that a visual selection excludes.
/// The dim keeps every image visible, while the selected set holds the eye.
const DIM_TINT: Color = Color::new(90, 90, 90, 255);
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
const THUMB_LONG_SIDE: LongSide = LongSide::new(1024);

/// Long side of the thumbnail the cache keeps on disk.
/// A cached entry is smaller than [`THUMB_LONG_SIDE`], so a warm cache shows a
/// softer cell than a fresh decode: the GPU scales the 512 px texture up in a
/// large cell.
/// The trade suits a disk cache: a 512 px thumb is a few hundred kilobytes per
/// file.
/// Bump it with the `thumbcache` format version when the size changes.
const CACHE_LONG_SIDE: LongSide = LongSide::new(512);

/// A finished background decode, matched to an entry by its unique id.
/// Indices shift when entries come and go.
/// Ids never shift.
#[derive(Debug)]
struct DecodeResult {
    id: u64,
    res: Result<DecodeOk, DecodeError>,
}

/// Successful decode payload: the thumbnail and the full-resolution image.
/// [`crate::decode_image`] caps the full image to the uploadable
/// [`crate::MAX_TEXTURE_SIDE`] before it arrives here.
/// The main thread uploads the thumb for every entry.
/// It uploads the full texture only for keep-set entries (selection or open
/// plus prefetched neighbors) and drops the rest.
/// The buffers live only until the next frame drains them, bounded by
/// `max_inflight()`.
#[derive(Debug)]
struct DecodeOk {
    /// The whole image with the aspect preserved, long side
    /// [`THUMB_LONG_SIDE`].
    thumb: DecodedImage,
    /// The source dimensions, as a decode capped at
    /// [`crate::MAX_TEXTURE_SIDE`] reports them.
    /// The entry keeps them as the image size, not the thumb size, so the
    /// image view fits and zooms the picture, not its thumbnail.
    image: (ImgW, ImgH),
    /// The tiny blurred copy for the `VV_BLUR_BG` gimmick (None when off).
    blur: Option<DecodedImage>,
    /// The untouched full-resolution image.
    /// The cache hit path sets it to None, because the entry needs only the
    /// thumbnail.
    full: Option<DecodedImage>,
}

/// A decode job for one entry, as the worker sees it.
/// The worker reads the cache first and decodes on a miss.
#[derive(Debug, Clone)]
struct Job {
    id: u64,
    path: PathBuf,
    /// The job must return the full-resolution image, because the entry holds
    /// a full-res texture: it belongs to the keep set (the selection or open
    /// entry and its prefetched neighbors).
    /// A thumb-only job decodes outside that set.
    want_full: bool,
    /// The `VV_BLUR_BG` gimmick settings, read once by the grid.
    blur_enabled: bool,
    blur_px: LongSide,
}

/// The three Vim visual selection styles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VisualMode {
    /// A contiguous index range from the anchor to the cursor.
    Char,
    /// Every whole grid row from the anchor row to the cursor row.
    Line,
    /// The rectangle spanned by the anchor column and the cursor column.
    Block,
}

/// The grid visual selection, in the style of Vim.
///
/// The code stores the anchor index and derives the selected set from the
/// current layout every frame.
/// A window resize or a +/- zoom therefore reflows the selection, exactly
/// like Vim reflowing a wrapped selection.
#[derive(Debug, Clone, Copy)]
struct Visual {
    mode: VisualMode,
    /// The index where the selection started.
    anchor: usize,
}

impl Visual {
    /// The selection after a visual key press for `mode` at `cursor`.
    ///
    /// A press for the active mode ends the selection, as in Vim.
    /// A press for another mode keeps the anchor, so `v` on row 1 then down
    /// two rows then `V` selects rows 1..=3.
    /// With no active selection the press starts one at `cursor`.
    #[must_use]
    fn pressed(current: Option<Self>, mode: VisualMode, cursor: usize) -> Option<Self> {
        match current {
            Some(v) if v.mode == mode => None,
            Some(v) => Some(Visual {
                mode,
                anchor: v.anchor,
            }),
            None => Some(Visual {
                mode,
                anchor: cursor,
            }),
        }
    }

    /// Is entry `i` inside the selection, with the cursor at `cursor`?
    /// `cols` is the current column count, which decides rows and columns.
    #[must_use]
    fn contains(self, i: usize, cursor: usize, cols: usize) -> bool {
        let (anchor, cursor) = (self.anchor, cursor);
        let (lo, hi) = (anchor.min(cursor), anchor.max(cursor));
        match self.mode {
            VisualMode::Char => (lo..=hi).contains(&i),
            VisualMode::Line => {
                let (first, last) = (lo / cols, hi / cols);
                (first..=last).contains(&(i / cols))
            }
            VisualMode::Block => {
                let (first_row, last_row) = (lo / cols, hi / cols);
                let (first_col, last_col) = (
                    (anchor % cols).min(cursor % cols),
                    (anchor % cols).max(cursor % cols),
                );
                (first_row..=last_row).contains(&(i / cols))
                    && (first_col..=last_col).contains(&(i % cols))
            }
        }
    }
}

/// Why the grid could not open a directory.
#[derive(Debug, Display, Error, From)]
pub enum GridError {
    /// The directory could not be read.
    #[display("failed to read directory")]
    ReadDir(std::io::Error),
}

/// Grid layout, in window pixels.
#[derive(Debug, Clone, Copy)]
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

/// Why a grid entry failed to load.
///
/// The code keeps the failed entry visible as a dimmed cell with an error
/// glyph, so the grid count never lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[error(ignore)]
pub enum DecodeFailure {
    /// The background decode failed (bad bytes, unsupported content, I/O).
    #[display("decode failed")]
    Decode,
    /// The GL texture upload failed (no context, driver refusal).
    #[display("texture upload failed")]
    TextureUpload,
}

/// One grid cell: the image file and its uploaded full-resolution texture.
/// The code draws thumbs by cropping a center square, so a resize never
/// needs a re-decode.
/// The GPU scales the texture down every frame.
#[derive(Debug)]
pub struct GridEntry {
    pub path: PathBuf,
    /// Full-resolution image dimensions (as decoded, not the thumb's).
    pub width: ImgW,
    pub height: ImgH,
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
    pub failed: Option<DecodeFailure>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridAction {
    /// Open the grid entry at this index in image mode.
    Open(usize),
    /// Quit the viewer.
    Quit,
}

/// The thumbnail grid, its entry list and its background decode workers.
#[derive(Debug)]
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
    /// The active Vim-style visual selection, if any.
    visual: Option<Visual>,
    /// The cached layout, keyed by (entry count, window size, zoom).
    /// The code recomputes it only when one of those changes, because the
    /// layout math is O(n).
    /// The field uses interior mutability because `draw` takes &self.
    layout_cache: RefCell<Option<(LayoutKey, Layout)>>,
    /// The `VV_BLUR_BG` gimmick settings, read once.
    /// Decode workers compute the tiny blurred copy only when enabled, at
    /// this texture long side.
    blur_enabled: bool,
    blur_px: LongSide,
    /// The thumbnail cache, shared with every decode worker.
    cache: Arc<Cache>,
}

/// Run one decode job on a worker: the cache first, the file on a miss.
fn run_job(job: &Job, cache: &Cache) -> Result<DecodeOk, DecodeError> {
    // A miss is normal and stays silent, because the first launch of a
    // directory misses once per file.
    if !job.want_full
        && let Some(hit) = cache.load(&job.path, CACHE_LONG_SIDE).ok()
    {
        // The cached thumb already sits below [`THUMB_LONG_SIDE`], so the
        // code uploads it as is and the GPU scales it in the cell.
        // A resample here would copy the buffer for nothing.
        let blur = job
            .blur_enabled
            .then(|| blurbg::small_blur(&hit.thumb, job.blur_px));
        return Ok(DecodeOk {
            thumb: hit.thumb,
            image: hit.image,
            blur,
            full: None,
        });
    }
    let full = crate::decode_image(&job.path)?;
    let blur = job
        .blur_enabled
        .then(|| blurbg::small_blur(&full, job.blur_px));
    // Borrow the decoded buffer for the thumb: a clone of a large image costs
    // hundreds of megabytes.
    let thumb = crate::downscale_rgba_ref(&full.data, full.width, full.height, THUMB_LONG_SIDE);
    store_thumb(cache, job, &thumb, full.width, full.height);
    let image = (full.width, full.height);
    // The keep-set refill is the only job that carries the full image to the
    // main thread; otherwise the large buffer drops here.
    let full = job.want_full.then_some(full);
    Ok(DecodeOk {
        thumb,
        image,
        blur,
        full,
    })
}

/// Put the thumbnail of a fresh decode into the cache.
/// A cache failure stays silent: the grid already holds the thumbnail, and a
/// missing cache home is a normal setup.
fn store_thumb(cache: &Cache, job: &Job, thumb: &DecodedImage, width: ImgW, height: ImgH) {
    // One resize for the disk copy: the runtime thumb stays at full quality.
    let cached = crate::downscale_rgba_ref(&thumb.data, thumb.width, thumb.height, CACHE_LONG_SIDE);
    let _ = cache.store(&job.path, CACHE_LONG_SIDE, &cached, (width, height));
}

impl Grid {
    /// List the images in a directory (sorted by file name) and start the
    /// background decode worker.
    ///
    /// # Errors
    ///
    /// The function errors when it cannot read the directory.
    pub fn from_dir(dir: &Path) -> Result<Grid, GridError> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    // Dotfiles are hidden, like nsxiv and file managers do.
                    && !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.'))
                    && is_image_path(p)
            })
            .collect();
        paths.sort_by(|a, b| {
            // Natural order, so img2 sorts before img10.
            let name = |p: &PathBuf| p.file_name().map(|n| n.to_string_lossy().to_lowercase());
            natord::compare_ignore_case(&name(a).unwrap_or_default(), &name(b).unwrap_or_default())
        });

        let (res_tx, res_rx) = channel::<DecodeResult>();

        Ok(Grid {
            entries: paths
                .into_iter()
                .enumerate()
                .map(|(i, path)| GridEntry {
                    path,
                    width: ImgW::new(0),
                    height: ImgH::new(0),
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
            visual: None,
            result_tx: res_tx,
            result_rx: res_rx,
            inflight: 0,
            layout_cache: RefCell::new(None),
            rep_dir: Default::default(),
            zoom: 1.0,
            scroll: 0.0,
            blur_enabled: blurbg::enabled(),
            blur_px: blurbg::blur_px(),
            cache: Arc::new(Cache::new()),
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
                    let DecodeOk {
                        thumb,
                        image,
                        blur,
                        full,
                    } = ok;
                    match crate::upload_rgba(rl, thread, &thumb.data, thumb.width, thumb.height) {
                        Ok(tex) => {
                            let e = &mut self.entries[i];
                            e.texture = Some(tex);
                            e.width = image.0;
                            e.height = image.1;
                            e.blur = blur;
                            e.queued = false;
                            if keep.contains(&e.id)
                                && let Some(full) = full
                            {
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
                            e.failed = Some(DecodeFailure::TextureUpload);
                        }
                    }
                }
                Ok(DecodeOk {
                    image,
                    full: Some(full),
                    ..
                }) => {
                    // Re-decode of a thumb-only entry (keep-set refill).
                    // The thumb already exists, so the code wants only the
                    // full-res texture, and only if the entry still stays
                    // in the keep set (the selection possibly moved on
                    // meanwhile).
                    // The dimensions land with a cache hit too, and they
                    // decide how the image view fits the picture.
                    self.entries[i].queued = false;
                    // The source dimensions land even when the entry left the
                    // keep set, because the image view reads them either way.
                    self.entries[i].width = image.0;
                    self.entries[i].height = image.1;
                    if keep.contains(&res.id)
                        && let Some(e) = self.entries.get_mut(i)
                        && e.full.is_none()
                    {
                        match crate::upload_rgba(rl, thread, &full.data, full.width, full.height) {
                            Ok(ft) => e.full = Some(ft),
                            Err(err) => {
                                eprintln!("vv: {}: {err:#}", e.path.display());
                                e.failed = Some(DecodeFailure::TextureUpload);
                            }
                        }
                    }
                }
                Ok(DecodeOk {
                    image, full: None, ..
                }) => {
                    // A cache hit for an entry that already shows a thumb:
                    // nothing is left to upload, but the source dimensions
                    // land, because the cached thumb carries them.
                    // The image view then fits the picture, not its thumb.
                    let e = &mut self.entries[i];
                    e.queued = false;
                    e.width = image.0;
                    e.height = image.1;
                }
                Err(err) => {
                    eprintln!(
                        "vv: {}: decode failed: {err}",
                        self.entries[i].path.display()
                    );
                    let e = &mut self.entries[i];
                    e.queued = false;
                    e.failed = Some(DecodeFailure::Decode);
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
            let job = Job {
                id: e.id,
                path: e.path.clone(),
                // Only a keep-set entry needs the full texture. A cold entry
                // outside it decodes for its thumbnail only.
                want_full: keep.contains(&e.id),
                blur_enabled: self.blur_enabled,
                blur_px: self.blur_px,
            };
            let tx = self.result_tx.clone();
            let cache = Arc::clone(&self.cache);
            rayon::spawn(move || {
                let id = job.id;
                let res = run_job(&job, &cache);
                // Receiver gone (grid dropped): the code discards the
                // result and the job simply ends.
                let _ = tx.send(DecodeResult { id, res });
            });
        }
    }

    /// Layout for the current state, via the cache (see `layout_cache`).
    fn layout(&self, win_w: WinW, win_h: WinH) -> Layout {
        let n = self.entries.len();
        let key = (
            n,
            win_w.get().to_bits(),
            win_h.get().to_bits(),
            self.zoom.to_bits(),
        );
        let mut cache = self.layout_cache.borrow_mut();
        if cache.as_ref().is_none_or(|(k, _)| *k != key) {
            *cache = Some((key, grid_layout_at(n, win_w.get(), win_h.get(), self.zoom)));
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
    pub fn mark_failed(&mut self, id: u64, err: DecodeFailure) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.queued = false;
            e.failed = Some(err);
        }
    }

    /// The indices of the grid neighbors of `sel` (left, right, up, down, in
    /// that order), as prefetch priority.
    /// The function needs the window size for the column count.
    /// It skips duplicates and out-of-range indices.
    #[must_use]
    pub fn prefetch_neighbors(&self, sel: usize, win_w: WinW, win_h: WinH) -> Vec<usize> {
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
    pub fn ensure_visible(&mut self, win_w: WinW, win_h: WinH) {
        if self.entries.is_empty() {
            return;
        }
        let l = self.layout(win_w, win_h);
        let (_, win_h) = (win_w.get(), win_h.get());
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
    /// `v`, `V`, and `Ctrl+v` start (or stop) the charwise, linewise, and
    /// blockwise visual selection at the current cell.
    /// A press for another mode keeps the anchor, so the line selection
    /// covers every row from the anchor row to the cursor row.
    /// `o` swaps the anchor and the cursor, as in Vim.
    /// ESC clears the visual selection, and stays inert when there is none,
    /// because the grid is the home view.
    /// `step` scales one press: Shift passes
    /// [`crate::SHIFT_SPEED_MULTIPLIER`], so the selection jumps that many
    /// cells.
    ///
    /// `None` means no action this frame.
    //
    // One frame input pipeline (queue drain, repeat, zoom, scroll,
    // navigation, mouse) is intentionally linear.
    // See main() too.
    #[allow(clippy::too_many_lines)]
    pub fn handle_input(
        &mut self,
        rl: &mut RaylibHandle,
        win: (WinW, WinH),
        step: u32,
    ) -> Option<GridAction> {
        let (win_w, win_h) = (win.0.get(), win.1.get());
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
        let mut escape = false;
        let mut start_visual: Option<VisualMode> = None;
        let mut swap_ends = false;
        while let Some(k) = rl.get_key_pressed() {
            match k {
                KeyboardKey::KEY_ENTER | KeyboardKey::KEY_KP_ENTER => enter = true,
                KeyboardKey::KEY_Q => quit = true,
                KeyboardKey::KEY_ESCAPE => escape = true,
                KeyboardKey::KEY_V => {
                    // Vim: `v` charwise, `V` linewise, `Ctrl+v` blockwise.
                    // The same key again ends the visual selection, and a
                    // different key switches its mode in place.
                    let mode = if rl.is_key_down(KeyboardKey::KEY_LEFT_CONTROL)
                        || rl.is_key_down(KeyboardKey::KEY_RIGHT_CONTROL)
                    {
                        VisualMode::Block
                    } else if rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
                        || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT)
                    {
                        VisualMode::Line
                    } else {
                        VisualMode::Char
                    };
                    start_visual = Some(mode);
                }
                KeyboardKey::KEY_O => swap_ends = true,
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
        if let Some(mode) = start_visual {
            self.visual = Visual::pressed(self.visual, mode, self.selected);
        }
        if escape && self.visual.is_some() {
            self.visual = None;
            // Consume the frame: ESC only clears the selection here.
            return None;
        }
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
        let l = self.layout(win.0, win.1);
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
        // `step` scales every press: Shift passes
        // [`crate::SHIFT_SPEED_MULTIPLIER`], so one press moves that many
        // cells.
        // The selection stays inside the entry list.
        let s = step.max(1) as usize;
        let cols_i = cols as i128;
        if left {
            sel = step_selection(sel, n, s, -1);
        }
        if right {
            sel = step_selection(sel, n, s, 1);
        }
        if up {
            sel = step_selection(sel, n, s, -cols_i);
        }
        if down {
            sel = step_selection(sel, n, s, cols_i);
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
        // `o` swaps the anchor and the cursor, so the far end keeps growing
        // the selection, as in Vim.
        if swap_ends && let Some(v) = self.visual.as_mut() {
            v.anchor = std::mem::replace(&mut self.selected, v.anchor);
        }
        // Keep the selection on screen after it moved or the layout changed
        // under it (+/- zoom).
        // The code leaves manual wheel scrolling untouched.
        if self.selected != old_sel || follow {
            self.ensure_visible(win.0, win.1);
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
            && let Some(idx) = self.cell_at(rl.get_mouse_position(), win.0, win.1)
        {
            if idx == self.selected {
                return Some(GridAction::Open(idx));
            }
            self.selected = idx;
            self.ensure_visible(win.0, win.1);
        }
        if open_at_cursor && let Some(idx) = self.cell_at(rl.get_mouse_position(), win.0, win.1) {
            self.selected = idx;
            self.ensure_visible(win.0, win.1);
            return Some(GridAction::Open(idx));
        }
        None
    }

    /// The grid cell under the window-space point, if any.
    fn cell_at(&self, m: Vector2, win_w: WinW, win_h: WinH) -> Option<usize> {
        let l = self.layout(win_w, win_h);
        let (cw, ch) = (l.cell_w, l.cell_h);
        // Grid coordinates: the code draws content at MARGIN + col*(cw+GAP)
        // minus scroll, so add scroll back to the cursor position.
        let (mx, my) = (m.x - MARGIN, m.y + self.scroll - MARGIN);
        let col = (mx / (cw + GAP)).floor();
        let row = (my / (ch + GAP)).floor();
        let cell = Rect::new(col * (cw + GAP), row * (ch + GAP), cw, ch);
        if col >= 0.0 && row >= 0.0 && col < l.cols as f32 && cell.contains(mx, my) {
            let idx = row as usize * l.cols + col as usize;
            if idx < self.entries.len() {
                return Some(idx);
            }
        }
        None
    }

    /// Draw the grid: the placeholder text or every visible thumbnail cell.
    pub fn draw(&self, d: &mut RaylibDrawHandle, win: (WinW, WinH)) {
        let (win_w, win_h) = (win.0.get(), win.1.get());
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
        let l = self.layout(win.0, win.1);
        let (cols, cw, ch, side) = (l.cols, l.cell_w, l.cell_h, l.side);
        let visual = self.visual;
        // A visual selection dims every cell outside it, so the selected set
        // stands out. Plain cursor marking dims nothing.
        let dim_others = visual.is_some();
        for (i, e) in self.entries.iter().enumerate() {
            let col = i % cols;
            let row = i / cols;
            let cx = MARGIN + col as f32 * (cw + GAP);
            let cy = MARGIN + row as f32 * (ch + GAP) - self.scroll;
            // Offscreen (scrolled out) cells: skip the draw work.
            if cy + ch < 0.0 || cy > win_h {
                continue;
            }
            let thumb = Rect::new(cx + (cw - side) / 2.0, cy + (ch - side) / 2.0, side, side);
            let in_selection =
                i == self.selected || visual.is_some_and(|v| v.contains(i, self.selected, cols));
            let tint = if dim_others && !in_selection {
                DIM_TINT
            } else {
                Color::WHITE
            };
            if let Some(tex) = &e.texture {
                // The thumb texture keeps the aspect ratio of the image.
                // The square cell shows its center crop (draw-time crop, so a
                // resize never needs a re-decode).
                let src = Rect::sized(tex.width() as f32, tex.height() as f32).center_square();
                d.draw_texture_pro(tex, src, thumb, Vector2::ZERO, 0.0, tint);
            } else if e.failed.is_some() {
                // Decode failed: dim red placeholder with an error glyph.
                // The file stays visible instead of vanishing silently.
                d.draw_rectangle_rec(thumb, Color::new(72, 26, 26, 255).tint(tint));
                let msg = "!";
                let tw = d.measure_text(msg, 24);
                d.draw_text(
                    msg,
                    (thumb.x + (thumb.width - tw as f32) / 2.0) as i32,
                    (thumb.y + (thumb.height - 24.0) / 2.0) as i32,
                    24,
                    Color::new(214, 92, 92, 255).tint(tint),
                );
            } else {
                // Still decoding: placeholder square.
                d.draw_rectangle_rec(thumb, Color::new(28, 28, 28, 255).tint(tint));
            }
            if in_selection {
                d.draw_rectangle_lines_ex(thumb, 4.0, Color::WHITE);
            }
        }
    }
}

/// Move the selection by one direction press, `step` times over.
///
/// `delta` is the index change of a single step: -1 and +1 for h and l, and
/// -cols and +cols for k and j.
/// The result stays inside `0..n`, and it clamps instead of wrapping.
/// A step of 0 counts as 1.
#[must_use]
fn step_selection(sel: usize, n: usize, step: usize, delta: i128) -> usize {
    if n == 0 {
        return 0;
    }
    let step = step.max(1);
    let moved = sel as i128 + delta * step as i128;
    moved.clamp(0, n as i128 - 1) as usize
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
    fn a_fresh_decode_fills_the_cache_and_hits_it_next() {
        let dir = TempDir::new().unwrap();
        let png = dir.path().join("a.png");
        image::DynamicImage::new_rgb8(64, 32)
            .save(&png)
            .expect("save");
        // A cache over the temporary directory, so the test never touches the
        // real thumbnail cache of the user.
        let cache = Cache::in_dir(dir.path().join("cache"));
        let job = Job {
            id: 0,
            path: png,
            want_full: false,
            blur_enabled: false,
            blur_px: LongSide::new(64),
        };
        let miss = run_job(&job, &cache).expect("decode");
        assert!(miss.full.is_none(), "a thumb-only job keeps no full image");
        assert_eq!(miss.image, (ImgW::new(64), ImgH::new(32)));
        assert!(
            cache.load(&job.path, CACHE_LONG_SIDE).is_ok(),
            "the decode stores a thumbnail"
        );
        // The second job takes the cached thumbnail and decodes nothing.
        let hit = run_job(&job, &cache).expect("cache hit");
        assert!(hit.full.is_none());
        assert_eq!(hit.thumb, miss.thumb);
        // The hit still reports the source dimensions, so the image view fits
        // the picture and not the cached thumb.
        assert_eq!(hit.image, miss.image);
        // A keep-set job ignores the cache, because it needs the full image.
        let keep = Job {
            want_full: true,
            ..job.clone()
        };
        let refill = run_job(&keep, &cache).expect("decode");
        assert!(refill.full.is_some());
        assert_eq!(refill.image, miss.image);
    }

    #[test]
    fn mark_failed_keeps_the_entry() {
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let png = image::DynamicImage::new_rgb8(1, 1);
        png.save(dir.join("a.png")).unwrap();
        let mut grid = Grid::from_dir(dir).unwrap();
        assert_eq!(grid.entries.len(), 1);
        let id = grid.entries[0].id;
        grid.mark_failed(id, DecodeFailure::Decode);
        assert_eq!(grid.entries.len(), 1, "failed entry stays in the grid");
        assert_eq!(grid.entries[0].failed, Some(DecodeFailure::Decode));
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
    fn visual_mode_switch_keeps_the_anchor() {
        // 3x3 grid: start charwise on row 0, move the cursor two rows down,
        // then press V. The line selection must cover rows 0..=2.
        let v = Visual::pressed(None, VisualMode::Char, 1).unwrap();
        let v = Visual::pressed(Some(v), VisualMode::Line, 7).unwrap();
        assert_eq!(v.anchor, 1);
        assert_eq!(v.mode, VisualMode::Line);
        let picked: Vec<usize> = (0..9).filter(|&i| v.contains(i, 7, 3)).collect();
        assert_eq!(picked, (0..9).collect::<Vec<_>>());
        // The same mode again ends the selection.
        assert!(Visual::pressed(Some(v), VisualMode::Line, 7).is_none());
    }

    #[test]
    fn visual_char_selects_a_contiguous_range() {
        let v = Visual {
            mode: VisualMode::Char,
            anchor: 4,
        };
        // 3x3 grid: cursor below and to the right of the anchor.
        let picked: Vec<usize> = (0..9).filter(|&i| v.contains(i, 7, 3)).collect();
        assert_eq!(picked, vec![4, 5, 6, 7]);
        // The range works in both directions.
        let picked: Vec<usize> = (0..9).filter(|&i| v.contains(i, 2, 3)).collect();
        assert_eq!(picked, vec![2, 3, 4]);
    }

    #[test]
    fn visual_line_selects_whole_rows() {
        let v = Visual {
            mode: VisualMode::Line,
            anchor: 4,
        };
        // 3x3 grid: rows 1..=2, every column.
        let picked: Vec<usize> = (0..9).filter(|&i| v.contains(i, 6, 3)).collect();
        assert_eq!(picked, vec![3, 4, 5, 6, 7, 8]);
        // The cursor in the same row selects that row only.
        let picked: Vec<usize> = (0..9).filter(|&i| v.contains(i, 3, 3)).collect();
        assert_eq!(picked, vec![3, 4, 5]);
    }

    #[test]
    fn visual_block_selects_a_rectangle() {
        let v = Visual {
            mode: VisualMode::Block,
            anchor: 1,
        };
        // 3x3 grid: anchor column 1, cursor column 2, rows 0..=2.
        let picked: Vec<usize> = (0..9).filter(|&i| v.contains(i, 8, 3)).collect();
        assert_eq!(picked, vec![1, 2, 4, 5, 7, 8]);
        // A narrower block over two rows.
        let picked: Vec<usize> = (0..9).filter(|&i| v.contains(i, 4, 3)).collect();
        assert_eq!(picked, vec![1, 4]);
    }

    #[test]
    fn visual_block_clamps_the_last_partial_row() {
        // 5 entries in a 3-column grid: the last row holds 2 cells.
        let v = Visual {
            mode: VisualMode::Block,
            anchor: 1,
        };
        let picked: Vec<usize> = (0..5).filter(|&i| v.contains(i, 4, 3)).collect();
        assert_eq!(picked, vec![1, 4]);
    }

    #[test]
    fn step_selection_doubles_with_shift() {
        // 3x3 grid over 9 entries, selection 4 (the middle cell).
        // A normal press moves one cell, and Shift moves two.
        assert_eq!(step_selection(4, 9, 1, 1), 5);
        assert_eq!(step_selection(4, 9, 2, 1), 6);
        assert_eq!(step_selection(4, 9, 1, -1), 3);
        assert_eq!(step_selection(4, 9, 2, -1), 2);
        // A row step is the column count apart, doubled with Shift.
        assert_eq!(step_selection(4, 9, 1, 3), 7);
        // The clamp holds at both ends instead of wrapping.
        assert_eq!(step_selection(8, 9, 2, 1), 8);
        assert_eq!(step_selection(0, 9, 2, -1), 0);
        assert_eq!(step_selection(1, 9, 2, -3), 0);
        // A step of zero still moves one cell.
        assert_eq!(step_selection(4, 9, 0, 1), 5);
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
        assert_eq!(
            grid.prefetch_neighbors(4, WinW::new(640.0), WinH::new(640.0)),
            vec![3, 5, 1, 7]
        );
        // Top-left corner: only right (1) and down (3) exist.
        assert_eq!(
            grid.prefetch_neighbors(0, WinW::new(640.0), WinH::new(640.0)),
            vec![1, 3]
        );
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

    #[test]
    fn from_dir_sorts_naturally() {
        // Natural order: img2 comes before img10, as in file managers.
        let dir = TempDir::new().unwrap();
        let dir = dir.path();
        let png = image::DynamicImage::new_rgb8(1, 1);
        for name in ["img10.png", "img2.png", "Img1.png"] {
            png.save(dir.join(name)).unwrap();
        }
        let grid = Grid::from_dir(dir).unwrap();
        let names: Vec<String> = grid
            .entries
            .iter()
            .map(|e| e.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["Img1.png", "img2.png", "img10.png"]);
    }

    proptest::proptest! {
        /// The layout always has a usable geometry: at least one column, a
        /// positive square side, and a content height that covers every row.
        #[test]
        fn grid_layout_is_always_usable(
            n in 1usize..200,
            win_w in 1.0f32..4000.0,
            win_h in 1.0f32..4000.0,
            zoom in 0.0f32..8.0,
        ) {
            let l = grid_layout_at(n, win_w, win_h, zoom);
            proptest::prop_assert!(l.cols >= 1, "cols {}", l.cols);
            proptest::prop_assert!(l.cols <= n, "cols {} > n {}", l.cols, n);
            proptest::prop_assert!(l.cell_w >= l.side && l.cell_h >= l.side);
            let rows = n.div_ceil(l.cols);
            proptest::prop_assert!(l.content_h >= 2.0 * MARGIN);
            proptest::prop_assert!(l.content_h >= rows as f32 * l.cell_h);
        }

        /// At the default zoom the grid spans the widest row exactly.
        #[test]
        fn grid_layout_default_spans_the_window(
            n in 1usize..60,
            win_w in 200.0f32..3000.0,
            win_h in 200.0f32..3000.0,
        ) {
            let l = grid_layout_at(n, win_w, win_h, 1.0);
            let span_w = 2.0 * MARGIN + l.cols as f32 * l.cell_w + (l.cols - 1) as f32 * GAP;
            proptest::prop_assert!((span_w - win_w).abs() < 0.05, "span_w {span_w} vs {win_w}");
        }

        /// A visual selection always contains the cursor, and the char mode
        /// contains exactly the inclusive range between the anchor and the
        /// cursor.
        #[test]
        fn visual_char_is_an_inclusive_range(
            anchor in 0usize..500,
            cursor in 0usize..500,
            cols in 1usize..30,
        ) {
            let v = Visual { mode: VisualMode::Char, anchor };
            let (lo, hi) = (anchor.min(cursor), anchor.max(cursor));
            for i in 0..1000usize {
                proptest::prop_assert_eq!(v.contains(i, cursor, cols), (lo..=hi).contains(&i));
            }
            proptest::prop_assert!(v.contains(cursor, cursor, cols));
        }

        /// A block selection covers whole rows and stays inside the column
        /// span of the anchor and the cursor.
        #[test]
        fn visual_block_is_inside_its_column_span(
            anchor in 0usize..500,
            cursor in 0usize..500,
            cols in 1usize..30,
        ) {
            let v = Visual { mode: VisualMode::Block, anchor };
            let (c0, c1) = ((anchor % cols).min(cursor % cols), (anchor % cols).max(cursor % cols));
            let (r0, r1) = ((anchor / cols).min(cursor / cols), (anchor / cols).max(cursor / cols));
            for i in 0..1000usize {
                let want = (r0..=r1).contains(&(i / cols)) && (c0..=c1).contains(&(i % cols));
                proptest::prop_assert_eq!(v.contains(i, cursor, cols), want);
            }
        }

        /// A selection step never leaves `0..n` and never wraps.
        #[test]
        fn step_selection_clamps(
            sel in 0usize..300,
            n in 1usize..300,
            step in 0usize..4,
            delta in -5i128..=5,
        ) {
            let out = step_selection(sel, n, step, delta);
            proptest::prop_assert!(out < n);
        }
    }
}
