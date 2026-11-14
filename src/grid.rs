//! Directory thumbnail grid.
//! Square center-crop thumbnails fill the whole window, unlike the fixed
//! thumbnail sizes of nsxiv.
//! A white border marks the selection.
//!
//! A cell is an [`EntrySource`]: a file on disk, opened lazily at decode
//! time.
//! One file is one cell, so a document shows its cover page and nothing else:
//! the image view is where its pages pair up into spreads.
//! The grid has a Vim-style visual selection (see [`Visual`]).
//! `v` selects charwise, `V` linewise, and `Ctrl+v` blockwise.
//! The same key toggles its mode off, `o` swaps the anchor and the cursor,
//! and ESC clears the selection.
//! Every selected cell draws the same white border as the cursor.
//! While a visual selection is active, every cell outside it draws dimmed
//! with the `DIM_TINT` grey, so the selected set stands out.
//! The plain cursor marking dims nothing.
//! A left-button drag over the grid starts a blockwise visual selection,
//! exactly like `Ctrl+v`.
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
//!
//! Only the image view keeps full-resolution textures, for the open entry and
//! its neighbors.
//! The grid view keeps none, so a grid-only session never uploads a source
//! texture of hundreds of megabytes.
//!
//! Beside the decodes, one cheap worker per entry reads only the header of
//! its cache entry and renders the [`Placeholder`] it holds (a `ThumbHash` of a
//! few bytes).
//! The grid uploads that tiny texture and draws it in the cell until the
//! decoded thumb lands, so a warm launch paints the whole window from headers
//! instead of a grey square per cell.
//! A cold launch has no cache entry, so it keeps the grey cell.

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
use winit::{event::MouseButton, keyboard::KeyCode};

use crate::{
    DIM_TINT, DecodedImage,
    blurbg::{self},
    dim::{ImgH, ImgW, LongSide, Rect, WinH, WinW},
    draw::{Color, Turn, Vec2},
    render::Renderer,
    surface::{Surface, SurfaceError, View, open_book_surface, open_surface},
    texture::{Filter, Ready, Texture, Uploader},
    thumbcache::Cache,
};

/// The source of a grid cell: a file on disk, opened (and sniffed) lazily at
/// decode time.
///
/// The type stays cheap to clone, and it keeps the seam for a future
/// in-memory or network source.
/// The path rides an [`Arc`], because a decode job outlives the borrow of the
/// grid that names it, and a refcount bump beats one path copy per job.
#[derive(Clone, Debug)]
pub struct EntrySource(Arc<PathBuf>);

impl EntrySource {
    /// A cell over the file at `path`.
    #[must_use]
    pub fn new(path: PathBuf) -> EntrySource {
        EntrySource(Arc::new(path))
    }

    /// The file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.0.as_path()
    }

    /// The path, shared so a worker can outlive this borrow.
    #[must_use]
    pub fn shared_path(&self) -> Arc<PathBuf> {
        Arc::clone(&self.0)
    }

    /// The path as text for a log line.
    #[must_use]
    pub fn log_label(&self) -> String {
        self.path().display().to_string()
    }

    /// The file, opened plain with its first plane.
    ///
    /// This is what a cell shows: a document cell holds page 0 and never its
    /// second page.
    ///
    /// # Errors
    ///
    /// The function errors when the file cannot be read or the content is not
    /// a decodable surface.
    pub fn open(&self) -> Result<(Arc<dyn Surface>, usize), SurfaceError> {
        let surface: Arc<dyn Surface> = open_surface(self.path())?.into();
        Ok((surface, 0))
    }

    /// The file, opened the way the image view shows it: a document arrives
    /// as a book (see [`crate::book`]), so plane 0 holds every spread.
    ///
    /// # Errors
    ///
    /// The function errors when the file cannot be read or the content is not
    /// a decodable surface.
    pub fn open_book(&self) -> Result<(Arc<dyn Surface>, usize), SurfaceError> {
        let surface: Arc<dyn Surface> = open_book_surface(self.path())?.into();
        Ok((surface, 0))
    }
}

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

/// Long side of the thumbnail texture stored per grid entry, in VRAM and on
/// disk.
/// The thumbnail holds the whole image with the aspect preserved, so the
/// square crop for a grid cell happens at draw time.
/// The decode worker downscales the full-resolution decode to this size, and
/// the cache keeps the same size, so a warm cell and a cold one match.
/// One size serves both, so a decode resizes once and a cache write copies
/// nothing.
/// The image view also shows the thumb as a full-frame placeholder while a
/// decode catches up.
/// The thumbnail must cover the whole image, not only its center square.
const THUMB_LONG_SIDE: LongSide = LongSide::new(512);

/// A finished background decode, matched to an entry by its unique id.
/// Indices shift when entries come and go.
/// Ids never shift.
#[derive(Debug)]
struct DecodeResult {
    id: u64,
    res: Result<DecodeOk, SurfaceError>,
}

/// A finished placeholder read, matched to an entry by its stable id.
#[derive(Debug)]
struct PlaceholderResult {
    id: u64,
    /// The source dimensions from the cache header, zero on a miss.
    image: (ImgW, ImgH),
    /// The rendered placeholder, or None on a miss or a render failure.
    /// The placeholder pass runs on a worker, so it uploads its own 32 px
    /// texture when the grid holds a device.
    ready: Option<Ready>,
}

/// Successful decode payload: the thumbnail and the full-resolution image.
/// [`crate::surface::Surface::render`] caps the full image to the uploadable
/// [`crate::MAX_TEXTURE_SIDE`] before it arrives here.
/// The main thread uploads the thumb for every entry.
/// It uploads the full texture only for keep-set entries (the open image and
/// its prefetched neighbors) and drops the rest.
/// The grid view keeps no keep set, so it uploads thumbs only.
/// The buffers live only until the next frame drains them, bounded by
/// `max_inflight()`.
#[derive(Debug)]
struct DecodeOk {
    /// The whole image with the aspect preserved, long side
    /// [`THUMB_LONG_SIDE`].
    thumb: Ready,
    /// The source dimensions, as [`crate::surface::Surface::bounds`] reports
    /// them (capped at [`crate::MAX_TEXTURE_SIDE`]).
    /// The entry keeps them as the image size, not the thumb size, so the
    /// image view fits and zooms the picture, not its thumbnail.
    image: (ImgW, ImgH),
    /// The tiny blurred copy for the `VV_BLUR_BG` gimmick (None when off).
    /// It stays pixels: it measures `VV_BLUR_PX` (128 by default) on its long
    /// side, so the viewer uploads it in one cheap call when it needs it.
    blur: Option<DecodedImage>,
    /// The untouched full-resolution image.
    /// The cache hit path sets it to None, because the entry needs only the
    /// thumbnail.
    /// A multi-plane source sets it to None too, because its cell holds page 0
    /// and the image view renders the spread itself.
    full: Option<Ready>,
    /// The source holds more than one plane, so no full-res texture helps this
    /// cell (see [`GridEntry::no_full`]).
    no_full: bool,
}

/// A decode job for one entry, as the worker sees it.
/// The worker reads the cache first and decodes on a miss.
/// The job holds an [`EntrySource`], so a job copy costs one refcount bump.
#[derive(Debug)]
struct Job {
    id: u64,
    source: EntrySource,
    /// The job must return the full-resolution image, because the entry holds
    /// a full-res texture: it belongs to the keep set of the image view (the
    /// open entry and its prefetched neighbors).
    /// A thumb-only job decodes outside that set, and the grid view has no set
    /// at all.
    want_full: bool,
    /// The `VV_BLUR_BG` gimmick settings, parsed once by the grid.
    /// A job blurs only when the gimmick is on, so one `Option` answers both
    /// the flag and the size.
    blur: Option<blurbg::Config>,
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

/// A left-button hold that started on a grid cell.
///
/// The travel over a second cell turns the hold into a blockwise visual
/// selection anchored at [`Drag::anchor`], which is the `Ctrl+v` style.
/// A hold that releases on its anchor cell stays a plain click.
#[derive(Debug, Clone, Copy)]
struct Drag {
    /// The entry index where the press landed.
    anchor: usize,
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
    /// The texture upload failed (a buffer that does not match its size).
    #[display("texture upload failed")]
    TextureUpload,
}

/// What one grid cell draws now.
///
/// One value says whether the cell shows a thumbnail, its few-byte placeholder
/// or an error square, so no cell holds a thumb and a failure at the same time.
#[derive(Debug)]
pub enum Cell {
    /// No decode has landed yet: the placeholder draws, or the cell stays
    /// grey.
    Pending {
        /// The uploaded [`Placeholder`], so a warm launch never shows a plain
        /// grey cell.
        /// The decoded thumb drops it.
        placeholder: Option<Texture>,
    },
    /// The decoded thumbnail.
    Thumb(Texture),
    /// The decode or the texture upload failed.
    /// The cell stays visible as a dimmed error square instead of vanishing,
    /// so the grid count never lies.
    Failed(DecodeFailure),
}

impl Cell {
    /// A cell with nothing decoded yet.
    #[must_use]
    pub const fn cold() -> Self {
        Self::Pending { placeholder: None }
    }

    /// Whether the cell still waits for its first decode.
    #[must_use]
    pub fn awaits_first_decode(&self) -> bool {
        matches!(self, Self::Pending { placeholder: None })
    }

    /// The decoded thumbnail, and never a placeholder.
    #[must_use]
    pub fn thumb(&self) -> Option<&Texture> {
        match self {
            Self::Thumb(tex) => Some(tex),
            Self::Pending { .. } | Self::Failed(_) => None,
        }
    }

    /// The placeholder of a cell that has not decoded yet.
    #[must_use]
    pub fn placeholder(&self) -> Option<&Texture> {
        match self {
            Self::Pending { placeholder } => placeholder.as_ref(),
            Self::Thumb(_) | Self::Failed(_) => None,
        }
    }

    /// The texture to draw now: the thumb, or else its placeholder.
    #[must_use]
    pub fn texture(&self) -> Option<&Texture> {
        self.thumb().or_else(|| self.placeholder())
    }

    /// Whether the cell shows an error square.
    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// Why the cell failed, if it did.
    #[must_use]
    pub fn failure(&self) -> Option<DecodeFailure> {
        match self {
            Self::Failed(err) => Some(*err),
            Self::Pending { .. } | Self::Thumb(_) => None,
        }
    }
}

/// One grid cell: the source it shows plus its uploaded full-resolution
/// texture.
/// The code draws thumbs by cropping a center square, so a resize never
/// needs a re-decode.
/// The GPU scales the texture down every frame.
///
/// The cell keeps an [`EntrySource`], so a file and a future in-memory source
/// both sit in a cell.
#[derive(Debug)]
pub struct GridEntry {
    pub source: EntrySource,
    /// Full-resolution image dimensions (as decoded, not the thumb's).
    pub width: ImgW,
    pub height: ImgH,
    /// What the cell draws now: nothing, its few-byte [`Placeholder`], the
    /// decoded thumbnail, or an error square.
    /// The thumbnail covers the whole image with the aspect preserved, long
    /// side `THUMB_LONG_SIDE`, and the grid cell draws it center-cropped.
    /// Full-resolution textures live in `full` only for the image-view keep
    /// set, so VRAM stays bounded on large directories.
    pub cell: Cell,
    /// The full-resolution texture.
    /// The code keeps it only for the open image and its prefetched neighbors
    /// (see `load_pending`), so opening them is instant.
    /// It drops as soon as the entry leaves that set.
    /// The grid view keeps none, so a grid-only session uploads no large
    /// texture.
    pub full: Option<Texture>,
    /// A unique, stable id used to match async decode results.
    pub id: u64,
    /// A decode job for this entry is queued or in flight.
    pub queued: bool,
    /// The texture the image view holds now (taken out of the grid).
    /// The grid puts it back when the view ends and never re-dispatches it
    /// in between.
    pub viewing: bool,
    /// The tiny blurred copy for the `VV_BLUR_BG` background.
    /// The grid keeps it while the texture is out in the image view.
    /// The view clones it.
    pub blur: Option<DecodedImage>,
    /// The source holds more than one plane, which the image view shows as
    /// book spreads (see [`crate::spread`]).
    /// The cell therefore never keeps a full-res texture: the spread the view
    /// shows is not the page this cell holds, and the view loads it itself.
    pub no_full: bool,
}

/// What the user asked the grid to do this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridAction {
    /// Open the grid entry at this index in image mode.
    Open(usize),
    /// Quit the viewer.
    Quit,
}

/// One frame's decode dispatch: which entries to decode first, whether the
/// wraparound scan runs, and what the keep set holds.
///
/// The four facts travel together because they answer one question, so a
/// caller names them instead of passing a bool beside a bool that compiles
/// just as well when swapped.
#[derive(Debug, Clone, Default)]
pub struct LoadPlan {
    /// The indices to decode first: the selection, or the neighbors of the
    /// open image.
    pub priority: Vec<usize>,
    /// The index the wraparound scan starts at (the open image, or the
    /// selection).
    pub scan_start: usize,
    /// Whether the wraparound scan runs at all.
    /// False dispatches only the `priority` entries and pauses the rest of the
    /// queue: the image view sets it while the open image still decodes, so
    /// the visible image gets all the cores.
    pub scan: bool,
    /// Whether the keep set holds full-resolution textures: the entry at
    /// `scan_start` plus the `priority` entries.
    /// The grid view answers false, so a grid-only session uploads thumbnails
    /// and nothing else: a 50 MP neighbor would otherwise cost 200 MB of VRAM
    /// and a 30 ms main-thread upload.
    /// The image view answers true, so the open image and its neighbors open
    /// at once.
    pub preload_full: bool,
}

/// The thumbnail grid, its entry list and its background decode workers.
#[derive(Debug)]
pub struct Grid {
    pub entries: Vec<GridEntry>,
    pub selected: usize,
    result_tx: Sender<DecodeResult>,
    result_rx: Receiver<DecodeResult>,
    /// The placeholder pass of a launch (see [`PlaceholderResult`]).
    /// The sender stays here so [`Grid::start`] can spawn the pass once the
    /// GPU exists, which is after the window opens.
    placeholder_tx: Sender<PlaceholderResult>,
    placeholder_rx: Receiver<PlaceholderResult>,
    /// The device handle that lets a decode worker upload its own texture.
    /// None until the window and the GPU exist, and None in every test, which
    /// then receives pixels.
    uploader: Option<Uploader>,
    inflight: usize,
    /// Thumbnail size zoom (1.0 = exact-fill default, and +/- steps it).
    zoom: f32,
    /// The vertical scroll offset, used only when zoomed in past the exact
    /// fill.
    scroll: f32,
    /// The active Vim-style visual selection, if any.
    visual: Option<Visual>,
    /// The left-button hold in progress, if any.
    drag: Option<Drag>,
    /// The cached layout, keyed by (entry count, window size, zoom).
    /// The code recomputes it only when one of those changes, because the
    /// layout math is O(n).
    /// The field uses interior mutability because `draw` takes &self.
    /// The grid is the only owner of the value, so this is a memo of a pure
    /// function of the key and not shared state: a `&mut self` layout would
    /// have to reach `draw`, which holds the grid immutably for the whole
    /// frame.
    layout_cache: RefCell<Option<(LayoutKey, Layout)>>,
    /// The `VV_BLUR_BG` gimmick settings, read once.
    /// Decode workers compute the tiny blurred copy only when the gimmick is
    /// on, at this size.
    blur: Option<blurbg::Config>,
    /// The thumbnail cache, shared with every decode worker.
    cache: Arc<Cache>,
}

/// Run one decode job on a worker: the cache first, the file on a miss.
fn run_job(
    job: &Job,
    cache: &Cache,
    uploader: Option<&Uploader>,
) -> Result<DecodeOk, SurfaceError> {
    // A miss is normal and stays silent, because the first launch of a
    // directory misses once per file.
    // A job that needs the full image skips the cache: the file decode
    // carries the thumb as well, so a cache read would only add a read.
    // An open file is sniffed at decode time, and its path keys the cache.
    let path = job.source.path();
    if !job.want_full
        && let Some(hit) = cache.load(path, THUMB_LONG_SIDE).ok()
    {
        // The cached thumb already measures [`THUMB_LONG_SIDE`], so the code
        // uploads it as is and the GPU scales it in the cell.
        // A resample here would copy the buffer for nothing.
        let blur = job.blur.map(|config| config.blur(&hit.thumb));
        return Ok(DecodeOk {
            thumb: ready(hit.thumb, uploader),
            image: hit.image,
            blur,
            full: None,
            // The header says nothing about the plane count, so a keep-set
            // refill asks the file once and learns it there.
            no_full: false,
        });
    }
    let (surface, plane) = job.source.open()?;
    // A multi-plane source (a document) opens as a spread in the image view,
    // which a cell does not hold: the cell keeps page 0, and the view renders
    // the spread through its own loader.
    let no_full = surface.plane_count() > 1;
    // The plane size is the source size the entry keeps, so a vector file
    // measures its base raster and never its thumbnail.
    let bounds = surface.bounds(plane)?;
    let image = (
        ImgW::new(bounds.w.round() as u32),
        ImgH::new(bounds.h.round() as u32),
    );
    if !job.want_full || no_full {
        // Thumbnail only, so no full-resolution texture is needed.
        // A vector file paints the thumbnail size directly; a raster format
        // decodes in full and downscales inside the surface render, on the
        // worker.
        let thumb = surface.render(View::whole(plane, thumb_scale(bounds)))?;
        let blur = job.blur.map(|config| config.blur(&thumb));
        store_thumb(cache, path, &thumb, image.0, image.1);
        return Ok(DecodeOk {
            thumb: ready(thumb, uploader),
            image,
            blur,
            full: None,
            no_full,
        });
    }
    let full = surface.render(View::whole(plane, 1.0))?;
    let blur = job.blur.map(|config| config.blur(&full));
    // Borrow the decoded buffer for the thumb: a clone of a large image costs
    // hundreds of megabytes.
    let thumb =
        crate::downscale_rgba_ref(full.as_rgba(), full.width(), full.height(), THUMB_LONG_SIDE);
    store_thumb(cache, path, &thumb, image.0, image.1);
    // The keep-set refill is the only job that carries the full image to the
    // main thread; otherwise the large buffer drops here.
    let full = job.want_full.then(|| ready(full, uploader));
    Ok(DecodeOk {
        thumb: ready(thumb, uploader),
        image,
        blur,
        full,
        no_full,
    })
}

/// The render scale that lands the long side of `bounds` on
/// [`THUMB_LONG_SIDE`].
///
/// The scale never goes past 1.0, so a small picture is not upscaled into its
/// thumbnail.
fn thumb_scale(bounds: crate::surface::Size) -> f32 {
    (THUMB_LONG_SIDE.raw() as f32 / bounds.w.max(bounds.h).max(1.0)).min(1.0)
}

/// Put the thumbnail of a fresh decode into the cache.
/// A cache failure stays silent: the grid already holds the thumbnail, and a
/// missing cache home is a normal setup.
fn store_thumb(cache: &Cache, path: &Path, thumb: &DecodedImage, width: ImgW, height: ImgH) {
    // The thumb already measures [`THUMB_LONG_SIDE`], which is the entry size,
    // so the store copies no pixels.
    let _ = cache.store(path, THUMB_LONG_SIDE, thumb, (width, height));
}

/// Run one placeholder job on a worker: the cache header, then the DCT
/// render.
/// A miss is normal (a cold directory, a changed source, the cache off).
/// A miss on a vector file still measures the source size, so its cell and
/// the image-view fit know the shape before the decode lands.
/// A miss anywhere else stays silent, and the grid keeps its grey cell.
fn run_placeholder(cache: &Cache, path: &Path) -> ((ImgW, ImgH), Option<DecodedImage>) {
    if let Ok((hash, image)) = cache.load_placeholder(path, THUMB_LONG_SIDE) {
        return (image, hash.render());
    }
    (
        crate::measure_vector(path).unwrap_or((ImgW::new(0), ImgH::new(0))),
        None,
    )
}

/// Upload a frame on the worker that produced it, or keep its pixels when no
/// device exists (a test).
fn ready(image: DecodedImage, uploader: Option<&Uploader>) -> Ready {
    match uploader {
        Some(uploader) => match uploader.image(&image) {
            Ok(texture) => Ready::Texture(texture),
            Err(err) => {
                eprintln!("vv: upload: {err}");
                Ready::Pixels(image)
            }
        },
        None => Ready::Pixels(image),
    }
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

        Ok(Self::from_paths(paths))
    }

    /// Build a grid over `paths`, in the order given, and start the background
    /// decode worker.
    ///
    /// [`Grid::from_dir`] hands it a sorted directory listing.
    /// A single-file launch hands it one path, which is a document that opens
    /// as spreads: the one cell shows the cover page and ESC still has a grid
    /// to return to.
    #[must_use]
    pub fn from_paths(paths: Vec<PathBuf>) -> Grid {
        let (res_tx, res_rx) = channel::<DecodeResult>();
        let cache = Arc::new(Cache::new());
        let entries: Vec<GridEntry> = paths
            .into_iter()
            .enumerate()
            .map(|(i, path)| GridEntry {
                source: EntrySource::new(path),
                width: ImgW::new(0),
                height: ImgH::new(0),
                cell: Cell::cold(),
                full: None,
                id: i as u64,
                queued: false,
                viewing: false,
                blur: None,
                no_full: false,
            })
            .collect();

        let (ph_tx, placeholder_rx) = channel::<PlaceholderResult>();
        Self::from_parts(entries, cache, res_tx, res_rx, ph_tx, placeholder_rx)
    }

    /// Start the background passes that need a device.
    ///
    /// The grid lists a directory before the window opens, because the launch
    /// title carries the count, and the GPU exists only after the window maps.
    /// This call therefore starts the placeholder pass, and every decode that
    /// follows uploads its own texture on its worker.
    ///
    /// The placeholder pass runs one cheap worker per entry, so the grid paints
    /// the window before the first decode lands.
    /// A job reads one header of about 200 bytes and renders 32 px, so the pass
    /// needs no inflight cap.
    /// Without a cache directory there is no header to read, so the pass stays
    /// off.
    pub fn start(&mut self, uploader: Uploader) {
        if self.cache.dir().is_some() {
            for e in &self.entries {
                let (id, path) = (e.id, e.source.path().to_path_buf());
                let cache = Arc::clone(&self.cache);
                let tx = self.placeholder_tx.clone();
                let uploader = uploader.clone();
                rayon::spawn(move || {
                    let (image, rgba) = run_placeholder(&cache, &path);
                    let ready = rgba.map(|rgba| ready(rgba, Some(&uploader)));
                    let _ = tx.send(PlaceholderResult { id, image, ready });
                });
            }
        }
        self.uploader = Some(uploader);
    }

    /// Assemble a grid from its entries and its worker channels.
    fn from_parts(
        entries: Vec<GridEntry>,
        cache: Arc<Cache>,
        result_tx: Sender<DecodeResult>,
        result_rx: Receiver<DecodeResult>,
        placeholder_tx: Sender<PlaceholderResult>,
        placeholder_rx: Receiver<PlaceholderResult>,
    ) -> Grid {
        Grid {
            entries,
            selected: 0,
            visual: None,
            drag: None,
            result_tx,
            result_rx,
            placeholder_tx,
            placeholder_rx,
            uploader: None,
            inflight: 0,
            layout_cache: RefCell::new(None),
            zoom: 1.0,
            scroll: 0.0,
            blur: blurbg::Config::from_env_if_enabled(),
            cache,
        }
    }

    /// Drain finished background decodes, which upload textures on the main
    /// thread, and hand out new jobs for one [`LoadPlan`].
    /// The function runs every frame and never blocks.
    /// It skips entries whose texture the image view holds now.
    pub fn load_pending(&mut self, plan: &LoadPlan) {
        // Ids that hold a full-resolution texture: the entry the
        // scan starts at (the open image) plus the prefetch priorities (its
        // neighbors).
        // Everything else stores only the thumbnail.
        let keep: HashSet<u64> = if plan.preload_full {
            plan.priority
                .iter()
                .filter_map(|&i| self.entries.get(i).map(|e| e.id))
                .chain(
                    self.entries
                        .get(plan.scan_start.min(self.entries.len().saturating_sub(1)))
                        .map(|e| e.id),
                )
                .collect()
        } else {
            HashSet::new()
        };

        self.drain_placeholders();
        self.drain_results(&keep);
        self.dispatch(plan, &keep);
    }

    /// Apply finished placeholder reads.
    /// The worker already uploaded its texture, so a whole directory of
    /// placeholders costs one channel drain and no pixel copy on this thread.
    /// An entry that already holds a thumb or a failure skips the texture, and
    /// the source dimensions still land, because the image view fits the
    /// picture before any decode.
    fn drain_placeholders(&mut self) {
        while let Ok(res) = self.placeholder_rx.try_recv() {
            let Some(i) = self.index_of(res.id) else {
                continue; // the entry left meanwhile, so the result is moot
            };
            let e = &mut self.entries[i];
            if e.width.raw() == 0 && res.image.0.raw() > 0 {
                // The header carries the source dimensions, so the image view
                // fits the picture before the first decode lands.
                e.width = res.image.0;
                e.height = res.image.1;
            }
            let Some(ready) = res.ready else {
                continue;
            };
            if !e.cell.awaits_first_decode() {
                continue;
            }
            e.cell = Cell::Pending {
                placeholder: ready.into_texture(),
            };
        }
    }

    /// Land the decode of entry `i`: its thumb, its dimensions, its blur copy
    /// and, for a keep-set entry, its full-resolution texture.
    ///
    /// The worker already uploaded both textures, so this call only moves
    /// handles: a cache hit and a fresh decode cost the same here.
    fn land_thumb(&mut self, i: usize, ok: DecodeOk, keep: &HashSet<u64>) {
        let DecodeOk {
            thumb,
            image,
            blur,
            full,
            no_full,
        } = ok;
        let texture = thumb.into_texture();
        let e = &mut self.entries[i];
        if let Some(tex) = texture {
            // The decoded thumb replaces the placeholder, so its texture drops
            // now.
            e.cell = Cell::Thumb(tex);
        } else {
            // No device (a test): the decode still carries its dimensions.
            e.cell = Cell::Failed(DecodeFailure::TextureUpload);
        }
        e.width = image.0;
        e.height = image.1;
        e.blur = blur;
        e.queued = false;
        e.no_full |= no_full;
        if keep.contains(&e.id)
            && let Some(full) = full
        {
            e.full = full.into_texture();
        }
    }

    /// Step 1 of [`Self::load_pending`]: apply finished decodes.
    /// A failed decode marks its entry as failed (the cell stays visible,
    /// dimmed, with an error mark) instead of deleting it, so the grid count
    /// never lies.
    /// Textures arrive already uploaded, because the decode worker holds the
    /// device.
    fn drain_results(&mut self, keep: &HashSet<u64>) {
        while let Ok(res) = self.result_rx.try_recv() {
            self.inflight -= 1;
            let Some(i) = self.entries.iter().position(|e| e.id == res.id) else {
                continue; // entry was spliced out meanwhile; drop the result
            };
            match res.res {
                Ok(ok) if self.entries[i].cell.thumb().is_none() => {
                    self.land_thumb(i, ok, keep);
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
                        e.full = full.into_texture();
                    }
                }
                Ok(DecodeOk {
                    image,
                    full: None,
                    no_full,
                    ..
                }) => {
                    // A cache hit for an entry that already shows a thumb:
                    // nothing is left to upload, but the source dimensions
                    // land, because the cached thumb carries them.
                    // The image view then fits the picture, not its thumb.
                    let e = &mut self.entries[i];
                    e.queued = false;
                    e.width = image.0;
                    e.height = image.1;
                    e.no_full |= no_full;
                }
                Err(err) => {
                    eprintln!(
                        "vv: {}: decode failed: {err}",
                        self.entries[i].source.log_label()
                    );
                    let e = &mut self.entries[i];
                    e.queued = false;
                    e.cell = Cell::Failed(DecodeFailure::Decode);
                }
            }
        }
        self.selected = self.selected.min(self.entries.len().saturating_sub(1));

        // Evict full-res textures from entries that left the keep set (the
        // open entry and its neighbors moved on).
        // Thumb textures stay.
        // The GPU context lives on this thread, so unloading is safe.
        for e in &mut self.entries {
            if e.full.is_some() && !keep.contains(&e.id) {
                drop(e.full.take());
            }
        }
    }

    /// Step 2 of [`Self::load_pending`]: dispatch new jobs.
    /// The code dispatches the `priority` indices of the plan first (the
    /// selection or the open image neighbors), then its wraparound scan.
    /// Jobs run on the shared rayon pool, so several decodes proceed in
    /// parallel.
    /// Each job sends its result back over the channel for `drain_results` to
    /// upload.
    /// The per-entry checks below (texture present, queued, viewing) make
    /// revisiting a priority index in the wraparound harmless.
    /// The code needs no extra dedup bookkeeping: O(1) per entry.
    fn dispatch(&mut self, plan: &LoadPlan, keep: &HashSet<u64>) {
        let n = self.entries.len();
        let start = plan.scan_start.min(n);
        let order = plan.priority.iter().copied().filter(|&i| i < n).chain(
            plan.scan
                .then(|| (start..n).chain(0..start))
                .into_iter()
                .flatten(),
        );

        for i in order {
            if self.inflight >= max_inflight() {
                break;
            }
            let e = &mut self.entries[i];
            if e.queued || e.viewing || e.cell.is_failed() {
                continue;
            }
            // Decode when there is no thumb yet.
            // Re-decode a thumb-only entry when it stays in the keep set but
            // its full-res texture is missing (its decode drained outside
            // the keep set, so the code dropped the full texture then).
            // A multi-plane source keeps no full texture at all, so it never
            // re-decodes for one.
            // The code never re-decodes thumb-only entries outside the keep
            // set.
            let wants_full = keep.contains(&e.id) && !e.no_full && e.full.is_none();
            if e.cell.thumb().is_some() && !wants_full {
                continue;
            }
            e.queued = true;
            self.inflight += 1;
            let job = Job {
                id: e.id,
                source: e.source.clone(),
                // Only a keep-set entry needs the full texture. A cold entry
                // outside it decodes for its thumbnail only.
                want_full: keep.contains(&e.id),
                blur: self.blur,
            };
            let tx = self.result_tx.clone();
            let cache = Arc::clone(&self.cache);
            let uploader = self.uploader.clone();
            rayon::spawn(move || {
                let id = job.id;
                let res = run_job(&job, &cache, uploader.as_ref());
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
            win_w.raw().to_bits(),
            win_h.raw().to_bits(),
            self.zoom.to_bits(),
        );
        let mut cache = self.layout_cache.borrow_mut();
        if cache.as_ref().is_none_or(|(k, _)| *k != key) {
            *cache = Some((key, grid_layout_at(n, win_w.raw(), win_h.raw(), self.zoom)));
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
    /// The cell stays, dimmed, with an error mark.
    pub fn mark_failed(&mut self, id: u64, err: DecodeFailure) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.id == id) {
            e.queued = false;
            e.cell = Cell::Failed(err);
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
        let (_, win_h) = (win_w.raw(), win_h.raw());
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
    /// A held left button that travels over the grid starts a blockwise
    /// visual selection from the press cell, like `Ctrl+v`.
    ///
    ///
    /// `None` means no action this frame.
    //
    // One frame input pipeline (queue drain, repeat, zoom, scroll,
    // navigation, mouse) is intentionally linear.
    // See main() too.
    #[allow(clippy::too_many_lines)]
    pub fn handle_input(
        &mut self,
        input: &crate::input::Input,
        win: (WinW, WinH),
        step: u32,
    ) -> Option<GridAction> {
        let (win_w, win_h) = (win.0.raw(), win.1.raw());
        if self.entries.is_empty() {
            return None;
        }
        // Read the frame's edge set, not a polled held state.
        // A press whose release lands within the same frame is invisible to a
        // held-state poll (0 -> 0 nets out), and a crisp Enter tap fits inside
        // one stalled frame.
        // The input layer records every event, so nothing is lost.
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
        for k in input.pressed_keys() {
            match k {
                KeyCode::Enter | KeyCode::NumpadEnter => enter = true,
                KeyCode::KeyQ => quit = true,
                KeyCode::Escape => escape = true,
                KeyCode::KeyV => {
                    // Vim: `v` charwise, `V` linewise, `Ctrl+v` blockwise.
                    // The same key again ends the visual selection, and a
                    // different key switches its mode in place.
                    let mode = if input.is_key_down(KeyCode::ControlLeft)
                        || input.is_key_down(KeyCode::ControlRight)
                    {
                        VisualMode::Block
                    } else if input.is_key_down(KeyCode::ShiftLeft)
                        || input.is_key_down(KeyCode::ShiftRight)
                    {
                        VisualMode::Line
                    } else {
                        VisualMode::Char
                    };
                    start_visual = Some(mode);
                }
                KeyCode::KeyO => swap_ends = true,
                KeyCode::KeyF => open_at_cursor = true,
                KeyCode::KeyH | KeyCode::ArrowLeft => left = true,
                KeyCode::KeyL | KeyCode::ArrowRight => right = true,
                KeyCode::KeyK | KeyCode::ArrowUp => up = true,
                KeyCode::KeyJ | KeyCode::ArrowDown => down = true,
                KeyCode::KeyG => {
                    if input.is_key_down(KeyCode::ShiftLeft)
                        || input.is_key_down(KeyCode::ShiftRight)
                    {
                        jump_last = true;
                    } else {
                        jump_first = true;
                    }
                }
                KeyCode::Equal | KeyCode::NumpadAdd => zoom_in = true,
                KeyCode::Minus | KeyCode::NumpadSubtract => zoom_out = true,
                _ => {}
            }
        }
        // An auto-repeat steps the selection again, at the repeat rate the X
        // server sends. The input layer separates it from the press edge, so a
        // one-shot key above never fires twice.
        for k in input.repeated_keys() {
            match k {
                KeyCode::KeyH | KeyCode::ArrowLeft => left = true,
                KeyCode::KeyL | KeyCode::ArrowRight => right = true,
                KeyCode::KeyK | KeyCode::ArrowUp => up = true,
                KeyCode::KeyJ | KeyCode::ArrowDown => down = true,
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
        for typed in input.typed() {
            match typed.text {
                '\n' | '\r' => enter = true,
                '+' => zoom_in = true,
                '-' => zoom_out = true,
                _ => {}
            }
        }
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
        let wheel = input.wheel_move();
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
        //
        // A held left button that travels over the grid starts a blockwise
        // visual selection from the cell where the press landed, so a drag
        // selects a rectangle exactly like `Ctrl+v`.
        // A drag that ends where it started is still a plain click, and a
        // click on the cursor cell still opens it.
        let mouse = input.cursor_position();
        let left_down = input.is_button_down(MouseButton::Left);
        // A release ends the hold. The code clears on the held state, not on
        // the release edge, so a frame that returns early never leaves a
        // stale hold behind.
        if !left_down {
            self.drag = None;
        }
        if input.is_button_pressed(MouseButton::Left)
            && let Some(idx) = self.cell_at(mouse, win.0, win.1)
        {
            self.drag = Some(Drag { anchor: idx });
            if idx == self.selected {
                return Some(GridAction::Open(idx));
            }
            self.selected = idx;
            self.ensure_visible(win.0, win.1);
        }
        if let Some(drag) = self.drag
            && let Some(idx) = self.cell_at(mouse, win.0, win.1)
            && idx != drag.anchor
        {
            // The press already selected the anchor cell, so a travel to a
            // second cell starts the block there.
            self.visual = Some(Visual {
                mode: VisualMode::Block,
                anchor: drag.anchor,
            });
            self.selected = idx;
            self.ensure_visible(win.0, win.1);
        }

        if open_at_cursor && let Some(idx) = self.cell_at(mouse, win.0, win.1) {
            self.selected = idx;
            self.ensure_visible(win.0, win.1);
            return Some(GridAction::Open(idx));
        }
        None
    }

    /// The grid cell under the window-space point, if any.
    fn cell_at(&self, m: Vec2, win_w: WinW, win_h: WinH) -> Option<usize> {
        let l = self.layout(win_w, win_h);
        let (cw, ch) = (l.cell_w, l.cell_h);
        // Grid coordinates: the code draws content at MARGIN + col*(cw+GAP)
        // minus scroll, so add scroll back to the cursor position.
        let (mx, my) = (m.x - MARGIN, m.y + self.scroll - MARGIN);
        let col = (mx / (cw + GAP)).floor();
        let row = (my / (ch + GAP)).floor();
        let cell = Rect::new(col * (cw + GAP), row * (ch + GAP), cw, ch);
        if col >= 0.0 && row >= 0.0 && col < l.cols as f32 && cell.contains_point(mx, my) {
            let idx = row as usize * l.cols + col as usize;
            if idx < self.entries.len() {
                return Some(idx);
            }
        }
        None
    }

    /// Draw every visible thumbnail cell.
    ///
    /// An empty grid draws nothing: the window title already carries the count
    /// (`(0 image(s)) – <dir> – versatile-viewer`), so the grid needs no text
    /// layer to say it.
    pub fn draw(&self, r: &mut Renderer, win: (WinW, WinH)) {
        let win_h = win.1.raw();
        if self.entries.is_empty() {
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
            if let Some(tex) = e.cell.thumb() {
                // The thumb texture keeps the aspect ratio of the image.
                // The square cell shows its center crop (draw-time crop, so a
                // resize never needs a re-decode).
                let src = Rect::sized(tex.width() as f32, tex.height() as f32).center_square();
                r.texture_pro(tex, src, thumb, Turn::NONE, tint, Filter::Linear);
            } else if e.cell.is_failed() {
                // Decode failed: dim red placeholder with an error mark.
                // The file stays visible instead of vanishing silently.
                r.fill_rect(thumb, Color::new(72, 26, 26, 255).tint(tint));
                draw_failure_mark(r, thumb, tint);
            } else if let Some(tex) = e.cell.placeholder() {
                // The few-byte ThumbHash placeholder, until the decoded thumb
                // lands.
                // It holds the whole image, so the draw crops it like a thumb.
                let src = Rect::sized(tex.width() as f32, tex.height() as f32).center_square();
                r.texture_pro(tex, src, thumb, Turn::NONE, tint, Filter::Linear);
            } else {
                // Still decoding: placeholder square.
                r.fill_rect(thumb, Color::new(28, 28, 28, 255).tint(tint));
            }
            if in_selection {
                r.stroke_rect(thumb, 4.0, Color::WHITE);
            }
        }
    }
}

/// Draw the failure mark of a cell whose decode failed.
///
/// The mark is an exclamation built from two rectangles: a bar and a dot.
/// The grid owns no font, so the cell shows a shape instead of a glyph.
fn draw_failure_mark(r: &mut Renderer, thumb: Rect, tint: Color) {
    let color = Color::new(214, 92, 92, 255).tint(tint);
    let stroke = (thumb.width * 0.08).max(2.0);
    let bar = thumb.height * 0.40;
    let gap = thumb.height * 0.12;
    let left = thumb.x + (thumb.width - stroke) / 2.0;
    let top = thumb.y + (thumb.height - (bar + gap + stroke)) / 2.0;
    r.fill_rect(Rect::new(left, top, stroke, bar), color);
    r.fill_rect(Rect::new(left, top + bar + gap, stroke, stroke), color);
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

/// Whether the directory filter of the grid keeps this file.
///
/// The list is the formats this build routes by content, plus every path a
/// converter rule of this process claims (`src/convert.rs`), so a configured
/// format appears in the grid too.
/// A PDF carries no image extension but opens as a page grid, so the filter
/// accepts it too.
/// A format with no rule of its own stays out: the viewer names no converter,
/// so a `.typ` source or a `.mmd` diagram lists only when a rule claims it.
fn is_image_path(path: &Path) -> bool {
    is_image_path_in(path, crate::convert::rules())
}

/// Whether `path` belongs in the grid, with `converters` as the rule table.
///
/// A test passes a table of its own, so the filter never reads the table of the
/// user.
fn is_image_path_in(path: &Path, converters: &crate::convert::Converters) -> bool {
    converters.claims(path) || is_image_extension(path)
}

/// Whether the extension of `path` names a format this build routes.
fn is_image_extension(path: &Path) -> bool {
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
                | "svg"
                | "svgz"
                | "pdf"
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

    /// The cache feature owns this path: without it every job decodes.
    #[cfg(feature = "cache")]
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
            source: EntrySource::new(png.clone()),
            want_full: false,
            blur: None,
        };
        let miss = run_job(&job, &cache, None).expect("decode");
        assert!(miss.full.is_none(), "a thumb-only job keeps no full image");
        assert_eq!(miss.image, (ImgW::new(64), ImgH::new(32)));
        assert!(
            cache.load(&png, THUMB_LONG_SIDE).is_ok(),
            "the decode stores a thumbnail"
        );
        // The second job takes the cached thumbnail and decodes nothing.
        let hit = run_job(&job, &cache, None).expect("cache hit");
        assert!(hit.full.is_none());
        // The entry is lossy, so the hit keeps the shape and the alpha plane
        // and only the color samples drift.
        assert_eq!(hit.thumb.dims(), miss.thumb.dims());
        assert_eq!(
            hit.thumb.pixels().unwrap().as_rgba().len(),
            miss.thumb.pixels().unwrap().as_rgba().len()
        );
        // The hit still reports the source dimensions, so the image view fits
        // the picture and not the cached thumb.
        assert_eq!(hit.image, miss.image);
        // A keep-set job ignores the cache, because it needs the full image.
        let keep = Job {
            want_full: true,
            ..job
        };
        let refill = run_job(&keep, &cache, None).expect("decode");
        assert!(refill.full.is_some());
        assert_eq!(refill.image, miss.image);
    }

    /// A vector entry measures its base raster, not its thumbnail.
    #[cfg(feature = "cache")]
    #[test]
    fn a_vector_entry_measures_its_base_raster() {
        let dir = TempDir::new().unwrap();
        let svg = dir.path().join("logo.svg");
        std::fs::write(
            &svg,
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
                <rect width="100" height="50" fill="#00ff00"/>
            </svg>"##,
        )
        .unwrap();
        let cache = Cache::in_dir(dir.path().join("cache"));
        let job = Job {
            id: 0,
            source: EntrySource::new(svg),
            want_full: false,
            blur: None,
        };
        let miss = run_job(&job, &cache, None).expect("decode");
        assert!(miss.full.is_none());
        // The thumbnail paints at the cell size, and the entry keeps the
        // base raster size, clamped up to the minimum side.
        assert_eq!(miss.thumb.dims(), (512, 256));
        assert_eq!(miss.image, (ImgW::new(1024), ImgH::new(512)));
        let hit = run_job(&job, &cache, None).expect("cache hit");
        assert_eq!(hit.image, miss.image);
        // The keep-set refill paints the base raster, so the open image zooms
        // on more pixels than the cell shows.
        let keep = Job {
            want_full: true,
            ..job
        };
        let refill = run_job(&keep, &cache, None).expect("decode");
        let full = refill.full.expect("the keep set keeps the full image");
        assert_eq!(full.dims(), (1024, 512));
        assert_eq!(refill.image, miss.image);
    }

    /// A cold vector entry learns its shape from the placeholder pass, so the
    /// cell and the image-view fit never wait for the decode.
    #[cfg(feature = "cache")]
    #[test]
    fn a_cold_vector_entry_measures_before_the_decode() {
        let dir = TempDir::new().unwrap();
        let svg = dir.path().join("logo.svg");
        std::fs::write(
            &svg,
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="300" height="100">
                <rect width="300" height="100" fill="#000000"/>
            </svg>"##,
        )
        .unwrap();
        // A cache directory with no entry for the file: a cold launch.
        let cache = Cache::in_dir(dir.path().join("cache"));
        let (image, rgba) = run_placeholder(&cache, &svg);
        assert_eq!(image, (ImgW::new(1024), ImgH::new(341)));
        assert!(rgba.is_none(), "a cold cell keeps the grey cell");
        // A raster file still reports nothing, because its decode carries the
        // size.
        let png = dir.path().join("photo.png");
        image::RgbaImage::from_pixel(4, 2, image::Rgba([9, 9, 9, 255]))
            .save(&png)
            .expect("png");
        let (image, rgba) = run_placeholder(&cache, &png);
        assert_eq!(image, (ImgW::new(0), ImgH::new(0)));
        assert!(rgba.is_none());
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
        assert_eq!(grid.entries[0].cell.failure(), Some(DecodeFailure::Decode));
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
        assert!(is_image_extension(Path::new("foo.png")));
        assert!(is_image_extension(Path::new("foo.PNG")));
        assert!(is_image_extension(Path::new("a/b/c.jpeg")));
        assert!(is_image_extension(Path::new("photo.JXL")));
        assert!(is_image_extension(Path::new("logo.svg")));
        assert!(is_image_extension(Path::new("logo.SVGZ")));
        // A PDF opens as a page grid, so the directory filter keeps it.
        assert!(is_image_extension(Path::new("doc.pdf")));
        assert!(is_image_extension(Path::new("DOC.PDF")));
        // A Typst source and a Mermaid source need a converter rule, so the
        // plain list leaves them out.
        assert!(!is_image_extension(Path::new("doc.typ")));
        assert!(!is_image_extension(Path::new("DOC.TYP")));
        // A Mermaid source needs a converter rule, so the plain list leaves
        // it out.
        assert!(!is_image_extension(Path::new("diagram.mmd")));
        assert!(!is_image_extension(Path::new("DIAGRAM.MMD")));
        assert!(!is_image_extension(Path::new("notes.txt")));
        assert!(!is_image_extension(Path::new("archive.tar.gz")));
        assert!(!is_image_extension(Path::new("noext")));
    }

    #[test]
    fn a_configured_extension_lists() {
        // A table of this test alone, so the filter never reads the table of
        // the user.
        let table = crate::convert::Converters::parse(
            "[[convert]]\next = [\"exr\", \"mmd\"]\ncommand = [\"exr2svg\"]\n",
        )
        .expect("a rule table");
        assert!(is_image_path_in(Path::new("photo.EXR"), &table));
        // The rule claims the Mermaid source too, so the filter keeps it.
        assert!(is_image_path_in(Path::new("diagram.mmd"), &table));
        assert!(!is_image_path_in(Path::new("notes.txt"), &table));
        // A table with no rule lists no converted format.
        let empty = crate::convert::Converters::empty();
        assert!(!is_image_path_in(Path::new("diagram.mmd"), &empty));
        assert!(is_image_path_in(Path::new("photo.png"), &empty));
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
            .map(|e| {
                e.source
                    .path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            })
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
            .map(|e| {
                e.source
                    .path()
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            })
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

    #[test]
    fn from_paths_builds_a_one_cell_grid() {
        // A single-file launch over a document keeps one cell, so ESC has a
        // grid to return to and the cell shows page 0.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.pdf");
        std::fs::write(&path, crate::pdf::TWO_PAGE_PDF).unwrap();
        let grid = Grid::from_paths(vec![path.clone()]);
        assert_eq!(grid.entries.len(), 1);
        assert_eq!(grid.entries[0].source.path(), path.as_path());
        assert_eq!(grid.entries[0].id, 0);
        assert!(
            !grid.entries[0].no_full,
            "no decode reported the planes yet"
        );
    }

    #[test]
    fn entry_source_open_reads_a_file_and_labels() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("a.png");
        image::DynamicImage::new_rgb8(3, 2).save(&path).unwrap();
        let source = EntrySource::new(path.clone());
        assert_eq!(source.path(), path.as_path());
        assert!(source.log_label().ends_with("a.png"));
        let (surface, plane) = source.open().unwrap();
        assert_eq!(plane, 0);
        assert_eq!(surface.plane_count(), 1);
        assert!(
            surface.progressive().unwrap().is_none(),
            "a PNG does not stream"
        );
    }

    #[test]
    fn entry_source_opens_a_document_as_page_zero_and_as_a_book() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("doc.pdf");
        std::fs::write(&path, crate::pdf::TWO_PAGE_PDF).unwrap();
        let source = EntrySource::new(path);
        // The cell keeps page 0, so a decode measures the page.
        let (plain, plane) = source.open().unwrap();
        assert_eq!((plain.plane_count(), plane), (2, 0));
        assert_eq!(
            plain.bounds(0).unwrap(),
            crate::surface::Size { w: 612.0, h: 792.0 }
        );
        // The image view shows the book: the cover spread over page 1 alone.
        let (book, plane) = source.open_book().unwrap();
        assert_eq!(plane, 0);
        assert_eq!(book.plane_count(), 1);
        assert_eq!(book.bounds(0).unwrap().w, 612.0 * 2.0 + crate::book::GUTTER);
        assert_eq!(book.bounds(0).unwrap().h, 792.0 * 2.0);
    }
}
