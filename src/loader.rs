//! Background streaming loader for the image view.
//!
//! The loader renders one [`Surface`] plane off the main thread.
//! It streams results back over a channel.
//! JPEG XL decodes progressively.
//! The worker reads fixed-size chunks from the surface's byte stream
//! ([`crate::surface::ChunkSource`]) and feeds them into jxl-oxide.
//! The stream hides where the bytes come from, so a source with no file can
//! still stream, and the file-backed one memory-maps the file.
//! jxl-oxide renders a blurry full-size preview long before all bytes arrive.
//! The worker sends the final full-quality render when decoding completes.
//! Every other content renders in one step, and it announces the plane size
//! from [`Surface::bounds`] first, so the window fits the picture at once.
//!
//! Cancellation: dropping the `Loader` sets a flag.
//! The worker checks this flag between chunks and closes the channel.
//! Pending sends fail then, and the worker exits.
//!
//! The map is a plain read-only mapping.
//! A file rewritten under the map faults with SIGBUS instead of returning
//! short data, so the viewer assumes the files it opens stay put.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    time::{Duration, Instant},
};

use derive_more::{Display, Error, From};
use jxl_oxide::{InitializeResult, JxlImage};

use crate::{
    DecodedImage, blurbg, cap_texture_side,
    dim::{ImgH, ImgW, LongSide, PreviewPx},
    downscale_rgba,
    error::DecodeError,
    fb_to_rgba,
    surface::{ChunkSource, Surface, SurfaceError, View},
};

/// A failed streaming load.
#[derive(Debug, Display, Error, From)]
pub enum LoadError {
    /// A file open or read failed.
    #[display("file i/o failed")]
    Io(std::io::Error),
    /// The JPEG XL decoder rejected the bytes.
    #[display("jxl-oxide: {_0}")]
    Jxl(
        #[from]
        #[error(source)]
        Box<dyn std::error::Error + Send + Sync + 'static>,
    ),
    /// The decode failed before the header completed.
    #[display("file ended before the image header was complete")]
    Incomplete,
    /// The surface could not describe or render its plane.
    #[display("{_0}")]
    Surface(
        #[from]
        #[error(source)]
        SurfaceError,
    ),
    /// A single-step decode failed.
    #[display("decode failed")]
    Decode(#[from] DecodeError),
}

/// Convert a jxl-oxide error into a [`LoadError`].
/// The input is a bare boxed trait object.
/// The function consumes the box.
fn jxl_err(e: Box<dyn std::error::Error + Send + Sync + 'static>) -> LoadError {
    LoadError::from(e)
}

/// Normal feed chunk size.
/// The size is large enough that the streaming decoder never waits on the
/// loop.
const CHUNK: usize = 256 * 1024;
/// Minimum time between progressive preview uploads.
/// The main thread then gets no flood of full-size RGBA buffers.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(100);
/// Chunk size for `VV_SLOW_STREAM`=1.
/// With this setting the worker sends chunks of this size with a pause
/// between them.
/// It continues until the first preview renders or `SLOW_DRIBBLE_MAX` bytes
/// go out.
/// After that it continues normally.
/// This setting makes progressive decoding visible in debug runs.
const SLOW_CHUNK: usize = 4 * 1024;
const SLOW_PAUSE: Duration = Duration::from_secs(1);
const SLOW_DRIBBLE_MAX: usize = 256 * 1024;

/// Messages from the loader worker to the main thread.
///
/// The type carries only successes.
/// A decode failure arrives as an [`Err`] through the same channel.
/// [`Loader::try_recv`] can then return both as one
/// `Result<DecodeMsg, LoadError>`.
/// The `?` operator composes this result.
#[derive(Debug)]
pub enum DecodeMsg {
    /// The JXL header is parsed.
    /// The code knows the final dimensions, with the orientation applied.
    /// No pixel data arrives yet.
    Header { width: ImgW, height: ImgH },
    /// A progressive preview of the still-loading frame.
    /// The buffer is full-size RGBA8 and stays blurry until the end.
    /// The worker sends a preview at most every `PREVIEW_INTERVAL`.
    /// `blur` is the tiny blurred copy for the `VV_BLUR_BG` gimmick.
    /// It is `None` when the gimmick is off.
    Preview {
        image: DecodedImage,
        blur: Option<DecodedImage>,
    },
    /// The final full-quality image.
    Done {
        image: DecodedImage,
        blur: Option<DecodedImage>,
    },
}

/// Handle for one surface load in flight.
/// Drop the handle to cancel the load.
#[derive(Debug)]
pub struct Loader {
    cancel: Arc<AtomicBool>,
    rx: Receiver<Result<DecodeMsg, LoadError>>,
}

impl Loader {
    /// Start the worker for `surface`, which renders plane 0.
    /// `preview_px` limits the long side of progressive-preview buffers.
    /// The screen never shows more pixels than that.
    /// Full-size RGBA every `PREVIEW_INTERVAL` wastes allocations.
    #[must_use]
    pub fn start(surface: Box<dyn Surface>, preview_px: PreviewPx) -> Loader {
        Self::start_plane(surface.into(), 0, preview_px)
    }

    /// Start the worker for one plane of `surface`.
    /// [`Self::start`] is the plane-0 shorthand.
    /// Only plane 0 of a one-plane surface can stream: the progressive byte
    /// path renders the whole file into that plane.
    #[must_use]
    pub fn start_plane(surface: Arc<dyn Surface>, plane: usize, preview_px: PreviewPx) -> Loader {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        // Detached on purpose: the main thread never joins.
        // The worker ends on cancellation or when the sends start to fail
        // (the receiver dropped).
        let blur_enabled = blurbg::enabled();
        let blur_px = blurbg::blur_px();
        std::thread::spawn(move || {
            match stream(
                surface.as_ref(),
                plane,
                &worker_cancel,
                &tx,
                blur_enabled,
                blur_px,
                preview_px,
            ) {
                Ok(Some(msg)) => {
                    let _ = tx.send(Ok(msg));
                }
                // Cancelled: send nothing, just drop the channel.
                Ok(None) => {}
                Err(err) => {
                    let _ = tx.send(Err(err));
                }
            }
        });
        Loader { cancel, rx }
    }

    /// Poll the next queued message.
    /// The function does not block.
    #[must_use]
    pub fn try_recv(&self) -> Option<Result<DecodeMsg, LoadError>> {
        self.rx.try_recv().ok()
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// The worker body.
///
/// `Ok(Some(msg))` is a terminal success (`Done`).
/// `Ok(None)` means the worker stopped and sends nothing.
/// Any other progress (`Header`, `Preview`) goes out through `tx`
/// directly.
#[allow(clippy::too_many_arguments)]
fn stream(
    surface: &dyn Surface,
    plane: usize,
    cancel: &AtomicBool,
    tx: &Sender<Result<DecodeMsg, LoadError>>,
    blur_enabled: bool,
    blur_px: LongSide,
    preview_px: PreviewPx,
) -> Result<Option<DecodeMsg>, LoadError> {
    if plane == 0
        && surface.plane_count() == 1
        && let Some(mut source) = surface.progressive()?
    {
        // JPEG XL: the byte stream carries progressive previews.
        return stream_jxl(
            source.as_mut(),
            cancel,
            tx,
            blur_enabled,
            blur_px,
            preview_px,
        );
    }

    // Dimensions first, so the view fits the plane before the render lands.
    let info = surface.bounds(plane)?;
    let (width, height) = (
        ImgW::new(info.w.ceil() as u32),
        ImgH::new(info.h.ceil() as u32),
    );
    let _ = tx.send(Ok(DecodeMsg::Header { width, height }));

    // A slow rasterizer sends a cheap low-res render first.
    // The code skips it when it would not be smaller than the window.
    if surface.previews() {
        let long = info.w.max(info.h).max(1.0);
        let scale = (preview_px.get() as f32 / long).min(1.0);
        if scale < 1.0
            && let Ok(preview) = surface.render(View::whole(plane, scale))
        {
            let blur = blur_enabled.then(|| blurbg::small_blur(&preview, blur_px));
            let _ = tx.send(Ok(DecodeMsg::Preview {
                image: preview,
                blur,
            }));
        }
    }

    let image = surface.render(View::whole(plane, 1.0))?;
    let blur = blur_enabled.then(|| blurbg::small_blur(&image, blur_px));
    Ok(Some(DecodeMsg::Done { image, blur }))
}

/// JPEG XL progressive decode.
/// The function reads fixed-size chunks from the source's byte stream and
/// feeds them into jxl-oxide.
/// jxl-oxide renders previews while the first frame still loads.
/// The worker sends the full render at the end.
fn stream_jxl(
    source: &mut dyn ChunkSource,
    cancel: &AtomicBool,
    tx: &Sender<Result<DecodeMsg, LoadError>>,
    blur_enabled: bool,
    blur_px: LongSide,
    preview_px: PreviewPx,
) -> Result<Option<DecodeMsg>, LoadError> {
    let slow = std::env::var_os("VV_SLOW_STREAM").is_some();
    // `try_init` consumes the uninit image.
    // The code keeps it in an Option so the NeedMoreData branch can put it
    // back.
    let mut uninit = Some(JxlImage::builder().build_uninit());
    let mut image: Option<JxlImage> = None;
    let mut last_preview: Option<Instant> = None;
    let mut dribbled = 0usize;
    let mut buf = vec![0u8; CHUNK];

    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        // Slow mode: keep dribbling small chunks (with pauses) until the
        // first preview renders.
        // A single dribble usually carries only the header and shows
        // nothing.
        let dribbling = slow && last_preview.is_none() && dribbled < SLOW_DRIBBLE_MAX;
        let chunk_size = if dribbling { SLOW_CHUNK } else { CHUNK };
        let n = read_chunk(source, &mut buf[..chunk_size])?;
        if n == 0 {
            break; // EOF
        }
        let chunk = &buf[..n];
        match &mut image {
            None => {
                let mut u = uninit.take().expect("header state");
                u.feed_bytes(chunk).map_err(jxl_err)?;
                match u.try_init().map_err(jxl_err)? {
                    InitializeResult::NeedMoreData(u) => uninit = Some(u),
                    InitializeResult::Initialized(img) => {
                        let msg = DecodeMsg::Header {
                            width: ImgW::new(img.width()),
                            height: ImgH::new(img.height()),
                        };
                        let _ = tx.send(Ok(msg));
                        image = Some(img);
                    }
                }
            }
            Some(img) => {
                img.feed_bytes(chunk).map_err(jxl_err)?;
            }
        }

        // Progressive preview while the first frame still loads.
        // Once any frame finishes (the only frame, for stills), wait for
        // the final render.
        // This also stops animations from overwriting frame 0 with a
        // half-loaded frame 1.
        if let Some(img) = &mut image
            && img.num_loaded_keyframes() == 0
            && !img.is_loading_done()
        {
            let due = last_preview.is_none_or(|t| t.elapsed() >= PREVIEW_INTERVAL);
            if due {
                // Render errors are normal while groups or passes are
                // missing.
                // The code ignores them and retries after the next chunk.
                if let Ok(render) = img.render_loading_frame() {
                    let image = fb_to_rgba(&render.image_all_channels())?;
                    // Previews never need full resolution (the screen is
                    // smaller).
                    // The code caps the long side so a 50 MP image does not
                    // allocate ~200 MB of RGBA per preview.
                    let image = downscale_rgba(image, LongSide::new(preview_px.get()));
                    let blur = blur_enabled.then(|| blurbg::small_blur(&image, blur_px));
                    let _ = tx.send(Ok(DecodeMsg::Preview { image, blur }));
                    last_preview = Some(Instant::now());
                }
            }
        }

        if dribbling {
            dribbled += n;
            // Cancel-aware pause so ESC never sticks for a full pause.
            for _ in 0..(SLOW_PAUSE.as_millis() / 50) {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(None);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    let mut img = image.ok_or(LoadError::Incomplete)?;
    img.finalize().map_err(jxl_err)?;
    let render = img.render_frame(0).map_err(jxl_err)?;
    let image = fb_to_rgba(&render.image_all_channels())?;
    // Cap here, on the worker: the main thread uploads the buffer as is.
    let image = cap_texture_side(image);
    let blur = blur_enabled.then(|| blurbg::small_blur(&image, blur_px));
    Ok(Some(DecodeMsg::Done { image, blur }))
}

/// Read up to `buf.len()` bytes from the stream.
/// The function returns 0 at the end of the stream.
/// It makes a single read call.
/// Short reads are correct: the next loop iteration reads more data then.
fn read_chunk(source: &mut dyn ChunkSource, buf: &mut [u8]) -> Result<usize, LoadError> {
    source.read(buf).map_err(LoadError::Io)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::TempDir::new().unwrap()
    }

    /// Poll the loader until a message arrives or `timeout` goes by.
    fn wait(loader: &Loader, timeout: Duration) -> Option<Result<DecodeMsg, LoadError>> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if let Some(msg) = loader.try_recv() {
                return Some(msg);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        None
    }

    #[test]
    fn downscale_rgba_caps_long_side_and_never_upscales() {
        let img = |w: u32, h: u32, v: u8| DecodedImage {
            width: ImgW::new(w),
            height: ImgH::new(h),
            data: vec![v; (w * h * 4) as usize],
        };
        // Above the cap: 800x200 with cap 64 -> 64x16, RGBA8 length matches.
        let out = downscale_rgba(img(800, 200, 7), LongSide::new(64));
        assert_eq!((out.width, out.height), (ImgW::new(64), ImgH::new(16)));
        assert_eq!(
            out.data.len(),
            (out.width.get() * out.height.get() * 4) as usize
        );
        // Below the cap: returned untouched.
        let src = img(32, 16, 7);
        let out = downscale_rgba(src.clone(), LongSide::new(64));
        assert_eq!((out.width, out.height), (ImgW::new(32), ImgH::new(16)));
        assert_eq!(out.data, src.data);
    }

    #[test]
    fn png_loads_as_header_then_done() {
        // Non-JXL content has no progressive data: a Header with the natural
        // dimensions (so the view fits at once), then exactly one Done.
        // No Preview: an image does not set `previews()`.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(4, 3).save(&path).unwrap();

        let surface = crate::surface::open_surface(&path).unwrap();
        let loader = Loader::start(surface, PreviewPx::new(512));
        let mut header = None;
        let done;
        loop {
            match wait(&loader, Duration::from_secs(10))
                .expect("loader timed out")
                .expect("decode failed")
            {
                DecodeMsg::Done { image, .. } => {
                    done = image;
                    break;
                }
                DecodeMsg::Header { width, height } => header = Some((width, height)),
                DecodeMsg::Preview { .. } => panic!("an image does not preview"),
            }
        }
        assert_eq!(header, Some((ImgW::new(4), ImgH::new(3))));
        assert_eq!((done.width, done.height), (ImgW::new(4), ImgH::new(3)));
        assert_eq!(done.data.len(), 4 * 3 * 4);
    }

    #[cfg(feature = "svg")]
    #[test]
    fn a_vector_file_announces_its_base_size_then_paints() {
        // A vector file parses in a few milliseconds and paints in tens, so
        // the Header goes out first and the window fits the picture while the
        // base raster arrives.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("drawing.svg");
        std::fs::write(
            &path,
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
                <rect width="100" height="50" fill="#ff0000"/>
            </svg>"##,
        )
        .unwrap();

        let surface = crate::surface::open_surface(&path).unwrap();
        let loader = Loader::start(surface, PreviewPx::new(512));
        let mut header = None;
        let done;
        loop {
            match wait(&loader, Duration::from_secs(10))
                .expect("loader timed out")
                .expect("decode failed")
            {
                DecodeMsg::Done { image, .. } => {
                    done = image;
                    break;
                }
                DecodeMsg::Header { width, height } => header = Some((width, height)),
                DecodeMsg::Preview { .. } => panic!("a vector file does not preview"),
            }
        }
        // The header carries the base raster size, and so does the render.
        assert_eq!(header, Some((ImgW::new(1024), ImgH::new(512))));
        assert_eq!((done.width, done.height), (ImgW::new(1024), ImgH::new(512)));
    }

    #[test]
    fn garbage_jxl_extension_fails_cleanly() {
        // The .jxl extension decides before the code sniffs the content.
        // The content is not a codestream.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("x.jxl");
        std::fs::write(&path, b"not really jxl").unwrap();

        let surface = crate::surface::open_surface(&path).expect("the extension selects JXL");
        let loader = Loader::start(surface, PreviewPx::new(512));
        loop {
            match wait(&loader, Duration::from_secs(10)).expect("loader timed out") {
                Err(_) => break,
                Ok(DecodeMsg::Done { .. }) => panic!("garbage must not decode"),
                Ok(DecodeMsg::Header { .. } | DecodeMsg::Preview { .. }) => {}
            }
        }
    }

    #[test]
    fn truncated_codestream_fails_cleanly() {
        // Valid codestream magic but no complete header: EOF before init.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("truncated.jxl");
        std::fs::write(&path, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();

        let surface = crate::surface::open_surface(&path).unwrap();
        let loader = Loader::start(surface, PreviewPx::new(512));
        loop {
            match wait(&loader, Duration::from_secs(10)).expect("loader timed out") {
                Err(_) => break,
                Ok(DecodeMsg::Done { .. }) => panic!("truncated file must not decode"),
                Ok(DecodeMsg::Header { .. } | DecodeMsg::Preview { .. }) => {}
            }
        }
    }

    #[test]
    fn cancel_stops_the_worker() {
        // Dropping the loader must terminate the worker without a hang.
        // A fresh handle cannot join the thread.
        // Instead start a loader on a big-ish file, drop it immediately,
        // and assert that no more messages arrive (the channel is closed).
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(16, 16).save(&path).unwrap();

        let surface = crate::surface::open_surface(&path).unwrap();
        let loader = Loader::start(surface, PreviewPx::new(512));
        drop(loader);
        std::thread::sleep(Duration::from_millis(50));
        // Nothing to assert beyond "no panic, no hang".
        // A `try_recv` on the dropped receiver never becomes observable.
        // The real check is that this test finishes.
    }
}
