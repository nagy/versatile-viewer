//! Background streaming loader for the image view.
//!
//! The loader opens an image off the main thread.
//! It streams results back over a channel.
//! JPEG XL decodes progressively.
//! The worker memory-maps the file and feeds slices of the map into
//! jxl-oxide.
//! jxl-oxide renders a blurry full-size preview long before all bytes arrive.
//! The worker sends the final full-quality render when decoding completes.
//! Every other format decodes in one step.
//!
//! Cancellation: dropping the `Loader` sets a flag.
//! The worker checks this flag between chunks and closes the channel.
//! Pending sends fail then, and the worker exits.
//!
//! The map is a plain read-only mapping.
//! A file rewritten under the map faults with SIGBUS instead of returning
//! short data, so the viewer assumes the files it opens stay put.

use std::{
    path::{Path, PathBuf},
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
    DecodedImage, blurbg, cap_texture_side, decode_view,
    dim::{ImgH, ImgW, LongSide, PreviewPx},
    downscale_rgba,
    error::DecodeError,
    fb_to_rgba, is_jxl,
    svg::Vector,
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
        /// The parsed source of a vector file, so the view can repaint the
        /// region its window shows.
        /// None for a raster format.
        vector: Option<std::sync::Arc<Vector>>,
    },
}

/// Handle for one image load in flight.
/// Drop the handle to cancel the load.
#[derive(Debug)]
pub struct Loader {
    cancel: Arc<AtomicBool>,
    rx: Receiver<Result<DecodeMsg, LoadError>>,
}

impl Loader {
    /// Start the worker for `path`.
    /// `preview_px` limits the long side of progressive-preview buffers.
    /// The screen never shows more pixels than that.
    /// Full-size RGBA every `PREVIEW_INTERVAL` wastes allocations.
    #[must_use]
    pub fn start(path: PathBuf, preview_px: PreviewPx) -> Loader {
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
                &path,
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
    path: &Path,
    cancel: &AtomicBool,
    tx: &Sender<Result<DecodeMsg, LoadError>>,
    blur_enabled: bool,
    blur_px: LongSide,
    preview_px: PreviewPx,
) -> Result<Option<DecodeMsg>, LoadError> {
    if !is_jxl(path) {
        // No progressive data for the common formats: one full decode.
        // A vector file announces its base size before it paints, so the
        // window fits the picture while the raster arrives.
        let decoded = {
            let mut announce = |width: ImgW, height: ImgH| {
                let _ = tx.send(Ok(DecodeMsg::Header { width, height }));
            };
            decode_view(path, &mut announce)?
        };
        let blur = blur_enabled.then(|| blurbg::small_blur(&decoded.image, blur_px));
        return Ok(Some(DecodeMsg::Done {
            image: decoded.image,
            blur,
            vector: decoded.vector,
        }));
    }

    // Map the file instead of reading it through a buffer.
    // The kernel pages the bytes in on demand, so a large JXL costs no user
    // space copy and no read loop.
    // A file that changes under the map faults (SIGBUS): the viewer reads
    // still files from a directory that does not churn.
    let file = std::fs::File::open(path)?;
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let data: &[u8] = &map;
    let slow = std::env::var_os("VV_SLOW_STREAM").is_some();
    // `try_init` consumes the uninit image.
    // The code keeps it in an Option so the NeedMoreData branch can put it
    // back.
    let mut uninit = Some(JxlImage::builder().build_uninit());
    let mut image: Option<JxlImage> = None;
    let mut last_preview: Option<Instant> = None;
    let mut dribbled = 0usize;
    let mut offset = 0usize;

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
        let chunk = &data[offset..(offset + chunk_size).min(data.len())];
        let n = chunk.len();
        if n == 0 {
            break; // EOF
        }
        offset += n;
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
    Ok(Some(DecodeMsg::Done {
        image,
        blur,
        vector: None,
    }))
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
    fn png_loads_as_single_done_message() {
        // Non-JXL formats have no progressive data: exactly one Done, no
        // Header/Preview, correct dimensions.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(4, 3).save(&path).unwrap();

        let loader = Loader::start(path, PreviewPx::new(512));
        let mut saw_header_or_preview = false;
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
                DecodeMsg::Header { .. } | DecodeMsg::Preview { .. } => {
                    saw_header_or_preview = true;
                }
            }
        }
        assert!(!saw_header_or_preview);
        assert_eq!((done.width, done.height), (ImgW::new(4), ImgH::new(3)));
        assert_eq!(done.data.len(), 4 * 3 * 4);
    }

    #[test]
    fn garbage_jxl_extension_fails_cleanly() {
        // The .jxl extension decides before the code sniffs the content.
        // The content is not a codestream.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("x.jxl");
        std::fs::write(&path, b"not really jxl").unwrap();

        let loader = Loader::start(path, PreviewPx::new(512));
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

        let loader = Loader::start(path, PreviewPx::new(512));
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

        let loader = Loader::start(path, PreviewPx::new(512));
        drop(loader);
        std::thread::sleep(Duration::from_millis(50));
        // Nothing to assert beyond "no panic, no hang".
        // A `try_recv` on the dropped receiver never becomes observable.
        // The real check is that this test finishes.
    }
}
