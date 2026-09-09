//! Background loader for the image view: renders one document page off the
//! main thread and streams results back over a channel.
//!
//! Two quality paths, per [`Document`]:
//! - JPEG XL (`stream_path`): the file is read in chunks and fed into
//!   jxl-oxide, which renders a blurry full-size preview long before all
//!   bytes arrive; the final full-quality render is sent when decoding
//!   completes.
//! - Everything else: dimensions first ([`LoaderMsg::Header`]), then — for
//!   documents that preview (PDFs) — a fast low-resolution render, then the
//!   full-quality render at the requested scale.
//!
//! Cancellation: dropping the `Loader` sets a flag the worker checks between
//! chunks and closes the channel, so pending sends fail and the worker exits.

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use jxl_oxide::{InitializeResult, JxlImage};

use crate::{
    blurbg::{self, BlurData},
    document::{Document, downscale_rgba, fb_to_rgba},
};

/// Wrap a jxl-oxide error (a bare boxed trait object) into anyhow,
/// consuming the box (anyhow adopts `Box<dyn Error + Send + Sync>`).
fn jxl_err(e: Box<dyn std::error::Error + Send + Sync + 'static>) -> anyhow::Error {
    anyhow::Error::from_boxed(e).context("jxl-oxide")
}

/// Normal read chunk size: big enough that file IO never bottlenecks.
const CHUNK: usize = 256 * 1024;
/// Minimum time between progressive preview uploads, so the main thread is
/// not flooded with full-size RGBA buffers.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(100);
/// Target for the cheap preview render's longest side, in pixels.
const PREVIEW_TARGET: f32 = 600.0;
/// VV_SLOW_STREAM=1: dribble chunks of this size with a pause between them
/// until the first preview renders (or `SLOW_DRIBBLE_MAX` bytes have been
/// dribbled), then continue normally — makes progressive decoding visible
/// to the eye in debug runs.
const SLOW_CHUNK: usize = 4 * 1024;
const SLOW_PAUSE: Duration = Duration::from_secs(1);
const SLOW_DRIBBLE_MAX: usize = 256 * 1024;

/// Messages from the loader worker to the main thread.
pub enum LoaderMsg {
    /// Page dimensions are known (in natural units: pixels or PDF points),
    /// before any pixel data arrives.
    Header { width: u32, height: u32 },
    /// Progressive preview of the still-loading frame (full-size RGBA8,
    /// blurry until done). Sent at most every `PREVIEW_INTERVAL`. `blur` is
    /// the tiny blurred copy for the `VV_BLUR_BG` gimmick (None when off).
    Preview {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        blur: Option<BlurData>,
    },
    /// Final full-quality image.
    Done {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        blur: Option<BlurData>,
    },
    /// Decoding failed; no image will arrive.
    Failed(String),
}

/// Handle for one in-flight page load. Drop to cancel.
pub struct Loader {
    cancel: Arc<AtomicBool>,
    rx: Receiver<LoaderMsg>,
}

impl Loader {
    /// Start rendering `page` of `doc` at `scale` (1.0 = natural size) on a
    /// worker thread. `preview_px` caps the long side of preview buffers:
    /// the screen never shows more pixels than that, so shipping full-size
    /// RGBA per preview is pure allocation churn.
    pub fn start_doc(doc: Arc<dyn Document>, page: usize, scale: f32, preview_px: u32) -> Loader {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        // Detached on purpose: the main thread never joins; the worker ends
        // on cancellation or when sends start failing (receiver dropped).
        let blur_enabled = blurbg::enabled();
        let blur_px = blurbg::blur_px();
        std::thread::spawn(move || {
            if let Err(err) = stream_doc(
                &doc,
                page,
                scale,
                &worker_cancel,
                &tx,
                blur_enabled,
                blur_px,
                preview_px,
            ) {
                let _ = tx.send(LoaderMsg::Failed(format!("{err:#}")));
            }
        });
        Loader { cancel, rx }
    }

    /// Poll the next queued message without blocking.
    pub fn try_recv(&self) -> Option<LoaderMsg> {
        self.rx.try_recv().ok()
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Worker body. The caller turns errors into `Failed` messages.
fn stream_doc(
    doc: &Arc<dyn Document>,
    page: usize,
    scale: f32,
    cancel: &AtomicBool,
    tx: &Sender<LoaderMsg>,
    blur_enabled: bool,
    blur_px: u32,
    preview_px: u32,
) -> Result<()> {
    if page == 0
        && let Some(path) = doc.stream_path()
    {
        // JPEG XL: progressive byte-stream decode with blurry previews (the
        // header message is sent when jxl-oxide parses the frame header).
        return stream_jxl(path, cancel, tx, blur_enabled, blur_px, preview_px);
    }

    // Dimensions first, so the view can fit and lay out before pixels land.
    let info = doc.page_info(page)?;
    let _ = tx.send(LoaderMsg::Header {
        width: info.width,
        height: info.height,
    });

    // Cheap low-res render first, so something shows up fast (PDFs); skip
    // it when the full render is no bigger (small pages, low fit scale).
    if doc.previews() {
        let long = info.width.max(info.height).max(1) as f32;
        let ps = scale
            .min(PREVIEW_TARGET / long)
            .min(preview_px.max(1) as f32 / long);
        if ps < scale
            && let Ok(preview) = doc.render(page, ps)
        {
            let blur = blur_enabled
                .then(|| blurbg::small_blur(&preview.rgba, preview.width, preview.height, blur_px));
            let _ = tx.send(LoaderMsg::Preview {
                rgba: preview.rgba,
                width: preview.width,
                height: preview.height,
                blur,
            });
        }
    }

    let decoded = doc.render(page, scale)?;
    let blur = blur_enabled
        .then(|| blurbg::small_blur(&decoded.rgba, decoded.width, decoded.height, blur_px));
    let _ = tx.send(LoaderMsg::Done {
        rgba: decoded.rgba,
        width: decoded.width,
        height: decoded.height,
        blur,
    });
    Ok(())
}

/// JPEG XL progressive decode: feed bytes in chunks, render previews while
/// the first frame is still loading, send the full render at the end.
fn stream_jxl(
    path: &Path,
    cancel: &AtomicBool,
    tx: &Sender<LoaderMsg>,
    blur_enabled: bool,
    blur_px: u32,
    preview_px: u32,
) -> Result<()> {
    let mut file = std::fs::File::open(path).with_context(|| format!("failed to open {path:?}"))?;
    let slow = std::env::var_os("VV_SLOW_STREAM").is_some();
    let mut buf = vec![0u8; CHUNK];
    // `try_init` consumes the uninit image; keep it in an Option so the
    // NeedMoreData branch can put it back.
    let mut uninit = Some(JxlImage::builder().build_uninit());
    let mut image: Option<JxlImage> = None;
    let mut last_preview: Option<Instant> = None;
    let mut dribbled = 0usize;

    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        // Slow mode: keep dribbling small chunks (with pauses) until the
        // first preview actually rendered; a single dribble would usually
        // only carry the header and show nothing.
        let dribbling = slow && last_preview.is_none() && dribbled < SLOW_DRIBBLE_MAX;
        let chunk_size = if dribbling { SLOW_CHUNK } else { CHUNK };
        let n = read_chunk(&mut file, &mut buf[..chunk_size])?;
        if n == 0 {
            break; // EOF
        }
        match &mut image {
            None => {
                let mut u = uninit.take().expect("header state");
                u.feed_bytes(&buf[..n]).map_err(jxl_err)?;
                match u.try_init().map_err(jxl_err)? {
                    InitializeResult::NeedMoreData(u) => uninit = Some(u),
                    InitializeResult::Initialized(img) => {
                        let msg = LoaderMsg::Header {
                            width: img.width(),
                            height: img.height(),
                        };
                        let _ = tx.send(msg);
                        image = Some(img);
                    }
                }
            }
            Some(img) => {
                img.feed_bytes(&buf[..n]).map_err(jxl_err)?;
            }
        }

        // Progressive preview while the first frame is still loading. Once
        // any frame has finished (the only frame, for stills), wait for the
        // final render — this also keeps animations from overwriting frame 0
        // with a half-loaded frame 1.
        if let Some(img) = &mut image
            && img.num_loaded_keyframes() == 0
            && !img.is_loading_done()
        {
            let due = last_preview.is_none_or(|t| t.elapsed() >= PREVIEW_INTERVAL);
            if due {
                // Render errors are expected while groups/passes are
                // missing; ignore them and retry after the next chunk.
                if let Ok(render) = img.render_loading_frame() {
                    let (rgba, width, height) = fb_to_rgba(&render.image_all_channels())?;
                    // Previews never need full resolution (the screen is
                    // smaller); cap the long side so a 50 MP image does not
                    // allocate ~200 MB of RGBA per preview.
                    let (rgba, width, height) = downscale_rgba(rgba, width, height, preview_px);
                    let blur =
                        blur_enabled.then(|| blurbg::small_blur(&rgba, width, height, blur_px));
                    let _ = tx.send(LoaderMsg::Preview {
                        rgba,
                        width,
                        height,
                        blur,
                    });
                    last_preview = Some(Instant::now());
                }
            }
        }

        if dribbling {
            dribbled += n;
            // Cancel-aware pause so ESC never sticks for a full pause.
            for _ in 0..(SLOW_PAUSE.as_millis() / 50) {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    let mut img = image.context("file ended before the image header was complete")?;
    img.finalize().map_err(jxl_err)?;
    let render = img
        .render_frame(0)
        .map_err(jxl_err)
        .with_context(|| format!("failed to render {}", path.display()))?;
    let (rgba, width, height) = fb_to_rgba(&render.image_all_channels())?;
    let blur = blur_enabled.then(|| blurbg::small_blur(&rgba, width, height, blur_px));
    let _ = tx.send(LoaderMsg::Done {
        rgba,
        width,
        height,
        blur,
    });
    Ok(())
}

/// Read up to `buf.len()` bytes; 0 at EOF. A single read call: regular files
/// normally fill the whole buffer, and short reads are fine anyway (the next
/// loop iteration just reads more).
fn read_chunk(file: &mut std::fs::File, buf: &mut [u8]) -> std::io::Result<usize> {
    use std::io::Read;
    file.read(buf)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::document::open_document;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::TempDir::new().unwrap()
    }

    /// Poll the loader until a message arrives or `timeout` elapses.
    fn wait(loader: &Loader, timeout: Duration) -> Option<LoaderMsg> {
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
        // Above the cap: 800x200 with cap 64 -> 64x16, RGBA8 length matches.
        let (out, w, h) = downscale_rgba(vec![7u8; 800 * 200 * 4], 800, 200, 64);
        assert_eq!((w, h), (64, 16));
        assert_eq!(out.len(), (w * h * 4) as usize);
        // Below the cap: returned untouched.
        let rgba = vec![7u8; 32 * 16 * 4];
        let (out, w, h) = downscale_rgba(rgba.clone(), 32, 16, 64);
        assert_eq!((w, h), (32, 16));
        assert_eq!(out, rgba);
    }

    #[test]
    fn png_loads_as_header_then_done() {
        // Non-JXL formats have no progressive data: Header, then exactly one
        // Done, correct dimensions.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(4, 3).save(&path).unwrap();

        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 1.0, 512);
        let mut saw_header = false;
        let done;
        loop {
            match wait(&loader, Duration::from_secs(10)).expect("loader timed out") {
                LoaderMsg::Done {
                    rgba,
                    width,
                    height,
                    ..
                } => {
                    done = (rgba, width, height);
                    break;
                }
                LoaderMsg::Header { width, height } => {
                    saw_header = true;
                    assert_eq!((width, height), (4, 3));
                }
                LoaderMsg::Preview { .. } => panic!("png must not preview"),
                LoaderMsg::Failed(err) => panic!("unexpected failure: {err}"),
            }
        }
        assert!(saw_header);
        let (rgba, width, height) = done;
        assert_eq!((width, height), (4, 3));
        assert_eq!(rgba.len(), 4 * 3 * 4);
    }

    #[test]
    fn garbage_jxl_extension_fails_cleanly() {
        // .jxl extension decides before sniffing; content is not a codestream.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("x.jxl");
        std::fs::write(&path, b"not really jxl").unwrap();

        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 1.0, 512);
        loop {
            match wait(&loader, Duration::from_secs(10)).expect("loader timed out") {
                LoaderMsg::Failed(_) => break,
                LoaderMsg::Done { .. } => panic!("garbage must not decode"),
                LoaderMsg::Header { .. } | LoaderMsg::Preview { .. } => {}
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

        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 1.0, 512);
        loop {
            match wait(&loader, Duration::from_secs(10)).expect("loader timed out") {
                LoaderMsg::Failed(_) => break,
                LoaderMsg::Done { .. } => panic!("truncated file must not decode"),
                LoaderMsg::Header { .. } | LoaderMsg::Preview { .. } => {}
            }
        }
    }

    #[test]
    fn cancel_stops_the_worker() {
        // Dropping the loader must terminate the worker without hanging:
        // start a loader on a big-ish file, drop it immediately, and assert
        // the test process survives (the worker exits on the cancel flag or
        // on closed-channel send failures). The real check is that this
        // test finishes.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(16, 16).save(&path).unwrap();

        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 1.0, 512);
        drop(loader);
        std::thread::sleep(Duration::from_millis(50));
    }

    #[test]
    fn pdf_loads_preview_then_done() {
        // Multi-page documents (PDF) preview before the full render.
        let tmp = temp_dir();
        let dir = tmp.path();
        let path = dir.join("doc.pdf");
        std::fs::write(&path, crate::pdf::tests::MINIMAL_PDF).unwrap();
        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 2.0, 512);
        let mut saw_preview = false;
        let done;
        loop {
            match wait(&loader, Duration::from_secs(20)).expect("loader timed out") {
                LoaderMsg::Done {
                    rgba,
                    width,
                    height,
                    ..
                } => {
                    done = (rgba, width, height);
                    break;
                }
                LoaderMsg::Header { width, height } => {
                    assert_eq!((width, height), (612, 792)); // letter, in points
                }
                LoaderMsg::Preview { width, height, .. } => {
                    saw_preview = true;
                    // Preview longest side capped near PREVIEW_TARGET.
                    assert!(width.max(height) <= 612); // 600 * 792/612 rounds up
                }
                LoaderMsg::Failed(err) => panic!("unexpected failure: {err}"),
            }
        }
        assert!(saw_preview);
        let (_, width, height) = done;
        assert_eq!((width, height), (1224, 1584)); // scale 2.0
    }
}
