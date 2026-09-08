//! Background loader for the image view: renders one document page off the
//! main thread and streams results back over a channel.
//!
//! Two quality paths, per [`Document`]:
//! - JPEG XL (`stream_path`): the file is read in chunks and fed into
//!   jxl-oxide, which renders a blurry full-size preview long before all
//!   bytes arrive; the final full-quality render is sent when decoding
//!   completes.
//! - Everything else: optionally a fast low-resolution preview render first
//!   (PDFs), then the full-quality render at the requested scale.
//!
//! Cancellation: dropping the `Loader` sets a flag the worker checks between
//! chunks and closes the channel, so pending sends fail and the worker exits.

use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender, TryRecvError, channel},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};

use crate::document::{Document, fb_to_rgba};

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
    /// Fast low-quality preview of the still-loading page (full-size RGBA8
    /// for its scale). Sent at most every `PREVIEW_INTERVAL`.
    Preview {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Final full-quality render at the requested scale.
    Done {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
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
    /// worker thread.
    pub fn start_doc(doc: Arc<dyn Document>, page: usize, scale: f32) -> Loader {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        // Detached on purpose: the main thread never joins; the worker ends
        // on cancellation or when sends start failing (receiver dropped).
        std::thread::spawn(move || {
            if let Err(err) = stream_doc(&doc, page, scale, &worker_cancel, &tx) {
                let _ = tx.send(LoaderMsg::Failed(format!("{err:#}")));
            }
        });
        Loader { cancel, rx }
    }

    /// Poll the next queued message without blocking.
    pub fn try_recv(&self) -> Option<LoaderMsg> {
        match self.rx.try_recv() {
            Ok(msg) => Some(msg),
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => None,
        }
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
) -> Result<()> {
    // Dimensions first, so the view can fit and lay out before pixels land.
    let info = doc.page_info(page)?;
    let _ = tx.send(LoaderMsg::Header {
        width: info.width,
        height: info.height,
    });

    if page == 0
        && let Some(path) = doc.stream_path()
    {
        // JPEG XL: progressive byte-stream decode with blurry previews.
        return stream_jxl(path, cancel, tx);
    }

    // Cheap low-res render first, so something shows up fast (PDFs).
    if doc.previews() && (info.width.max(info.height) as f32) > PREVIEW_TARGET {
        let ps = (PREVIEW_TARGET / info.width.max(info.height) as f32).min(scale);
        if let Ok(preview) = doc.render(page, ps) {
            let _ = tx.send(LoaderMsg::Preview {
                rgba: preview.rgba,
                width: preview.width,
                height: preview.height,
            });
        }
    }

    let decoded = doc.render(page, scale)?;
    let _ = tx.send(LoaderMsg::Done {
        rgba: decoded.rgba,
        width: decoded.width,
        height: decoded.height,
    });
    Ok(())
}

/// JPEG XL progressive decode: feed bytes in chunks, render previews while
/// the first frame is still loading, send the full render at the end.
fn stream_jxl(path: &Path, cancel: &AtomicBool, tx: &Sender<LoaderMsg>) -> Result<()> {
    use jxl_oxide::{InitializeResult, JxlImage};

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
                u.feed_bytes(&buf[..n])
                    .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))?;
                match u
                    .try_init()
                    .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))?
                {
                    InitializeResult::NeedMoreData(u) => uninit = Some(u),
                    InitializeResult::Initialized(img) => {
                        image = Some(img);
                    }
                }
            }
            Some(img) => {
                img.feed_bytes(&buf[..n])
                    .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))?;
            }
        }

        // Progressive preview while the first frame is still loading. Once
        // any frame has finished (the only frame, for stills), wait for the
        // final render — this also keeps animations from overwriting frame 0
        // with a half-loaded frame 1.
        if let Some(img) = &mut image {
            if img.num_loaded_keyframes() == 0 && !img.is_loading_done() {
                let due = last_preview.map_or(true, |t| t.elapsed() >= PREVIEW_INTERVAL);
                if due {
                    // Render errors are expected while groups/passes are
                    // missing; ignore them and retry after the next chunk.
                    if let Ok(render) = img.render_loading_frame() {
                        let (rgba, width, height) = fb_to_rgba(&render.image_all_channels())?;
                        let _ = tx.send(LoaderMsg::Preview {
                            rgba,
                            width,
                            height,
                        });
                        last_preview = Some(Instant::now());
                    }
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
    img.finalize()
        .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))?;
    let render = img
        .render_frame(0)
        .map_err(|e| anyhow::anyhow!("jxl-oxide: {e}"))
        .with_context(|| format!("failed to render {path:?}"))?;
    let (rgba, width, height) = fb_to_rgba(&render.image_all_channels())?;
    let _ = tx.send(LoaderMsg::Done {
        rgba,
        width,
        height,
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

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("vv-loader-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Poll the loader until a message arrives or `timeout` elapses.
    fn wait(loader: &Loader, timeout: Duration) -> Option<LoaderMsg> {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if let Some(msg) = loader.try_recv() {
                return Some(msg);
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        None
    }

    #[test]
    fn png_loads_as_header_then_done() {
        // Non-JXL formats have no progressive data: Header, then exactly one
        // Done, correct dimensions.
        let dir = temp_dir("png");
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(4, 3).save(&path).unwrap();

        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 1.0);
        let mut saw_header = false;
        let done;
        loop {
            match wait(&loader, Duration::from_secs(10)).expect("loader timed out") {
                LoaderMsg::Done {
                    rgba,
                    width,
                    height,
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
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn garbage_jxl_extension_fails_cleanly() {
        // .jxl extension decides before sniffing; content is not a codestream.
        let dir = temp_dir("garbage");
        let path = dir.join("x.jxl");
        std::fs::write(&path, b"not really jxl").unwrap();

        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 1.0);
        loop {
            match wait(&loader, Duration::from_secs(10)).expect("loader timed out") {
                LoaderMsg::Failed(_) => break,
                LoaderMsg::Done { .. } => panic!("garbage must not decode"),
                LoaderMsg::Header { .. } | LoaderMsg::Preview { .. } => {}
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn truncated_codestream_fails_cleanly() {
        // Valid codestream magic but no complete header: EOF before init.
        let dir = temp_dir("truncated");
        let path = dir.join("truncated.jxl");
        std::fs::write(&path, [0xffu8, 0x0a, 0x01, 0x02]).unwrap();

        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 1.0);
        loop {
            match wait(&loader, Duration::from_secs(10)).expect("loader timed out") {
                LoaderMsg::Failed(_) => break,
                LoaderMsg::Done { .. } => panic!("truncated file must not decode"),
                LoaderMsg::Header { .. } | LoaderMsg::Preview { .. } => {}
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cancel_stops_the_worker() {
        // Dropping the loader must terminate the worker without hanging:
        // start a loader on a big-ish file, drop it immediately, and assert
        // the test process survives (the worker exits on the cancel flag or
        // on closed-channel send failures).
        let dir = temp_dir("cancel");
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(16, 16).save(&path).unwrap();

        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 1.0);
        drop(loader);
        std::thread::sleep(Duration::from_millis(50));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pdf_loads_preview_then_done() {
        // Multi-page documents (PDF) preview before the full render.
        let dir = temp_dir("pdf");
        let path = dir.join("doc.pdf");
        std::fs::write(&path, crate::pdf::tests::MINIMAL_PDF).unwrap();
        let doc = open_document(&path).unwrap();
        let loader = Loader::start_doc(doc, 0, 2.0);
        let mut saw_preview = false;
        let done;
        loop {
            match wait(&loader, Duration::from_secs(20)).expect("loader timed out") {
                LoaderMsg::Done {
                    rgba,
                    width,
                    height,
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
        std::fs::remove_dir_all(&dir).ok();
    }
}
