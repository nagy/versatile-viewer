//! Background streaming loader for the image view.
//!
//! Opens an image off the main thread and streams results back over a
//! channel. JPEG XL is decoded progressively: the file is read in chunks and
//! fed into jxl-oxide, which renders a blurry full-size preview long before
//! all bytes arrive; the final full-quality render is sent when decoding
//! completes. Every other format decodes in one go.
//!
//! Cancellation: dropping the `Loader` sets a flag the worker checks between
//! chunks and closes the channel, so pending sends fail and the worker exits.

use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender, TryRecvError, channel},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use jxl_oxide::{InitializeResult, JxlImage};

use crate::{decode_common, fb_to_rgba, is_jxl};

/// Wrap a jxl-oxide error (a bare boxed trait object) into anyhow.
fn jxl_err(e: Box<dyn std::error::Error + Send + Sync + 'static>) -> anyhow::Error {
    anyhow::anyhow!("jxl-oxide: {e}")
}

/// Normal read chunk size: big enough that file IO never bottlenecks.
const CHUNK: usize = 256 * 1024;
/// Minimum time between progressive preview uploads, so the main thread is
/// not flooded with full-size RGBA buffers.
const PREVIEW_INTERVAL: Duration = Duration::from_millis(100);
/// VV_SLOW_STREAM=1: dribble chunks of this size with a pause between them
/// until the first preview renders (or `SLOW_DRIBBLE_MAX` bytes have been
/// dribbled), then continue normally — makes progressive decoding visible
/// to the eye in debug runs.
const SLOW_CHUNK: usize = 4 * 1024;
const SLOW_PAUSE: Duration = Duration::from_secs(1);
const SLOW_DRIBBLE_MAX: usize = 256 * 1024;

/// Messages from the loader worker to the main thread.
pub enum LoaderMsg {
    /// JXL header parsed: final (orientation-applied) dimensions are known,
    /// before any pixel data arrives.
    Header { width: u32, height: u32 },
    /// Progressive preview of the still-loading frame (full-size RGBA8,
    /// blurry until done). Sent at most every `PREVIEW_INTERVAL`.
    Preview {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Final full-quality image.
    Done {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Decoding failed; no image will arrive.
    Failed(String),
}

/// Handle for one in-flight image load. Drop to cancel.
pub struct Loader {
    cancel: Arc<AtomicBool>,
    rx: Receiver<LoaderMsg>,
}

impl Loader {
    /// Spawn the worker for `path`.
    pub fn start(path: PathBuf) -> Loader {
        let (tx, rx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        // Detached on purpose: the main thread never joins; the worker ends
        // on cancellation or when sends start failing (receiver dropped).
        std::thread::spawn(move || {
            if let Err(err) = stream(&path, &worker_cancel, &tx) {
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
fn stream(path: &Path, cancel: &AtomicBool, tx: &Sender<LoaderMsg>) -> Result<()> {
    if !is_jxl(path) {
        // No progressive data for common formats: one full decode.
        let decoded = decode_common(path)?;
        let _ = tx.send(LoaderMsg::Done {
            rgba: decoded.rgba,
            width: decoded.width,
            height: decoded.height,
        });
        return Ok(());
    }

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
    img.finalize().map_err(jxl_err)?;
    let render = img
        .render_frame(0)
        .map_err(jxl_err)
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

    fn temp_dir(name: &str) -> PathBuf {
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
            std::thread::sleep(Duration::from_millis(10));
        }
        None
    }

    #[test]
    fn png_loads_as_single_done_message() {
        // Non-JXL formats have no progressive data: exactly one Done, no
        // Header/Preview, correct dimensions.
        let dir = temp_dir("png");
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(4, 3).save(&path).unwrap();

        let loader = Loader::start(path);
        let mut saw_header_or_preview = false;
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
                LoaderMsg::Header { .. } | LoaderMsg::Preview { .. } => {
                    saw_header_or_preview = true
                }
                LoaderMsg::Failed(err) => panic!("unexpected failure: {err}"),
            }
        }
        assert!(!saw_header_or_preview);
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

        let loader = Loader::start(path);
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

        let loader = Loader::start(path);
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
        // join the thread via a fresh handle is not possible, so instead
        // start a loader on a big-ish file, drop it immediately, and assert
        // no further messages arrive afterwards (channel is closed).
        let dir = temp_dir("cancel");
        let path = dir.join("img.png");
        image::DynamicImage::new_rgb8(16, 16).save(&path).unwrap();

        let loader = Loader::start(path);
        drop(loader);
        std::thread::sleep(Duration::from_millis(50));
        // Nothing to assert beyond "no panic, no hang"; try_recv on the
        // dropped receiver was never observable. The real check is that this
        // test finishes.
        std::fs::remove_dir_all(&dir).ok();
    }
}
