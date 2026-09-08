//! versatile-viewer — image viewer (JXL first-class, plus PNG/JPEG) with a
//! directory thumbnail grid. q quits; ESC/Enter toggle grid ↔ image view.
//!
//! Usage: versatile-viewer <image-path-or-directory>

use std::{env, path::Path};

use anyhow::{Context, Result, bail};
use raylib::{
    color::Color,
    consts::{KeyboardKey, PixelFormat},
    prelude::*,
};

mod document;
mod grid;
mod keyrepeat;
mod loader;
mod pdf;

use std::sync::Arc;

use document::{DecodedImage, Document, PageInfo, open_document};
use grid::{EntrySource, Grid, GridAction};
use loader::{Loader, LoaderMsg};

/// How the image is scaled to the window. Scale is recomputed every frame,
/// so resizing always stays correct.
#[derive(Clone, Copy, PartialEq)]
enum ZoomMode {
    /// Fit down to the window, centered; never upscaled (default).
    FitDown,
    /// Fit all sides: scale up or down until the image first touches a
    /// border (Shift+W).
    FitAll,
    /// Fit to the window width (e).
    FitWidth,
    /// Fit to the window height (Shift+E).
    FitHeight,
    /// Fill the window: scale until the image covers every side (t);
    /// the shorter relative dimension overflows and is cropped.
    Fill,
    /// Free zoom factor, set with +/- (multiples of the last fit scale).
    Free(f32),
}

/// Viewer screen: thumbnail grid or single image.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Grid,
    Image,
}

/// Decode an image and upload it as a GPU texture. The raw RGBA buffer is
/// wrapped in a raylib Image without re-encoding; the ffi::Image only
/// borrows it, so the wrapper is forgotten after upload and the Vec stays
/// Upload a raw RGBA8 buffer as a GPU texture. Must be called on the main
/// thread (GL context lives there). The buffer is only borrowed for the
/// upload; the ffi::Image wrapper is forgotten so raylib never frees the
/// caller's Vec.
fn upload_rgba(
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<Texture2D> {
    let ffi_image = raylib::ffi::Image {
        data: rgba.as_ptr() as *mut std::os::raw::c_void,
        width: width as i32,
        height: height as i32,
        mipmaps: 1,
        format: PixelFormat::PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 as i32,
    };
    let image = unsafe { Image::from_raw(ffi_image) };
    let texture = rl.load_texture_from_image(thread, &image)?;
    image.to_raw(); // forget: drop would MemFree our borrowed Vec
    Ok(texture)
}

/// Show a new RGBA8 frame in image view. When a texture with the same
/// dimensions already exists (progressive previews of the same image), its
/// pixels are updated in place instead of reallocating a GPU texture.
fn show_frame(
    rl: &mut RaylibHandle,
    thread: &RaylibThread,
    view_tex: &mut Option<Texture2D>,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<()> {
    match view_tex {
        Some(tex) if tex.width() == width as i32 && tex.height() == height as i32 => {
            use raylib::texture::RaylibTexture2D;
            tex.update_texture(rgba)?;
        }
        _ => {
            *view_tex = Some(upload_rgba(rl, thread, rgba, width, height)?);
        }
    }
    Ok(())
}

/// Put the viewed texture back into its grid entry (if it came from one)
/// and clear the view texture. Called when leaving image mode or switching
/// to another grid entry.
fn put_back_view(
    grid: &mut Option<Grid>,
    view_from_grid: &mut Option<u64>,
    view_tex: &mut Option<Texture2D>,
) {
    if let Some(id) = view_from_grid.take() {
        if let Some(e) = grid
            .as_mut()
            .and_then(|g| g.entries.iter_mut().find(|e| e.id == id))
        {
            e.texture = view_tex.take();
            e.viewing = false;
        }
    }
    *view_tex = None;
}

/// Reset fit/pan state for a freshly shown image.
/// Document-backed view state for a grid entry source (None for plain image
/// files; doc + page for pages of multi-page documents).
fn view_doc_of(source: &EntrySource) -> Option<(Arc<dyn Document>, usize)> {
    match source {
        EntrySource::File(_) => None,
        EntrySource::DocPage { doc, page } => Some((doc.clone(), *page)),
    }
}

/// Fit-all scale for a page of natural size `info` inside a win_w x win_h
/// window, clamped so renders neither vanish nor explode (see pdf.rs caps).
fn fit_scale(info: PageInfo, win_w: f32, win_h: f32) -> f32 {
    (win_w / info.width as f32)
        .min(win_h / info.height as f32)
        .clamp(0.01, 8.0)
}

/// Open + start the loader for a grid entry with no ready texture. Returns
/// (loader, document-backed view if any, initial render scale). The document
/// opens here because texture-cached entries never need it; a failure aborts
/// the open and the caller removes the entry.
fn start_entry_loader(
    source: &EntrySource,
    win_w: f32,
    win_h: f32,
) -> Result<(Loader, Option<(Arc<dyn Document>, usize)>, f32)> {
    match source {
        EntrySource::File(path) => {
            let doc = open_document(path)?;
            if doc.page_count() > 1 {
                // Multi-page (PDF): start on page 0 at the window's fit
                // scale; n/p moves between pages, zoom re-renders sharper.
                let fit = fit_scale(doc.page_info(0)?, win_w, win_h);
                Ok((Loader::start_doc(doc.clone(), 0, fit), Some((doc, 0)), fit))
            } else {
                Ok((Loader::start_doc(doc, 0, 1.0), None, 1.0))
            }
        }
        EntrySource::DocPage { doc, page } => {
            let fit = fit_scale(doc.page_info(*page)?, win_w, win_h);
            Ok((
                Loader::start_doc(doc.clone(), *page, fit),
                Some((doc.clone(), *page)),
                fit,
            ))
        }
    }
}

fn reset_view(
    zoom: &mut ZoomMode,
    pan: &mut Vector2,
    target_pan: &mut Vector2,
    view_scale: &mut Option<f32>,
) {
    // Freshly shown images open fit-all (Shift+W behavior): upscale or
    // downscale until the image first touches a window border.
    *zoom = ZoomMode::FitAll;
    *pan = Vector2::ZERO;
    *target_pan = Vector2::ZERO;
    *view_scale = None;
}

fn main() -> Result<()> {
    let arg = env::args()
        .nth(1)
        .context("usage: versatile-viewer <image-path-or-directory>")?;
    let path = Path::new(&arg);

    // Directory launch: list the directory before opening the window (the
    // title shows the image count) — no GL involved yet. Single-file launch:
    // no grid.
    let dir_grid = if path.is_dir() {
        Some(Grid::from_dir(path)?)
    } else if path.is_file() {
        None
    } else {
        bail!("no such file or directory: {path:?}");
    };

    // Single-file launch: open the document before the window so it can be
    // sized to page 0. Single-page documents decode up front (window = image
    // size); multi-page documents (PDF) start on the loader and get a
    // default window fitted to page 0. Directory launch: fixed window size.
    let mut single_doc = if dir_grid.is_none() {
        Some(open_document(path)?)
    } else {
        None
    };
    let single_decoded = match &single_doc {
        Some(doc) if doc.page_count() == 1 => Some(doc.render(0, 1.0)?),
        _ => None,
    };
    let (win0_w, win0_h) = match (&single_decoded, &single_doc) {
        (Some(d), _) => (d.width as i32, d.height as i32),
        (None, Some(doc)) => {
            let info = doc.page_info(0)?;
            // Fit page 0 into ~1024x768 (never upscale past 1 pt = 2 px).
            let f = (1024.0 / info.width as f32)
                .min(768.0 / info.height as f32)
                .min(2.0);
            let w = (info.width as f32 * f).round() as i32;
            let h = (info.height as f32 * f).round() as i32;
            (w.max(320), h.max(240))
        }
        _ => (1024, 768),
    };

    let (mut rl, thread) = raylib::init()
        .size(win0_w, win0_h)
        .title(&format!(
            "versatile-viewer — {}",
            if dir_grid.is_none() {
                path.display().to_string()
            } else {
                format!(
                    "{} ({} images)",
                    path.display(),
                    dir_grid.as_ref().unwrap().entries.len()
                )
            }
        ))
        .resizable()
        .build();
    // We quit via the q key handling ourselves (set_exit_key would make ESC
    // close the window outright instead of returning to the grid).
    rl.set_exit_key(None);

    // Take the grid AFTER the window exists: it holds GPU textures, and
    // being declared after `rl` it is dropped (and unloaded) BEFORE the
    // window closes — both on normal exit and on panic unwinding.
    // Active grid (directory grid, or a PDF's page-overview grid on top of
    // it). `home_grid` holds the directory grid while a page grid is open.
    let mut grid = dir_grid;
    let mut home_grid: Option<Grid> = None;
    // Background streamer for the open image (no GL objects inside, but like
    // `grid` declared after `rl` so it drops before the window closes). Drop
    // cancels the worker.
    let mut loader: Option<Loader> = None;

    // Texture currently shown in image mode (single-file launch sets it up
    // front; grid-opened images stream in via the loader).
    let mut view_tex: Option<Texture2D> = None;
    let mut view_loading = false;
    // When the current load started (rl.get_time()); the "decoding..."
    // indicator only appears once the load exceeds 1 s so fast loads never
    // flash text on screen.
    let mut view_loading_since = 0.0f64;
    let mut img_w = 0.0f32;
    let mut img_h = 0.0f32;
    if let Some(decoded) = single_decoded {
        let ffi_image = raylib::ffi::Image {
            data: decoded.rgba.as_ptr() as *mut std::os::raw::c_void,
            width: decoded.width as i32,
            height: decoded.height as i32,
            mipmaps: 1,
            format: PixelFormat::PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 as i32,
        };
        let image = unsafe { Image::from_raw(ffi_image) };
        view_tex = Some(rl.load_texture_from_image(&thread, &image)?);
        image.to_raw(); // forget: drop would MemFree our borrowed Vec
        drop(decoded.rgba);
        img_w = win0_w as f32;
        img_h = win0_h as f32;
    }

    // Document-backed view: Some((doc, page)) while the open view shows a
    // page of a multi-page document (PDF). Enables in-document page
    // navigation (n/p) and scale-matched re-rendering (sharp zoom).
    let mut view_doc: Option<(Arc<dyn Document>, usize)> = None;
    // Scale the current view texture was rendered at (1.0 = natural size):
    // images render once at native size; PDF pages re-render at the settled
    // zoom scale (the refine logic in the frame loop).
    let mut view_render_scale: f32 = 1.0;
    // In-flight refine render: (entry id, page, scale, result channel).
    let mut refine: Option<
        std::sync::mpsc::Receiver<(u64, usize, f32, Result<DecodedImage, String>)>,
    > = None;

    // A single-file multi-page document (PDF) launches into its page grid;
    // single-file images launch into the image view.
    if single_doc.as_ref().is_some_and(|d| d.page_count() > 1) {
        let doc = single_doc.take().unwrap();
        grid = Some(Grid::from_document(doc));
    }

    let mut mode = if grid.is_none() {
        Mode::Image
    } else {
        Mode::Grid
    };
    // Which grid entry is open in image mode, by stable id (None when
    // launched with a file).
    let mut open_id: Option<u64> = None;
    // Set when the texture currently shown was taken out of a grid entry
    // (its id); it goes back when leaving image mode or switching entries.
    let mut view_from_grid: Option<u64> = None;
    // Set when the viewer should exit entirely.
    let mut quit = false;
    // VV_DEBUG=1: trace grid open/return events to stderr.
    let debug = std::env::var_os("VV_DEBUG").is_some();
    // Previous per-key down state for the VV_DEBUG event trace.
    let mut prev_down = [false; 349];

    rl.set_target_fps(60);
    let mut zoom = ZoomMode::FitDown;
    // Target pan offset; on-screen pan eases toward it (same easing as zoom).
    let mut target_pan = Vector2::ZERO;
    let mut pan = Vector2::ZERO;
    // On-screen scale, eased toward the target scale each frame.
    let mut view_scale: Option<f32> = None;
    // Auto-repeat state for image-mode prev/next (xset r rate values).
    let mut rep_nav_fwd = keyrepeat::RepeatState::new();
    let mut rep_nav_back = keyrepeat::RepeatState::new();

    while !rl.window_should_close() && !quit {
        // VV_DEBUG: trace every key raylib sees (keycode per raylib/GLFW:
        // 257=Enter, 335=KpEnter, 256=Esc, 262/263=Right/Left,
        // 264/265=Up/Down, 72/74/75/76=h/j/k/l).
        if debug {
            for k in 32u32..=348 {
                let (down, pressed) = unsafe {
                    (
                        raylib::ffi::IsKeyDown(k as i32),
                        raylib::ffi::IsKeyPressed(k as i32),
                    )
                };
                if pressed {
                    eprintln!("vv: PRESSED key {k}");
                }
                if down != prev_down[k as usize] {
                    eprintln!("vv: key {k} {}", if down { "DOWN" } else { "UP" });
                    prev_down[k as usize] = down;
                }
            }
            // Frames longer than a key tap can swallow IsKeyPressed edge
            // detection entirely (press+release between two polls); log them.
            let ft = rl.get_frame_time();
            if ft > 0.1 {
                eprintln!("vv: slow frame {ft:.0} ms");
            }
        }
        let win_w = rl.get_screen_width() as f32;
        let win_h = rl.get_screen_height() as f32;

        // Prefetch priority: in grid mode the selected entry's neighbors
        // (left/right/up/down); in image mode the previous/next entries.
        // load_pending drains finished decodes (texture uploads happen here,
        // on the main thread) and dispatches new ones to the rayon pool.
        let (priority, scan_start) = if mode == Mode::Image {
            match open_id.and_then(|id| grid.as_ref().and_then(|g| g.index_of(id))) {
                Some(i) => {
                    let n = grid.as_ref().unwrap().entries.len();
                    let mut v = Vec::new();
                    if i > 0 {
                        v.push(i - 1);
                    }
                    if i + 1 < n {
                        v.push(i + 1);
                    }
                    (v, i)
                }
                None => (Vec::new(), 0),
            }
        } else if let Some(g) = grid.as_ref() {
            (g.prefetch_neighbors(g.selected, win_w, win_h), g.selected)
        } else {
            (Vec::new(), 0)
        };
        if let Some(g) = grid.as_mut() {
            g.load_pending(&mut rl, &thread, &priority, scan_start);
        }

        // Image mode: drain the streaming loader first; texture uploads need
        // the main thread.
        if mode == Mode::Image {
            let mut open_failed = false;
            // Waiting on a grid decode (no streaming loader for this open):
            // when its texture lands, take it over as the view.
            if view_tex.is_none() && loader.is_none() {
                if let (Some(g), Some(id)) = (grid.as_mut(), open_id) {
                    match g.entries.iter_mut().find(|e| e.id == id) {
                        Some(e) => {
                            if let Some(tex) = e.texture.take() {
                                e.viewing = true;
                                view_from_grid = Some(id);
                                view_tex = Some(tex);
                                img_w = e.width as f32;
                                img_h = e.height as f32;
                                zoom = ZoomMode::FitAll;
                                pan = Vector2::ZERO;
                                target_pan = Vector2::ZERO;
                                view_scale = None;
                                view_loading = false;
                            }
                        }
                        None => open_failed = true, // entry vanished (decode failed)
                    }
                }
            }
            if let Some(loader) = &loader {
                while let Some(msg) = loader.try_recv() {
                    match msg {
                        LoaderMsg::Header { width, height } => {
                            // Dimensions known: fit-down immediately (the
                            // ease block only runs from the next frame on).
                            img_w = width as f32;
                            img_h = height as f32;
                            zoom = ZoomMode::FitAll;
                            pan = Vector2::ZERO;
                            target_pan = Vector2::ZERO;
                            view_scale = Some((win_w / img_w).min(win_h / img_h));
                        }
                        LoaderMsg::Preview {
                            rgba,
                            width,
                            height,
                        } => {
                            // Texture px per natural unit (points for PDFs).
                            if img_w > 0.0 && img_h > 0.0 && width > 0 {
                                view_render_scale = width as f32 / img_w;
                            }
                            // Progressively better render of the same image.
                            show_frame(&mut rl, &thread, &mut view_tex, &rgba, width, height)?;
                        }
                        LoaderMsg::Done {
                            rgba,
                            width,
                            height,
                        } => {
                            // Non-JXL formats only learn dimensions here.
                            if img_w == 0.0 {
                                img_w = width as f32;
                                img_h = height as f32;
                                zoom = ZoomMode::FitAll;
                                pan = Vector2::ZERO;
                                target_pan = Vector2::ZERO;
                                view_scale = Some((win_w / img_w).min(win_h / img_h));
                            }
                            show_frame(&mut rl, &thread, &mut view_tex, &rgba, width, height)?;
                            if img_w > 0.0 && img_h > 0.0 && width > 0 {
                                view_render_scale = width as f32 / img_w;
                            }
                            view_loading = false;
                        }
                        LoaderMsg::Failed(err) => {
                            eprintln!("vv: {err}");
                            open_failed = true;
                        }
                    }
                }
            }
            if open_failed {
                loader = None; // cancel the worker
                if let Some(id) = open_id.take() {
                    grid.as_mut().unwrap().remove_entry_by_id(id);
                }
                view_from_grid = None;
                view_doc = None;
                refine = None;
                view_render_scale = 1.0;
                mode = Mode::Grid;
                view_tex = None;
                view_loading = false;
            }
        }

        if mode == Mode::Grid {
            // Grid navigation: h/j/k/l + arrows move the selection,
            // Enter opens the selected image, q quits (ESC is inert here;
            // the grid is the home view).
            match grid.as_mut().unwrap().handle_input(&mut rl, win_w, win_h) {
                GridAction::Open(i) => {
                    // PDF entries open a page-overview grid (one entry per
                    // page, decoded on the same rayon pool) instead of the
                    // image view; ESC pops back to the directory grid.
                    if let EntrySource::File(p) = &grid.as_ref().unwrap().entries[i].source {
                        if crate::document::is_pdf(p) {
                            match open_document(p) {
                                Ok(doc) if doc.page_count() > 1 => {
                                    home_grid = grid.take();
                                    grid = Some(Grid::from_document(doc));
                                    continue;
                                }
                                Ok(_) => {} // single-page PDF: open like an image
                                Err(err) => {
                                    eprintln!("vv: {err:#}");
                                    grid.as_mut().unwrap().remove_entry(i);
                                    continue;
                                }
                            }
                        }
                    }
                    if debug {
                        eprintln!("vv: enter pressed -> open idx {i}");
                    }
                    // If the entry's decode already finished, its full-res
                    // texture is ready: take it and show it this frame — no
                    // decode, no black gap. If a decode is already in flight,
                    // just wait for it (no duplicate work). Otherwise fall
                    // back to the streaming loader (JXL: blurry preview fast).
                    let (id, tex, w, h, source, queued) = {
                        let g = grid.as_mut().unwrap();
                        let e = &mut g.entries[i];
                        (
                            e.id,
                            e.texture.take(),
                            e.width,
                            e.height,
                            e.source.clone(),
                            e.queued,
                        )
                    };
                    open_id = Some(id);
                    view_doc = view_doc_of(&source);
                    view_render_scale = 1.0;
                    refine = None;
                    mode = Mode::Image;
                    reset_view(&mut zoom, &mut pan, &mut target_pan, &mut view_scale);
                    if let Some(tex) = tex {
                        grid.as_mut().unwrap().entries[i].viewing = true;
                        view_from_grid = Some(id);
                        view_tex = Some(tex);
                        img_w = w as f32;
                        img_h = h as f32;
                        // The image-mode scale math runs only from the next
                        // frame on (this frame took the grid branch); set the
                        // initial fit-all scale here so the draw this frame
                        // has it.
                        view_scale = Some((win_w / img_w).min(win_h / img_h));
                        view_loading = false;
                    } else if queued {
                        view_from_grid = None;
                        loader = None;
                        view_tex = None;
                        view_loading = true;
                        view_loading_since = rl.get_time();
                        img_w = 0.0; // dimensions arrive with the texture
                        img_h = 0.0;
                    } else {
                        view_from_grid = None;
                        match start_entry_loader(&source, win_w, win_h) {
                            Ok((l, vd, rs)) => {
                                loader = Some(l);
                                view_doc = vd;
                                view_render_scale = rs;
                            }
                            Err(err) => {
                                eprintln!("vv: {err:#}");
                                grid.as_mut().unwrap().remove_entry_by_id(id);
                                open_id = None;
                                mode = Mode::Grid;
                            }
                        }
                        view_tex = None;
                        view_loading = loader.is_some();
                        view_loading_since = rl.get_time();
                        img_w = 0.0; // dimensions arrive with the header message
                        img_h = 0.0;
                    }
                }
                GridAction::Back => {
                    // ESC on a page-overview grid: pop it (its textures drop
                    // with it) and restore the directory grid.
                    if let Some(home) = home_grid.take() {
                        grid = Some(home);
                    }
                }
                GridAction::Quit => quit = true,
                GridAction::None => {}
            }
        } else {
            // Image mode.
            // Drain the raw key/char queues every frame (same rationale as
            // grid.rs handle_input): is_key_pressed misses a press+release
            // pair that lands inside one frame — easy here while frames
            // stall on texture uploads or decode. Leftover queue entries
            // would otherwise leak into grid mode and act there (e.g. a
            // missed Enter immediately reopening the just-viewed image).
            // State queries (is_key_down panning, is_key_pressed W/E/=/-)
            // are unaffected: the queue is separate from the key snapshot.
            // Enter/ESC return to the grid when one exists (nsxiv-like:
            // Enter toggles between grid and the open image); q quits,
            // ESC never quits the program (inert in single-file launches,
            // where there is no grid to return to). Space/Backspace switch
            // to the next/previous image (nsxiv-style nav; Space/n next,
            // Backspace/p previous; arrows and h/j/k/l stay panning).
            // Nav keys auto-repeat while held, at the X server's rate
            // (xset r rate; keyrepeat::settings). g/G jump to the
            // first/last image.
            let mut enter = false;
            let mut quit_pressed = false;
            let mut nav_next_edge = false;
            let mut nav_prev_edge = false;
            let mut jump_first = false;
            let mut jump_last = false;
            let shift = rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
                || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT);
            while let Some(k) = rl.get_key_pressed() {
                match k {
                    KeyboardKey::KEY_ENTER | KeyboardKey::KEY_KP_ENTER => enter = true,
                    KeyboardKey::KEY_ESCAPE => enter = true,
                    KeyboardKey::KEY_Q => quit_pressed = true,
                    KeyboardKey::KEY_SPACE | KeyboardKey::KEY_N => nav_next_edge = true,
                    KeyboardKey::KEY_BACKSPACE | KeyboardKey::KEY_P => nav_prev_edge = true,
                    KeyboardKey::KEY_G => {
                        if shift {
                            jump_last = true;
                        } else {
                            jump_first = true;
                        }
                    }
                    _ => {}
                }
            }
            // Auto-repeat: the initial press fires immediately (edge from
            // the queue above), holding fires at the X server's repeat rate
            // after its delay.
            let now = rl.get_time();
            let (delay, rate) = keyrepeat::settings();
            let mut nav = if rep_nav_fwd.tick(
                nav_next_edge,
                rl.is_key_down(KeyboardKey::KEY_N) || rl.is_key_down(KeyboardKey::KEY_SPACE),
                now,
                delay,
                rate,
            ) {
                Some(1)
            } else if rep_nav_back.tick(
                nav_prev_edge,
                rl.is_key_down(KeyboardKey::KEY_P) || rl.is_key_down(KeyboardKey::KEY_BACKSPACE),
                now,
                delay,
                rate,
            ) {
                Some(-1)
            } else {
                None
            };
            // g/G: jump to the first/last image (as a nav delta).
            if jump_first || jump_last {
                if let Some(cur) = open_id.and_then(|id| grid.as_ref().and_then(|g| g.index_of(id)))
                {
                    let n = grid.as_ref().unwrap().entries.len();
                    let t = if jump_first { 0 } else { n - 1 };
                    nav = Some(t as i64 - cur as i64);
                }
            }
            // Some input setups (IMEs, unusual X11 input methods) deliver
            // Enter as a character event ('\n'/'\r'); accept both. This
            // drain also covers the +/- zoom chars for every frame, so
            // nothing piles up while the header has not arrived yet.
            let mut zoom_in_char = false;
            let mut zoom_out_char = false;
            while let Some(c) = rl.get_char_pressed() {
                match c {
                    '\n' | '\r' => enter = true,
                    '+' => zoom_in_char = true,
                    '-' => zoom_out_char = true,
                    _ => {}
                }
            }

            let mut return_to_grid = false;
            if quit_pressed {
                quit = true; // single-file launch: no grid to fall back to
            }
            if grid.is_some() && enter {
                return_to_grid = true;
            }
            if return_to_grid {
                mode = Mode::Grid;
                // Hand the shown texture back to its grid entry and make
                // that entry the grid selection (nsxiv-like).
                put_back_view(&mut grid, &mut view_from_grid, &mut view_tex);
                if let Some(id) = open_id.take() {
                    if let Some(g) = grid.as_mut() {
                        if let Some(i) = g.index_of(id) {
                            g.selected = i;
                            // Scrolled (zoomed-in) grids: bring it back on screen.
                            g.ensure_visible(win_w, win_h);
                        }
                    }
                }
                loader = None; // cancels a still-running stream
                view_loading = false;
                view_loading_since = 0.0;
                view_doc = None;
                refine = None;
                view_render_scale = 1.0;
            }

            // Prev/next while viewing (only with a grid to navigate). A
            // ready texture swaps in the same frame; a decode in flight is
            // waited on (auto-swap when it lands); otherwise the streaming
            // loader takes over. The new entry's own neighbors are prefetched
            // via the priority list at the top of the loop.
            if !return_to_grid {
                if let Some(delta) = nav {
                    if let Some((doc, page)) = view_doc.clone() {
                        // Multi-page document: n/p (and g/G) move between
                        // pages. The shown texture goes back to its entry,
                        // the target page streams in via the loader.
                        let n = doc.page_count();
                        let mut t = page as i64 + delta;
                        if jump_first {
                            t = 0;
                        }
                        if jump_last {
                            t = n as i64 - 1;
                        }
                        if t >= 0 && t < n as i64 {
                            let t = t as usize;
                            put_back_view(&mut grid, &mut view_from_grid, &mut view_tex);
                            refine = None;
                            loader = None;
                            let fit = doc
                                .page_info(t)
                                .map(|i| fit_scale(i, win_w, win_h))
                                .unwrap_or(1.0);
                            view_doc = Some((doc.clone(), t));
                            view_render_scale = fit;
                            view_loading = true;
                            view_loading_since = rl.get_time();
                            img_w = 0.0;
                            img_h = 0.0;
                            loader = Some(Loader::start_doc(doc, t, fit));
                        }
                    } else if let Some(cur) =
                        open_id.and_then(|id| grid.as_ref().and_then(|g| g.index_of(id)))
                    {
                        let cur = cur;
                        let n = grid.as_ref().unwrap().entries.len();
                        let t = cur as i64 + delta;
                        if t >= 0 && (t as usize) < n {
                            let j = t as usize;
                            put_back_view(&mut grid, &mut view_from_grid, &mut view_tex);
                            loader = None;
                            let (id, tex, w, h, source, queued) = {
                                let g = grid.as_mut().unwrap();
                                let e = &mut g.entries[j];
                                (
                                    e.id,
                                    e.texture.take(),
                                    e.width,
                                    e.height,
                                    e.source.clone(),
                                    e.queued,
                                )
                            };
                            open_id = Some(id);
                            view_doc = view_doc_of(&source);
                            view_render_scale = 1.0;
                            refine = None;
                            reset_view(&mut zoom, &mut pan, &mut target_pan, &mut view_scale);
                            if let Some(tex) = tex {
                                grid.as_mut().unwrap().entries[j].viewing = true;
                                view_from_grid = Some(id);
                                view_tex = Some(tex);
                                img_w = w as f32;
                                img_h = h as f32;
                                // Set the initial fit-all scale here: the
                                // scale math below already ran past the nav
                                // code only afterwards, and this frame's draw
                                // needs a scale now.
                                view_scale = Some((win_w / img_w).min(win_h / img_h));
                                view_loading = false;
                            } else if queued {
                                view_from_grid = None;
                                view_tex = None;
                                view_loading = true;
                                view_loading_since = rl.get_time();
                                img_w = 0.0;
                                img_h = 0.0;
                            } else {
                                view_from_grid = None;
                                match start_entry_loader(&source, win_w, win_h) {
                                    Ok((l, vd, rs)) => {
                                        loader = Some(l);
                                        view_doc = vd;
                                        view_render_scale = rs;
                                    }
                                    Err(err) => {
                                        eprintln!("vv: {err:#}");
                                        grid.as_mut().unwrap().remove_entry_by_id(id);
                                        open_id = None;
                                        mode = Mode::Grid;
                                    }
                                }
                                view_tex = None;
                                view_loading = loader.is_some();
                                view_loading_since = rl.get_time();
                                img_w = 0.0;
                                img_h = 0.0;
                            }
                        }
                    }
                }
            }

            // Keyboard shortcuts. Capital W / capital E arrive as W/E + shift.
            if rl.is_key_pressed(KeyboardKey::KEY_W) {
                zoom = if shift {
                    ZoomMode::FitAll
                } else {
                    ZoomMode::FitDown
                };
                target_pan = Vector2::ZERO;
            } else if rl.is_key_pressed(KeyboardKey::KEY_E) {
                zoom = if shift {
                    ZoomMode::FitHeight
                } else {
                    ZoomMode::FitWidth
                };
                target_pan = Vector2::ZERO;
            } else if rl.is_key_pressed(KeyboardKey::KEY_T) {
                // Two-state toggle: whole image visible (fit-all) vs window
                // completely covered (fill). Distinct for any image/window
                // combination, unlike a fit-width/fit-height cycle.
                zoom = if zoom == ZoomMode::Fill {
                    ZoomMode::FitAll
                } else {
                    ZoomMode::Fill
                };
                target_pan = Vector2::ZERO;
            }

            // Before the header arrives the image dimensions are unknown;
            // skip all scale/pan math (it divides by img_w/img_h).
            if img_w > 0.0 {
                // Scale for the current mode (fit modes recompute every frame, so
                // resizing stays correct).
                let target_scale = match zoom {
                    ZoomMode::FitDown => (win_w / img_w).min(win_h / img_h).min(1.0),
                    ZoomMode::FitAll => (win_w / img_w).min(win_h / img_h),
                    ZoomMode::FitWidth => win_w / img_w,
                    ZoomMode::FitHeight => win_h / img_h,
                    // Cover: the larger ratio wins, so the image fills the
                    // window and the other axis is cropped.
                    ZoomMode::Fill => (win_w / img_w).max(win_h / img_h),
                    ZoomMode::Free(scale) => scale,
                };

                // Ease the on-screen scale toward the target so zoom steps animate
                // smoothly (~95% of the way after 150 ms; snap when close enough).
                let prev_scale = view_scale;
                let alpha = 1.0 - (-rl.get_frame_time() / 0.05).exp();
                view_scale = Some(match view_scale {
                    None => target_scale,
                    Some(s) => {
                        let s = s + (target_scale - s) * alpha;
                        if (target_scale - s).abs() < target_scale * 0.001 {
                            target_scale
                        } else {
                            s
                        }
                    }
                });

                // Free zoom: +/- steps the scale up/down by 25%, starting from the
                // scale currently on screen. Detected two ways: the keycode of the
                // US-layout =/- keys (incl. numpad) and the typed character, which
                // covers non-US layouts where '+' lives on another physical key.
                let mut zoom_in = rl.is_key_pressed(KeyboardKey::KEY_EQUAL)
                    || rl.is_key_pressed(KeyboardKey::KEY_KP_ADD)
                    || zoom_in_char;
                let mut zoom_out = rl.is_key_pressed(KeyboardKey::KEY_MINUS)
                    || rl.is_key_pressed(KeyboardKey::KEY_KP_SUBTRACT)
                    || zoom_out_char;
                if zoom_in || zoom_out {
                    let factor = if zoom_in { 1.25 } else { 1.0 / 1.25 };
                    zoom = ZoomMode::Free((target_scale * factor).clamp(0.01, 100.0));
                }

                // Window-center-anchored zoom (free zoom only): while the on-screen
                // scale eases, shift the pan each frame so the image point under
                // the window center stays fixed. offset = center + pan, so keeping
                // the anchor's image point put gives
                //   offset' = anchor - (anchor - offset) * (scale'/scale).
                let scale = view_scale.unwrap();
                if matches!(zoom, ZoomMode::Free(_)) {
                    if let Some(s_old) = prev_scale {
                        if (scale - s_old).abs() > f32::EPSILON && s_old > 0.0 {
                            let r = scale / s_old;
                            let ax = win_w / 2.0;
                            let ay = win_h / 2.0;
                            let ox = ax - (ax - (win_w - img_w * s_old) / 2.0 - pan.x) * r;
                            let oy = ay - (ay - (win_h - img_h * s_old) / 2.0 - pan.y) * r;
                            pan.x = ox - (win_w - img_w * scale) / 2.0;
                            pan.y = oy - (win_h - img_h * scale) / 2.0;
                            // Pin the target too, so pan easing doesn't fight the anchor.
                            target_pan.x = pan.x;
                            target_pan.y = pan.y;
                        }
                    }
                }

                // Vim-style panning (h/j/k/l + arrow keys); held keys move the
                // target offset, the on-screen pan eases after it (same exponential
                // easing as zoom), so taps glide and holds scroll smoothly.
                // Input read before begin_drawing borrows rl mutably.
                // Per-second speed: matches the old 3%-of-window-per-frame pace
                // (3% × 60 fps = 180% per second), now frame-time aware.
                let speed = win_w.max(win_h) * 1.8 * rl.get_frame_time();
                let pan_left =
                    rl.is_key_down(KeyboardKey::KEY_H) || rl.is_key_down(KeyboardKey::KEY_LEFT);
                let pan_right =
                    rl.is_key_down(KeyboardKey::KEY_L) || rl.is_key_down(KeyboardKey::KEY_RIGHT);
                let pan_up =
                    rl.is_key_down(KeyboardKey::KEY_K) || rl.is_key_down(KeyboardKey::KEY_UP);
                let pan_down =
                    rl.is_key_down(KeyboardKey::KEY_J) || rl.is_key_down(KeyboardKey::KEY_DOWN);
                if pan_left {
                    target_pan.x += speed;
                }
                if pan_right {
                    target_pan.x -= speed;
                }
                if pan_up {
                    target_pan.y += speed;
                }
                if pan_down {
                    target_pan.y -= speed;
                }
                let pan_alpha = 1.0 - (-rl.get_frame_time() / 0.05).exp();
                pan = Vector2 {
                    x: pan.x + (target_pan.x - pan.x) * pan_alpha,
                    y: pan.y + (target_pan.y - pan.y) * pan_alpha,
                };
                // Snap when the residual glide is sub-pixel.
                if (target_pan.x - pan.x).abs() < 0.25 {
                    pan.x = target_pan.x;
                }
                if (target_pan.y - pan.y).abs() < 0.25 {
                    pan.y = target_pan.y;
                }
                // Document pages (PDF): once the zoom animation settles and
                // the settled scale differs enough from the scale the current
                // texture was rendered at, re-render the page at that scale.
                // Text stays sharp at every resting zoom level (fresh
                // rasterization at screen resolution — same trick zathura/
                // mupdf use). While the render runs, the old texture keeps
                // showing (slightly soft, never blank).
                if let Some((doc, page)) = view_doc.as_ref() {
                    let settled = view_scale
                        .map(|s| (s - target_scale).abs() <= target_scale * 0.001)
                        .unwrap_or(false);
                    let scale = view_scale.unwrap_or(1.0);
                    let rel_diff = (scale - view_render_scale).abs() / view_render_scale.max(1e-3);
                    if settled && refine.is_none() && rel_diff > 0.15 {
                        let (tx, rx) = std::sync::mpsc::channel();
                        let doc = doc.clone();
                        let page = *page;
                        let id = open_id.unwrap_or(u64::MAX);
                        let rs = scale.clamp(0.05, 8.0);
                        rayon::spawn(move || {
                            let res = doc.render(page, rs).map_err(|e| format!("{e:#}"));
                            // Receiver gone (view left): result discarded.
                            let _ = tx.send((id, page, rs, res));
                        });
                        refine = Some(rx);
                    }
                }
                // Poll a finished refine render; accept it only if it still
                // matches what is on screen (same entry, page and zoom).
                if let Some(rx) = &refine {
                    if let Ok((id, page, rs, res)) = rx.try_recv() {
                        refine = None;
                        if open_id == Some(id)
                            && view_doc.as_ref().is_some_and(|(_, p)| *p == page)
                            && view_scale
                                .map(|s| (s - rs).abs() <= rs * 0.2)
                                .unwrap_or(false)
                        {
                            if let Ok(d) = res {
                                show_frame(
                                    &mut rl,
                                    &thread,
                                    &mut view_tex,
                                    &d.rgba,
                                    d.width,
                                    d.height,
                                )?;
                                view_render_scale = rs;
                            }
                        }
                    }
                }
            } // img_w > 0.0: fit/pan math needs known dimensions
        }

        // Whether the "decoding..." indicator should show this frame: only
        // once the load has taken over a second; brief loads would otherwise
        // flash the text for a few frames. Computed before begin_drawing
        // (rl is mutably borrowed by the draw handle).
        let show_decoding = view_loading && rl.get_time() - view_loading_since > 1.0;

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);

        if mode == Mode::Grid {
            grid.as_ref().unwrap().draw(&mut d, win_w, win_h);
        } else {
            if let Some(texture) = &view_tex {
                let dw = img_w * view_scale.unwrap();
                let dh = img_h * view_scale.unwrap();

                // Source rect in TEXTURE pixels, not natural units: PDF
                // pages re-render at the settled zoom scale, so the texture
                // holds points * scale pixels while img_w/img_h stay in
                // points (the fit math works in natural units). Sampling
                // img_w here would crop the top-left fraction of a refined
                // texture and stretch it over the full dest — the page
                // would look zoomed-in after every refine. For plain images
                // texture pixels equal natural pixels, so this is a no-op
                // there.
                let src = Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: texture.width() as f32,
                    height: texture.height() as f32,
                };
                let dest = Rectangle {
                    x: (win_w - dw) / 2.0 + pan.x,
                    y: (win_h - dh) / 2.0 + pan.y,
                    width: dw,
                    height: dh,
                };
                d.draw_texture_pro(texture, src, dest, Vector2::ZERO, 0.0, Color::WHITE);
            }
            // Threshold check happens above, before begin_drawing.
            if show_decoding {
                let msg = "decoding...";
                let tw = d.measure_text(msg, 20);
                d.draw_text(
                    msg,
                    (win_w as i32 - tw) / 2,
                    win_h as i32 / 2 - 10,
                    20,
                    Color::GRAY,
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{fb_to_rgba, to_u8};
}
