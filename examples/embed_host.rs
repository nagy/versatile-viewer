// The example spells out the X11 calls a host makes, and a protocol value
// (a window id, a size) is a `u32`/`u16` on the wire. The casts between those
// and the local names are the point of the file, so the pedantic cast lints are
// noise here, exactly as they are in `main.rs`.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

//! A minimal X11 host for an embedded pane.
//!
//! ```text
//! frame          the top-level window the window manager decorates
//! └── container  a child the host sizes to the pane area
//!     └── vv     the window `vv --wid <container>` creates, a child of it
//! ```
//!
//! Run it with `cargo run --example embed_host -- [--child-size WxH] <path>`.
//!
//! The host owes the pane two things, and this example does both:
//!
//! - **Size.** It resizes the container, and nothing else. The pane reads the
//!   container geometry itself and follows it, the way `mpv --wid` does, so no
//!   host call ever reaches the child window.
//! - **Focus.** It calls `XSetInputFocus` on the child of the container when
//!   the pane is clicked, because keys follow focus. The pane never focuses
//!   itself, so without this it would show a picture and ignore every key.
//!
//! The top bar of the frame is the part a click reaches: the pane's own window
//! covers the container, so a click there goes to the pane and never to the
//! host. Click the bar to toggle focus between the pane and the frame, and read
//! `xwininfo -tree` to see the three levels.

use std::{
    path::{Path, PathBuf},
    process::{Child, Command},
    thread::sleep,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use x11rb::{
    COPY_FROM_PARENT, CURRENT_TIME,
    connection::Connection,
    protocol::xproto::{
        AtomEnum, ClientMessageEvent, ConfigureWindowAux, ConnectionExt as _, CreateWindowAux,
        EventMask, InputFocus, PropMode, Window, WindowClass,
    },
    rust_connection::RustConnection,
    wrapper::ConnectionExt as _,
};

/// The top-level frame size, which is what the window manager places.
const FRAME_SIZE: (u16, u16) = (900, 700);

/// The bar at the top of the frame that a click can reach, in pixels.
const BAR_HEIGHT: u16 = 40;

/// The gap a click loop sleeps for, so it reaps the child and reads the X
/// connection without burning a core.
const LOOP_PAUSE: Duration = Duration::from_millis(16);

/// The command line of the example host.
#[derive(Debug)]
struct Args {
    /// The pane size that `--child-size` asks for, which becomes a frame
    /// resize.
    child_size: Option<(u16, u16)>,
    /// The `vv` binary, when `--vv` names one.
    vv: Option<PathBuf>,
    /// The image file or directory the pane shows.
    path: PathBuf,
}

fn main() -> Result<()> {
    let args = parse_args(std::env::args().skip(1))?;
    let (conn, screen) = RustConnection::connect(None)
        .context("cannot open the X display: an embedding host needs X11")?;
    let root = conn.setup().roots[screen].root;
    let atoms = Atoms::new(&conn)?;
    let frame = conn.generate_id()?;
    let container = conn.generate_id()?;
    create_window(&conn, root, frame, FRAME_SIZE, Some("vv embed host"))?;
    // The container starts under the bar, so a click on the bar reaches the
    // frame and a click below it reaches the pane.
    create_window(
        &conn,
        frame,
        container,
        (FRAME_SIZE.0, FRAME_SIZE.1 - BAR_HEIGHT),
        None,
    )?;
    conn.configure_window(
        container,
        &ConfigureWindowAux::new().y(i32::from(BAR_HEIGHT)),
    )?;
    // The pane id goes to the child process, so the container has to exist
    // before the spawn.
    conn.map_window(frame)?;
    conn.map_window(container)?;
    // A close request arrives as a client message, so the frame names the
    // protocol it speaks.
    conn.change_property(
        PropMode::REPLACE,
        frame,
        atoms.wm_protocols,
        AtomEnum::ATOM,
        32,
        1,
        &atoms.wm_delete.to_ne_bytes(),
    )?;
    // `--child-size` scripts a resize without a mouse, and it goes through the
    // frame: the container follows the frame through the same `ConfigureNotify`
    // path a user drag takes, so the pane follows it in turn.
    if let Some((width, height)) = args.child_size {
        conn.configure_window(
            frame,
            &ConfigureWindowAux::new()
                .width(u32::from(width))
                .height(u32::from(height.saturating_add(BAR_HEIGHT))),
        )?;
    }
    conn.flush()?;

    let mut child = spawn_pane(&args, container)?;
    eprintln!(
        "vv embed host: frame 0x{frame:x}, container 0x{container:x}, pane {} in it",
        child.id()
    );
    eprintln!("vv embed host: click the bar to focus the pane, Escape on the bar quits");

    loop {
        while let Some(event) = conn.poll_for_event()? {
            if handle_event(&conn, &event, frame, container, &atoms)? {
                stop(&mut child);
                return Ok(());
            }
        }
        if let Some(status) = child.try_wait()? {
            eprintln!("vv embed host: the pane exited with {status}");
            if !status.success() {
                bail!("the pane reported a failure: {status}");
            }
            return Ok(());
        }
        sleep(LOOP_PAUSE);
    }
}

/// The atoms the host sets and reads.
struct Atoms {
    /// The property that names the protocols a window speaks.
    wm_protocols: u32,
    /// The close request a window manager sends instead of destroying a
    /// window.
    wm_delete: u32,
}

impl Atoms {
    fn new(conn: &RustConnection) -> Result<Self> {
        Ok(Self {
            wm_protocols: intern(conn, b"WM_PROTOCOLS")?,
            wm_delete: intern(conn, b"WM_DELETE_WINDOW")?,
        })
    }
}

fn intern(conn: &RustConnection, name: &[u8]) -> Result<u32> {
    Ok(conn.intern_atom(false, name)?.reply()?.atom)
}

/// Create one input/output child with the parent's visual and a black
/// background, so the picture has no undefined pixels before the pane paints.
fn create_window(
    conn: &RustConnection,
    parent: Window,
    window: Window,
    size: (u16, u16),
    title: Option<&str>,
) -> Result<()> {
    conn.create_window(
        COPY_FROM_PARENT as u8,
        window,
        parent,
        0,
        0,
        size.0,
        size.1,
        0,
        WindowClass::INPUT_OUTPUT,
        COPY_FROM_PARENT,
        &CreateWindowAux::new()
            .background_pixel(0x0000_0000)
            .event_mask(
                EventMask::EXPOSURE
                    | EventMask::STRUCTURE_NOTIFY
                    | EventMask::BUTTON_PRESS
                    | EventMask::KEY_PRESS,
            ),
    )?;
    conn.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        title.unwrap_or("vv embed pane").as_bytes(),
    )?;
    let class = b"vv-embed-host\0vv-embed-host\0";
    conn.change_property8(
        PropMode::REPLACE,
        window,
        AtomEnum::WM_CLASS,
        AtomEnum::STRING,
        class,
    )?;
    Ok(())
}

/// Start `vv` on the container.
fn spawn_pane(args: &Args, container: Window) -> Result<Child> {
    let vv = match &args.vv {
        Some(vv) => vv.clone(),
        None => vv_binary()?,
    };
    let child = Command::new(&vv)
        .arg("--wid")
        .arg(format!("0x{container:x}"))
        .arg(&args.path)
        .spawn()
        .with_context(|| format!("cannot start {}", vv.display()))?;
    Ok(child)
}

/// The `vv` binary beside this example, in the same cargo target directory.
fn vv_binary() -> Result<PathBuf> {
    let example = std::env::current_exe().context("cannot name this example")?;
    let dir = example
        .parent()
        .and_then(Path::parent)
        .context("the example does not live in a cargo target directory")?;
    let vv = dir.join("vv");
    if vv.is_file() {
        Ok(vv)
    } else {
        bail!("no vv binary at {}, run `cargo build` first", vv.display())
    }
}

/// Read one X event, and answer whether the host must stop.
///
/// A `ConfigureNotify` of the frame resizes the container, which is the whole
/// size duty: the pane follows the container by itself.
fn handle_event(
    conn: &RustConnection,
    event: &x11rb::protocol::Event,
    frame: Window,
    container: Window,
    atoms: &Atoms,
) -> Result<bool> {
    match event {
        x11rb::protocol::Event::ConfigureNotify(event) if event.window == frame => {
            let height = event.height.saturating_sub(BAR_HEIGHT);
            conn.configure_window(
                container,
                &ConfigureWindowAux::new()
                    .width(u32::from(event.width))
                    .height(u32::from(height)),
            )?;
            conn.flush()?;
            eprintln!(
                "vv embed host: frame {}x{} -> container {}x{}",
                event.width, event.height, event.width, height
            );
        }
        x11rb::protocol::Event::ButtonPress(event) if event.event == frame => {
            // The bar belongs to the host, so a click there hands the keyboard
            // to the pane or takes it back.
            let pane = pane_window(conn, container)?;
            let focus = conn.get_input_focus()?.reply()?.focus;
            match pane {
                Some(pane) if focus == pane => {
                    conn.set_input_focus(InputFocus::PARENT, frame, CURRENT_TIME)?;
                    eprintln!("vv embed host: focus -> frame");
                }
                Some(pane) => {
                    conn.set_input_focus(InputFocus::PARENT, pane, CURRENT_TIME)?;
                    eprintln!("vv embed host: focus -> pane 0x{pane:x}");
                }
                None => eprintln!("vv embed host: no pane window yet"),
            }
            conn.flush()?;
        }
        x11rb::protocol::Event::KeyPress(event)
            if event.detail == ESCAPE_KEYCODE && event.event == frame =>
        {
            return Ok(true);
        }
        x11rb::protocol::Event::ClientMessage(event) if is_close_request(event, atoms) => {
            return Ok(true);
        }
        _ => {}
    }
    Ok(false)
}

/// The keycode of Escape in the core keyboard mapping, which is 9 on every
/// normal X server.
const ESCAPE_KEYCODE: u8 = 9;

/// The pane's own window, which is the child of the container that `vv`
/// created.
///
/// A host never learns that id: `vv` creates it, and the host finds it the way
/// this function does.
fn pane_window(conn: &RustConnection, container: Window) -> Result<Option<Window>> {
    let tree = conn.query_tree(container)?.reply()?;
    Ok(tree.children.first().copied())
}

/// Does this client message ask the host to close?
fn is_close_request(event: &ClientMessageEvent, atoms: &Atoms) -> bool {
    event.type_ == atoms.wm_protocols && event.data.as_data32()[0] == atoms.wm_delete
}

/// End the pane and leave the X server as it was.
fn stop(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn parse_args<I: Iterator<Item = String>>(args: I) -> Result<Args> {
    let mut child_size = None;
    let mut vv = None;
    let mut path: Option<PathBuf> = None;
    let mut args = args;
    while let Some(arg) = args.next() {
        if arg == "--child-size" {
            let value = args.next().context("--child-size needs WxH")?;
            child_size = Some(parse_size(&value)?);
        } else if arg == "--vv" {
            vv = Some(PathBuf::from(args.next().context("--vv needs a path")?));
        } else if path.is_some() {
            bail!("a second path: {arg}");
        } else {
            path = Some(PathBuf::from(arg));
        }
    }
    Ok(Args {
        child_size,
        vv,
        path: path.context("usage: embed_host [--child-size WxH] [--vv <binary>] <path>")?,
    })
}

/// Read a `WxH` pair of pixel sizes.
fn parse_size(text: &str) -> Result<(u16, u16)> {
    let (width, height) = text
        .split_once('x')
        .with_context(|| format!("--child-size needs WxH, not {text:?}"))?;
    Ok((
        width
            .parse()
            .with_context(|| format!("bad width in {text:?}"))?,
        height
            .parse()
            .with_context(|| format!("bad height in {text:?}"))?,
    ))
}
