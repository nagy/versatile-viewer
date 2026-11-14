//! The embedding pane: the X11 window a host hands over with `--wid`.
//!
//! A host application shows the viewer inside a window of its own by creating
//! a container window and spawning `vv --wid <container>`. `XCreateWindow`
//! then runs with that container as the parent, so the viewer never becomes a
//! top-level window and no window manager frames it.
//!
//! [`Wid`] is the parsed id, and it is the one place a bare window number
//! becomes a value with a meaning.
//!
//! The pane lives in the binary (see `Pane` in `src/main.rs`): this module
//! only names the id and reads the host geometry, so no other module learns
//! that a host exists.

use std::time::{Duration, Instant};

use derive_more::{Display, Error, From};
use x11rb::{
    protocol::xproto::{ConnectionExt as _, Window},
    rust_connection::RustConnection,
};

/// How often an embedded pane reads the size of the host window it fills.
///
/// A host resizes the container it created, and X sends that `ConfigureNotify`
/// to the container: the child window the viewer draws into never hears about
/// it, and a clip is all it would see.
/// The pane therefore reads the parent geometry itself, like `mpv --wid` does,
/// at a rate that costs one X round trip per poll and shows no lag on a
/// splitter drag.
pub const PANE_POLL: Duration = Duration::from_millis(50);

/// The X11 window a host handed to `--wid`.
///
/// An X11 window id is a `u32` (`winit::platform::x11::XWindow`), and the
/// parsed id never travels as a bare number, so a wrong value cannot reach a
/// window call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wid(u32);

impl Wid {
    /// Read the decimal or `0x` hexadecimal form that the command line carries.
    ///
    /// `None` covers a non-numeric value and `0`: the X server never issues
    /// window id 0, so a zero names no window and no host can hand one over.
    ///
    /// # Examples
    ///
    /// ```
    /// use vv_view::pane::Wid;
    ///
    /// assert_eq!(Wid::parse("74").map(Wid::raw), Some(74));
    /// assert_eq!(Wid::parse("0x4a").map(Wid::raw), Some(74));
    /// assert_eq!(Wid::parse("0x4A").map(Wid::raw), Some(74));
    /// assert_eq!(Wid::parse("0"), None);
    /// assert_eq!(Wid::parse("zz"), None);
    /// ```
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let id = match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => text.parse().ok(),
        };
        id.filter(|id| *id != 0).map(Self)
    }

    /// The raw id, which a parent window or a geometry read takes.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Why an embedded pane cannot read the host window.
#[derive(Debug, Display, Error, From)]
pub enum PaneError {
    /// The X display of `$DISPLAY` could not be reached.
    #[display("cannot connect to the X display")]
    Connect(#[from] x11rb::errors::ConnectError),
    /// The connection to the X server failed.
    #[display("the X connection failed")]
    Connection(#[from] x11rb::errors::ConnectionError),
    /// The X server rejected the geometry request, for example because the host
    /// window is gone.
    #[display("the host window answered no geometry")]
    Reply(#[from] x11rb::errors::ReplyError),
}

/// The host window an embedded pane fills.
///
/// One X connection, one parent id, and the last size this pane acted on, so an
/// unchanged host costs one round trip and no resize request.
pub struct PaneGeometry {
    conn: RustConnection,
    parent: Window,
    last: Option<(u16, u16)>,
    next: Instant,
}

impl PaneGeometry {
    /// Open a connection to the display of `$DISPLAY` for the host window.
    ///
    /// The pane draws the host window itself, so the id is all it needs: no
    /// reparent, no window manager, and no call that writes to the host.
    ///
    /// # Errors
    ///
    /// [`PaneError::Connect`] when `$DISPLAY` names no reachable X server.
    pub fn new(parent: Wid) -> Result<Self, PaneError> {
        Ok(Self {
            // The screen number that comes back names no window: the pane
            // addresses the host id it was given.
            conn: RustConnection::connect(None)?.0,
            parent: parent.raw(),
            last: None,
            next: Instant::now(),
        })
    }

    /// The size the pane must take, in physical pixels.
    ///
    /// `None` means the poll is not due yet, the host window kept its size, or
    /// the host collapsed the pane (a zero side has no swapchain).
    ///
    /// # Errors
    ///
    /// [`PaneError::Connection`] when the X connection failed, and
    /// [`PaneError::Reply`] when the server rejects the request, which is what
    /// a host window that went away answers.
    pub fn poll(&mut self) -> Result<Option<(u32, u32)>, PaneError> {
        let now = Instant::now();
        if now < self.next {
            return Ok(None);
        }
        self.next = now + PANE_POLL;
        let reply = self.conn.get_geometry(self.parent)?.reply()?;
        let size = (reply.width, reply.height);
        if size.0 == 0 || size.1 == 0 || self.last == Some(size) {
            return Ok(None);
        }
        self.last = Some(size);
        Ok(Some((u32::from(size.0), u32::from(size.1))))
    }
}

#[cfg(test)]
mod tests {
    use super::Wid;

    #[test]
    fn a_window_id_reads_in_both_bases() {
        assert_eq!(Wid::parse("74").map(Wid::raw), Some(74));
        assert_eq!(Wid::parse("0x4a").map(Wid::raw), Some(74));
        assert_eq!(Wid::parse("0X4A").map(Wid::raw), Some(74));
        // A real id from an X server: eight hex digits.
        assert_eq!(Wid::parse("0x3a00007").map(Wid::raw), Some(0x03a0_0007));
    }

    #[test]
    fn a_non_window_reads_as_no_id() {
        // Zero names no window, and X11 never issues it.
        assert_eq!(Wid::parse("0"), None);
        assert_eq!(Wid::parse("0x0"), None);
        // Not a number, or not a number in either base.
        assert_eq!(Wid::parse("zz"), None);
        assert_eq!(Wid::parse(""), None);
        assert_eq!(Wid::parse("0x"), None);
        assert_eq!(Wid::parse("74a"), None);
        assert_eq!(Wid::parse("-74"), None);
        // Past the 32-bit range of an X window id.
        assert_eq!(Wid::parse("0x1ffffffff"), None);
    }
}
