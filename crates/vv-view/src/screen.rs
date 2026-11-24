//! Whether the window the viewer draws into is on screen.
//!
//! A window manager hides a window by unmapping it, and X reports that as an
//! `UnmapNotify`: a minimize sends one, and so does the hide of a tiling window
//! manager that keeps no iconified state at all (`EXWM` calls `UnmapWindow` and
//! sets `WM_STATE` to `IconicState` beside it).
//! Neither signal reaches this process as a window event.
//! winit's X11 backend handles `MapNotify` and `VisibilityNotify` and drops
//! `UnmapNotify`, and a window manager may clear the window's event mask around
//! its own unmap, so even a listener that selected `StructureNotifyMask` misses
//! the event: `EXWM` sets the mask to `NoEvent`, unmaps, and restores it after.
//! The graphics stack says nothing either, because a Vulkan surface carries no
//! notion of occlusion: an unmapped window keeps completing presents at the
//! display's refresh rate, so every frame of the loop still rasterizes,
//! uploads and composites a picture that no display shows.
//!
//! The map state is the one fact that cannot be lost, because it is read
//! instead of delivered: [`OnScreen`] asks for the window's attributes and
//! reads `map_state` from the reply.
//! `IsUnviewable` covers a window whose host window is unmapped, so an
//! embedded pane needs no read of its own.
//! One read is one round trip, so it happens at most once per [`MAP_POLL`]
//! while the window is on screen and on every frame while it is not: the frame
//! that carries the `Expose` of a window coming back reads the state that
//! follows it and paints at once, where a hidden window's own rate would leave
//! the picture stale until the next poll.

use std::time::{Duration, Instant};

use x11rb::{
    protocol::xproto::{ConnectionExt as _, MapState, Window},
    rust_connection::RustConnection,
};

use crate::pane::PaneError;

/// How often a window that is on screen is read again.
///
/// A hidden window is read on every frame instead, so this rate only says how
/// late a hide is noticed while the loop still paints: one frame, because the
/// loop runs faster than this.
/// The read is one round trip, and ten of them a second cost less than the
/// frame they ride on.
pub const MAP_POLL: Duration = Duration::from_millis(100);

/// Whether the window is on screen, read from the X server.
///
/// A display that goes away ends the reading instead of reporting on every
/// frame, and the window then counts as on screen for good: painting a picture
/// nobody sees costs one frame, and leaving a visible one stale costs the
/// user's trust in it.
pub struct OnScreen {
    /// The display, or `None` once a read failed.
    conn: Option<RustConnection>,
    /// The X11 window the viewer draws into, as a top-level window or as the
    /// child of a host.
    window: Window,
    /// The answer of the last read.
    shown: bool,
    /// When the next read is due while the window is on screen.
    next: Instant,
}

impl OnScreen {
    /// Read the state of `window` once.
    ///
    /// # Errors
    ///
    /// [`PaneError::Connect`] when `$DISPLAY` names no reachable X server, and
    /// [`PaneError::Reply`] when the server rejects the attributes request.
    pub fn new(window: u32) -> Result<Self, PaneError> {
        let conn = RustConnection::connect(None)?.0;
        let shown = read(&conn, window)?;
        Ok(Self {
            conn: Some(conn),
            window,
            shown,
            next: Instant::now() + MAP_POLL,
        })
    }

    /// Is the window on screen?
    ///
    /// The answer is the last read, so a frame pays a round trip only when one
    /// is due.
    pub fn on_screen(&mut self) -> bool {
        let now = Instant::now();
        if self.due(now) {
            self.next = now + MAP_POLL;
            self.shown = match self.read() {
                Ok(shown) => shown,
                Err(err) => {
                    tracing::warn!(err = %err, "window state");
                    self.conn = None;
                    true
                }
            };
        }
        self.shown
    }

    /// Is a read due?
    ///
    /// A window that is off screen is read on every frame, so the frame that
    /// brings it back paints in the same frame as the `Expose` that woke it.
    fn due(&self, now: Instant) -> bool {
        !self.shown || now >= self.next
    }

    /// Read the map state once, or answer `true` when the display was lost.
    fn read(&self) -> Result<bool, PaneError> {
        match self.conn.as_ref() {
            Some(conn) => read(conn, self.window),
            None => Ok(true),
        }
    }
}

/// Read the map state of one window.
///
/// `IsViewable` is the whole answer: an unmapped window and one whose host
/// window is unmapped both show no pixel of the picture.
fn read(conn: &RustConnection, window: Window) -> Result<bool, PaneError> {
    let attributes = conn.get_window_attributes(window)?.reply()?;
    Ok(matches!(attributes.map_state, MapState::VIEWABLE))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::{MAP_POLL, OnScreen};

    /// A watcher with no display, which answers what it was built with until a
    /// read replaces it: a lost display answers `true` and clears the state.
    fn without_a_display(shown: bool, next: Instant) -> OnScreen {
        OnScreen {
            conn: None,
            window: 1,
            shown,
            next,
        }
    }

    #[test]
    fn a_window_off_screen_is_read_on_every_frame() {
        let now = Instant::now();
        // The deadline of the last read still stands, and the read runs anyway,
        // because the frame that carries the `Expose` of the returning window
        // must find the map state that follows it.
        let hidden = without_a_display(false, now + MAP_POLL);
        assert!(hidden.due(now));
    }

    #[test]
    fn a_window_on_screen_is_read_once_a_poll() {
        let now = Instant::now();
        let shown = without_a_display(true, now + MAP_POLL);
        assert!(!shown.due(now));
        // A frame that is later than the poll reads again, and the deadline
        // keeps one read per poll however fast the frames come.
        assert!(shown.due(now + MAP_POLL));
    }

    #[test]
    fn a_lost_display_never_hides_the_window() {
        let mut state = without_a_display(false, Instant::now());
        assert!(state.on_screen());
        // One failed read is the whole loss, and no later frame hides the
        // picture on a display that says nothing.
        assert!(state.on_screen());
    }
}
