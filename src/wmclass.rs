//! Set the X11 WM_CLASS hint on the raylib window.
//!
//! raylib (via GLFW) derives WM_CLASS from the window title at creation
//! time — here that is the full `versatile-viewer — <path>` string, which
//! is useless for window rules — and never exposes GLFW's class-name
//! hints. Instead we open our own libX11 connection and call
//! `XSetClassHint` directly. No-op on Wayland (app_id there comes from the
//! desktop entry) and whenever no X display is reachable.

use std::{ffi::CString, os::raw::c_void, ptr};

use x11::xlib;

/// WM_CLASS value: both res_name (instance) and res_class, matching the
/// binary name.
const CLASS: &str = "vv";

/// Stamp `WM_CLASS = vv` on the window behind raylib's `get_window_handle()`.
///
/// Best effort: silently does nothing on Wayland or if Xlib cannot open a
/// display; the title-derived class from GLFW stays in effect then.
pub fn set_class(handle: *mut c_void) {
    // GetWindowHandle() returns a pointer to the XID on X11 but an opaque
    // Wayland window struct pointer otherwise — only deref it under a
    // real X session.
    if env_is_set("WAYLAND_DISPLAY") || !env_is_set("DISPLAY") {
        return;
    }
    if handle.is_null() {
        return;
    }
    // SAFETY: on X11, raylib stores the XID in static storage and returns
    // a pointer to it; it stays valid for the window's lifetime.
    let xid = unsafe { *(handle.cast::<xlib::XID>()) };
    if xid == 0 {
        return;
    }

    // SAFETY: plain Xlib usage — a dedicated connection used only to set
    // the class-hint property (any connection may do that); every pointer
    // checked before use.
    unsafe {
        let display = xlib::XOpenDisplay(ptr::null());
        if display.is_null() {
            return;
        }

        let name = CString::new(CLASS).unwrap_or_default();
        let hint = xlib::XAllocClassHint();
        if !hint.is_null() {
            (*hint).res_name = name.as_ptr().cast_mut();
            (*hint).res_class = name.as_ptr().cast_mut();
            xlib::XSetClassHint(display, xid, hint);
            xlib::XFree(hint.cast());
        }
        xlib::XCloseDisplay(display);
    }
}

/// Whether the env var exists and is non-empty.
fn env_is_set(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|v| !v.is_empty())
}
