//! Set the X11 `WM_CLASS` hint on the raylib window.
//!
//! raylib uses GLFW.
//! GLFW derives `WM_CLASS` from the window title at creation time.
//! Here the title holds the full `versatile-viewer` name and the path.
//! Window rules need a different value.
//! GLFW never exposes its class-name hints.
//! This module opens its own libX11 connection and calls `XSetClassHint`
//! directly.
//! The module does nothing on Wayland.
//! There the desktop entry supplies `app_id`.
//! The module also does nothing when it cannot reach an X display.

use std::{ffi::CString, os::raw::c_void, ptr};

use x11::xlib;

/// The `WM_CLASS` value.
/// It sets both `res_name` (instance) and `res_class`.
/// Both match the binary name.
const CLASS: &str = "vv";

/// Set `WM_CLASS = vv` on the window behind raylib's `get_window_handle()`.
///
/// The function never reports an error.
/// It does nothing on Wayland.
/// It does nothing if Xlib cannot open a display.
/// The title-derived class from GLFW stays in effect then.
pub fn set_class(handle: *mut c_void) {
    // On X11, GetWindowHandle() returns a pointer to the XID.
    // On Wayland, it returns an opaque window struct pointer.
    // Dereference the pointer only under a real X session.
    if env_is_set("WAYLAND_DISPLAY") || !env_is_set("DISPLAY") {
        return;
    }
    if handle.is_null() {
        return;
    }
    // SAFETY: on X11, raylib stores the XID in static storage.
    // It returns a pointer to that storage.
    // The pointer stays valid for the window lifetime.
    let xid = unsafe { *(handle.cast::<xlib::XID>()) };
    if xid == 0 {
        return;
    }

    // SAFETY: this is plain Xlib usage.
    // The code opens a dedicated connection to set the class-hint property.
    // Any connection can set that property.
    // The code checks every pointer before use.
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

/// Return true when the environment variable exists and holds a value.
fn env_is_set(key: &str) -> bool {
    std::env::var_os(key).is_some_and(|v| !v.is_empty())
}
