//! Auto-repeat for the keys the viewer handles manually (nav keys, grid
//! moves).
//!
//! Raylib reports key presses as edges only.
//! `IsKeyPressed` fires once per physical press.
//! Raylib discards GLFW repeat events.
//! Holding a key then does nothing.
//! This module reimplements the X server auto-repeat on top of
//! `IsKeyDown`.
//! It uses the same delay and rate values that `xset r rate` configures.
//! The module reads these values once from `xset q`.
//! It falls back to the Xorg defaults (660 ms delay, 20 repeats/s) when it
//! cannot read them (Wayland, no X).

use std::{process::Command, sync::OnceLock};

/// The initial delay in seconds and the repeat rate in repeats per second.
/// The function reads these values once, on first use.
pub fn settings() -> (f32, f32) {
    static CACHE: OnceLock<(f32, f32)> = OnceLock::new();
    *CACHE.get_or_init(|| xset_settings().unwrap_or((0.66, 20.0)))
}

/// Per-key auto-repeat state.
/// Feed it once per frame with `tick`.
#[derive(Clone, Copy, Default)]
pub struct RepeatState {
    held: bool,
    next: f64,
}

impl RepeatState {
    /// Feed this key's state once per frame.
    ///
    /// `edge`: the raw event queue reports the key press this frame.
    /// `IsKeyPressed` can miss taps that fit inside one frame.
    /// `down`: the key is held now.
    /// `now`: the value from `rl.get_time()`.
    /// The function returns true exactly when the action fires.
    /// It fires on the initial press.
    /// It then fires every 1/rate seconds after the hold outlasts the delay.
    ///
    /// # Examples
    ///
    /// ```
    /// # use versatile_viewer::keyrepeat::RepeatState;
    /// let mut st = RepeatState::default();
    /// // The initial press fires immediately.
    /// assert!(st.tick(true, true, 0.0, 0.66, 20.0));
    /// // Repeats wait out the delay (660 ms), then fire every 1/rate (50 ms
    /// // at 20/s).
    /// // The schedule starts at the last fire, not at `now`.
    /// assert!(!st.tick(false, true, 0.65, 0.66, 20.0));
    /// assert!(st.tick(false, true, 0.67, 0.66, 20.0));
    /// assert!(st.tick(false, true, 0.72, 0.66, 20.0));
    /// // Release stops the repeats. A fresh edge fires again.
    /// assert!(!st.tick(false, false, 0.8, 0.66, 20.0));
    /// assert!(st.tick(true, true, 1.0, 0.66, 20.0));
    /// ```
    pub fn tick(&mut self, edge: bool, down: bool, now: f64, delay: f32, rate: f32) -> bool {
        if edge {
            self.held = true;
            self.next = now + f64::from(delay).max(0.0);
            return true;
        }
        if !down || !self.held {
            self.held = false;
            return false;
        }
        if now >= self.next {
            // Schedule relative to the last fire (not to `now`).
            // The repeats then stay at the configured rate.
            // The code never fires twice in one frame.
            self.next = (self.next + 1.0 / f64::from(rate).max(0.001)).max(now);
            return true;
        }
        false
    }
}

/// Ask the X server for its keyboard auto-repeat settings.
/// The command `xset q` shows these values.
/// The command `xset r rate delay rate` sets them.
fn xset_settings() -> Option<(f32, f32)> {
    let out = Command::new("xset").arg("q").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut delay = None;
    let mut rate = None;
    for line in text.lines() {
        if delay.is_none() {
            delay = num_after(line, "auto repeat delay:").map(|ms| ms / 1000.0);
        }
        if rate.is_none() {
            rate = num_after(line, "repeat rate:");
        }
    }
    match (delay, rate) {
        (Some(d), Some(r)) if d > 0.0 && r > 0.0 => Some((d, r)),
        _ => None,
    }
}

/// The first integer after `key` on this line.
/// "    auto repeat delay:  660    repeat rate:  20" becomes 660.
fn num_after(line: &str, key: &str) -> Option<f32> {
    let rest = line.split_once(key)?.1;
    let num: String = rest
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    num.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_fires_on_edge_then_at_rate() {
        let mut st = RepeatState::default();
        let (d, r) = (0.66, 20.0);
        assert!(st.tick(true, true, 0.0, d, r)); // initial press fires
        assert!(!st.tick(false, true, 0.1, d, r)); // held, before delay
        assert!(!st.tick(false, true, 0.65, d, r));
        assert!(st.tick(false, true, 0.67, d, r)); // first repeat past delay
        assert!(!st.tick(false, true, 0.70, d, r));
        assert!(st.tick(false, true, 0.72, d, r)); // next repeat 1/20 s later
        assert!(!st.tick(false, false, 0.8, d, r)); // released: nothing
        assert!(st.tick(true, true, 1.0, d, r)); // fresh press fires again
    }

    #[test]
    fn repeat_press_release_within_one_frame() {
        // Edge caught by the event queue fires once even though the key is
        // already up when polled (IsKeyDown false).
        let mut st = RepeatState::default();
        assert!(st.tick(true, false, 0.0, 0.66, 20.0));
        assert!(!st.tick(false, false, 0.01, 0.66, 20.0));
        assert!(!st.tick(false, false, 1.0, 0.66, 20.0));
    }

    #[test]
    fn repeat_never_fires_without_an_edge() {
        // Without an observed press edge the key does not count as held.
        // No repeat then fires.
        // The first edge starts the cycle.
        let mut st = RepeatState::default();
        assert!(!st.tick(false, true, 0.0, 0.66, 20.0));
        assert!(!st.tick(false, true, 10.0, 0.66, 20.0));
        assert!(st.tick(true, true, 10.0, 0.66, 20.0));
        assert!(!st.tick(false, true, 10.5, 0.66, 20.0));
        assert!(st.tick(false, true, 10.7, 0.66, 20.0)); // delay elapsed
    }

    #[test]
    fn xset_delay_and_rate_parse() {
        let line = "    auto repeat delay:  660    repeat rate:  20";
        assert_eq!(num_after(line, "auto repeat delay:"), Some(660.0));
        assert_eq!(num_after(line, "repeat rate:"), Some(20.0));
        assert_eq!(num_after("nothing here", "repeat rate:"), None);
    }
}
