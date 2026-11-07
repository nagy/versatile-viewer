//! The keyboard and the pointer, as one frame's worth of state.
//!
//! The module replaces the input calls of raylib: `IsKeyDown`,
//! `IsKeyPressed`, `GetKeyPressed`, `GetCharPressed`, `GetMousePosition`,
//! `GetMouseDelta`, `GetMouseWheelMove` and the three button states.
//!
//! # Held state, frame edges and repeats
//!
//! The view ramps advance on held state, so a hold must read `true` every frame
//! between its press and its release: [`Input::is_key_down`] answers that.
//!
//! A one-shot action reads an edge, which [`Input::pressed_keys`] answers for
//! one frame only.
//!
//! X11 auto-repeat arrives as a fresh press event with `repeat` set. raylib
//! dropped those events, so `src/keyrepeat.rs` rebuilt the repeat rate from
//! `xset q`. winit reports the flag, so the grid steps its selection on
//! [`Input::repeated_keys`] and every one-shot action skips it.
//!
//! # The pointer
//!
//! Two sources exist and they do not measure the same travel.
//! [`Input::motion`] takes the raw device deltas of `DeviceEvent::MouseMotion`,
//! which carry no pointer acceleration: a captured drag reads that source, and
//! the pan multiplies it by the viewer's own factor.
//! [`Input::cursor`] takes the window positions of `CursorMoved`, which carry
//! the acceleration and the desktop scale.
//!
//! # Typed text
//!
//! A `+` or `-` on a non-US layout names its physical key only through the
//! keyboard event that produced it, so [`Input::typed`] pairs the character
//! with that key's code. [`crate::zoomkeys`] learns the pair from it.

use std::collections::HashSet;

use winit::{
    event::{ElementState, MouseButton, MouseScrollDelta},
    keyboard::KeyCode,
};

use crate::draw::Vec2;

/// One wheel notch, in pixels of a pixel-precision scroll device.
const WHEEL_PIXEL_NOTCH: f64 = 100.0;

/// The most recent typed character and the physical key that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Typed {
    /// The key that went down.
    pub key: KeyCode,
    /// The character it typed.
    pub text: char,
}

/// One frame's keyboard, pointer and wheel state.
#[derive(Debug, Default)]
pub struct Input {
    /// The keys held now.
    held: HashSet<KeyCode>,
    /// The keys that went down this frame, without a repeat.
    pressed: HashSet<KeyCode>,
    /// The keys that came back as an auto-repeat this frame.
    repeated: HashSet<KeyCode>,
    /// The keys that came up this frame.
    released: HashSet<KeyCode>,
    /// The typed characters of this frame, each with its physical key.
    typed: Vec<Typed>,
    /// The pointer position, in window pixels.
    cursor: Vec2,
    /// The raw device travel since the last frame.
    delta: Vec2,
    /// The wheel travel since the last frame, in notches.
    wheel: f32,
    /// The buttons held now.
    buttons: HashSet<MouseButton>,
    /// The buttons that went down this frame.
    pressed_buttons: HashSet<MouseButton>,
    /// The buttons that came up this frame.
    released_buttons: HashSet<MouseButton>,
}

impl Input {
    /// Empty the frame edges, so the next frame reads a fresh set.
    ///
    /// The held state, the cursor position and the buttons stay, because they
    /// describe the world and not the frame.
    pub fn begin_frame(&mut self) {
        self.pressed.clear();
        self.repeated.clear();
        self.released.clear();
        self.typed.clear();
        self.delta = Vec2::ZERO;
        self.wheel = 0.0;
        self.pressed_buttons.clear();
        self.released_buttons.clear();
    }

    /// Take one keyboard event.
    ///
    /// `text` is the character the event produced, if any: a modifier press
    /// carries none.
    pub fn key(&mut self, key: KeyCode, state: ElementState, repeat: bool, text: Option<char>) {
        match state {
            ElementState::Pressed => {
                self.held.insert(key);
                if repeat {
                    self.repeated.insert(key);
                } else {
                    self.pressed.insert(key);
                }
                if let Some(text) = text {
                    self.typed.push(Typed { key, text });
                }
            }
            ElementState::Released => {
                self.held.remove(&key);
                self.released.insert(key);
            }
        }
    }

    /// Take one pointer position, in window pixels.
    pub fn cursor(&mut self, position: Vec2) {
        self.cursor = position;
    }

    /// Move the pointer by the raw device travel of one event.
    ///
    /// A captured drag reads this: the pointer itself barely moves, because the
    /// grab keeps it in place.
    pub fn motion(&mut self, delta: Vec2) {
        self.delta += delta;
        self.cursor += delta;
    }

    /// Put the pointer at a position without reading it as travel.
    ///
    /// The capture recenters the pointer, and that warp is not a move.
    pub fn set_cursor(&mut self, position: Vec2) {
        self.cursor = position;
    }

    /// Take one wheel report.
    pub fn wheel(&mut self, delta: MouseScrollDelta) {
        match delta {
            MouseScrollDelta::LineDelta(_, y) => self.wheel += y,
            MouseScrollDelta::PixelDelta(position) => {
                self.wheel += (position.y / WHEEL_PIXEL_NOTCH) as f32;
            }
        }
    }

    /// Take one mouse button event.
    pub fn button(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if self.buttons.insert(button) {
                    self.pressed_buttons.insert(button);
                }
            }
            ElementState::Released => {
                self.buttons.remove(&button);
                self.released_buttons.insert(button);
            }
        }
    }

    /// Forget every held key and button.
    ///
    /// A lost focus never delivers the releases of the keys that were down, so
    /// a hold would otherwise stay on after the window came back.
    pub fn forget_held(&mut self) {
        self.held.clear();
        self.buttons.clear();
    }

    /// Is the key held now?
    #[must_use]
    pub fn is_key_down(&self, key: KeyCode) -> bool {
        self.held.contains(&key)
    }

    /// Did the key go down this frame, without a repeat?
    #[must_use]
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    /// Did the key come up this frame?
    #[must_use]
    pub fn is_key_released(&self, key: KeyCode) -> bool {
        self.released.contains(&key)
    }

    /// The keys that went down this frame, without a repeat.
    pub fn pressed_keys(&self) -> impl Iterator<Item = KeyCode> + '_ {
        self.pressed.iter().copied()
    }

    /// The keys that came back as an auto-repeat this frame.
    pub fn repeated_keys(&self) -> impl Iterator<Item = KeyCode> + '_ {
        self.repeated.iter().copied()
    }

    /// The typed characters of this frame, each with its physical key.
    #[must_use]
    pub fn typed(&self) -> &[Typed] {
        &self.typed
    }

    /// The pointer position, in window pixels.
    #[must_use]
    pub fn cursor_position(&self) -> Vec2 {
        self.cursor
    }

    /// The raw device travel since the last frame.
    #[must_use]
    pub fn delta(&self) -> Vec2 {
        self.delta
    }

    /// The wheel travel since the last frame, in notches.
    #[must_use]
    pub fn wheel_move(&self) -> f32 {
        self.wheel
    }

    /// Is the button held now?
    #[must_use]
    pub fn is_button_down(&self, button: MouseButton) -> bool {
        self.buttons.contains(&button)
    }

    /// Did the button go down this frame?
    #[must_use]
    pub fn is_button_pressed(&self, button: MouseButton) -> bool {
        self.pressed_buttons.contains(&button)
    }

    /// Did the button come up this frame?
    #[must_use]
    pub fn is_button_released(&self, button: MouseButton) -> bool {
        self.released_buttons.contains(&button)
    }

    /// Did any of these buttons go down this frame?
    #[must_use]
    pub fn any_button_pressed(&self, buttons: &[MouseButton]) -> bool {
        buttons.iter().any(|&b| self.is_button_pressed(b))
    }

    /// Is any of these buttons held now?
    #[must_use]
    pub fn any_button_down(&self, buttons: &[MouseButton]) -> bool {
        buttons.iter().any(|&b| self.is_button_down(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hold reads true every frame, and an edge only once.
    #[test]
    fn a_hold_outlives_its_edge() {
        let mut input = Input::default();
        input.key(KeyCode::KeyH, ElementState::Pressed, false, None);
        assert!(input.is_key_down(KeyCode::KeyH));
        assert!(input.is_key_pressed(KeyCode::KeyH));
        input.begin_frame();
        assert!(input.is_key_down(KeyCode::KeyH));
        assert!(!input.is_key_pressed(KeyCode::KeyH));
        input.key(KeyCode::KeyH, ElementState::Released, false, None);
        assert!(!input.is_key_down(KeyCode::KeyH));
        assert!(input.is_key_released(KeyCode::KeyH));
    }

    /// An auto-repeat is not an edge, and the grid reads it on its own path.
    #[test]
    fn a_repeat_is_not_a_press() {
        let mut input = Input::default();
        input.key(KeyCode::KeyJ, ElementState::Pressed, false, None);
        input.begin_frame();
        input.key(KeyCode::KeyJ, ElementState::Pressed, true, None);
        assert!(!input.is_key_pressed(KeyCode::KeyJ));
        assert_eq!(
            input.repeated_keys().collect::<Vec<_>>(),
            vec![KeyCode::KeyJ]
        );
        assert!(input.is_key_down(KeyCode::KeyJ));
    }

    /// A lost focus drops every hold.
    #[test]
    fn a_lost_focus_drops_the_holds() {
        let mut input = Input::default();
        input.key(KeyCode::KeyF, ElementState::Pressed, false, None);
        input.button(MouseButton::Left, ElementState::Pressed);
        input.forget_held();
        assert!(!input.is_key_down(KeyCode::KeyF));
        assert!(!input.is_button_down(MouseButton::Left));
    }

    /// The raw travel moves the pointer, and the wheel counts notches.
    #[test]
    fn the_pointer_and_the_wheel_accumulate() {
        let mut input = Input::default();
        input.cursor(Vec2::new(10.0, 10.0));
        input.motion(Vec2::new(3.0, -2.0));
        input.motion(Vec2::new(1.0, 1.0));
        assert_eq!(input.cursor_position(), Vec2::new(14.0, 9.0));
        assert_eq!(input.delta(), Vec2::new(4.0, -1.0));
        input.wheel(MouseScrollDelta::LineDelta(0.0, 1.0));
        input.wheel(MouseScrollDelta::PixelDelta(
            winit::dpi::PhysicalPosition::new(0.0, 200.0),
        ));
        assert!((input.wheel_move() - 3.0).abs() < 1e-6);
        input.begin_frame();
        assert_eq!(input.delta(), Vec2::ZERO);
        assert_eq!(input.wheel_move(), 0.0);
        assert_eq!(input.cursor_position(), Vec2::new(14.0, 9.0));
    }

    /// A typed character keeps the key that produced it, for the zoom keys.
    #[test]
    fn a_typed_character_carries_its_key() {
        let mut input = Input::default();
        input.key(KeyCode::Equal, ElementState::Pressed, false, Some('+'));
        assert_eq!(
            input.typed(),
            &[Typed {
                key: KeyCode::Equal,
                text: '+'
            }]
        );
    }
}
