//! The physical key behind a typed `+` or `-`.
//!
//! The image zoom needs the held state of the `+` and `-` keys.
//! A US layout puts them on the `=` and `-` keycodes, which
//! [`raylib::consts::KeyboardKey`] names.
//! Other layouts move `+` to another physical key.
//!
//! A typed character names the key by what it produces, but it carries no
//! keycode.
//! [`TypedKeys`] learns the keycode instead.
//! When a typed `+` arrives, the key that went down in the same frame is the
//! `+` key.
//! The code records that keycode and polls it with `is_key_down` from then on.
//!
//! A time window cannot do this job.
//! The X server auto-repeats only the key that went down last.
//! A held `+` therefore stops repeating as soon as another key goes down, and
//! a deadline that waits for the next repeat lapses while the `+` key still
//! is down.
//! Polling the keycode reads the real held state and ignores the repeat rate.

use raylib::consts::KeyboardKey;

/// The physical keycode that a typed `+` or `-` belongs to.
/// The value comes from `rl.get_key_pressed()` in the same frame as the
/// character.
#[derive(Debug, Clone, Copy, Default)]
pub struct TypedKeys {
    /// The keycode learned for `+`, if any.
    zoom_in: Option<KeyboardKey>,
    /// The keycode learned for `-`, if any.
    zoom_out: Option<KeyboardKey>,
}

impl TypedKeys {
    /// Record a typed `+`.
    /// `key`: the key that went down this frame, if one did.
    ///
    /// The first frame that pairs a typed `+` with a key press teaches the
    /// `+` keycode.
    /// Later frames keep it.
    pub fn typed_zoom_in(&mut self, key: Option<KeyboardKey>) {
        if self.zoom_in.is_none() {
            self.zoom_in = key;
        }
    }

    /// Record a typed `-`.
    /// `key`: the key that went down this frame, if one did.
    pub fn typed_zoom_out(&mut self, key: Option<KeyboardKey>) {
        if self.zoom_out.is_none() {
            self.zoom_out = key;
        }
    }

    /// The keycode learned for `+`, if any.
    #[must_use]
    pub fn zoom_in_key(&self) -> Option<KeyboardKey> {
        self.zoom_in
    }

    /// The keycode learned for `-`, if any.
    #[must_use]
    pub fn zoom_out_key(&self) -> Option<KeyboardKey> {
        self.zoom_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first typed character pairs with the key that went down.
    #[test]
    fn a_typed_character_learns_its_key() {
        let mut keys = TypedKeys::default();
        keys.typed_zoom_in(Some(KeyboardKey::KEY_V));
        assert_eq!(keys.zoom_in_key(), Some(KeyboardKey::KEY_V));
    }

    /// The directions learn independently.
    #[test]
    fn the_directions_learn_independently() {
        let mut keys = TypedKeys::default();
        keys.typed_zoom_in(Some(KeyboardKey::KEY_V));
        assert_eq!(keys.zoom_out_key(), None);
        keys.typed_zoom_out(Some(KeyboardKey::KEY_W));
        assert_eq!(keys.zoom_in_key(), Some(KeyboardKey::KEY_V));
        assert_eq!(keys.zoom_out_key(), Some(KeyboardKey::KEY_W));
    }

    /// A later typed character does not overwrite a learned keycode.
    /// The auto-repeat delivers characters without a press event.
    #[test]
    fn a_learned_key_stays() {
        let mut keys = TypedKeys::default();
        keys.typed_zoom_in(Some(KeyboardKey::KEY_V));
        keys.typed_zoom_in(None);
        assert_eq!(keys.zoom_in_key(), Some(KeyboardKey::KEY_V));
    }

    /// A typed character without a press event teaches nothing.
    #[test]
    fn nothing_is_learned_without_a_press() {
        let mut keys = TypedKeys::default();
        keys.typed_zoom_in(None);
        keys.typed_zoom_out(None);
        assert_eq!(keys.zoom_in_key(), None);
        assert_eq!(keys.zoom_out_key(), None);
    }
}
