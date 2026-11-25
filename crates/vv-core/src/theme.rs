//! The recolor theme: the two colors every picture maps onto.
//!
//! A page drawn on white paper with black ink is the one picture a dark window
//! cannot show, so the quad shader maps it: black becomes [`Theme::ink`], white
//! becomes [`Theme::paper`], and every color between them keeps its hue while
//! its lightness travels between the two.
//! The map runs per fragment as the picture is composited, so nothing decodes
//! twice and the toggle costs one uniform write.
//!
//! # The map
//!
//! A color is one lightness on the grey axis plus a direction off it:
//!
//! ```text
//! l  = 0.30 r + 0.59 g + 0.11 b     the weighted lightness
//! h  = rgb - l                      the hue, a vector in the equal-lightness plane
//! u  = colorumax(h, l, 0, 1)        how far that direction may walk before it
//!                                   leaves the RGB cube
//! s  = 1 / u                        the saturation, 0 on the grey axis, 1 at the edge
//! l' = l (l2 - l1) + l1             the lightness between paper and ink
//! su = s colorumax(h, l', l1, l2)   the same walk, clamped at the new lightness
//! rgb' = (1 - max rgb) (ink - l1) + min rgb (paper - l2) + l' + su h
//! ```
//!
//! `l1` and `l2` are the lightnesses of [`Theme::ink`] and [`Theme::paper`].
//! The third term of the last line is the theme's own chroma, weighted by how
//! black or how white the pixel is, and it is what lands a black pixel on the
//! ink itself and a white one on the paper itself; without it a grey page would
//! recolor to grey and never pick up the theme's color.
//! The hue is kept because a page is not a photograph: without it a red chart
//! line and a blue one of the same lightness would both turn one shade of the
//! paper, and a legend would lose the only thing it says.
//!
//! # A paper that is the window background
//!
//! [`THEME`]'s paper is the window background, so a recolored page runs to the
//! edge of the window without a seam. In the image view that is the whole
//! point; in the grid it costs the one cue that says where a cell starts, since
//! the gaps between cells are that same background and a white page recolors
//! onto it exactly. [`PLATE`] is the shade a cell draws under its thumbnail
//! while the theme is on, so a wall of recolored pages still shows its cells.
//!
//! The arithmetic matches Zathura's `recolor_slow`, which is the reference this
//! viewer follows, so a page reads the same in both.
//! The GPU runs it on the encoded bytes the decoders write, never on linear
//! light, because the swapchain stays non-sRGB.

use crate::draw::Color;

/// The weights of the lightness, which must sum to one.
///
/// The same three Zathura uses, so the two viewers agree on where a color sits
/// between paper and ink.
pub const LUMA: [f32; 3] = [0.30, 0.59, 0.11];

/// Whether [`THEME`] bends the lightness curve toward the ink.
///
/// `l` becomes `(l - l⁴) s/2 + l⁴`: the ends of the range keep their contrast
/// and the middle travels toward the ink, which makes a grey rule or a grey
/// glyph bolder on the page and flattens the middle of a photograph.
/// Off, which is what the reference viewer does by default, so a page reads the
/// same in both.
/// No key selects it today: the theme carries the flag so a later one can, and
/// a paint drives the other curve so the branch stays proven.
pub const ADJUST_LIGHTNESS: bool = false;

/// The theme the `Shift+T` toggle turns on: ink on a dark blue-green page.
///
/// [`Theme::paper`] is the window background itself, so a recolored page fills
/// the window without a letterbox of another shade at its edge.
pub const THEME: Theme = Theme::new(Color::from_hex("#062329"), Color::from_hex("#d1b897"));

/// The plate a grid cell draws under its thumbnail while the theme is on.
///
/// A theme's paper is the window background, and the gaps between cells are
/// that same background, so a recolored page has nothing to end against: two
/// white pages side by side would read as one continuous dark field, and a grid
/// of documents would show no cell boundary at all.
/// The plate is a shade off the paper, so a recolored page sits in a cell the
/// way a printed page sits on a desk.
/// The grid draws it only while a theme is on, because a plain grid already
/// separates its cells with white paper.
pub const PLATE: Color = Color::from_hex("#0b3335");

/// What a recolored picture maps its dark and light ends onto.
///
/// The names say what a page shows: [`Self::ink`] is what black becomes and
/// [`Self::paper`] what white becomes.
/// The renderer takes the two as one uniform, because the map needs a lightness
/// per color and the CPU computes that once for the whole frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    /// What a white pixel becomes.
    pub paper: Color,
    /// What a black pixel becomes.
    pub ink: Color,
    /// Whether the map bends the lightness curve, which [`ADJUST_LIGHTNESS`]
    /// sets for [`THEME`].
    ///
    /// The flag rides the theme so a caller can pick the other curve without a
    /// second constant, and so a test drives the branch the default turns off.
    pub adjust_lightness: bool,
}

impl Theme {
    /// A theme from the color the paper becomes and the color the ink becomes.
    ///
    /// # Panics
    ///
    /// Panics when either color is not opaque.
    /// The map keeps the alpha of the pixel it recolors and carries the
    /// lightness of each theme color in the fourth uniform channel, so a
    /// translucent theme would recolor exactly as if it were opaque: the check
    /// refuses that picture instead of painting a wrong one.
    #[must_use]
    pub const fn new(paper: Color, ink: Color) -> Self {
        assert!(paper.a == 255, "the recolor paper is opaque");
        assert!(ink.a == 255, "the recolored ink is opaque");
        Self {
            paper,
            ink,
            adjust_lightness: ADJUST_LIGHTNESS,
        }
    }

    /// The same two colors with the lightness curve flipped.
    #[must_use]
    pub const fn with_adjust_lightness(self, adjust_lightness: bool) -> Self {
        Self {
            adjust_lightness,
            ..self
        }
    }

    /// The paper as the shader takes it: three channels and their lightness.
    ///
    /// The lightness rides the fourth channel because the map interpolates
    /// between the two lightnesses on every fragment, so computing it once per
    /// frame is the difference between one multiply and a million.
    #[must_use]
    pub const fn paper_uniform(&self) -> [f32; 4] {
        channels_and_luma(self.paper)
    }

    /// The ink as the shader takes it: three channels and their lightness.
    #[must_use]
    pub const fn ink_uniform(&self) -> [f32; 4] {
        channels_and_luma(self.ink)
    }
}

/// The three channels of `color` as 0..1 floats, then their weighted lightness.
const fn channels_and_luma(color: Color) -> [f32; 4] {
    let r = color.r as f32 / 255.0;
    let g = color.g as f32 / 255.0;
    let b = color.b as f32 / 255.0;
    [r, g, b, LUMA[0] * r + LUMA[1] * g + LUMA[2] * b]
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // the asserted values are exact divisions by 255
mod tests {
    use super::*;

    /// The palette reads as the hexadecimal it was picked in.
    #[test]
    fn the_theme_round_trips_through_its_channels() {
        assert_eq!(THEME.paper, Color::new(0x06, 0x23, 0x29, 255));
        assert_eq!(THEME.ink, Color::new(0xd1, 0xb8, 0x97, 255));
    }

    /// A translucent theme is refused, because the map cannot carry its alpha.
    #[test]
    #[should_panic(expected = "the recolor paper is opaque")]
    fn a_translucent_theme_is_refused() {
        let _ = Theme::new(Color::new(0, 0, 0, 128), THEME.ink);
    }

    /// The plate stands off the paper, or a cell would end where it starts.
    #[test]
    fn the_plate_stands_off_the_paper() {
        let (paper, plate) = (THEME.paper, PLATE);
        let luma = |color: Color| {
            0.30 * f32::from(color.r) + 0.59 * f32::from(color.g) + 0.11 * f32::from(color.b)
        };
        assert!(
            luma(plate) > luma(paper) + 8.0,
            "the plate reads against the page: {plate:?} vs {paper:?}"
        );
        assert_ne!((plate.r, plate.g, plate.b), (paper.r, paper.g, paper.b));
    }

    /// The fourth channel is the lightness, and ink stands above paper.
    ///
    /// A theme whose ink is darker than its paper would invert every page, so
    /// the ordering is the one property the map cannot recover from.
    #[test]
    fn the_lightness_rides_the_fourth_channel() {
        let paper = THEME.paper_uniform();
        let ink = THEME.ink_uniform();
        assert_eq!(
            paper[3],
            LUMA[0] * paper[0] + LUMA[1] * paper[1] + LUMA[2] * paper[2]
        );
        assert_eq!(
            ink[3],
            0.30 * 209.0 / 255.0 + 0.59 * 184.0 / 255.0 + 0.11 * 151.0 / 255.0
        );
        assert!(ink[3] > paper[3] + 0.4, "{ink:?} vs {paper:?}");
    }
}
