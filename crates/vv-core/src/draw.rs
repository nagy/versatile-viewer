//! The drawing values of the viewer: a point, a color and a quad.
//!
//! [`Vec2`] is the one point type of the code. It measures window pixels in the
//! draw code, surface units in the image view and pixels of a plane in the
//! decoders, and every one of those units stays an `f32` pair.
//! The type is `glam`'s, re-exported here, so the arithmetic, the rounding, the
//! length and the interpolation come from that crate instead of a local copy.
//! [`turned`] is the one helper beside it: `glam` measures a turn in radians
//! counter-clockwise in a y-up space, and every draw rotation of the viewer
//! names degrees clockwise in a y-down space.
//! [`Color`] keeps the eight-bit channels the decoders write, so a tint or a
//! background passes to a texture or to a fill without a conversion.
//!
//! Both types are plain values with no GPU handle, so the whole module is
//! testable without a display.

pub use glam::Vec2;

use crate::dim::{self, Rect};

/// The same vector turned by `degrees`, clockwise in a y-down space.
///
/// The space grows down, so a positive angle turns toward the right of the
/// screen, which is the sign every draw rotation of the viewer uses.
/// `glam` measures radians counter-clockwise in a y-up space, so the two signs
/// meet here, in one place.
///
/// # Examples
///
/// ```
/// # use vv_core::draw::{Vec2, turned};
/// let x = Vec2::new(1.0, 0.0);
/// assert!(turned(x, 90.0).abs_diff_eq(Vec2::new(0.0, 1.0), 1e-6));
/// assert!(turned(x, 360.0).abs_diff_eq(x, 1e-5));
/// ```
#[must_use]
pub fn turned(v: Vec2, degrees: f32) -> Vec2 {
    v.rotate_angle(degrees.to_radians())
}

/// One color of the viewer, as four eight-bit channels.
///
/// The channels are straight alpha: [`Color::a`] says how much of the color the
/// draw keeps, and the renderer blends with `SRC_ALPHA` over
/// `ONE_MINUS_SRC_ALPHA`.
///
/// # Examples
///
/// ```
/// # use vv_core::draw::Color;
/// // A tint keeps the share of every channel it names.
/// assert_eq!(Color::WHITE.tint(Color::WHITE), Color::WHITE);
/// assert_eq!(
///     Color::WHITE.tint(Color::new(90, 90, 90, 255)),
///     Color::new(90, 90, 90, 255)
/// );
/// // The clear color takes the float channels a swapchain wants.
/// let clear = Color::new(0x00, 0x33, 0xff, 0xff).to_unit();
/// assert!((clear[1] - 0.2).abs() < 1e-9);
/// assert_eq!(clear[3], 1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Color {
    /// The red channel.
    pub r: u8,
    /// The green channel.
    pub g: u8,
    /// The blue channel.
    pub b: u8,
    /// The alpha channel, where 255 is opaque.
    pub a: u8,
}

impl Color {
    /// A color from its four channels.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque white, which keeps a texture as it was decoded.
    pub const WHITE: Self = Self::new(255, 255, 255, 255);
    /// Opaque black, which hides what it covers.
    pub const BLACK: Self = Self::new(0, 0, 0, 255);
    /// The yellow of the debug crosshair.
    pub const YELLOW: Self = Self::new(253, 249, 0, 255);
    /// The sky blue of the debug grab ring.
    pub const SKYBLUE: Self = Self::new(102, 191, 255, 255);

    /// This color multiplied by `tint`, channel by channel.
    ///
    /// The alpha of the tint scales nothing: a dimmed cell keeps its opacity
    /// and darkens its channels, which is what a grid selection needs.
    #[must_use]
    pub fn tint(self, tint: Self) -> Self {
        Self::new(
            mul_channel(self.r, tint.r),
            mul_channel(self.g, tint.g),
            mul_channel(self.b, tint.b),
            mul_channel(self.a, tint.a),
        )
    }

    /// The four channels, in the byte order a texture and a clear color take.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// The four channels as 0..1 floats, which a vertex tint and a shader
    /// constant take.
    #[must_use]
    pub fn to_f32(self) -> [f32; 4] {
        self.to_unit().map(|channel| channel as f32)
    }

    /// The four channels as the 0..1 floats a swapchain clear takes.
    ///
    /// The values stay non-linear, because the swapchain format stays
    /// non-sRGB: the bytes reach the screen as the decoder wrote them.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn to_unit(self) -> [f64; 4] {
        [
            f64::from(self.r) / 255.0,
            f64::from(self.g) / 255.0,
            f64::from(self.b) / 255.0,
            f64::from(self.a) / 255.0,
        ]
    }
}

/// The pivot of a turned draw: a point of the destination and an angle.
///
/// Raylib's `DrawTexturePro` took the two as loose arguments, and a caller that
/// swapped them drew a turned rect around the origin of the window.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Turn {
    /// The pivot, in window pixels.
    pub origin: Vec2,
    /// The clockwise angle, in degrees.
    pub angle: f32,
}

impl Turn {
    /// An unturned draw: no angle, and the origin names nothing.
    pub const NONE: Self = Self {
        origin: Vec2::ZERO,
        angle: 0.0,
    };

    /// A turn about `origin`.
    #[must_use]
    pub const fn new(origin: Vec2, angle: f32) -> Self {
        Self { origin, angle }
    }
}

/// Four corners of one quad, in perimeter order.
///
/// The renderer splits a quad into the two triangles `p0 p1 p2` and
/// `p0 p2 p3`, so the order goes around the shape and never crosses itself.
/// A triangle is a quad whose last corner repeats the first: the second
/// triangle then has no area and draws nothing.
///
/// # Examples
///
/// ```
/// # use vv_core::dim;
/// # use vv_core::draw::{Quad, Vec2};
/// let quad = Quad::rect(dim::rect(10.0, 20.0, 30.0, 40.0));
/// assert_eq!(quad.p[0], Vec2::new(10.0, 20.0));
/// assert_eq!(quad.p[2], Vec2::new(40.0, 60.0));
/// // A triangle repeats its first corner, so its second triangle is empty.
/// let tri = Quad::triangle(Vec2::ZERO, Vec2::new(4.0, 0.0), Vec2::new(0.0, 4.0));
/// assert_eq!(tri.p[3], tri.p[0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    /// The corners, around the perimeter.
    pub p: [Vec2; 4],
}

impl Quad {
    /// A quad from its four corners, around the perimeter.
    #[must_use]
    pub const fn new(p: [Vec2; 4]) -> Self {
        Self { p }
    }

    /// The quad of an axis-aligned rectangle.
    #[must_use]
    pub fn rect(rect: Rect) -> Self {
        let (x, y) = (rect.min_x(), rect.min_y());
        let (right, bottom) = (rect.max_x(), rect.max_y());
        Self::new([
            Vec2::new(x, y),
            Vec2::new(right, y),
            Vec2::new(right, bottom),
            Vec2::new(x, bottom),
        ])
    }

    /// The quad of a destination rect and its turn.
    ///
    /// This is the mapping the draw had before the port: the rect is pulled
    /// back by the origin, so the point the origin names lands on the dest
    /// position, and the rect turns around that point.
    /// A zero angle and a half-size origin therefore draw the plain rect.
    #[must_use]
    pub fn pro(dest: Rect, turn: Turn) -> Self {
        let pulled = dim::rect(
            dest.min_x() - turn.origin.x,
            dest.min_y() - turn.origin.y,
            dest.width(),
            dest.height(),
        );
        Self::turned(pulled, Vec2::new(dest.min_x(), dest.min_y()), turn.angle)
    }

    /// The quad of a rectangle turned `degrees` clockwise around `pivot`.
    ///
    /// Raylib turned the quad around its destination position and pulled the
    /// rect back by half its size, which is a turn around the rect center.
    /// This call names the pivot, so the image turn and the Shift+R turn
    /// around the pointer are the same helper.
    #[must_use]
    pub fn turned(rect: Rect, pivot: Vec2, degrees: f32) -> Self {
        let axis = Self::rect(rect);
        if degrees == 0.0 {
            return axis;
        }
        Self::new(axis.p.map(|corner| pivot + turned(corner - pivot, degrees)))
    }

    /// The quad of a triangle, with the last corner repeated.
    #[must_use]
    pub const fn triangle(a: Vec2, b: Vec2, c: Vec2) -> Self {
        Self::new([a, b, c, a])
    }

    /// The quad of a straight line of `thickness`, from `a` to `b`.
    ///
    /// A zero-length line answers an empty quad at `a`, so the draw skips it
    /// instead of dividing by zero.
    #[must_use]
    pub fn line(a: Vec2, b: Vec2, thickness: f32) -> Self {
        let along = b - a;
        let length = along.length();
        if length == 0.0 {
            return Self::new([a, a, a, a]);
        }
        let normal = Vec2::new(-along.y, along.x) * (thickness / 2.0 / length);
        Self::new([a + normal, b + normal, b - normal, a - normal])
    }
}

/// One channel of a tint multiply, in integer arithmetic.
#[allow(clippy::cast_possible_truncation)]
const fn mul_channel(channel: u8, tint: u8) -> u8 {
    ((channel as u16 * tint as u16) / 255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ops follow the arithmetic a caller writes.
    #[test]
    fn the_vector_ops_agree_with_the_parts() {
        let a = Vec2::new(2.0, 3.0);
        let b = Vec2::new(-1.0, 5.0);
        assert_eq!(a + b, Vec2::new(1.0, 8.0));
        assert_eq!(a - b, Vec2::new(3.0, -2.0));
        assert_eq!(a * 2.0, Vec2::new(4.0, 6.0));
        assert_eq!(a / 2.0, Vec2::new(1.0, 1.5));
        assert_eq!(a.dot(b), 13.0);
        assert_eq!(a.distance(a), 0.0);
        assert_eq!(a.lerp(b, 0.5), Vec2::new(0.5, 4.0));
    }

    /// A dest rect and its origin land where the draw puts them.
    #[test]
    fn a_turned_dest_keeps_its_origin_on_the_position() {
        let dest = dim::rect(100.0, 50.0, 40.0, 20.0);
        let plain = Quad::pro(dest, Turn::new(Vec2::new(20.0, 10.0), 0.0));
        assert_eq!(plain.p[0], Vec2::new(80.0, 40.0));
        assert_eq!(plain.p[2], Vec2::new(120.0, 60.0));
        // A quarter turn keeps the origin at the dest position and moves the
        // corners a quarter around it.
        let turned = Quad::pro(dest, Turn::new(Vec2::new(20.0, 10.0), 90.0));
        assert!(turned.p[0].abs_diff_eq(Vec2::new(110.0, 30.0), 1e-5));
        assert!(turned.p[2].abs_diff_eq(Vec2::new(90.0, 70.0), 1e-5));
    }

    /// A quarter turn maps the axes onto each other in a y-down space.
    #[test]
    fn a_quarter_turn_moves_the_axes() {
        let x = Vec2::new(1.0, 0.0);
        let quarter = turned(x, 90.0);
        assert!(quarter.abs_diff_eq(Vec2::new(0.0, 1.0), 1e-6));
        assert!(turned(x, 360.0).abs_diff_eq(x, 1e-5));
    }

    /// A line is a quad of `thickness` around its segment, and a point is no
    /// quad at all.
    #[test]
    fn a_line_spans_its_thickness() {
        let line = Quad::line(Vec2::new(10.0, 20.0), Vec2::new(30.0, 20.0), 4.0);
        // A horizontal line grows up and down by half its thickness, so the
        // corners sit on 18 and 22, in perimeter order from the start point.
        assert!(line.p[0].abs_diff_eq(Vec2::new(10.0, 22.0), 1e-6));
        assert!(line.p[1].abs_diff_eq(Vec2::new(30.0, 22.0), 1e-6));
        assert!(line.p[2].abs_diff_eq(Vec2::new(30.0, 18.0), 1e-6));
        assert!(line.p[3].abs_diff_eq(Vec2::new(10.0, 18.0), 1e-6));
        // A zero-length line answers an empty quad at its point, so the draw
        // never divides by zero.
        let point = Quad::line(Vec2::new(5.0, 5.0), Vec2::new(5.0, 5.0), 4.0);
        assert!(
            point
                .p
                .iter()
                .all(|p| p.abs_diff_eq(Vec2::new(5.0, 5.0), 1e-6))
        );
    }

    /// A tint scales every channel in integer arithmetic.
    #[test]
    fn a_tint_scales_each_channel() {
        let half = Color::new(128, 128, 128, 255);
        assert_eq!(Color::WHITE.tint(half), Color::new(128, 128, 128, 255));
        assert_eq!(
            Color::new(10, 20, 30, 255).tint(half),
            Color::new(5, 10, 15, 255)
        );
        assert_eq!(Color::WHITE.tint(Color::new(0, 0, 0, 255)), Color::BLACK);
    }

    /// The clear color keeps the bytes in 0..1.
    #[test]
    fn the_clear_color_is_the_byte_over_two_fifty_five() {
        let clear = Color::new(0, 51, 255, 255).to_unit();
        assert!((clear[1] - 0.2).abs() < 1e-12);
        assert_eq!(clear[0], 0.0);
        assert_eq!(clear[3], 1.0);
    }
}
