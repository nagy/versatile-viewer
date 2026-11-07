//! Gimmick: a blurred copy of the viewed image as the image-view background.
//!
//! The gimmick is off unless `VV_BLUR_BG=1` is set.
//! The background is a tiny blurred copy of the image.
//! Its long side is 128 pixels by default.
//! The code uploads this copy as a texture.
//! The GPU upscales the texture with bilinear filtering.
//! The upscale itself is the blur.
//! The CPU cost is one downscale and blur pass on a worker thread.
//! The background is drawn scaled to cover the whole window.
//! It fits on the narrower side and the code crops the overflow.
//! It sits behind the sharp image and is dimmed, so the image stands out.
//! `VV_BG_DIM` sets the background brightness as a 0..=1 multiplier.
//! The default is 0.6.

use std::time::Instant;

use crate::{
    DecodedImage,
    dim::{LongSide, Rect, WinH, WinW},
    downscale_rgba_ref,
    draw::{Color, Vec2},
    render::Renderer,
    texture::{Texture, Uploader},
};

/// Long side of the blurred background texture.
/// The GPU upscales from this size.
/// `VV_BLUR_PX` configures it.
/// Fewer pixels give a blurrier result.
const DEFAULT_LONG_SIDE: LongSide = LongSide::new(128);
/// Gaussian sigma applied to the tiny image, in its own pixels.
const BLUR_SIGMA: f32 = 8.0;
/// Background brightness when `VV_BG_DIM` is unset.
const DEFAULT_DIM: f32 = 0.6;

/// Crossfade duration for grid-view background changes, in seconds.
const FADE_SECS: f64 = 0.4;

/// The `VV_BLUR_BG` gimmick settings, parsed once at the process boundary.
///
/// `None` means the gimmick is off, so a caller carries one `Option` instead of
/// a flag beside a size that means nothing while the flag is false.
/// The decode workers read the long side, and the GPU side reads the
/// brightness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// The long side of the tiny blurred copy, in pixels.
    px: LongSide,
    /// The brightness the background keeps, as a 0..=1 multiplier.
    dim: f32,
}

impl Config {
    /// Read the settings from the environment, or `None` when the gimmick is
    /// off.
    ///
    /// The environment is read here and nowhere else, so a worker never
    /// re-parses it.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        // A strict opt-in: exactly `VV_BLUR_BG=1`.
        std::env::var_os("VV_BLUR_BG")
            .is_some_and(|v| v == "1")
            .then(|| Self {
                px: parse_px(std::env::var("VV_BLUR_PX").ok().as_deref()),
                dim: parse_dim(std::env::var("VV_BG_DIM").ok().as_deref()),
            })
    }

    /// The long side of the tiny blurred copy, in pixels.
    #[must_use]
    pub const fn px(self) -> LongSide {
        self.px
    }

    /// The brightness the background keeps, as a 0..=1 multiplier.
    #[must_use]
    pub const fn dim(self) -> f32 {
        self.dim
    }

    /// The tiny blurred copy of `image`, at the configured long side.
    #[must_use]
    pub fn blur(self, image: &DecodedImage) -> DecodedImage {
        small_blur(image, self.px)
    }
}

/// Long side of the background texture, from `VV_BLUR_PX`.
/// The value is clamped to 8..=1024.
/// The default 128 is already far below any window size.
fn parse_px(raw: Option<&str>) -> LongSide {
    LongSide::new(
        raw.and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(DEFAULT_LONG_SIDE.raw())
            .clamp(8, 1024),
    )
}

/// Background brightness from `VV_BG_DIM`.
/// The value is a 0..=1 multiplier and is clamped.
fn parse_dim(s: Option<&str>) -> f32 {
    s.and_then(|s| s.trim().parse::<f32>().ok())
        .unwrap_or(DEFAULT_DIM)
        .clamp(0.05, 1.0)
}

/// Compute the tiny blurred copy of an RGBA8 image.
///
/// `long_side` is the long side of the texture (`VV_BLUR_PX`, default 128).
/// The code downscales with the SIMD resampler, then blurs on the tiny image.
/// The result holds a few KB.
///
/// # Panics
///
/// The function panics if `rgba` does not hold exactly `width * height * 4`
/// bytes.
///
/// # Examples
///
/// ```
/// # use versatile_viewer::{DecodedImage, blurbg::small_blur};
/// # use versatile_viewer::dim::{ImgH, ImgW, LongSide};
/// // Landscape: the long side lands exactly on `long_side`.
/// // The aspect stays and each pixel holds 4 bytes.
/// let img = DecodedImage::new(ImgW::new(200), ImgH::new(100), vec![0; 4 * 200 * 100]).unwrap();
/// let img = small_blur(&img, LongSide::new(128));
/// assert_eq!((img.width().raw(), img.height().raw()), (128, 64));
/// assert_eq!(
///     img.as_rgba().len(),
///     4 * img.width().raw() as usize * img.height().raw() as usize
/// );
///
/// // Portrait: the long side switches to the height.
/// let img = DecodedImage::new(ImgW::new(100), ImgH::new(200), vec![0; 4 * 100 * 200]).unwrap();
/// let img = small_blur(&img, LongSide::new(128));
/// assert_eq!((img.width().raw(), img.height().raw()), (64, 128));
/// assert_eq!(
///     img.as_rgba().len(),
///     4 * img.width().raw() as usize * img.height().raw() as usize
/// );
/// ```
#[must_use]
pub fn small_blur(image: &DecodedImage, long_side: LongSide) -> DecodedImage {
    // The SIMD resample reads the borrowed buffer, so the code never clones
    // the input.
    let small = downscale_rgba_ref(image.as_rgba(), image.width(), image.height(), long_side);
    let (width, height) = (small.width(), small.height());
    let pixels: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(width.raw(), height.raw(), small.into_rgba())
            .expect("resampled buffer matches dims");
    let blurred = image::imageops::blur(&pixels, BLUR_SIGMA);
    DecodedImage::from_exact(width, height, blurred.into_raw())
}

/// GPU side of the gimmick.
/// It holds the blurred background texture for the image view.
#[derive(Debug)]
pub struct BlurBg {
    dim: f32,
    /// The current background texture.
    tex: Option<Texture>,
    /// Source tag of the current texture (grid entry id).
    /// None means unknown, for example a single-file launch.
    /// The code uses the tag to skip redundant transitions.
    current: Option<u64>,
    /// The previous texture, kept while a crossfade to `tex` runs.
    old: Option<Texture>,
    /// The start time of the running crossfade.
    fade_start: Option<Instant>,
}

impl BlurBg {
    /// None when the gimmick is off.
    #[must_use]
    pub fn from_env() -> Option<BlurBg> {
        let config = Config::from_env()?;
        Some(BlurBg {
            dim: config.dim(),
            tex: None,
            current: None,
            old: None,
            fade_start: None,
        })
    }

    /// Upload a tiny blurred copy as the background texture.
    ///
    /// The copy measures `VV_BLUR_PX` on its long side (128 by default), so a
    /// fresh upload costs a few kilobytes and the previous texture simply
    /// drops.
    fn upload(&mut self, uploader: &Uploader, data: &DecodedImage) {
        match uploader.image(data) {
            // The linear filter is the default of an upload, and the GPU
            // upscale of this tiny texture is the blur.
            Ok(texture) => self.tex = Some(texture),
            Err(err) => {
                eprintln!("vv: blur background: {err}");
                self.tex = None;
            }
        }
    }

    /// Replace the background texture from a tiny blurred copy, or update
    /// it in place.
    /// This path has no fade (same image: streaming previews).
    /// The code cuts a running crossfade short.
    /// Errors are non-fatal: the background just stays transparent, and the
    /// window background shows through.
    pub fn attach(&mut self, uploader: &Uploader, data: &DecodedImage, tag: Option<u64>) {
        self.upload(uploader, data);
        if tag.is_some() {
            self.current = tag;
        }
    }

    /// Swap the background to `data` with a slow crossfade.
    /// The function does nothing while `tag` already matches.
    /// The grid calls this function every frame for the selected entry.
    /// Without an existing background, the new texture fades in from the
    /// window background.
    pub fn transition(&mut self, uploader: &Uploader, data: &DecodedImage, tag: u64) {
        if self.current == Some(tag) {
            return;
        }
        // First background ever (empty window at startup): show it
        // at once, with no fade-in from the window background.
        let first = self.tex.is_none();
        match uploader.image(data) {
            Ok(t) => {
                if first {
                    self.old = None;
                    self.fade_start = None;
                } else {
                    // The current texture becomes the fade-out layer.
                    // The code drops the older fade-out layer (max two
                    // alive).
                    self.old = self.tex.take();
                    self.fade_start = Some(Instant::now());
                }
                self.tex = Some(t);
                self.current = Some(tag);
            }
            Err(err) => eprintln!("vv: blur background: {err}"),
        }
    }

    /// Draw the background (if any) scaled to cover the window.
    /// During a crossfade the old texture shows at full opacity underneath.
    /// The new texture fades in on top.
    /// Call this function every frame so the fades complete and the code
    /// frees the old texture.
    pub fn draw(&mut self, r: &mut Renderer, win_w: WinW, win_h: WinH) {
        let progress = self
            .fade_start
            .map_or(1.0, |s| (s.elapsed().as_secs_f64() / FADE_SECS).min(1.0));
        if progress >= 1.0 {
            self.old = None;
            self.fade_start = None;
        }
        let Some(tex) = &self.tex else {
            return;
        };
        let rgb = (self.dim * 255.0) as u8;
        // Old texture first, at full opacity...
        if let Some(old) = &self.old {
            r.texture_pro(
                old,
                Rect::sized(old.width() as f32, old.height() as f32),
                cover_rect(old.width(), old.height(), win_w, win_h),
                Vec2::ZERO,
                0.0,
                Color::new(rgb, rgb, rgb, 255),
            );
        }
        // ...then the new one, fading in (from the window background when
        // there is no old).
        let t = progress * progress * (3.0 - 2.0 * progress); // smoothstep
        r.texture_pro(
            tex,
            Rect::sized(tex.width() as f32, tex.height() as f32),
            cover_rect(tex.width(), tex.height(), win_w, win_h),
            Vec2::ZERO,
            0.0,
            Color::new(rgb, rgb, rgb, (t * 255.0) as u8),
        );
    }
}

/// Dest rect for a texture of w x h scaled to cover the window.
/// The texture fits on the narrower side and stays centered.
/// The other axis overflows and the code crops it.
const fn cover_rect(w: u32, h: u32, win_w: WinW, win_h: WinH) -> Rect {
    let (fw, fh) = (w as f32, h as f32);
    let (win_w, win_h) = (win_w.raw(), win_h.raw());
    let s = (win_w / fw).max(win_h / fh);
    Rect::new(
        (win_w - fw * s) / 2.0,
        (win_h - fh * s) / 2.0,
        fw * s,
        fh * s,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::{ImgH, ImgW};

    // Compile-time check that these stay const-evaluable.
    const _: f32 = cover_rect(64, 32, WinW::new(800.0), WinH::new(600.0)).width;

    #[test]
    fn small_blur_dims_and_size() {
        let img = |w: u32, h: u32, v: u8| {
            DecodedImage::from_exact(ImgW::new(w), ImgH::new(h), vec![v; (w * h * 4) as usize])
        };
        // Long side 64 (VV_BLUR_PX): 800x200 in -> 64x16 out, RGBA8 matches.
        let out = small_blur(&img(800, 200, 128), LongSide::new(64));
        assert_eq!((out.width(), out.height()), (ImgW::new(64), ImgH::new(16)));
        assert_eq!(out.as_rgba().len(), (64 * 16 * 4) as usize);
        // Portrait input keeps the aspect the other way around.
        let out = small_blur(&img(100, 400, 0), LongSide::new(64));
        assert_eq!((out.width(), out.height()), (ImgW::new(16), ImgH::new(64)));
        // The code honors the configurable long side (VV_BLUR_PX).
        let out = small_blur(&img(800, 200, 0), LongSide::new(256));
        assert_eq!((out.width(), out.height()), (ImgW::new(256), ImgH::new(64)));
        // Degenerate 1xN input still yields at least 1px on each side.
        let out = small_blur(&img(1, 9, 0), LongSide::new(64));
        assert!(out.width().raw() >= 1 && out.height().raw() >= 1);
        assert_eq!(
            out.as_rgba().len(),
            (out.width().raw() * out.height().raw() * 4) as usize
        );
    }

    #[test]
    fn cover_rect_fits_narrower_side() {
        // 2:1 texture in an 800x600 window: height limits -> 1200x600,
        // centered (x overflows equally to both sides).
        let r = cover_rect(64, 32, WinW::new(800.0), WinH::new(600.0));
        assert_eq!((r.width, r.height), (1200.0, 600.0));
        assert_eq!((r.x, r.y), (-200.0, 0.0));
        // Tall texture: width limits instead.
        let r = cover_rect(32, 64, WinW::new(800.0), WinH::new(600.0));
        assert_eq!((r.width, r.height), (800.0, 1600.0));
        assert_eq!((r.x, r.y), (0.0, -500.0));
    }

    #[test]
    fn blur_px_defaults_and_clamps() {
        // No var set: the 128px default.
        assert_eq!(parse_px(None), LongSide::new(128));
        assert_eq!(parse_px(Some("128")), LongSide::new(128));
        assert_eq!(parse_px(Some(" 32 ")), LongSide::new(32));
        assert_eq!(parse_px(Some("4")), LongSide::new(8)); // clamped up
        assert_eq!(parse_px(Some("99999")), LongSide::new(1024)); // clamped down
        assert_eq!(parse_px(Some("junk")), LongSide::new(128));
    }

    #[test]
    fn parse_dim_defaults_and_clamps() {
        assert!((parse_dim(None) - 0.6).abs() < 1e-6);
        assert!((parse_dim(Some("0.3")) - 0.3).abs() < 1e-6);
        assert!((parse_dim(Some(" 0.75 ")) - 0.75).abs() < 1e-6);
        assert!((parse_dim(Some("2.0")) - 1.0).abs() < 1e-6);
        assert!((parse_dim(Some("0")) - 0.05).abs() < 1e-6);
        assert!((parse_dim(Some("junk")) - 0.6).abs() < 1e-6);
    }
}
