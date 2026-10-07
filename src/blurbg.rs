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

use raylib::{color::Color, consts::TextureFilter, prelude::*, texture::RaylibTexture2D};

use crate::DecodedImage;

/// Long side of the blurred background texture.
/// The GPU upscales from this size.
/// `VV_BLUR_PX` configures it.
/// Fewer pixels give a blurrier result.
const DEFAULT_LONG_SIDE: u32 = 128;
/// Gaussian sigma applied to the tiny image, in its own pixels.
const BLUR_SIGMA: f32 = 8.0;
/// Background brightness when `VV_BG_DIM` is unset.
const DEFAULT_DIM: f32 = 0.6;

/// Crossfade duration for grid-view background changes, in seconds.
const FADE_SECS: f64 = 0.4;

/// Return true when the gimmick is enabled.
/// This is a strict opt-in: exactly `VV_BLUR_BG=1`.
#[must_use]
pub fn enabled() -> bool {
    std::env::var_os("VV_BLUR_BG").is_some_and(|v| v == "1")
}

/// Long side of the background texture, from `VV_BLUR_PX`.
/// The value is clamped to 8..=1024.
/// The default 128 is already far below any window size.
#[must_use]
pub fn blur_px() -> u32 {
    std::env::var("VV_BLUR_PX")
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(DEFAULT_LONG_SIDE)
        .clamp(8, 1024)
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
/// The function is cheap enough for a decode worker.
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
/// // Landscape: the long side lands exactly on `long_side`.
/// // The aspect stays and each pixel holds 4 bytes.
/// let img = DecodedImage {
///     width: 200,
///     height: 100,
///     data: vec![0; 4 * 200 * 100],
/// };
/// let img = small_blur(&img, 128);
/// assert_eq!((img.width, img.height), (128, 64));
/// assert_eq!(img.data.len(), 4 * img.width as usize * img.height as usize);
///
/// // Portrait: the long side switches to the height.
/// let img = DecodedImage {
///     width: 100,
///     height: 200,
///     data: vec![0; 4 * 100 * 200],
/// };
/// let img = small_blur(&img, 128);
/// assert_eq!((img.width, img.height), (64, 128));
/// assert_eq!(img.data.len(), 4 * img.width as usize * img.height as usize);
/// ```
#[must_use]
pub fn small_blur(image: &DecodedImage, long_side: u32) -> DecodedImage {
    let (width, height) = (image.width, image.height);
    let (tw, th) = if width >= height {
        (
            long_side,
            ((long_side as f32 * height as f32 / width as f32).round() as u32).max(1),
        )
    } else {
        (
            ((long_side as f32 * width as f32 / height as f32).round() as u32).max(1),
            long_side,
        )
    };
    let img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(width, height, image.data.clone())
            .expect("rgba buffer matches dimensions");
    let small = image::imageops::resize(&img, tw, th, image::imageops::FilterType::Triangle);
    let blurred = image::imageops::blur(&small, BLUR_SIGMA);
    DecodedImage {
        width: tw,
        height: th,
        data: blurred.into_raw(),
    }
}

/// GPU side of the gimmick.
/// It holds the blurred background texture for the image view.
/// It lives between `rl` and its users in main.
/// It then drops and unloads before the window closes.
pub struct BlurBg {
    dim: f32,
    /// The current background texture.
    tex: Option<Texture2D>,
    /// Source tag of the current texture (grid entry id).
    /// None means unknown, for example a single-file launch.
    /// The code uses the tag to skip redundant transitions.
    current: Option<u64>,
    /// The previous texture, kept while a crossfade to `tex` runs.
    old: Option<Texture2D>,
    /// The start time of the running crossfade.
    fade_start: Option<Instant>,
}

impl BlurBg {
    /// None when the gimmick is off.
    #[must_use]
    pub fn from_env() -> Option<BlurBg> {
        if !enabled() {
            return None;
        }
        Some(BlurBg {
            dim: parse_dim(std::env::var("VV_BG_DIM").ok().as_deref()),
            tex: None,
            current: None,
            old: None,
            fade_start: None,
        })
    }

    /// Upload a tiny blurred copy as the background texture.
    /// The function updates the texture in place when the dimensions match
    /// (streaming previews).
    fn upload(&mut self, rl: &mut RaylibHandle, thread: &RaylibThread, data: DecodedImage) {
        let DecodedImage {
            width,
            height,
            data: rgba,
        } = data;
        let same = matches!(&self.tex, Some(t) if t.width() == width as i32 && t.height() == height as i32);
        if same {
            if let Some(t) = &mut self.tex
                && t.update_texture(&rgba).is_err()
            {
                self.tex = None;
            }
            return;
        }
        match crate::upload_rgba(rl, thread, &rgba, width, height) {
            Ok(t) => {
                // Bilinear filtering turns the tiny copy into a smooth blur
                // when the GPU upscales it every frame.
                t.set_texture_filter(thread, TextureFilter::TEXTURE_FILTER_BILINEAR);
                self.tex = Some(t); // drops (unloads) any previous texture
            }
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
    /// Errors are non-fatal: the background just stays black.
    pub fn attach(
        &mut self,
        rl: &mut RaylibHandle,
        thread: &RaylibThread,
        data: DecodedImage,
        tag: Option<u64>,
    ) {
        self.upload(rl, thread, data);
        if tag.is_some() {
            self.current = tag;
        }
    }

    /// Swap the background to `data` with a slow crossfade.
    /// The function does nothing while `tag` already matches.
    /// The grid calls this function every frame for the selected entry.
    /// Without an existing background, the new texture fades in from black.
    pub fn transition(
        &mut self,
        rl: &mut RaylibHandle,
        thread: &RaylibThread,
        data: &DecodedImage,
        tag: u64,
    ) {
        if self.current == Some(tag) {
            return;
        }
        let (rgba, width, height) = (&data.data, data.width, data.height);
        // First background ever (empty window at startup): show it
        // at once, with no fade-in from black.
        let first = self.tex.is_none();
        match crate::upload_rgba(rl, thread, rgba, width, height) {
            Ok(t) => {
                t.set_texture_filter(thread, TextureFilter::TEXTURE_FILTER_BILINEAR);
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
    pub fn draw(&mut self, d: &mut RaylibDrawHandle, win_w: f32, win_h: f32) {
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
            d.draw_texture_pro(
                old,
                Rectangle {
                    x: 0.0,
                    y: 0.0,
                    width: old.width() as f32,
                    height: old.height() as f32,
                },
                cover_rect(old.width() as u32, old.height() as u32, win_w, win_h),
                Vector2::ZERO,
                0.0,
                Color::new(rgb, rgb, rgb, 255),
            );
        }
        // ...then the new one, fading in (from black when there is no old).
        let t = progress * progress * (3.0 - 2.0 * progress); // smoothstep
        d.draw_texture_pro(
            tex,
            Rectangle {
                x: 0.0,
                y: 0.0,
                width: tex.width() as f32,
                height: tex.height() as f32,
            },
            cover_rect(tex.width() as u32, tex.height() as u32, win_w, win_h),
            Vector2::ZERO,
            0.0,
            Color::new(rgb, rgb, rgb, (t * 255.0) as u8),
        );
    }
}

/// Dest rect for a texture of w x h scaled to cover the window.
/// The texture fits on the narrower side and stays centered.
/// The other axis overflows and the code crops it.
const fn cover_rect(w: u32, h: u32, win_w: f32, win_h: f32) -> Rectangle {
    let (fw, fh) = (w as f32, h as f32);
    let s = (win_w / fw).max(win_h / fh);
    Rectangle {
        x: (win_w - fw * s) / 2.0,
        y: (win_h - fh * s) / 2.0,
        width: fw * s,
        height: fh * s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time check that these stay const-evaluable.
    const _: f32 = cover_rect(64, 32, 800.0, 600.0).width;

    #[test]
    fn small_blur_dims_and_size() {
        let img = |w: u32, h: u32, v: u8| DecodedImage {
            width: w,
            height: h,
            data: vec![v; (w * h * 4) as usize],
        };
        // Long side 64 (VV_BLUR_PX): 800x200 in -> 64x16 out, RGBA8 matches.
        let out = small_blur(&img(800, 200, 128), 64);
        assert_eq!((out.width, out.height), (64, 16));
        assert_eq!(out.data.len(), (64 * 16 * 4) as usize);
        // Portrait input keeps the aspect the other way around.
        let out = small_blur(&img(100, 400, 0), 64);
        assert_eq!((out.width, out.height), (16, 64));
        // The code honors the configurable long side (VV_BLUR_PX).
        let out = small_blur(&img(800, 200, 0), 256);
        assert_eq!((out.width, out.height), (256, 64));
        // Degenerate 1xN input still yields at least 1px on each side.
        let out = small_blur(&img(1, 9, 0), 64);
        assert!(out.width >= 1 && out.height >= 1);
        assert_eq!(out.data.len(), (out.width * out.height * 4) as usize);
    }

    #[test]
    fn cover_rect_fits_narrower_side() {
        // 2:1 texture in an 800x600 window: height limits -> 1200x600,
        // centered (x overflows equally to both sides).
        let r = cover_rect(64, 32, 800.0, 600.0);
        assert_eq!((r.width, r.height), (1200.0, 600.0));
        assert_eq!((r.x, r.y), (-200.0, 0.0));
        // Tall texture: width limits instead.
        let r = cover_rect(32, 64, 800.0, 600.0);
        assert_eq!((r.width, r.height), (800.0, 1600.0));
        assert_eq!((r.x, r.y), (0.0, -500.0));
    }

    #[test]
    fn blur_px_defaults_and_clamps() {
        // No var set: the 128px default (parse of None-like input).
        let p = |s: Option<&str>| -> u32 {
            s.and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(DEFAULT_LONG_SIDE)
                .clamp(8, 1024)
        };
        assert_eq!(p(None), 128);
        assert_eq!(p(Some("128")), 128);
        assert_eq!(p(Some(" 32 ")), 32);
        assert_eq!(p(Some("4")), 8); // clamped up
        assert_eq!(p(Some("99999")), 1024); // clamped down
        assert_eq!(p(Some("junk")), 128);
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
