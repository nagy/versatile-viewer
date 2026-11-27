//! The display-list pass: `vello` paints a [`Primitive`] list into a
//! window-sized target, and the quad pipeline composites the result.
//!
//! # Why a second renderer
//!
//! The viewer draws everything through one instanced-quad pipeline, and a
//! primitive is not a quad: a path needs analytic anti-aliasing, a gradient
//! needs a ramp, a clip needs a stack and a glyph needs an outline. `vello`
//! answers all four on the device and queue this crate already holds, so no
//! second device and no second `wgpu` exists in the process.
//!
//! # The target
//!
//! `vello` renders with compute work into an `Rgba8Unorm` texture that carries
//! `STORAGE_BINDING`, and [`SceneView::draw`] answers the same texture in the
//! form a quad draw samples.
//! The target is the size of the window and the transform maps surface units
//! onto window pixels, so a zoom or a pan re-renders *cheap* primitives rather
//! than re-rasterizing a patch: the list itself never changes.
//!
//! # Encoding once
//!
//! A list carries no scale and no region, so the pass encodes it in surface
//! units and places the whole scene per frame with one [`Vello::append`], which
//! multiplies the transform of every draw. A zoom or a pan therefore costs a copy
//! of the encoded stream (about a fiftieth of the encode) and never another walk
//! of the paths. `vello` reads the encoded curves and flattens them on the
//! device during the render, so the placed list is exactly the picture a list
//! encoded for that zoom would have drawn — not a magnified polyline.
//!
//! # The band
//!
//! The encode covers the window and no more (see [`band_of`]): a plane taller
//! than the window, and a book is one plane of every page, carries work the
//! frame never shows, and `vello` pays for all of it anyway — it flattens every
//! encoded curve and its bump allocators are fixed sizes picked for a bench
//! scene, so a book zoomed a few times overflows them and paints *nothing*, with
//! no error to report.
//! A frame therefore encodes the primitives whose boxes meet the window's own
//! box in surface units, padded by [`BAND_MARGIN`] so a pan inside the band costs
//! nothing, and a frame whose window has left the band re-encodes.
//! A clip, a layer and a marker always stay: the stack must keep its shape, and
//! a clip carries no ink of its own.
//!
//! # The primitive that does not fit
//!
//! A band bounds one document's worth of work, not one primitive's: a path holds
//! whatever the producer wrote, and `vello` flattens it whole at the frame's
//! scale, culling only a draw whose box is entirely outside the target. A
//! page-spanning path of twenty thousand curves therefore starves the same
//! allocators at a deep zoom, and by the same silent path (see above).
//! The encoder answers that one shape: a path that alone asks for more than
//! [`PRIMITIVE_BUDGET`] is flattened to the band and clipped to it, so a frame
//! that shows a corner of the page asks `vello` for the corner. The clip is
//! exact where it matters — a curve whose hull misses the band becomes the chord
//! between its own ends, and only the curves that reach the band are flattened —
//! and it is the one place where a curve stops being flattened by the device at
//! the frame's own zoom, so a band remembers the scale it was flattened for and a
//! frame past [`LADDER`] times it re-encodes (see [`CLIP_PX`]).
//!
//! # The frame that does not fit
//!
//! [`Work`] is the count of lines that frame would flatten, in the tolerance law
//! `vello` itself sizes its buffers with, and it over-reports: Wang's formula
//! bounds a curve the flattener approximates by an Euler spiral and then caps at
//! a hundred lines a piece, by 1.2 times at the tightest and 4.5 at a deep zoom
//! (measured against the allocator the shader writes, see
//! `docs/measurements/display-budget.md`). So the budget
//! is a fraction of the real buffer that the smallest of those over-counts still
//! holds, and a frame that will not fit is refused and shown as whatever the
//! caller keeps behind the list: a caller with a raster
//! of the plane draws that instead, which the source rasterizes at the window's
//! own scale and which has no such limit, and a caller with nothing behind the
//! list asks for [`Fallback::LastPaint`] and gets the paint on screen again,
//! placed where the frame's transform puts the same surface units. The second one
//! is stale in sharpness alone, which beats the window background a source-less
//! list would otherwise show.
//!
//! A refusal gives up the band before it answers, when the box this frame shows
//! is worth giving it up for ([`SceneView::tighten`]).
//! A band outlives the frame that made it and still covers every frame the window
//! later zooms into, while [`Work`] counts a primitive whole however little of it
//! the window shows, so a plane the window once held in one frame would keep that
//! frame's bill for every zoom that followed: the refusal would become permanent,
//! and a zoom out and back in would leave the list for the raster for good.
//! Counting the box of the refused frame says whether the renderer can hold the
//! picture that frame actually shows, and a frame it can hold re-encodes for that
//! box and paints. That count takes a primitive whole, which is what the clip of
//! the encode takes back, so a box with a primitive past [`PRIMITIVE_BUDGET`] is
//! re-encoded whatever the count says, and the clip says whether it fits.
//!
//! # Image leaves
//!
//! A [`Primitive::Image`] carries pixels and not a texture, so this pass uploads
//! each leaf once and keeps it in a table keyed by the leaf's address. The list
//! lives in an `Arc<[Primitive]>` that the view holds for as long as the plane
//! does, so an address stays stable, and a different list clears the table.

use std::{collections::HashMap, sync::Arc};

use derive_more::{Display, Error, From};
use vello::{
    AaConfig, AaSupport, Glyph, RenderParams, Renderer as VelloRenderer, RendererOptions,
    Scene as Vello,
    kurbo::{self, Affine, PathEl, Shape},
    peniko::{
        self as peniko, Blob, Color as VelloColor, ColorStop, Extend, Fill, FontData, Gradient,
        GradientKind, ImageAlphaType, ImageBrush, ImageData, ImageFormat, LinearGradientPosition,
        RadialGradientPosition,
    },
};

use crate::{
    DecodedImage,
    dim::{self, Rect},
    draw::{Color, Quad, Vec2},
    region,
    scene::{FillRule, GlyphRun, Paint, Primitive, Spread, Stop},
    texture::{Texture, Uploader},
};

/// How many parsed faces the pass keeps.
///
/// A document names a handful of faces and a treemap names one, so the table
/// stays small; a session that walks a directory of PDFs would otherwise pin
/// the fonts of every document it ever showed.
const FACES: usize = 8;

/// The anti-aliasing a scene renders with.
///
/// Area sampling is the one mode that is always available, and the fine
/// variants cost pipelines at startup that this viewer would not switch to.
const AA: AaConfig = AaConfig::Area;

/// Why the display-list pass did not come up.
#[derive(Debug, Display, Error, From)]
pub enum SceneError {
    /// `vello` refused the device, which happens when a required feature or
    /// limit is missing.
    #[display("vello refused the device: {_0}")]
    Renderer(#[from] vello::Error),
}

/// One display-list pass: the `vello` renderer, its window-sized target and the
/// pictures of the list it draws.
pub struct SceneView {
    renderer: VelloRenderer,
    /// The part of the list being drawn, encoded once in surface units.
    ///
    /// A list carries no scale and no region, so its geometry never changes: the
    /// band encodes once and every frame places it with one [`Vello::append`],
    /// which multiplies the transform of every draw and leaves the paths, the
    /// ramps and the pictures alone. `vello` flattens a curve on the device
    /// during the render, so a placed list is as sharp at a 10x zoom as one
    /// encoded for it.
    base: Vello,
    /// What `base` holds: the list, the plane its layers are clipped to, and the
    /// surface box it kept the primitives for.
    encoded: Option<Base>,
    /// What every primitive of the list asks `vello` to flatten, kept for the
    /// life of the list.
    ///
    /// A band change reads it instead of walking the paths again, and a book
    /// changes its band three times on the way through one spread step (see
    /// [`Counts`]).
    counts: Option<Counts>,
    /// The scene of this frame: `base` placed by the frame's transform, reused
    /// so a frame allocates no scene.
    framed: Vello,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    /// The composite side of `view`, which a quad draw samples.
    texture: Texture,
    /// The size of `target`, in window pixels.
    size: (u32, u32),
    uploader: Uploader,
    /// The picture of each [`Primitive::Image`] leaf, keyed by its address.
    images: HashMap<usize, ImageBrush>,
    /// The parsed face of each [`Primitive::Glyphs`] run, keyed by its bytes.
    ///
    /// Each entry keeps the bytes it parsed, so the address it is keyed on
    /// cannot be handed to another face while the entry lives.
    /// The table also outlives the list that named it: a run carries its own
    /// bytes, and every label of a map names one shared face for the process,
    /// so a relaid map draws its labels without parsing the face again.
    fonts: HashMap<usize, (Arc<[u8]>, FontData)>,
    /// The frame `target` holds: its list, its transform and its size.
    ///
    /// A frame that asks for exactly this one draws the texture on screen
    /// again instead of painting the list, because a paint parses and encodes
    /// every primitive of it and the answer would be the same picture.
    drawn: Option<Drawn>,
    /// The window size the frame before this one asked for.
    ///
    /// A size that moved since then means the window itself is moving, and a
    /// drag of a window changes it on every frame: the paint of each one would
    /// re-encode the whole list, so the frame that finds a moved size reuses
    /// the texture it has (see [`Self::place_stale`]) and the paint lands once
    /// the size holds still.
    asked: Option<(u32, u32)>,
}

/// The list, the plane and the band a [`SceneView::base`] holds.
#[derive(Clone)]
struct Base {
    /// The list it encoded.
    list: Arc<[Primitive]>,
    /// The plane, in surface units, which bounds an opacity layer.
    ///
    /// A layer is a buffer of the shape vello is handed, and the list is
    /// encoded without a window, so the plane is the one shape that bounds it
    /// for every frame the list draws in.
    plane: (f32, f32),
    /// The surface box it kept the primitives for, padded.
    band: kurbo::Rect,
    /// What `vello` has to flatten for the primitives it kept, in surface units.
    ///
    /// It is counted once per band, and a frame only applies its own scale to it
    /// instead of walking the paths again (see [`Work`]).
    work: Work,
    /// The scale a primitive was flattened to the band at, or `None` while the
    /// encode kept every path whole (see [`CLIP_PX`]).
    ///
    /// A flattened primitive is edges once and for all, so a zoom past
    /// [`LADDER`] times this scale shows the facets and the band is encoded
    /// again.
    clip_scale: Option<f64>,
}

impl Base {
    /// Whether this base is the answer for `list` on `plane`.
    ///
    /// The handle and not the contents: a list is immutable and lives for as
    /// long as the plane does, so the same handle is the same geometry.
    fn holds(&self, list: &Arc<[Primitive]>, plane: (f32, f32)) -> bool {
        self.plane == plane && Arc::ptr_eq(&self.list, list)
    }

    /// Whether the band holds every bit of the window `visible` names.
    ///
    /// A frame inside the band draws the encode it has; a frame that left it
    /// needs the primitives it now shows, and only a fresh encode has them.
    fn covers(&self, visible: kurbo::Rect) -> bool {
        self.band.x0 <= visible.x0
            && self.band.y0 <= visible.y0
            && visible.x1 <= self.band.x1
            && visible.y1 <= self.band.y1
    }

    /// Whether the flattening this base holds still reads sharp at `scale`.
    ///
    /// A clipped primitive is a polyline in surface units, so a frame that
    /// magnified it shows the facets. [`CLIP_PX`] of deviation at the scale the
    /// encode was flattened for becomes [`LADDER`] times that at the top of the
    /// ladder, which is the most this pass lets a clipped edge drift.
    fn holds_scale(&self, scale: f64) -> bool {
        self.clip_scale
            .is_none_or(|flattened| scale <= flattened * LADDER)
    }
}

/// What one encode wrote.
#[derive(Clone, Copy, Default)]
struct Encoded {
    /// What the frame that draws this encode asks `vello` to flatten.
    work: Work,
    /// Whether the encode flattened a path to the band (see [`CLIP_PX`]).
    clipped: bool,
}

/// The frame the target holds.
#[derive(Clone)]
struct Drawn {
    /// The list it painted.
    ///
    /// The handle and not its address: a freed list hands its address to the
    /// next allocation, and a reuse that read it as the same list would draw a
    /// picture that is not there.
    list: Arc<[Primitive]>,
    /// The surface-to-window transform it painted with.
    to_window: Affine,
    /// The target size it painted at.
    size: (u32, u32),
}

/// What a frame the pass cannot hold answers instead of a paint.
///
/// A frame past the estimate is one vello paints *nothing* of, and it says so
/// to nobody, so the pass refuses it here. What the refusal shows is the
/// caller's to state, because only the caller knows what it keeps behind the
/// list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    /// Nothing: the frame answers no paint, and a caller that holds a raster of
    /// the plane draws that instead, at the window's own scale.
    Nothing,
    /// The paint the pass already made, placed where this frame's transform
    /// puts it (see [`SceneView::place_stale`]).
    ///
    /// This is the answer for a list with no source behind it: the alternative
    /// is the window background, and a paint that lags in sharpness beats no
    /// picture at all.
    LastPaint,
}

/// What one frame asks the pass for: where the list lands, at what size, and
/// what a frame past the budget answers.
///
/// The three travel together because a pass is asked once per frame and every
/// one of them is a fact about that frame alone: the placement moves with the
/// zoom, the size with the window, and the fallback with what the caller keeps
/// behind the list.
#[derive(Clone, Copy)]
pub struct Request<'a> {
    /// The transform that maps surface units onto the window.
    pub frame: &'a region::Frame,
    /// The target size, in pixels.
    pub size: (u32, u32),
    /// What a frame past the budget answers (see [`Fallback`]).
    pub fallback: Fallback,
}

/// What one display pass answers: the texture to composite, the part of it the
/// picture fills, and where that part lands.
///
/// A painted frame fills its texture and lands on the whole window. A frame
/// that reused the paint of an earlier one answers that paint, scaled, turned
/// and moved by the transform between the two windows, so a window drag shows a
/// picture that sits where the drag put it and only lags in sharpness.
#[derive(Clone, Copy)]
pub struct Painted<'a> {
    /// The rendered list.
    pub texture: &'a Texture,
    /// The part of `texture` the picture fills, in its own pixels.
    pub src: Rect,
    /// Where that part lands, in window pixels.
    pub quad: Quad,
}

impl SceneView {
    /// Build the pass on `device`.
    ///
    /// `vello` compiles its compute pipelines here, which costs a moment, so a
    /// caller builds this the first time a source answers a list and never for
    /// a picture.
    /// The pass is `Send` and the device is a shared handle, so a frame loop
    /// asks a worker for one and the frames before it lands paint their raster
    /// instead, the way `vv-view`'s frame body does.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError`] when vello refuses the device.
    pub fn new(device: &wgpu::Device, uploader: Uploader) -> Result<Self, SceneError> {
        let renderer = VelloRenderer::new(
            device,
            RendererOptions {
                use_cpu: false,
                antialiasing_support: AaSupport::area_only(),
                num_init_threads: None,
                pipeline_cache: None,
            },
        )?;
        let (target, view, texture) = target_of(device, &uploader, (1, 1));
        Ok(Self {
            renderer,
            base: Vello::new(),
            encoded: None,
            counts: None,
            framed: Vello::new(),
            target,
            view,
            texture,
            size: (1, 1),
            uploader,
            images: HashMap::new(),
            fonts: HashMap::new(),
            drawn: None,
            asked: None,
        })
    }

    /// Draw `list` in window pixels and answer the texture to composite.
    ///
    /// `frame` maps surface units onto the window, so the same list draws
    /// sharply at every zoom and under any turn.
    ///
    /// The list encodes once, in surface units, and each frame places the whole
    /// of it with one [`Vello::append`], so a zoom and a pan cost a copy and not
    /// an encode (see [`Self::base`]).
    ///
    /// A frame that asks for the paint of the frame before it reuses that paint,
    /// and a frame whose window moved since the last one reuses it too and
    /// places it where the new window puts the same surface units (see
    /// [`Self::place_stale`]). The paint is the one expensive step of this pass
    /// and none of those frames changes a pixel of it.
    ///
    /// A frame whose geometry would run past vello's fixed allocators answers
    /// what [`Request::fallback`] names: [`Fallback::Nothing`] paints nothing, and the
    /// caller draws the raster it keeps for that plane instead (see the module
    /// docstring), while [`Fallback::LastPaint`] hands back the paint on screen,
    /// placed where this frame puts the same surface units.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError`] when vello refuses the frame, which a missing
    /// device feature causes.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        list: &Arc<[Primitive]>,
        request: Request<'_>,
    ) -> Result<Option<Painted<'_>>, SceneError> {
        let Request {
            frame,
            size,
            fallback,
        } = request;
        let to_window = to_window(frame);
        let moved = self.asked != Some(size);
        self.asked = Some(size);
        if let Some(drawn) = &self.drawn {
            // The same list under the same transform at the same size is the
            // picture the target already holds: the draw composites it again
            // and the pass does nothing.
            if !moved
                && drawn.size == size
                && drawn.to_window == to_window
                && Arc::ptr_eq(&drawn.list, list)
            {
                return Ok(Some(self.painted_whole()));
            }
            // A window that moved since the last frame is still moving, so this
            // frame places the paint it has and leaves the paint of the settled
            // size to the frame that finds the size at rest.
            // A list this pass never painted is painted: a new list is the one
            // thing a window drag cannot answer from the paint it has.
            if moved && drawn.size != size && Arc::ptr_eq(&drawn.list, list) {
                return Ok(Some(self.place_stale(drawn, to_window)));
            }
        }
        let plane = (frame.img.width, frame.img.height);
        let visible = window_hull(to_window, size);
        let band = band_of(visible);
        // What the frame asks `vello` to flatten is bounded by the largest
        // factor of the placing transform, which is its largest singular value:
        // the linear part is a turn composed with the two factors, so its two
        // column norms are those factors and a turn leaves them alone.
        // A budget must over-report and never under-report, so the larger one
        // is the answer even when the window shows more of the coarser axis.
        let [xx, yx, xy, yy, _, _] = to_window.as_coeffs();
        let scale = xx.hypot(yx).max(xy.hypot(yy));
        if !self.encoded.as_ref().is_some_and(|base| {
            base.holds(list, plane) && base.covers(visible) && base.holds_scale(scale)
        }) {
            self.keep(list, plane, band, scale);
        }
        // A frame past the budget is one vello paints *nothing* of, and it says
        // so to nobody: refuse it here, where the caller can still show the
        // plane, and forget the paint on screen so no later frame reuses it.
        if self
            .encoded
            .as_ref()
            .is_some_and(|base| base.work.too_heavy(scale))
        {
            // A band the window outgrew keeps the work of everything it covered,
            // so this frame asks the renderer for far more than it paints and no
            // later frame lowers that bill on its own. Give the band up for the
            // box this frame shows when that box fits, so a refusal is a frame
            // and not the state the view stays in (see [`Self::tighten`]).
            if self.tighten(list, band, scale) {
                self.keep(list, plane, band, scale);
            }
        }
        if self
            .encoded
            .as_ref()
            .is_some_and(|base| base.work.too_heavy(scale))
        {
            // A caller that holds a raster of this plane asks for nothing: its
            // patch is sharp, and a stretched paint under it would only delay
            // it. A list with nothing behind it has no such raster, so the
            // frame that would show the window background shows the last encode
            // instead, placed by the transform between the two windows. That
            // paint is stale in sharpness only, which is the trade this pass
            // takes over a window that goes blank.
            let kept = matches!(
                (fallback, &self.drawn),
                (Fallback::LastPaint, Some(drawn)) if Arc::ptr_eq(&drawn.list, list)
            );
            tracing::debug!(budget = LINE_BUDGET, kept, "display list too heavy");
            if kept && let Some(drawn) = &self.drawn {
                return Ok(Some(self.place_stale(drawn, to_window)));
            }
            self.drawn = None;
            return Ok(None);
        }
        self.resize(device, size);
        self.framed.reset();
        self.framed.append(&self.base, Some(to_window));
        self.renderer.render_to_texture(
            device,
            queue,
            &self.framed,
            &self.view,
            &RenderParams {
                // Transparent, so the letterbox bars and the window background
                // stay the window's own color and the scene only paints where a
                // primitive is.
                base_color: VelloColor::TRANSPARENT,
                width: self.size.0,
                height: self.size.1,
                antialiasing_method: AA,
            },
        )?;
        self.drawn = Some(Drawn {
            list: Arc::clone(list),
            to_window,
            size: self.size,
        });
        Ok(Some(self.painted_whole()))
    }

    /// The paint on screen as the whole window it covers.
    fn painted_whole(&self) -> Painted<'_> {
        let whole = dim::whole(self.size.0 as f32, self.size.1 as f32);
        Painted {
            texture: &self.texture,
            src: whole,
            quad: Quad::rect(whole),
        }
    }

    /// The paint on screen placed where `to_window` puts the frame it painted.
    ///
    /// The paint of `drawn` covers the window of that frame, so the picture
    /// lands where the mapping `to_window ∘ drawn.to_window⁻¹` puts it: a
    /// window drag changes the fit scale and a frame refused past the budget may
    /// have zoomed as well, so that mapping is a scale, a turn and a move, and
    /// the texture fills the rect it names.
    fn place_stale(&self, drawn: &Drawn, to_window: Affine) -> Painted<'_> {
        let (src, quad) = stale_place(drawn.size, drawn.to_window, to_window);
        Painted {
            texture: &self.texture,
            src,
            quad,
        }
    }

    /// Encode `list` for `band` and keep it as the base every frame places.
    ///
    /// A base is the encode of one box, so a frame that outgrew it or shrank
    /// back into it hands its own box here (see [`Self::draw`]).
    fn keep(&mut self, list: &Arc<[Primitive]>, plane: (f32, f32), band: kurbo::Rect, scale: f64) {
        if self
            .encoded
            .as_ref()
            .is_some_and(|base| !Arc::ptr_eq(&base.list, list))
        {
            // Another list is another set of leaves, so the pictures of the
            // last one cannot be trusted and the addresses they were keyed
            // on may now mean something else.
            // The faces stay: a run carries its bytes, so a face of this
            // tree outlives the list that drew it, and vello caches the
            // outlines of a face by its bytes. Dropping them would make the
            // first frame of every relaid map outline every label again.
            self.images.clear();
        }
        self.base.reset();
        let (kept, _) = self.survey(list, band);
        let encoded = self.encode(list, plane, &kept, band, scale);
        self.encoded = Some(Base {
            list: Arc::clone(list),
            plane,
            band,
            work: encoded.work,
            clip_scale: encoded.clipped.then_some(scale),
        });
    }

    /// The primitives of `list` whose ink reaches `band`, in list order, each
    /// with what it asks `vello` to flatten, and the sum of those asks.
    ///
    /// The counts of a list are kept, so only the primitives no band has
    /// reached for are walked (see [`Counts`]). A primitive keeps one count for
    /// every band, because the count is what the whole path asks for: a band
    /// that keeps a `vello` path whole pays all of it (see [`Self::encode`]).
    fn survey(&mut self, list: &Arc<[Primitive]>, band: kurbo::Rect) -> (Vec<(usize, Work)>, Work) {
        let counts = match &mut self.counts {
            Some(counts) if counts.holds(list) => counts,
            room => room.insert(Counts::new(list)),
        };
        counts.survey(band)
    }

    /// Whether the box this frame shows is a box the pass should encode for.
    ///
    /// A base the frame outgrew is still the base of every frame that shrank back
    /// into it, and [`Work`] counts a primitive whole however little of it the
    /// window shows. So the band of a plane the window once held in one frame
    /// keeps that frame's bill for every zoom that follows, and it covers those
    /// frames, so no frame ever encodes again. This is the way out: count what the
    /// box of *this* frame asks `vello` to flatten, and encode for it when the
    /// renderer can hold it.
    ///
    /// The first count of a primitive walks its path, so it is spent only where
    /// it can pay: a document's ink scales with the area a band reaches, so a box
    /// the band shrank `shrunk` times smaller than holds about the base's work
    /// over `shrunk`. A box whose bound would still run past [`LINE_BUDGET`] is
    /// refused at once, which keeps a pan while a frame is refused free of the
    /// walk instead of paying it every frame. The counts that survive are the
    /// ones a refusal did ask for, and the next box of that band reads them (see
    /// [`Counts`]).
    ///
    /// A count is what a primitive asks for whole, and a primitive the encoder
    /// clips asks for the corner the box shows instead (see [`Self::encode`]), so
    /// a box the survey calls too heavy is still worth an encode when a
    /// primitive over [`PRIMITIVE_BUDGET`] has a clip at all: the clip is what
    /// takes that count back. A dashed stroke is the case without one, and a
    /// document of those is refused without the walk.
    fn tighten(&mut self, list: &Arc<[Primitive]>, band: kurbo::Rect, scale: f64) -> bool {
        let Some(base) = self.encoded.as_ref() else {
            return false;
        };
        let (had, wants) = (base.band.area(), band.area());
        // A band this frame already is, and a box of no area or no bound, hold
        // nothing a second walk could bound.
        if base.band == band || !(had.is_finite() && wants > 0.0) {
            return false;
        }
        if base.work.lines_at(scale) / (had / wants) > LINE_BUDGET {
            return false;
        }
        let (kept, work) = self.survey(list, band);
        !work.too_heavy(scale)
            || kept.iter().any(|&(index, own)| {
                own.lines_at(scale) > PRIMITIVE_BUDGET && clippable(&list[index])
            })
    }

    /// Encode the primitives of `list` whose ink reaches `band`, in surface
    /// units.
    ///
    /// Nothing here is placed by a window: the frame that draws the band places
    /// all of it at once (see [`Self::draw`]), so the geometry of a zoom or a pan
    /// inside the band costs no encode.
    /// A primitive the band does not reach is dropped, because `vello` would
    /// pay for it in every frame: it flattens the curves of the whole encode and
    /// its bump allocators are fixed (see the module docstring).
    /// A path that alone asks for more than [`PRIMITIVE_BUDGET`] is the other
    /// half of that rule: it is flattened to the band and clipped to it, so the
    /// frame that shows a corner of it pays for the corner and not for the path
    /// (see [`Self::clipped`]).
    fn encode(
        &mut self,
        list: &[Primitive],
        plane: (f32, f32),
        kept: &[(usize, Work)],
        band: kurbo::Rect,
        scale: f64,
    ) -> Encoded {
        let mut work = Work::default();
        let mut clipped = false;
        for &(index, own) in kept {
            // The frame's own scale is what says whether a path is too heavy to
            // hand over whole: a curve costs more the deeper the window zooms
            // into it (see [`Work::lines_at`]).
            if let Primitive::Path { .. } = &list[index]
                && own.lines_at(scale) > PRIMITIVE_BUDGET
                && let Some(flattened) = self.clipped(&list[index], band, CLIP_PX / scale)
            {
                work += flattened;
                clipped = true;
                continue;
            }
            work += own;
            match &list[index] {
                Primitive::Rect { rect, paint } => {
                    // A rect carries no transform, so both its shape and its
                    // paint live in surface units.
                    self.paint(Fill::NonZero, Affine::IDENTITY, paint, None, rect);
                }
                Primitive::Path {
                    path,
                    transform,
                    fill,
                    fill_rule,
                    stroke,
                } => {
                    // The shape carries its own transform, which is what puts a
                    // leaf of a document in surface units.
                    // A ramp is written in surface units and the shape in its
                    // own, so the brush takes the inverse the shape applied.
                    let paint_at = Some(transform.inverse());
                    if let Some(paint) = fill {
                        self.paint(
                            vello_fill_rule(*fill_rule),
                            *transform,
                            paint,
                            paint_at,
                            path,
                        );
                    }
                    if let Some(outline) = stroke {
                        self.outline(outline, *transform, paint_at, path);
                    }
                }
                Primitive::Glyphs {
                    run,
                    transform,
                    paint,
                } => {
                    self.glyphs(run, transform, paint);
                }
                Primitive::PushClip {
                    path,
                    transform,
                    fill_rule,
                } => {
                    self.base
                        .push_clip_layer(vello_fill_rule(*fill_rule), *transform, path);
                }
                Primitive::PopClip | Primitive::PopLayer => self.base.pop_layer(),
                Primitive::PushLayer { alpha } => {
                    // vello opens a layer only around a clip shape, and the list
                    // is encoded without a window, so the layer is bounded by
                    // the plane: no window shows more of it than that, and the
                    // `append` of the frame carries the clip into the window with
                    // the rest of the list.
                    let whole = kurbo::Rect::new(0.0, 0.0, f64::from(plane.0), f64::from(plane.1));
                    self.base.push_layer(
                        Fill::NonZero,
                        peniko::Mix::Normal,
                        *alpha,
                        Affine::IDENTITY,
                        &whole,
                    );
                }
                Primitive::Image { pixels, transform } => {
                    let key = pixels.as_rgba().as_ptr() as usize;
                    let brush = self
                        .images
                        .entry(key)
                        .or_insert_with(|| image_brush(pixels));
                    self.base.draw_image(brush.as_ref(), *transform);
                }
            }
        }
        Encoded { work, clipped }
    }

    /// Encode one path as the band's own part of it, and answer what the frame
    /// then asks `vello` to flatten.
    ///
    /// `tolerance` is in surface units: the deviation a flattened curve may show
    /// at the scale this encode is made for.
    ///
    /// The path is flattened to edges and clipped to the band, so the ink the
    /// window cannot show is never handed to the renderer: `vello` flattens a
    /// path whole at the frame's scale whatever part of it the window holds (see
    /// the module docstring), and a page-spanning path is what that costs.
    ///
    /// A fill is clipped as the closed shape it is. A stroke is outlined first
    /// and the outline is clipped as a fill, because the outline is what paints
    /// and the centre line is not: `vello` expands a stroke on the device, and
    /// this CPU outline is the one thing that makes a clipped stroke possible.
    ///
    /// `None` answers a path this pass cannot clip and the caller keeps whole: a
    /// dashed stroke, whose dashes are laid along the whole of the path and would
    /// shift when a piece of it is left out, or a placement no box in the path's
    /// own units bounds.
    fn clipped(
        &mut self,
        primitive: &Primitive,
        band: kurbo::Rect,
        tolerance: f64,
    ) -> Option<Work> {
        if !clippable(primitive) {
            return None;
        }
        let Primitive::Path {
            path,
            transform,
            fill,
            fill_rule,
            stroke,
        } = primitive
        else {
            return None;
        };
        let local = tolerance / longest_axis(transform);
        let mut work = Work::default();
        if let Some(paint) = fill {
            let edges = edges(path, local_box(band, transform, 0.0)?, local);
            let shape = clip_polygons(&placed(&edges, *transform), band);
            work += edges_work(&shape);
            if !shape.is_empty() {
                self.paint(
                    vello_fill_rule(*fill_rule),
                    Affine::IDENTITY,
                    paint,
                    Some(Affine::IDENTITY),
                    &shape,
                );
            }
        }
        if let Some(outline) = stroke {
            // A join reaches past the outline by the miter limit, so the box
            // that decides which curves to flatten holds the widest one.
            let reach = outline.style.width * outline.style.miter_limit.clamp(1.0, 16.0);
            let edges = edges(path, local_box(band, transform, reach)?, local);
            let outline_shape = kurbo::stroke(
                edges.iter(),
                &outline.style,
                &kurbo::StrokeOpts::default(),
                local,
            );
            let shape = clip_polygons(&placed(&outline_shape, *transform), band);
            work += edges_work(&shape);
            if !shape.is_empty() {
                self.paint(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    &outline.paint,
                    Some(Affine::IDENTITY),
                    &shape,
                );
            }
        }
        Some(work)
    }

    /// Encode one glyph run, or nothing when its bytes name no face.
    ///
    /// The face is parsed once per run, keyed by its bytes, so a label never
    /// re-parses a font per frame, and the table outlives the list: a face comes
    /// from `sans_serif_face`, one `Arc` for the process, so every label of every
    /// map names the same bytes and vello keeps the outlines it resolved for
    /// them.
    ///
    /// A run carries a producer's bytes, and `draw_glyphs` unwraps the same parse
    /// this check makes, so a face that does not resolve is refused here: a frame
    /// that drops one run still paints, where a panic there paints nothing ever
    /// again. The table takes no entry for a refused run, so a list of bad runs
    /// costs no face parse per frame.
    fn glyphs(&mut self, run: &GlyphRun, transform: &Affine, paint: &Paint) {
        if skrifa::FontRef::from_index(run.font.as_ref(), run.face).is_err() {
            tracing::warn!(
                face = run.face,
                bytes = run.font.len(),
                "a glyph run names no face"
            );
            return;
        }
        let key = run.font.as_ptr() as usize;
        if self.fonts.len() >= FACES && !self.fonts.contains_key(&key) {
            self.fonts.clear();
        }
        let (_, font) = self.fonts.entry(key).or_insert_with(|| {
            (
                Arc::clone(&run.font),
                FontData::new(Blob::from(run.font.to_vec()), run.face),
            )
        });
        let placed = *transform;
        let glyphs = run.glyphs.iter().map(|glyph| Glyph {
            id: glyph.id,
            x: glyph.x,
            y: glyph.y,
        });
        match brush(paint) {
            Brush::Solid(color) => {
                self.base
                    .draw_glyphs(font)
                    .font_size(run.size)
                    .brush(color)
                    .transform(placed)
                    .draw(Fill::NonZero, glyphs);
            }
            Brush::Ramp { gradient, space } => {
                // A glyph run's coordinates are in its own space, so the ramp
                // reaches it the way it reaches a path.
                let at = brush_at(Some(placed.inverse()), space);
                self.base
                    .draw_glyphs(font)
                    .font_size(run.size)
                    .brush(&gradient)
                    .brush_transform(Some(at))
                    .transform(placed)
                    .draw(Fill::NonZero, glyphs);
            }
        }
    }

    /// Take a window size, rebuilding the target when it moved.
    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        let size = (size.0.max(1), size.1.max(1));
        if self.size == size {
            return;
        }
        let (target, view, texture) = target_of(device, &self.uploader, size);
        self.target = target;
        self.view = view;
        self.texture = texture;
        self.size = size;
    }
}

/// The render target, its view and the composite form of that view.
fn target_of(
    device: &wgpu::Device,
    uploader: &Uploader,
    size: (u32, u32),
) -> (wgpu::Texture, wgpu::TextureView, Texture) {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vv scene"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // The one format vello takes, and the format of every other texture in
        // this tree, so the bytes reach the screen unchanged.
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            // vello paints with compute work, so the target is a storage
            // binding and not only a render attachment.
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let texture = uploader.wrap(&view, size);
    (target, view, texture)
}

/// The transform that maps surface units onto window pixels.
///
/// This is the draw transform of [`region::Frame`]: the plane scales on each
/// axis, turns clockwise by the frame's angle about its own center, and that
/// center sits at the window middle plus the pan.
#[must_use]
pub fn to_window(frame: &region::Frame) -> Affine {
    let (w, h) = (f64::from(frame.img.width), f64::from(frame.img.height));
    let scale = (f64::from(frame.scale.x), f64::from(frame.scale.y));
    let center = (
        f64::from(frame.win.width) / 2.0 + f64::from(frame.pan.x),
        f64::from(frame.win.height) / 2.0 + f64::from(frame.pan.y),
    );
    Affine::translate(center)
        * Affine::rotate(f64::from(frame.angle).to_radians())
        * Affine::scale_non_uniform(scale.0, scale.1)
        * Affine::translate((-w / 2.0, -h / 2.0))
}

/// How much of the window each side adds to the encoded band.
///
/// A pan or a nudge that stays inside the padding costs no encode, and a zoom
/// that shrinks the window keeps drawing the band it already has.
/// The margin is a fraction of the window, so a deep zoom pads a small box and
/// a fit view pads a large one: the band tracks what a gesture is likely to
/// show next, and never grows past the work of one and a half windows per side.
const BAND_MARGIN: f64 = 0.5;

/// The surface box the window `to_window` names.
///
/// The window's four corners mapped back through the frame, so a turn and a pan
/// both land in the box as they read: a turned window covers the box its
/// corners span, which is what a primitive must meet to be visible.
/// A frame with no drawable plane answers a box that is not finite, and
/// [`band_of`] then keeps everything.
#[must_use]
fn window_hull(to_window: Affine, size: (u32, u32)) -> kurbo::Rect {
    let inverse = to_window.inverse();
    let (w, h) = (f64::from(size.0), f64::from(size.1));
    let corners =
        [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| inverse * kurbo::Point::new(x, y));
    let mut hull = kurbo::Rect::new(corners[0].x, corners[0].y, corners[0].x, corners[0].y);
    for point in &corners[1..] {
        hull = hull.union(kurbo::Rect::new(point.x, point.y, point.x, point.y));
    }
    hull
}

/// The box of a plane that `frame` shows, padded by [`BAND_MARGIN`] on each
/// side, or `None` for a frame whose window maps onto no box.
///
/// This is the box [`SceneView`] culls its encode to, and the box a source that
/// lists one part of its plane is asked for (see
/// [`vv_source::surface::Surface::display_box`]): one policy for both, so a
/// source never builds a list of ink the frame cannot show.
///
/// The frame names its own window, so the caller needs nothing beside it.
#[must_use]
pub fn band_for(frame: &region::Frame) -> Option<region::Crop> {
    let size = (
        frame.win.width.max(0.0).round() as u32,
        frame.win.height.max(0.0).round() as u32,
    );
    let band = band_of(window_hull(to_window(frame), size));
    let finite =
        band.x0.is_finite() && band.y0.is_finite() && band.x1.is_finite() && band.y1.is_finite();
    finite.then(|| {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a surface box is f32 everywhere else in the tree, and a box that does not \
                      fit an f32 is a frame no source can list"
        )]
        region::crop(
            band.x0 as f32,
            band.y0 as f32,
            band.width() as f32,
            band.height() as f32,
        )
    })
}

/// The band an encode keeps for the window `visible` names: that box, padded by
/// [`BAND_MARGIN`] of its size on each side.
///
/// A box that is not finite (a frame whose scale is zero or NaN) answers a band
/// that holds everything, because the frame is already unusable and dropping
/// primitives would only add a second failure to it.
#[must_use]
fn band_of(visible: kurbo::Rect) -> kurbo::Rect {
    let finite = visible.x0.is_finite()
        && visible.y0.is_finite()
        && visible.x1.is_finite()
        && visible.y1.is_finite();
    if !finite {
        return kurbo::Rect::new(
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::INFINITY,
        );
    }
    visible.inflate(
        visible.width() * BAND_MARGIN,
        visible.height() * BAND_MARGIN,
    )
}

/// Whether a primitive of `bounds` must enter the encode for `band`.
///
/// `bounds` is [`bounds_of`]'s answer, kept beside the count in [`Counts`]
/// because measuring it walks every element of the path.
/// A clip, a layer and a marker carry no ink of their own and the stack must
/// keep the order the producer wrote, so they always stay: dropping a
/// `PushClip` would leave its `PopClip` to close a clip it never opened and
/// move every primitive after it into another stack.
#[must_use]
fn reaches(bounds: Option<kurbo::Rect>, band: kurbo::Rect) -> bool {
    match bounds {
        // The pass cannot state the box, so it keeps the primitive.
        None => true,
        // A shape of no area is one `vello` may still draw (a zero-width stroke
        // is a hairline), so it never culls.
        Some(bounds) if bounds.is_zero_area() => true,
        Some(bounds) => !bounds.intersect(band).is_zero_area(),
    }
}

/// The box a primitive paints inside, in surface units, or `None` when the pass
/// must keep it.
///
/// The box is conservative on purpose: a glyph's own outline is not in the run,
/// so a run's box is the hull of its origins padded by two ems and a stroked
/// path's box reaches half its width past the centre line. A box that is too
/// large only costs the encode of a primitive the frame does not show.
#[must_use]
fn bounds_of(primitive: &Primitive) -> Option<kurbo::Rect> {
    match primitive {
        Primitive::Rect { rect, .. } => Some(*rect),
        Primitive::Path {
            path,
            transform,
            stroke,
            ..
        } => {
            let reach = stroke.as_ref().map_or(0.0, |outline| {
                outline.style.width * longest_axis(transform) / 2.0
            });
            Some(
                transform
                    .transform_rect_bbox(path.bounding_box())
                    .inflate(reach, reach),
            )
        }
        Primitive::Glyphs { run, transform, .. } => {
            let reach = f64::from(run.size) * longest_axis(transform) * 2.0;
            let mut hull: Option<kurbo::Rect> = None;
            for glyph in run.glyphs.iter() {
                let origin = *transform * kurbo::Point::new(f64::from(glyph.x), f64::from(glyph.y));
                let point = kurbo::Rect::new(origin.x, origin.y, origin.x, origin.y);
                // The hull of the origins, and never a box that also holds the
                // origin of the plane: a run that landed somewhere else would
                // keep itself in every band.
                hull = Some(hull.map_or(point, |hull| hull.union(point)));
            }
            Some(hull.unwrap_or(kurbo::Rect::ZERO).inflate(reach, reach))
        }
        Primitive::Image { pixels, transform } => {
            let (w, h) = (
                f64::from(pixels.width().raw()),
                f64::from(pixels.height().raw()),
            );
            Some(transform.transform_rect_bbox(kurbo::Rect::new(0.0, 0.0, w, h)))
        }
        Primitive::PushClip { .. }
        | Primitive::PopClip
        | Primitive::PushLayer { .. }
        | Primitive::PopLayer => None,
    }
}

/// The longest axis a transform maps a unit vector onto.
///
/// A stroke width and an em are stated in the shape's own units, so a box that
/// reaches for them has to ask the transform how far they land.
#[must_use]
fn longest_axis(transform: &Affine) -> f64 {
    let [xx, yx, xy, yy, _, _] = transform.as_coeffs();
    xx.hypot(yx).max(xy.hypot(yy))
}

/// Whether [`SceneView::clipped`] has an answer for `primitive`.
///
/// A fill always does, and so does a stroke whose outline is laid along the path
/// alone. A dashed stroke is the one shape the clip cannot state, and a
/// placement that maps no length at all has no box to flatten to.
#[must_use]
fn clippable(primitive: &Primitive) -> bool {
    let Primitive::Path {
        transform, stroke, ..
    } = primitive
    else {
        return false;
    };
    let stretch = longest_axis(transform);
    stretch.is_finite()
        && stretch > 0.0
        && stroke
            .as_ref()
            .is_none_or(|outline| outline.style.dash_pattern.is_empty())
}

/// The most flattened lines one frame may ask `vello` for.
///
/// `vello_encoding::BufferSizes::new` fixes that buffer at `1 << 21` entries
/// ("hand picked to accommodate the vello test scenes as well as paris-30k"),
/// and a frame that runs past it is abandoned without an error, so this pass
/// refuses one that would (see the module docstring).
/// [`Work`] over-reports what the flattener writes, by a factor that grows with
/// the zoom: a range of curve shapes measured against `BumpAllocators::lines`
/// answers 1.16 at an eighth of the fit scale, 2.2 at one to one and 4.5 at a
/// deep zoom, because Wang's formula bounds a curve that `vello` approximates by
/// an Euler spiral and then caps at a hundred lines a piece. So the smallest
/// over-count leaves an estimate of three quarters of the real buffer still
/// under what the renderer holds, and the buffer itself too near it: this budget
/// keeps a third of the buffer clear in the worst measured case (see
/// [`Self::lines_at`] and `docs/measurements/display-budget.md`).
const LINE_BUDGET: f64 = ((1u32 << 21) * 3 / 4) as f64;

/// The most lines one primitive may ask `vello` to flatten before the encoder
/// clips it to the band.
///
/// A path is a document's own geometry, so nothing bounds what a producer wrote:
/// one page-spanning path of twenty thousand curves asks for more than a whole
/// band. A primitive past [`LINE_BUDGET`] alone is one no band holding it can
/// paint, so clipping it is the only way that frame paints at all, while a
/// primitive under the budget is handed over whole: the clip walks and flattens
/// the path on the frame thread, which is 20 to 45 ms on the fixtures this tree
/// tests, and a band that fits must not pay for one.
const PRIMITIVE_BUDGET: f64 = LINE_BUDGET;

/// The deviation, in device pixels, a clipped curve may show at the scale its
/// band was encoded for.
///
/// A clipped primitive becomes edges, so this is the one place where a curve
/// stops being flattened by `vello` at the frame's own zoom and becomes a
/// polyline. A quarter pixel is under what area sampling resolves, and
/// [`LADDER`] bounds how far a later zoom may stretch it.
const CLIP_PX: f64 = 0.25;

/// The factor a frame's scale may grow past the one its band was flattened for
/// before the band is encoded again.
///
/// [`CLIP_PX`] is the deviation at the encode's scale, so the top of the ladder
/// shows half a pixel and a deeper frame is refused by nothing: it re-encodes,
/// and only the curves that reach the new band are walked again.
const LADDER: f64 = 2.0;

/// The reciprocal square root of the tolerance `vello` flattens to.
///
/// It is the tolerance `vello_encoding` sizes its own buffers with — 0.2 device
/// pixels, so the factor of the line count below is `1 / sqrt(0.2)`.
const RSQRT_OF_TOLERANCE: f64 = 2.236_067_977_5;

/// `sqrt(n * (n - 1) / 8)` of Wang's formula, for each curve degree.
const QUAD_DEGREE: f64 = 0.5;
const CUBIC_DEGREE: f64 = 0.866_025_403_78;

/// The floor this pass puts under one curve, in lines.
///
/// `vello_encoding` floors its own count at five, and the sweep recorded in
/// `docs/measurements/display-budget.md`
/// found that short: a wave of eight units at a quarter of the fit scale
/// flattens to six lines a curve, so an estimate floored at five answered 0.84
/// of the lines the renderer wrote, and a budget must never answer less. Eight
/// is flat over the sweep's small scales, where the flattener's count stops
/// falling because a curve is a few Euler-spiral pieces and each piece draws at
/// least one line.
const LINES_PER_CURVE: f64 = 8.0;

/// How many curves a glyph's outline stands for.
///
/// A run carries no outline, so the pass counts one em-sized curve per glyph
/// several times over: a bound rather than a measurement, and one that grows
/// with the zoom the way an outline does.
const CURVES_PER_GLYPH: f64 = 40.0;

/// What one band asks `vello` to flatten.
///
/// `vello` flattens in device units, and the lines a curve becomes grow with the
/// square root of the scale it lands at — Wang's formula, which `vello_encoding`
/// itself sizes its buffers with, and which bounds a curve the renderer
/// approximates by an Euler spiral and then caps at a hundred lines a piece, so
/// it answers more than the flattener writes (see [`LINE_BUDGET`]). So the
/// counts are kept in surface units and the frame applies its own square root,
/// and a zoom or a pan costs no walk of the paths.
#[derive(Clone, Copy, Debug, Default)]
struct Work {
    /// The segments that flatten to one line each, at any scale.
    lines: f64,
    /// The lines the curve segments become at one to one.
    curves: f64,
    /// The number of curve segments, which carry [`LINES_PER_CURVE`] each.
    curve_count: f64,
}

impl Work {
    /// The lines this work becomes when the band lands at `scale`.
    #[must_use]
    fn lines_at(&self, scale: f64) -> f64 {
        if !(scale.is_finite() && scale > 0.0) {
            return f64::INFINITY;
        }
        let curves = self.curves * scale.sqrt() * RSQRT_OF_TOLERANCE;
        self.lines + curves.max(LINES_PER_CURVE * self.curve_count)
    }

    /// Whether `vello` cannot hold the band at `scale`.
    #[must_use]
    fn too_heavy(&self, scale: f64) -> bool {
        self.lines_at(scale) > LINE_BUDGET
    }

    /// Add one curve, whose Wang bound at one to one is `lines` lines.
    fn curve(&mut self, lines: f64) {
        if lines.is_finite() && lines > 0.0 {
            self.curves += lines;
            self.curve_count += 1.0;
        } else {
            // A segment the bound cannot state still occupies the flatten pass.
            self.lines += 1.0;
        }
    }
}

impl std::ops::AddAssign for Work {
    fn add_assign(&mut self, other: Self) {
        self.lines += other.lines;
        self.curves += other.curves;
        self.curve_count += other.curve_count;
    }
}

/// What the pass has measured of one primitive.
///
/// The box and the count are pure functions of the primitive and its own
/// transform, so they never change while the list lives. They are kept rather
/// than measured again because measuring them walks every element of the path,
/// and a book's page is one merged glyph-outline path of a hundred thousand
/// curves: a spread step changes the band three times on its way, and a frame
/// the band refuses probes a box of its own (see [`SceneView::tighten`]).
#[derive(Clone, Copy, Default)]
struct Counted {
    /// The box the primitive's ink reaches ([`bounds_of`]), once a band has
    /// reached for it.
    reach: Reach,
    /// What it asks `vello` to flatten ([`count`]), or `None` until a band keeps
    /// it.
    ///
    /// A list measures every box on its first band, so the band can tell what
    /// it reaches, but it counts only the ink it keeps: a frame that shows one
    /// page of a book pays for that page and never for the whole document.
    work: Option<Work>,
}

/// The box of one primitive, as far as the pass has got with it.
#[derive(Clone, Copy, Default)]
enum Reach {
    /// No band has asked for it yet.
    #[default]
    Unmeasured,
    /// [`bounds_of`]'s answer, and `None` for a primitive the pass keeps
    /// whatever the band: a clip, a layer or a marker.
    Measured(Option<kurbo::Rect>),
}

/// The counts of one list, one per primitive.
///
/// The list is the key, the rule [`Base`] and the image table follow: another
/// list is another set of paths, and the counts of one say nothing about
/// another.
struct Counts {
    /// The list every entry was counted from.
    list: Arc<[Primitive]>,
    /// One per primitive of `list`, in its order.
    per: Vec<Counted>,
}

impl Counts {
    /// Measure nothing yet of `list`: both halves are filled the first time a
    /// band asks for them.
    fn new(list: &Arc<[Primitive]>) -> Self {
        Self {
            list: Arc::clone(list),
            per: vec![Counted::default(); list.len()],
        }
    }

    /// Whether these are the counts of `list`.
    fn holds(&self, list: &Arc<[Primitive]>) -> bool {
        Arc::ptr_eq(&self.list, list)
    }

    /// The primitives whose ink reaches `band`, in list order, each with what it
    /// asks `vello` to flatten, and the sum of those asks.
    fn survey(&mut self, band: kurbo::Rect) -> (Vec<(usize, Work)>, Work) {
        let Counts { list, per } = self;
        let mut kept = Vec::new();
        let mut work = Work::default();
        for (index, (primitive, counted)) in list.iter().zip(per).enumerate() {
            let bounds = match counted.reach {
                Reach::Measured(bounds) => bounds,
                Reach::Unmeasured => {
                    let bounds = bounds_of(primitive);
                    counted.reach = Reach::Measured(bounds);
                    bounds
                }
            };
            if !reaches(bounds, band) {
                continue;
            }
            let asks = *counted.work.get_or_insert_with(|| {
                let mut asks = Work::default();
                count(&mut asks, primitive);
                asks
            });
            kept.push((index, asks));
            work += asks;
        }
        (kept, work)
    }

    /// How many primitives are counted, which a test reads to prove that a
    /// second survey of the same list counts nothing again.
    #[cfg(test)]
    fn counted(&self) -> usize {
        self.per
            .iter()
            .filter(|counted| counted.work.is_some())
            .count()
    }

    /// How many primitives are measured, which a test reads to prove that one
    /// survey measures the boxes of the whole list and counts only its band.
    #[cfg(test)]
    fn measured(&self) -> usize {
        self.per
            .iter()
            .filter(|counted| matches!(counted.reach, Reach::Measured(_)))
            .count()
    }
}

/// Add what `vello` flattens for `primitive` to `work`.
///
/// The counts land in surface units — a primitive's own transform is applied
/// here — so a frame scales them without a second walk of the paths.
fn count(work: &mut Work, primitive: &Primitive) {
    match primitive {
        // A rect carries no transform, so its four sides are surface units
        // already, plus the close a fill adds.
        Primitive::Rect { .. } => work.lines += 5.0,
        Primitive::Path {
            path,
            transform,
            stroke,
            ..
        } => {
            let width = stroke.as_ref().map_or(0.0, |outline| outline.style.width);
            count_path(work, path, Some(transform), width);
        }
        // A clip path is flattened like a fill, so it is counted like one.
        Primitive::PushClip {
            path, transform, ..
        } => {
            count_path(work, path, Some(transform), 0.0);
        }
        Primitive::Glyphs { run, transform, .. } => {
            // A run carries no outline, so each glyph stands for
            // [`CURVES_PER_GLYPH`] curves across its em box: a bound rather than
            // a measurement, and one that grows with the zoom the way an
            // outline does.
            let em = f64::from(run.size) * longest_axis(transform);
            let glyphs = run.glyphs.len() as f64;
            let curves = glyphs * CURVES_PER_GLYPH;
            work.curves += curves * CUBIC_DEGREE * em.sqrt();
            work.curve_count += curves;
        }
        // A picture is a brush: it uploads and samples, and flattens nothing.
        Primitive::Image { .. }
        | Primitive::PushLayer { .. }
        | Primitive::PopLayer
        | Primitive::PopClip => {}
    }
}

/// Add the segments of `path`, placed by `transform`, to `work`.
///
/// `width` is the stroke the path is outlined by, in the path's own units, and
/// zero for a fill: a stroke is two offset curves of its centre line, and every
/// vertex adds a join or a cap of its own width, which flattens like a curve
/// spanning that width.
fn count_path(work: &mut Work, path: &kurbo::BezPath, transform: Option<&Affine>, width: f64) {
    let at = transform.copied();
    let place = |point: kurbo::Point| match at {
        Some(affine) => affine * point,
        None => point,
    };
    let sides = if width > 0.0 { 2.0 } else { 1.0 };
    let join = width * at.map_or(1.0, |affine| longest_axis(&affine));
    let mut start: Option<kurbo::Point> = None;
    let mut last: Option<kurbo::Point> = None;
    let vertex = |work: &mut Work| {
        work.lines += sides;
        if width > 0.0 {
            work.curve(QUAD_DEGREE * join.sqrt());
        }
    };
    for element in path.elements() {
        match *element {
            PathEl::MoveTo(point) => {
                if last.is_some() && start.is_some() {
                    vertex(work);
                }
                start = Some(point);
                last = Some(point);
            }
            PathEl::LineTo(point) => {
                vertex(work);
                last = Some(point);
            }
            PathEl::QuadTo(control, point) => {
                let from = last.or(start).unwrap_or(point);
                let deviation = (place(from).to_vec2() - place(control).to_vec2() * 2.0
                    + place(point).to_vec2())
                .hypot();
                work.curve(sides * QUAD_DEGREE * deviation.max(0.0).sqrt());
                last = Some(point);
            }
            PathEl::CurveTo(control_a, control_b, point) => {
                let from = last.or(start).unwrap_or(point);
                let first = (place(from).to_vec2() - place(control_a).to_vec2() * 2.0
                    + place(control_b).to_vec2())
                .hypot();
                let second = (place(control_a).to_vec2() - place(control_b).to_vec2() * 2.0
                    + place(point).to_vec2())
                .hypot();
                work.curve(sides * CUBIC_DEGREE * first.max(second).max(0.0).sqrt());
                last = Some(point);
            }
            PathEl::ClosePath => {
                if last.is_some() && start.is_some() {
                    vertex(work);
                }
                last = start;
            }
        }
    }
}

/// `path` as edges: every curve the box `local` misses becomes the chord
/// between its own ends, and every curve that reaches the box is flattened to
/// `tolerance`.
///
/// `local` is the band in the path's own units, and `tolerance` is in those
/// units too, so one walk of the path answers edges a single clip can use at any
/// placement.
///
/// The substitution is exact rather than an approximation: a curve lies inside
/// the hull of its control points, so a hull the box misses cannot bulge into it,
/// and a chord between those ends lies in the same hull. Neither the curve nor its
/// chord reaches the band, so the clip drops both, and the framed part of the
/// path is the same either way. Only the curves that do reach the band are
/// walked, which is what keeps a deep zoom cheap.
fn edges(path: &kurbo::BezPath, local: kurbo::Rect, tolerance: f64) -> kurbo::BezPath {
    let mut out = kurbo::BezPath::new();
    let mut start = kurbo::Point::ZERO;
    let mut from = kurbo::Point::ZERO;
    for element in path.elements() {
        match *element {
            PathEl::MoveTo(point) => {
                out.move_to(point);
                start = point;
                from = point;
            }
            PathEl::LineTo(point) => {
                out.line_to(point);
                from = point;
            }
            PathEl::QuadTo(control, point) => {
                let hull = hull_of(&[from, control, point]);
                if reaches(Some(hull), local) {
                    flatten_into(
                        &[PathEl::MoveTo(from), PathEl::QuadTo(control, point)],
                        tolerance,
                        &mut out,
                    );
                } else {
                    out.line_to(point);
                }
                from = point;
            }
            PathEl::CurveTo(control_a, control_b, point) => {
                let hull = hull_of(&[from, control_a, control_b, point]);
                if reaches(Some(hull), local) {
                    flatten_into(
                        &[
                            PathEl::MoveTo(from),
                            PathEl::CurveTo(control_a, control_b, point),
                        ],
                        tolerance,
                        &mut out,
                    );
                } else {
                    out.line_to(point);
                }
                from = point;
            }
            PathEl::ClosePath => {
                out.close_path();
                from = start;
            }
        }
    }
    out
}

/// The box that holds `points`.
fn hull_of(points: &[kurbo::Point]) -> kurbo::Rect {
    let mut hull = kurbo::Rect::new(points[0].x, points[0].y, points[0].x, points[0].y);
    for point in &points[1..] {
        hull = hull.union(kurbo::Rect::new(point.x, point.y, point.x, point.y));
    }
    hull
}

/// Push the lines of `elements` flattened to `tolerance` into `out`.
fn flatten_into(elements: &[PathEl], tolerance: f64, out: &mut kurbo::BezPath) {
    kurbo::flatten(elements.iter().copied(), tolerance, |element| {
        if let PathEl::LineTo(point) = element {
            out.line_to(point);
        }
    });
}

/// `path` placed in surface units.
///
/// The copy is what `apply_affine` needs and what the clip keeps: the edges of a
/// path are walked once per encode and dropped once the encode is made.
fn placed(path: &kurbo::BezPath, transform: Affine) -> kurbo::BezPath {
    let mut out = path.clone();
    out.apply_affine(transform);
    out
}

/// The band in the units of a path placed by `transform`, reached by `reach`,
/// or `None` for a placement no box bounds.
///
/// A box of the band's inverse image is a superset of it, so a curve this box
/// misses misses the band too, and one it keeps is only walked and clipped: the
/// answer is conservative in cost and exact in geometry.
fn local_box(band: kurbo::Rect, transform: &Affine, reach: f64) -> Option<kurbo::Rect> {
    let inverse = transform.inverse();
    if inverse
        .as_coeffs()
        .iter()
        .any(|coefficient| !coefficient.is_finite())
    {
        return None;
    }
    let boxed = inverse.transform_rect_bbox(band);
    boxed.is_finite().then(|| boxed.inflate(reach, reach))
}

/// The polygons of `path` that lie inside `band`, `path` read as closed.
///
/// Sutherland–Hodgman, one side of the rect at a time: a rect is convex, so each
/// pass is a walk of the polygon's edges and the four of them answer exactly the
/// part of it the rect holds.
/// A subpath survives as one polygon or as nothing, so a hole stays a hole under
/// either fill rule, and the order of the vertices never reverses.
fn clip_polygons(path: &kurbo::BezPath, band: kurbo::Rect) -> kurbo::BezPath {
    let mut out = kurbo::BezPath::new();
    let mut polygon: Vec<kurbo::Point> = Vec::new();
    for element in path.elements() {
        match *element {
            PathEl::MoveTo(point) => {
                push_polygon(&mut polygon, band, &mut out);
                polygon.push(point);
            }
            // `edges` writes lines, so a curve here is the one shape this clip
            // cannot split: its chord is the edge a fill would draw between the
            // same ends.
            PathEl::LineTo(point) | PathEl::QuadTo(_, point) | PathEl::CurveTo(_, _, point) => {
                polygon.push(point);
            }
            PathEl::ClosePath => push_polygon(&mut polygon, band, &mut out),
        }
    }
    push_polygon(&mut polygon, band, &mut out);
    out
}

/// Clip `polygon` to `band` and append it to `out`, then empty it.
fn push_polygon(polygon: &mut Vec<kurbo::Point>, band: kurbo::Rect, out: &mut kurbo::BezPath) {
    let clipped = clip_polygon(polygon, band);
    polygon.clear();
    if clipped.len() < 3 {
        return;
    }
    out.move_to(clipped[0]);
    for point in &clipped[1..] {
        out.line_to(*point);
    }
    out.close_path();
}

/// The part of `points` inside `band`, read as a closed polygon.
fn clip_polygon(points: &[kurbo::Point], band: kurbo::Rect) -> Vec<kurbo::Point> {
    if points.len() < 3 {
        return Vec::new();
    }
    let mut polygon = points.to_vec();
    polygon = clip_half(
        &polygon,
        |point| point.x >= band.x0,
        |a, b| cross_x(a, b, band.x0),
    );
    polygon = clip_half(
        &polygon,
        |point| point.x <= band.x1,
        |a, b| cross_x(a, b, band.x1),
    );
    polygon = clip_half(
        &polygon,
        |point| point.y >= band.y0,
        |a, b| cross_y(a, b, band.y0),
    );
    polygon = clip_half(
        &polygon,
        |point| point.y <= band.y1,
        |a, b| cross_y(a, b, band.y1),
    );
    polygon
}

/// The part of `polygon` on the side `inside` names, one pass of a clip.
fn clip_half(
    polygon: &[kurbo::Point],
    inside: impl Fn(kurbo::Point) -> bool,
    cross: impl Fn(kurbo::Point, kurbo::Point) -> kurbo::Point,
) -> Vec<kurbo::Point> {
    let mut out = Vec::with_capacity(polygon.len() + 4);
    for (index, &current) in polygon.iter().enumerate() {
        // The edge before the first vertex closes the polygon, because a fill
        // closes a subpath whether the producer wrote it or not.
        let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
        let (current_inside, previous_inside) = (inside(current), inside(previous));
        if current_inside != previous_inside {
            out.push(cross(previous, current));
        }
        if current_inside {
            out.push(current);
        }
    }
    out
}

/// Where the edge `a` to `b` crosses the vertical line `x`.
fn cross_x(a: kurbo::Point, b: kurbo::Point, x: f64) -> kurbo::Point {
    let t = (x - a.x) / (b.x - a.x);
    kurbo::Point::new(x, a.y + t * (b.y - a.y))
}

/// Where the edge `a` to `b` crosses the horizontal line `y`.
fn cross_y(a: kurbo::Point, b: kurbo::Point, y: f64) -> kurbo::Point {
    let t = (y - a.y) / (b.y - a.y);
    kurbo::Point::new(a.x + t * (b.x - a.x), y)
}

/// What `vello` flattens for a path that is already edges, in surface units.
///
/// Every vertex is one line, and the close of a subpath is one more, which is
/// what [`count_path`] answers for a path of no curves.
fn edges_work(path: &kurbo::BezPath) -> Work {
    let mut work = Work::default();
    count_path(&mut work, path, None, 0.0);
    work
}

/// The one-color brush of a solid paint.
/// The brush of a paint, whether a flat color or a ramp this call owns.
enum Brush {
    /// One color.
    Solid(VelloColor),
    /// A ramp, and the space its own geometry is written in.
    ///
    /// A linear ramp is written in surface units, so its space is the identity
    /// and its two points already land where they belong; a radial keeps its
    /// circles in its own space and lets that space squash them, so the space
    /// travels with the ramp and the draw composes it (see [`brush_at`]).
    Ramp { gradient: Gradient, space: Affine },
}

impl SceneView {
    /// Fill a shape of the list with `paint`, in surface units.
    ///
    /// `paint_at` maps the paint's own space onto the shape's, which is the
    /// inverse of the shape transform because every ramp of this tree is
    /// written in surface units.
    fn paint(
        &mut self,
        style: Fill,
        placed: Affine,
        paint: &Paint,
        paint_at: Option<Affine>,
        shape: &impl kurbo::Shape,
    ) {
        match brush(paint) {
            Brush::Solid(color) => self.base.fill(style, placed, color, None, shape),
            Brush::Ramp { gradient, space } => {
                let at = Some(brush_at(paint_at, space));
                self.base.fill(style, placed, &gradient, at, shape);
            }
        }
    }

    /// Stroke `path` with the outline `outline` names.
    fn outline(
        &mut self,
        outline: &crate::scene::Stroke,
        placed: Affine,
        paint_at: Option<Affine>,
        path: &kurbo::BezPath,
    ) {
        match brush(&outline.paint) {
            Brush::Solid(color) => self.base.stroke(&outline.style, placed, color, None, path),
            Brush::Ramp { gradient, space } => {
                let at = Some(brush_at(paint_at, space));
                self.base
                    .stroke(&outline.style, placed, &gradient, at, path);
            }
        }
    }
}

/// The brush of `paint`.
fn brush(paint: &Paint) -> Brush {
    match paint {
        Paint::Solid(color) => Brush::Solid(vello_color(*color)),
        Paint::Linear {
            from,
            to,
            stops,
            spread,
        } => Brush::Ramp {
            gradient: Gradient {
                kind: GradientKind::Linear(LinearGradientPosition {
                    start: kurbo::Point::new(from.0, from.1),
                    end: kurbo::Point::new(to.0, to.1),
                }),
                extend: extend(*spread),
                stops: ramp(stops),
                ..Gradient::default()
            },
            // A linear ramp of this list is stated in surface units.
            space: Affine::IDENTITY,
        },
        Paint::Radial {
            center,
            radius,
            focus,
            focus_radius,
            transform,
            stops,
            spread,
        } => Brush::Ramp {
            gradient: Gradient {
                kind: GradientKind::Radial(RadialGradientPosition {
                    start_center: kurbo::Point::new(focus.0, focus.1),
                    start_radius: *focus_radius as f32,
                    end_center: kurbo::Point::new(center.0, center.1),
                    end_radius: *radius as f32,
                }),
                extend: extend(*spread),
                stops: ramp(stops),
                ..Gradient::default()
            },
            space: *transform,
        },
    }
}

/// The brush transform of a ramp whose own space lands on the plane by `space`.
///
/// `shape_at` is the inverse of the shape's transform, which is the brush
/// transform a ramp written in surface units takes: it hands the ramp to the
/// shape's own space, where the fill reads it.
/// A ramp with a space of its own goes through that space first, so `space`
/// multiplies in on the right and the ramp still lands in surface units before
/// the shape takes it back.
fn brush_at(shape_at: Option<Affine>, space: Affine) -> Affine {
    shape_at.map_or(space, |inverse| inverse * space)
}

/// The stops of a ramp, as vello reads them.
fn ramp(stops: &[Stop]) -> peniko::ColorStops {
    let mut out = peniko::ColorStops::new();
    for stop in stops {
        out.push(ColorStop {
            offset: stop.offset,
            color: vello_color(stop.color).into(),
        });
    }
    out
}

/// What a ramp does past its last stop.
const fn extend(spread: Spread) -> Extend {
    match spread {
        Spread::Pad => Extend::Pad,
        Spread::Repeat => Extend::Repeat,
        // A mirror is a reflect: the ramp runs backwards past its last stop.
        Spread::RepeatMirror => Extend::Reflect,
    }
}

/// The fill rule of a shape.
const fn vello_fill_rule(rule: FillRule) -> Fill {
    match rule {
        FillRule::NonZero => Fill::NonZero,
        FillRule::EvenOdd => Fill::EvenOdd,
    }
}

/// One of this tree's colors as vello's.
fn vello_color(color: Color) -> VelloColor {
    VelloColor::from_rgba8(color.r, color.g, color.b, color.a)
}

/// Where the paint of a frame lands in a window that another frame names.
///
/// The paint covers the window of `drawn_to_window`, and the picture of it is
/// the one `to_window` puts in the same place, so the whole texture fills the
/// rect that `to_window ∘ drawn_to_window⁻¹` names: its middle is the image of
/// the old window's middle, and its scale and turn are the ones of that mapping.
/// Both transforms belong to a frame with a drawable plane, so the mapping is a
/// similarity and one rect beside one angle says it.
#[must_use]
fn stale_place(drawn_size: (u32, u32), drawn_to_window: Affine, to_window: Affine) -> (Rect, Quad) {
    let (width, height) = (f64::from(drawn_size.0), f64::from(drawn_size.1));
    let src = dim::whole(drawn_size.0 as f32, drawn_size.1 as f32);
    let delta = to_window * drawn_to_window.inverse();
    // The matrix of an affine is `[a c e; b d f]`, so this reads the picture's
    // own x and y coefficients by the axis they move.
    let [xx, yx, xy, yy, tx, ty] = delta.as_coeffs();
    let scale = xx.hypot(yx);
    let (scale, angle) = if scale.is_finite() && scale > 0.0 {
        (scale, yx.atan2(xx).to_degrees() as f32)
    } else {
        // A degenerate mapping cannot place a picture, and the window it names
        // shows nothing either: the paint lands where it was.
        (1.0, 0.0)
    };
    let center = Vec2::new(
        (xx * width / 2.0 + xy * height / 2.0 + tx) as f32,
        (yx * width / 2.0 + yy * height / 2.0 + ty) as f32,
    );
    let (dest_w, dest_h) = ((width * scale) as f32, (height * scale) as f32);
    let dest = dim::rect(
        center.x - dest_w / 2.0,
        center.y - dest_h / 2.0,
        dest_w,
        dest_h,
    );
    (src, Quad::turned(dest, center, angle))
}

/// The picture of an image leaf, as a brush a fill samples.
///
/// The bytes stay straight alpha, which is the convention of the whole tree:
/// every decoder writes straight bytes and the SVG backend un-premultiplies
/// before it hands a picture over.
fn image_brush(pixels: &DecodedImage) -> ImageBrush {
    let (width, height) = (pixels.width().raw(), pixels.height().raw());
    let data = ImageData {
        data: Blob::from(pixels.as_rgba().to_vec()),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width,
        height,
    };
    ImageBrush::new(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{PlacedGlyph, Stroke};

    /// A solid paint, for the primitives the band tests build.
    fn solid() -> Paint {
        Paint::Solid(Color::WHITE)
    }

    /// Whether the band keeps `primitive`, measured as it comes.
    ///
    /// The pass reads the same answer off its kept counts; this is the walk
    /// those counts stand in for.
    fn hits(primitive: &Primitive, band: kurbo::Rect) -> bool {
        reaches(bounds_of(primitive), band)
    }

    /// A rect as a path, which is what a producer that places a rect by a
    /// transform writes.
    fn rect_path(rect: kurbo::Rect) -> kurbo::BezPath {
        let mut path = kurbo::BezPath::new();
        path.move_to(rect.origin());
        path.line_to((rect.x1, rect.y0));
        path.line_to((rect.x1, rect.y1));
        path.line_to((rect.x0, rect.y1));
        path.close_path();
        path
    }

    /// A frame that shows a window of `win` over a plane of `plane` at `scale`,
    /// panned by `pan` and turned by `angle`.
    fn frame(plane: (f32, f32), win: (f32, f32), pan: (f32, f32), scale: f32) -> region::Frame {
        region::Frame {
            img: dim::SurfaceSize::new(plane.0, plane.1),
            win: dim::WindowSize::new(win.0, win.1),
            pan: dim::WindowVector::new(pan.0, pan.1),
            scale: region::Scale::uniform(scale),
            angle: 0.0,
        }
    }

    /// A band covers its own window and the pad around it, so a pan inside the
    /// pad costs no encode and one that left it asks for a new one.
    #[test]
    fn a_band_holds_the_window_and_the_pan_around_it() {
        let plane = (400.0, 8000.0);
        let win = (800.0, 600.0);
        let hull =
            |pan_y| window_hull(to_window(&frame(plane, win, (0.0, pan_y), 8.0)), (800, 600));
        let visible = hull(0.0);
        assert!(
            (visible.width() - 100.0).abs() < 1e-9 && (visible.height() - 75.0).abs() < 1e-9,
            "a window of 800 by 600 at scale 8 covers 100 by 75 surface units: {visible:?}"
        );
        let base = Base {
            list: Arc::from(Vec::<Primitive>::new()),
            plane,
            band: band_of(visible),
            work: Work::default(),
            clip_scale: None,
        };
        assert!(
            base.covers(visible),
            "a band covers the window it was made for"
        );
        assert!(
            base.covers(hull(200.0)),
            "the pad holds a pan of half a window"
        );
        assert!(
            !base.covers(hull(1000.0)),
            "a window that left the band is not covered"
        );
    }

    /// A frame whose two axes carry different factors maps each axis at its
    /// own, so the box a stated view asks for is the box the window shows.
    #[test]
    fn a_frame_scales_each_axis_at_its_own_factor() {
        let mut squashed = frame((100.0, 400.0), (800.0, 600.0), (0.0, 0.0), 1.0);
        squashed.scale = region::Scale::axes(8.0, 1.5);
        let mapped = to_window(&squashed);
        assert_eq!(
            mapped * kurbo::Point::new(0.0, 0.0),
            kurbo::Point::new(0.0, 0.0)
        );
        assert_eq!(
            mapped * kurbo::Point::new(100.0, 400.0),
            kurbo::Point::new(800.0, 600.0),
            "the box fills the window on both axes"
        );
        // The window covers 100 by 400 surface units, so the band keeps a
        // primitive a uniform fit of the same plane would have thrown away.
        let visible = window_hull(mapped, (800, 600));
        assert!(
            (visible.width() - 100.0).abs() < 1e-9 && (visible.height() - 400.0).abs() < 1e-9,
            "the two axes bound different spans: {visible:?}"
        );
    }

    /// A turned window covers the box its corners span, because that is what a
    /// primitive has to meet to be visible.
    #[test]
    fn a_turned_window_covers_the_box_of_its_corners() {
        let mut turned = frame((400.0, 8000.0), (800.0, 600.0), (0.0, 0.0), 8.0);
        turned.angle = 90.0;
        let hull = window_hull(to_window(&turned), (800, 600));
        assert!(
            (hull.width() - 75.0).abs() < 1e-9 && (hull.height() - 100.0).abs() < 1e-9,
            "a quarter turn swaps the sides: {hull:?}"
        );
    }

    /// The band keeps the ink it reaches and drops the rest, and it always keeps
    /// a clip, a layer and their markers.
    #[test]
    fn a_band_keeps_the_primitives_it_reaches() {
        let band = kurbo::Rect::new(0.0, 0.0, 100.0, 100.0);
        let rect = |x: f64| Primitive::Rect {
            rect: kurbo::Rect::new(x, 10.0, x + 20.0, 30.0),
            paint: solid(),
        };
        assert!(hits(&rect(10.0), band), "a rect inside the band stays");
        assert!(!hits(&rect(200.0), band), "a rect outside the band drops");
        let flat = Primitive::Rect {
            rect: kurbo::Rect::new(10.0, 10.0, 90.0, 10.0),
            paint: solid(),
        };
        assert!(
            hits(&flat, band),
            "a shape of no area may still draw a hairline"
        );

        // A path carries its own transform, so its box is the one it lands in.
        let path = |at: Affine, stroke: Option<Stroke>, fill: bool| Primitive::Path {
            path: rect_path(kurbo::Rect::new(0.0, 0.0, 10.0, 10.0)),
            transform: at,
            fill: fill.then(solid),
            fill_rule: FillRule::NonZero,
            stroke,
        };
        assert!(hits(
            &path(Affine::translate((50.0, 50.0)), None, true),
            band
        ));
        assert!(
            !hits(&path(Affine::translate((150.0, 50.0)), None, true), band),
            "a path that landed outside the band drops"
        );
        // A stroke reaches half its width past the centre line.
        let outline = |width| {
            Some(Stroke {
                style: kurbo::Stroke::new(width),
                paint: solid(),
            })
        };
        let hairline = path(Affine::translate((120.0, 50.0)), outline(1.0), false);
        assert!(!hits(&hairline, band), "a hairline well outside is culled");
        let wide = path(Affine::translate((120.0, 50.0)), outline(60.0), false);
        assert!(hits(&wide, band), "a wide stroke reaches into the band");

        // A picture lands by its own pixels.
        let picture = |at: Affine| Primitive::Image {
            pixels: DecodedImage::from_exact(dim::ImgW::new(10), dim::ImgH::new(10), vec![0; 400]),
            transform: at,
        };
        assert!(hits(&picture(Affine::translate((50.0, 50.0))), band));
        assert!(!hits(&picture(Affine::translate((500.0, 500.0))), band));

        // A run states no outline, so its box pads two ems around its origins.
        let run = |at: Affine| Primitive::Glyphs {
            run: GlyphRun {
                font: Arc::from(Vec::<u8>::new()),
                face: 0,
                size: 10.0,
                glyphs: Arc::from(vec![
                    PlacedGlyph {
                        id: 0,
                        x: 0.0,
                        y: 0.0,
                    },
                    PlacedGlyph {
                        id: 0,
                        x: 50.0,
                        y: 0.0,
                    },
                ]),
            },
            transform: at,
            paint: solid(),
        };
        assert!(hits(&run(Affine::IDENTITY), band));
        assert!(!hits(&run(Affine::translate((300.0, 0.0))), band));

        // The stack stays whole: a clip carries no ink of its own, and a
        // `PopClip` without its `PushClip` would close some other clip.
        let stack = [
            Primitive::PushClip {
                path: rect_path(kurbo::Rect::new(0.0, 0.0, 10.0, 10.0)),
                transform: Affine::IDENTITY,
                fill_rule: FillRule::NonZero,
            },
            Primitive::PopClip,
            Primitive::PushLayer { alpha: 0.5 },
            Primitive::PopLayer,
        ];
        for primitive in stack {
            assert!(
                hits(&primitive, band),
                "the stack stays whole: {primitive:?}"
            );
        }
    }

    /// A path of `curves` cubics, each spanning `reach` surface units, as a
    /// producer that draws a coastline or a flattened export writes it.
    fn wave(curves: usize, reach: f64) -> Primitive {
        let mut path = kurbo::BezPath::new();
        path.move_to((0.0, 0.0));
        for step in 0..curves {
            let x = step as f64 * reach / 4.0;
            path.curve_to(
                (x, reach),
                (x + reach / 4.0, -reach),
                (x + reach / 2.0, 0.0),
            );
        }
        Primitive::Path {
            path,
            transform: Affine::IDENTITY,
            fill: Some(solid()),
            fill_rule: FillRule::NonZero,
            stroke: None,
        }
    }

    /// The estimate grows with the zoom for a curve and stays put for a
    /// straight line, because a flatten pass divides a curve by the device
    /// tolerance and a line needs no division at all.
    #[test]
    fn a_curve_costs_more_as_the_window_zooms() {
        let mut straight = Work::default();
        count(
            &mut straight,
            &Primitive::Path {
                path: rect_path(kurbo::Rect::new(0.0, 0.0, 10.0, 10.0)),
                transform: Affine::IDENTITY,
                fill: Some(solid()),
                fill_rule: FillRule::NonZero,
                stroke: None,
            },
        );
        let mut curved = Work::default();
        count(&mut curved, &wave(100, 400.0));

        assert_eq!(
            straight.lines_at(1.0),
            straight.lines_at(64.0),
            "four sides flatten the same at any zoom"
        );
        assert!(
            curved.lines_at(64.0) > curved.lines_at(1.0) * 2.0,
            "a hundred curves of a deep zoom cost more than twice the lines"
        );
    }

    /// A band whose estimate crosses the budget is refused before it paints, and
    /// one that stays under it is not: `vello` answers *nothing* on a frame past
    /// its lines buffer and reports nothing at all, so the caller must be told.
    ///
    /// What `LINE_BUDGET` is worth against the real buffer is a measurement, not
    /// an assertion this test can make: `a_band_past_the_budget_is_the_frame_the_renderer_blanks_on`
    /// pins it against the renderer's own blank.
    #[test]
    fn a_heavy_band_is_refused() {
        let mut light = Work::default();
        count(&mut light, &wave(4, 10.0));
        let mut heavy = Work::default();
        // Two thousand curves across a page, the shape a flattened export hands
        // over: a band keeps it whole, and the flatten pass divides it whole.
        count(&mut heavy, &wave(2000, 400.0));

        for scale in [1.0, 4.0, 16.0] {
            assert!(
                !light.too_heavy(scale),
                "a handful of small curves paints at {scale}x"
            );
            assert!(
                !heavy.too_heavy(scale),
                "the page-spanning path still fits at {scale}x"
            );
        }
        assert!(
            heavy.too_heavy(256.0),
            "a deep zoom of that path cannot fit the renderer"
        );
    }

    /// A page of twenty thousand curves is handed over whole at the zoom where a
    /// reader first looks at it, and clipped only past the budget.
    ///
    /// The clip walks and flattens the path on the frame thread, so it is paid
    /// only where the primitive alone is more than a band can hold: eight times
    /// the fit is a frame the renderer holds whole, and four hundred times is
    /// past [`PRIMITIVE_BUDGET`] and is the frame the clip exists for.
    #[test]
    fn a_page_is_clipped_only_past_the_budget() {
        let mut page = Work::default();
        // The waves of the display pass's heavy page: a sixty-unit swing over a
        // fifteen-unit step, twenty thousand of them across the plane.
        count(&mut page, &wave(20_000, 60.0));

        assert!(
            page.lines_at(8.0) < PRIMITIVE_BUDGET,
            "eight times the fit is handed over whole: {} lines",
            page.lines_at(8.0)
        );
        assert!(
            page.lines_at(400.0) > PRIMITIVE_BUDGET,
            "four hundred times is past it: {} lines",
            page.lines_at(400.0)
        );
    }

    /// A curve shrunk small is counted at the floor and never below what the
    /// flattener writes of it.
    ///
    /// A flat curve needs fewer lines as it shrinks, until the count stops at the
    /// pieces `vello` fits it with: the sweep in
    /// `docs/measurements/display-budget.md` measured six lines a
    /// curve for a wave of eight units at a quarter of the fit scale, and six
    /// again at a sixteenth. `vello_encoding` floors its own count at five, which
    /// answered less than the renderer wrote, so the floor here is eight.
    #[test]
    fn a_small_curve_is_counted_at_the_floor() {
        let mut small = Work::default();
        count(&mut small, &wave(20_000, 8.0));

        assert!(
            small.lines_at(0.25) >= 6.0 * 20_000.0,
            "a quarter of the fit counts the six lines a curve the sweep measured: {}",
            small.lines_at(0.25)
        );
        assert!(
            small.lines_at(0.015_625) >= 6.0 * 20_000.0,
            "and four more shrinks do not count less: {}",
            small.lines_at(0.015_625)
        );
    }

    /// What `list` asks `vello` to flatten for `band`, walked from scratch.
    ///
    /// The oracle the kept counts are checked against: the same two steps
    /// [`Counts::survey`] takes, with nothing carried between calls.
    fn walked(list: &[Primitive], band: kurbo::Rect) -> Work {
        let mut work = Work::default();
        for primitive in list.iter().filter(|p| reaches(bounds_of(p), band)) {
            count(&mut work, primitive);
        }
        work
    }

    /// A page of a book: a wave of curves inside its own slot of the plane.
    fn page(slot: usize, curves: usize) -> Primitive {
        let mut primitive = wave(curves, 40.0);
        if let Primitive::Path { transform, .. } = &mut primitive {
            *transform = Affine::translate((0.0, f64::from(slot as u32) * 200.0));
        }
        primitive
    }

    /// A curve the box misses becomes the chord between its own ends, and the
    /// substitution is exact: the curve lies in the hull of its control points, so
    /// a hull the box misses cannot reach it.
    #[test]
    fn a_curve_the_box_misses_becomes_its_chord() {
        let mut path = kurbo::BezPath::new();
        path.move_to((0.0, 0.0));
        path.curve_to((1.0, 10.0), (9.0, 10.0), (10.0, 0.0));
        // A box that holds the curve, and one a thousand units away from it.
        let near = kurbo::Rect::new(-1.0, -1.0, 11.0, 11.0);
        let far = kurbo::Rect::new(1000.0, 1000.0, 1001.0, 1001.0);

        let flattened = edges(&path, near, 0.01);
        assert!(
            flattened
                .elements()
                .iter()
                .all(|element| !matches!(element, PathEl::CurveTo(..))),
            "a curve the box reaches is flattened"
        );
        assert!(
            flattened.elements().len() > 4,
            "a flattened curve is a run of lines, not one of them"
        );

        let chaffed = edges(&path, far, 0.01);
        let elements: Vec<PathEl> = chaffed.elements().to_vec();
        assert_eq!(
            elements,
            vec![
                PathEl::MoveTo((0.0, 0.0).into()),
                PathEl::LineTo((10.0, 0.0).into())
            ],
            "a curve the box misses is the chord between its ends"
        );
    }

    /// The clip keeps the part of a polygon the band holds, and the whole of a
    /// polygon the band holds.
    #[test]
    fn a_polygon_is_kept_to_the_band() {
        let band = kurbo::Rect::new(0.0, 0.0, 100.0, 100.0);
        let inside = rect_path(kurbo::Rect::new(25.0, 25.0, 75.0, 75.0));
        let kept = clip_polygons(&inside, band);
        assert_eq!(
            kept.elements().len(),
            5,
            "a polygon inside the band is four sides and a close: {kept:?}"
        );
        for element in kept.elements() {
            let (PathEl::MoveTo(point) | PathEl::LineTo(point)) = *element else {
                continue;
            };
            assert!(
                band.contains(point),
                "every vertex of a kept polygon is inside the band: {point:?}"
            );
        }

        // A square that straddles the right side is cut to the band, and the cut
        // keeps its area: the part of the square the band holds is half of it.
        let straddling = rect_path(kurbo::Rect::new(50.0, 25.0, 150.0, 75.0));
        let kept = clip_polygons(&straddling, band);
        assert!(
            (area(&kept) - 2500.0).abs() < 1e-9,
            "the clip keeps the half of the square the band holds: {}",
            area(&kept)
        );
    }

    /// A subpath the band misses is dropped and one it holds is kept, so a hole
    /// stays a hole and the fill rule still says what the producer meant.
    #[test]
    fn a_clipped_hole_survives_the_clip() {
        let band = kurbo::Rect::new(0.0, 0.0, 100.0, 100.0);
        let mut path = rect_path(kurbo::Rect::new(10.0, 10.0, 90.0, 90.0));
        // A hole of the opposite turn, and a subpath the band never reaches.
        path.extend(
            rect_path(kurbo::Rect::new(30.0, 30.0, 50.0, 50.0))
                .elements()
                .iter()
                .rev()
                .copied(),
        );
        path.extend(
            rect_path(kurbo::Rect::new(300.0, 300.0, 400.0, 400.0))
                .elements()
                .iter()
                .copied(),
        );

        let kept = clip_polygons(&path, band);
        let mut turns: Vec<f64> = Vec::new();
        let mut polygon: Vec<kurbo::Point> = Vec::new();
        for element in kept.elements() {
            match *element {
                PathEl::MoveTo(point) => {
                    if polygon.len() > 2 {
                        turns.push(turn(&polygon));
                    }
                    polygon.clear();
                    polygon.push(point);
                }
                PathEl::LineTo(point) => polygon.push(point),
                PathEl::ClosePath => {
                    turns.push(turn(&polygon));
                    polygon.clear();
                }
                PathEl::QuadTo(..) | PathEl::CurveTo(..) => {}
            }
        }
        assert_eq!(turns.len(), 2, "the outer shape and its hole survive");
        assert!(
            turns[0].signum() != turns[1].signum(),
            "the hole still turns against its shape: {turns:?}"
        );
    }

    /// The band of a flattened encode is encoded again when a zoom doubles the
    /// scale it was flattened for, and not before.
    #[test]
    fn a_flattened_band_is_encoded_again_past_the_ladder() {
        let base = |clip_scale| Base {
            list: Arc::from(Vec::<Primitive>::new()),
            plane: (400.0, 8000.0),
            band: kurbo::Rect::new(0.0, 0.0, 100.0, 75.0),
            work: Work::default(),
            clip_scale,
        };
        let flattened = base(Some(4.0));
        assert!(
            flattened.holds_scale(4.0),
            "a band is sharp at its own scale"
        );
        assert!(
            flattened.holds_scale(4.0 * LADDER),
            "and at the top of the ladder its deviation allows"
        );
        assert!(
            !flattened.holds_scale(4.0 * LADDER * 1.01),
            "past it the facets show and the band is encoded again"
        );
        assert!(
            base(None).holds_scale(4096.0),
            "a band that clipped nothing keeps every frame"
        );
    }

    /// The signed area of a closed path, in square surface units.
    fn area(path: &kurbo::BezPath) -> f64 {
        kurbo::Shape::area(path)
    }

    /// The signed area of a polygon, whose turn names the winding of a subpath.
    fn turn(points: &[kurbo::Point]) -> f64 {
        let mut path = kurbo::BezPath::new();
        path.move_to(points[0]);
        for point in &points[1..] {
            path.line_to(*point);
        }
        path.close_path();
        area(&path)
    }

    /// A kept count answers what a walk of the same band answers, for every band
    /// a glide and a zoom ask for.
    ///
    /// The cache is the whole of the spread step's cost: a band change used to
    /// walk every path element of the band, and a book's page is one path of a
    /// hundred thousand curves.
    #[test]
    fn a_kept_count_answers_what_a_walk_of_the_band_does() {
        let list: Arc<[Primitive]> = Arc::from(vec![page(0, 400), page(1, 400), page(2, 400)]);
        let mut counts = Counts::new(&list);
        let bands = [
            kurbo::Rect::new(-400.0, -300.0, 400.0, 300.0),
            // A band of the second page alone.
            kurbo::Rect::new(-400.0, 100.0, 400.0, 300.0),
            // A band that reaches every page.
            kurbo::Rect::new(-400.0, -300.0, 400.0, 900.0),
            // A band that reaches nothing.
            kurbo::Rect::new(-4000.0, -4000.0, -3000.0, -3000.0),
            // The first band again, which the counts now answer from memory.
            kurbo::Rect::new(-400.0, -300.0, 400.0, 300.0),
        ];
        for band in bands {
            let (kept, work) = counts.survey(band);
            // A tolerance and not an equality: the kept counts add a primitive's
            // total, and the walk adds its curves one by one, so the two sums
            // differ in the last bits.
            let (kept_lines, walked_lines) =
                (work.lines_at(1.0), walked(&list, band).lines_at(1.0));
            assert!(
                (kept_lines - walked_lines).abs() <= walked_lines * 1e-9,
                "the kept count of {band:?} is the count a walk answers: {kept_lines} against \
                 {walked_lines}"
            );
            assert!(
                kept.iter()
                    .all(|&(index, _)| reaches(bounds_of(&list[index]), band)),
                "a kept primitive is one the band reaches, at {band:?}"
            );
            assert!(
                (0..list.len()).all(|index| {
                    kept.iter().any(|&(kept, _)| kept == index)
                        == reaches(bounds_of(&list[index]), band)
                }),
                "every primitive the band reaches is kept, at {band:?}"
            );
        }
        assert_eq!(
            counts.counted(),
            list.len(),
            "each primitive is counted once and kept"
        );
        assert_eq!(
            counts.measured(),
            list.len(),
            "every box of the list is measured once"
        );
    }

    /// A band of one page measures the boxes of the whole list but counts only
    /// the page it reaches, so the frame that opens a book spends its walk on
    /// the ink it shows.
    #[test]
    fn a_band_counts_only_the_ink_it_reaches() {
        let list: Arc<[Primitive]> = Arc::from(vec![page(0, 400), page(1, 400), page(2, 400)]);
        let mut counts = Counts::new(&list);
        let first_page = kurbo::Rect::new(-400.0, -100.0, 400.0, 100.0);
        let (kept, _) = counts.survey(first_page);
        assert_eq!(
            kept.iter().map(|&(index, _)| index).collect::<Vec<_>>(),
            vec![0],
            "the band keeps the page it shows"
        );
        assert_eq!(counts.counted(), 1, "and counts that page alone");
        assert_eq!(counts.measured(), 3, "while it measured every box it asks");
    }

    /// A second survey of a band counts no path again, and another list starts
    /// from nothing: the counts of one list say nothing about another.
    #[test]
    fn a_survey_counts_each_primitive_once() {
        let list: Arc<[Primitive]> = Arc::from(vec![page(0, 40)]);
        let mut counts = Counts::new(&list);
        let band = kurbo::Rect::new(-400.0, -300.0, 400.0, 300.0);
        assert_eq!(counts.counted(), 0, "a fresh list carries no count");
        let (_, first) = counts.survey(band);
        assert_eq!(
            counts.counted(),
            1,
            "the band counted the primitive it kept"
        );
        let (_, again) = counts.survey(band);
        assert_eq!(
            first.lines_at(1.0),
            again.lines_at(1.0),
            "a second survey answers the work the first did"
        );
        assert_eq!(
            counts.counted(),
            1,
            "the second survey counted no path a second time"
        );

        let other: Arc<[Primitive]> = Arc::from(vec![page(3, 40), page(4, 40)]);
        assert!(
            !counts.holds(&other),
            "another list is another set of paths"
        );
        let fresh = Counts::new(&other);
        assert_eq!(fresh.counted(), 0, "another list starts uncounted");
    }

    /// A frame that names no drawable plane keeps everything, because a band it
    /// cannot state would drop primitives a working frame shows.
    #[test]
    fn a_frame_with_no_window_keeps_everything() {
        let hull = window_hull(
            to_window(&frame((400.0, 800.0), (800.0, 600.0), (0.0, 0.0), 0.0)),
            (800, 600),
        );
        let band = band_of(hull);
        let far = Primitive::Rect {
            rect: kurbo::Rect::new(1e6, 1e6, 2e6, 2e6),
            paint: solid(),
        };
        assert!(
            hits(&far, band),
            "a band that is not finite holds everything"
        );
    }

    /// A frame that fits a plane of `img` into a window of `win`, unturned and
    /// unpanned.
    fn fit(win: (f32, f32), img: (f32, f32)) -> Affine {
        to_window(&region::Frame {
            img: dim::SurfaceSize::new(img.0, img.1),
            win: dim::WindowSize::new(win.0, win.1),
            pan: dim::WindowVector::new(0.0, 0.0),
            scale: region::Scale::uniform((win.0 / img.0).min(win.1 / img.1)),
            angle: 0.0,
        })
    }

    /// The four corners of a window, in the order [`Quad::rect`] names them.
    fn window_corners(size: (u32, u32)) -> [kurbo::Point; 4] {
        let (w, h) = (f64::from(size.0), f64::from(size.1));
        [
            kurbo::Point::new(0.0, 0.0),
            kurbo::Point::new(w, 0.0),
            kurbo::Point::new(w, h),
            kurbo::Point::new(0.0, h),
        ]
    }

    /// A reused paint lands exactly where the mapping of the two windows puts
    /// it, corner for corner.
    ///
    /// The pass answers one rect and one angle, so this pins the reading of the
    /// similarity out of the affine: a sign, a corner order or a turn direction
    /// that disagreed would land a corner somewhere else.
    #[test]
    fn a_reused_paint_follows_the_mapping_of_the_two_windows() {
        let plane = (400.0, 300.0);
        let cases = [
            // The same window twice.
            ((800.0, 600.0), (800.0, 600.0), 0.0),
            // A window that grew on both axes.
            ((800.0, 600.0), (1200.0, 900.0), 0.0),
            // A turned one.
            ((800.0, 600.0), (800.0, 600.0), 90.0),
            // A grown and turned one.
            ((900.0, 700.0), (1400.0, 1000.0), 30.0),
            // A shrunk one.
            ((900.0, 700.0), (500.0, 900.0), -15.0),
        ];
        for (from, to, angle) in cases {
            let drawn = fit(from, plane);
            let now = to_window(&region::Frame {
                img: dim::SurfaceSize::new(plane.0, plane.1),
                win: dim::WindowSize::new(to.0, to.1),
                pan: dim::WindowVector::new(10.0, -20.0),
                scale: region::Scale::uniform((to.0 / plane.0).min(to.1 / plane.1)),
                angle,
            });
            let size = (from.0 as u32, from.1 as u32);
            let (src, quad) = stale_place(size, drawn, now);
            assert_eq!(
                src,
                dim::whole(from.0, from.1),
                "the paint fills its own texture"
            );
            let delta = now * drawn.inverse();
            for (got, corner) in quad.p.into_iter().zip(window_corners(size)) {
                let want = delta * corner;
                assert!(
                    got.abs_diff_eq(Vec2::new(want.x as f32, want.y as f32), 0.01),
                    "a corner of {from:?} -> {to:?} at {angle} deg landed at {got:?} instead of \
                     ({:.2}, {:.2})",
                    want.x,
                    want.y
                );
            }
        }
    }

    /// A window that doubles its size on the plane's fit covers the new window
    /// with the old paint, so the paint is not letterboxed by its own growth.
    #[test]
    fn a_doubled_window_stretches_the_fit_over_it() {
        let plane = (400.0, 300.0);
        let drawn = fit((800.0, 600.0), plane);
        let now = fit((1600.0, 1200.0), plane);
        let (_, quad) = stale_place((800, 600), drawn, now);
        let want = Quad::rect(dim::whole(1600.0, 1200.0));
        for (got, want) in quad.p.into_iter().zip(want.p) {
            assert!(
                got.abs_diff_eq(want, 0.01),
                "a corner landed at {got:?} instead of {want:?}"
            );
        }
    }
}
