//! A scene document that a producer outside this process wrote.
//!
//! `vv --scene <file>` reads one: a version, the plane extent, the optional box
//! the window shows and the display list, so a program that owns the layout of a
//! picture hands over the shapes and this viewer draws them at any zoom.
//! The list is the picture and there is no raster behind it, so a scene
//! document is not a [`Surface`](crate::surface::Surface): the two entrances
//! here answer a plane and a list, and nothing in this module can rasterize.
//!
//! # The document
//!
//! ```json
//! {
//!   "version": 2,
//!   "plane": { "width": 800.0, "height": 480.0 },
//!   "viewport": { "x0": 100.0, "y0": 40.0, "x1": 300.0, "y1": 160.0 },
//!   "primitives": [{ "Rect": { "rect": { "x0": 0.0, "y0": 0.0, "x1": 8.0, "y1": 3.0 },
//!                              "paint": { "Solid": { "r": 1, "g": 2, "b": 3, "a": 255 } } } }]
//! }
//! ```
//!
//! The vocabulary is [`crate::scene::Primitive`] itself, so a variant this
//! tree adds is a variant a document may state, and a field it renames breaks
//! a document loudly instead of painting the wrong thing.
//! `version` is the one guard on the shape of the document, and it is read on
//! its own before the rest: a producer that writes another number is refused by
//! name, so it reads why, and a scene this viewer cannot read never half-paints.
//!
//! # The viewport
//!
//! A document may name the box of its plane the window shows, and then the
//! window shows exactly that box: each axis takes the ratio of the window to the
//! box, so a producer whose two axes scale differently states it here and the
//! picture is right and not merely whole.
//! A document that states no viewport frames the whole plane, one factor on both
//! axes, which is what a picture does.
//! The box is in the plane's own units and says nothing about window pixels, so
//! a resize re-derives both ratios and the view keeps showing the box the
//! producer named.
//!
//! # What a document cannot state
//!
//! A [`Primitive::Image`] leaf stays off the wire, because it carries decoded
//! pixels and a picture is bytes rather than a shape. A document that names one
//! is refused as an unknown variant, the same answer a misspelled primitive
//! gets, and a list that holds one does not serialise either.
//! A document that leaves a field out or invents one is refused too: every
//! struct of the vocabulary denies an unknown field and none of them defaults,
//! so a producer and this reader agree on the whole shape or on nothing.

use std::sync::Arc;

use derive_more::{Display, Error, From};
use serde::{Deserialize, Serialize};

use crate::{
    region::{self, Crop},
    scene::Primitive,
    surface::Size,
};

/// The document version this tree writes and reads.
///
/// A producer that moves to another shape bumps this, and an older viewer
/// refuses the document by name instead of reading the fields it recognizes.
/// Version 2 added the viewport, so a producer that states one writes 2 and a
/// version 1 viewer refuses it by version rather than by an unknown field.
pub const VERSION: u32 = 2;

/// The oldest document version this tree still reads.
///
/// Version 1 is a version 2 document without a viewport: every other field
/// means what it meant, so a document written before the viewport existed still
/// paints.
pub const OLDEST: u32 = 1;

/// One scene document as a producer states it.
///
/// The extent is the plane the list was written for, in the producer's own
/// units: the list is in those units and carries no scale, so the viewer frames
/// the extent — or the box the document states inside it — and a zoom is a
/// change of the transform alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDoc {
    /// The document version, which must be [`OLDEST`] to [`VERSION`].
    pub version: u32,
    /// The plane the list describes.
    pub plane: Extent,
    /// The box of the plane the window shows, or none for the whole plane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viewport: Option<Viewport>,
    /// The primitives, in paint order.
    pub primitives: Vec<Primitive>,
}

/// The box of the plane a window shows, in the plane's own units.
///
/// The box fills the window, so the two axes take their own ratio: a document
/// whose axes are independent (a plot with a price axis and a time axis) states
/// the range of each and the viewer maps each range onto the window's own span.
/// The box needs no relation to the plane's bounds: a producer may show the
/// empty space past the end of its data, and the viewer paints the background
/// there.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Viewport {
    /// The left edge of the box, in surface units.
    pub x0: f32,
    /// The top edge of the box, in surface units.
    pub y0: f32,
    /// The right edge of the box, in surface units.
    pub x1: f32,
    /// The bottom edge of the box, in surface units.
    pub y1: f32,
}

/// The one field a document is checked on before the rest of it is read.
///
/// A producer that moved to a shape this tree does not know states another
/// `version`, and the answer has to say so. Reading the whole document first
/// would stop on the first field it no longer recognizes and answer a shape
/// error, which names neither the version nor the producer's problem, and it
/// would allocate a list it is about to drop.
#[derive(Debug, Deserialize)]
struct Header {
    /// The document version, which must be [`OLDEST`] to [`VERSION`].
    version: u32,
}

/// The box a scene covers, in the units of its list.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extent {
    /// The width of the plane.
    pub width: f32,
    /// The height of the plane.
    pub height: f32,
}

/// A document this tree read and accepted: the plane and the list to draw.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    /// The plane extent, in surface units.
    pub plane: Size,
    /// The box of the plane a window shows, in surface units.
    pub viewport: Option<Crop>,
    /// The primitives, in paint order.
    pub primitives: Arc<[Primitive]>,
}

/// Why a scene document could not be read.
#[derive(Debug, Display, Error, From)]
pub enum SceneError {
    /// The file could not be opened or read.
    #[display("failed to read the scene file")]
    Open(#[from] std::io::Error),
    /// The bytes are not a scene document.
    #[display("scene document: {_0}")]
    Parse(#[from] serde_json::Error),
    /// The document states a version this tree does not read.
    #[display("scene document version {found}, and this tree reads {oldest} to {newest}")]
    Version {
        /// The version the document states.
        found: u32,
        /// The oldest version this tree reads.
        oldest: u32,
        /// The newest version this tree reads.
        newest: u32,
    },
    /// The viewport is not a box this viewer can frame.
    #[display("scene viewport {x0},{y0} to {x1},{y1} is not a positive box")]
    Viewport {
        /// The left edge the document states.
        x0: f32,
        /// The top edge the document states.
        y0: f32,
        /// The right edge the document states.
        x1: f32,
        /// The bottom edge the document states.
        y1: f32,
    },
    /// The extent is not a plane this viewer can frame.
    #[display("scene plane {width}x{height} is not a positive extent")]
    Plane {
        /// The width the document states.
        width: f32,
        /// The height the document states.
        height: f32,
    },
}

impl Scene {
    /// Read a document's bytes into the plane and the list they describe.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::Parse`] when the bytes are not a document of this
    /// shape, [`SceneError::Version`] when they state another version,
    /// [`SceneError::Plane`] when the extent it names cannot be framed, and
    /// [`SceneError::Viewport`] when the box it names cannot be.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SceneError> {
        // The version first, on its own: a document of another shape is refused
        // by version and not by whichever field this tree happens to read
        // before it.
        let header: Header = serde_json::from_slice(data)?;
        Self::check_version(header.version)?;
        Self::checked(serde_json::from_slice(data)?)
    }

    /// Read a document that arrived as a JSON value into the plane and the list
    /// it describes.
    ///
    /// This is the entrance of a document a protocol carried, where the reader
    /// already holds the fields and the bytes of the same document exist only
    /// on the writer's side.
    ///
    /// # Errors
    ///
    /// Returns every error [`Self::from_bytes`] answers, and they mean the same
    /// thing: the version is read out of the value on its own, so a document of
    /// another shape is refused by version rather than by whichever field this
    /// tree reaches first.
    pub fn from_value(document: serde_json::Value) -> Result<Self, SceneError> {
        // The one field is copied out, so the list is never cloned to read a
        // number. A document that states no version at all falls through to the
        // shape's own error, which names the field it is missing.
        if let Some(found) = document.get("version").and_then(serde_json::Value::as_u64) {
            Self::check_version(u32::try_from(found).unwrap_or(u32::MAX))?;
        }
        Self::checked(serde_json::from_value(document)?)
    }

    /// Read the scene document at `path`.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::Open`] when the file cannot be read, and every
    /// error [`Self::from_bytes`] answers.
    pub fn read(path: &std::path::Path) -> Result<Self, SceneError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// Refuse a document whose version this tree does not read.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::Version`] when `found` is outside [`OLDEST`] to
    /// [`VERSION`].
    fn check_version(found: u32) -> Result<(), SceneError> {
        if (OLDEST..=VERSION).contains(&found) {
            return Ok(());
        }
        Err(SceneError::Version {
            found,
            oldest: OLDEST,
            newest: VERSION,
        })
    }

    /// Build the plane and the list of a document whose version already read.
    ///
    /// # Errors
    ///
    /// Returns [`SceneError::Plane`] when the extent it names cannot be framed
    /// and [`SceneError::Viewport`] when the box it names cannot be.
    fn checked(doc: SceneDoc) -> Result<Self, SceneError> {
        let plane = Size {
            w: doc.plane.width,
            h: doc.plane.height,
        };
        // A list with no plane to land on paints nothing and a fit over it has
        // no ratio to compute, so the document is refused before a window opens
        // rather than after a blank frame.
        if !(plane.w.is_finite() && plane.h.is_finite() && plane.w > 0.0 && plane.h > 0.0) {
            return Err(SceneError::Plane {
                width: plane.w,
                height: plane.h,
            });
        }
        // A box of no width, of no height or of no number gives the fit a ratio
        // it cannot compute, so the document is refused before a window opens
        // over it rather than after a blank frame.
        // The extents are checked and not only the edges: two corners far enough
        // apart overflow to an infinity, and a box of infinite width fits at
        // zero and paints nothing.
        let viewport = match doc.viewport {
            Some(view) => {
                let (w, h) = (view.x1 - view.x0, view.y1 - view.y0);
                let boxed = w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0;
                if !boxed {
                    return Err(SceneError::Viewport {
                        x0: view.x0,
                        y0: view.y0,
                        x1: view.x1,
                        y1: view.y1,
                    });
                }
                Some(region::crop(view.x0, view.y0, w, h))
            }
            None => None,
        };
        Ok(Self {
            plane,
            viewport,
            primitives: Arc::from(doc.primitives),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        draw::Color,
        scene::{FillRule, Paint, Stroke},
    };

    /// A document with one primitive of every kind a producer states, written as
    /// text a producer would send rather than as this tree's own output, so a
    /// renamed variant or a moved field fails here.
    ///
    /// A path is kurbo's `Vec<PathEl>`, so a stroked line is a `MoveTo` and a
    /// `LineTo` under one placement transform.
    const DOC: &str = r#"{
      "version": 1,
      "plane": { "width": 800.0, "height": 480.0 },
      "primitives": [
        { "Rect": { "rect": { "x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0 },
                    "paint": { "Solid": { "r": 10, "g": 20, "b": 30, "a": 255 } } } },
        { "PushClip": { "path": [],
                        "transform": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        "fill_rule": "EvenOdd" } },
        { "Path": { "path": [ { "MoveTo": { "x": 5.0, "y": 6.0 } },
                              { "LineTo": { "x": 5.0, "y": 9.0 } } ],
                    "transform": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    "fill": null,
                    "fill_rule": "NonZero",
                    "stroke": { "style": { "width": 2.0, "join": "Miter",
                                           "miter_limit": 4.0, "start_cap": "Butt",
                                           "end_cap": "Butt", "dash_offset": 0.0,
                                           "dash_pattern": [] },
                                "paint": { "Solid": { "r": 255, "g": 0, "b": 0, "a": 255 } } } } },
        { "PushLayer": { "alpha": 0.5 } },
        "PopLayer",
        "PopClip"
      ]
    }"#;

    /// A document reads into the plane and the list it states, and the
    /// vocabulary arrives as the variants a producer emits.
    #[test]
    fn a_document_reads_into_a_plane_and_a_list() {
        let scene = Scene::from_bytes(DOC.as_bytes()).expect("the document reads");
        assert_eq!(scene.plane, Size { w: 800.0, h: 480.0 });
        assert_eq!(scene.primitives.len(), 6);
        let Primitive::Rect { rect, paint } = &scene.primitives[0] else {
            panic!("the first primitive is the rect the document states");
        };
        assert_eq!((rect.x0, rect.y0, rect.x1, rect.y1), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(paint, &Paint::Solid(Color::new(10, 20, 30, 255)));
        // A solid stroke is a path with a style and a dashed one the same path
        // with a dash pattern, so the width and the dashes travel with it.
        let Primitive::Path {
            path,
            stroke: Some(stroke),
            fill_rule,
            ..
        } = &scene.primitives[2]
        else {
            panic!("the third primitive is the stroked path");
        };
        assert_eq!(path.elements().len(), 2, "a line is a move and a line");
        assert_eq!(*fill_rule, FillRule::NonZero);
        assert_eq!(stroke.style.width, 2.0);
        assert_eq!(&stroke.paint, &Paint::Solid(Color::new(255, 0, 0, 255)));
        // The stack replays in paint order, so the clip closes last.
        assert!(matches!(scene.primitives[5], Primitive::PopClip));
    }

    /// A file on disk reads the same as its bytes, which is the route the
    /// command line takes.
    #[test]
    fn a_file_reads_the_same_as_its_bytes() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let path = dir.path().join("scene.json");
        std::fs::write(&path, DOC).expect("the document writes");
        let scene = Scene::read(&path).expect("the document reads");
        let bytes = Scene::from_bytes(DOC.as_bytes()).expect("the same document reads");
        assert_eq!(scene, bytes);
    }

    /// A document that arrives as JSON reads the same as its bytes, which is
    /// the route a protocol takes: the fields are already parsed on the reader's
    /// side and no second copy of the document exists.
    #[test]
    fn a_value_reads_the_same_as_its_bytes() {
        let value: serde_json::Value = serde_json::from_str(DOC).expect("the document parses");
        let scene = Scene::from_value(value).expect("the same document reads");
        let bytes = Scene::from_bytes(DOC.as_bytes()).expect("the same document reads");
        assert_eq!(scene, bytes);
    }

    /// The version of a value is read before the shape, exactly as it is on the
    /// wire: a producer that moved first reads why, and not whichever field
    /// this tree stopped at.
    #[test]
    fn a_value_of_another_version_is_refused_by_version() {
        let value = serde_json::json!({
            "version": 3,
            "plane": { "width": 1.0, "height": 1.0 },
            "primitives": [ { "FancyShape": { "nothing": true } } ],
        });
        let err = Scene::from_value(value).expect_err("version 3 is not read here");
        assert_eq!(
            err.to_string(),
            format!("scene document version 3, and this tree reads {OLDEST} to {VERSION}")
        );
        // A value with no version at all fails on the shape, which names the
        // field it is missing rather than a version nobody stated.
        let err = Scene::from_value(serde_json::json!({
            "plane": { "width": 4.0, "height": 2.0 },
            "primitives": [],
        }))
        .expect_err("a document states its version");
        assert!(err.to_string().contains("version"), "{err}");
    }

    /// A version this tree does not read is refused by name, so a producer that
    /// moved first never half-paints.
    #[test]
    fn another_version_is_refused_by_name() {
        let doc = DOC.replace("\"version\": 1", "\"version\": 3");
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("version 3 is not read here");
        assert_eq!(
            err.to_string(),
            format!("scene document version 3, and this tree reads {OLDEST} to {VERSION}")
        );
    }

    /// The version is read before the shape, so a producer that moved first
    /// reads why, and not whichever field this tree stopped at.
    #[test]
    fn another_version_is_refused_before_the_shape_is_read() {
        let doc = r#"{ "version": 3, "plane": { "width": 1.0, "height": 1.0 },
                       "primitives": [ { "FancyShape": { "nothing": true } } ] }"#;
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("version 3 is not read here");
        assert!(
            err.to_string().contains("version 3"),
            "the answer names the version: {err}"
        );
    }

    /// A document that states the box its window shows reads as that box, and a
    /// document that states none frames the whole plane.
    #[test]
    fn a_viewport_names_the_box_the_window_shows() {
        let doc = r#"{ "version": 2, "plane": { "width": 800.0, "height": 480.0 },
                       "viewport": { "x0": 100.0, "y0": 40.0, "x1": 300.0, "y1": 160.0 },
                       "primitives": [] }"#;
        let scene = Scene::from_bytes(doc.as_bytes()).expect("the box reads");
        assert_eq!(
            scene.viewport,
            Some(region::crop(100.0, 40.0, 200.0, 120.0)),
            "the box reads as the crop the view fits"
        );
        // A version 1 document states no viewport at all, and that is the
        // whole plane.
        let old = Scene::from_bytes(DOC.as_bytes()).expect("a version 1 document reads");
        assert_eq!(old.viewport, None);
    }

    /// A box with no width, no height or no number is refused before a window
    /// opens over it: the fit has no ratio to compute.
    #[test]
    fn a_viewport_of_no_extent_is_refused() {
        let doc = r#"{ "version": 2, "plane": { "width": 8.0, "height": 4.0 },
                       "viewport": { "x0": 2.0, "y0": 1.0, "x1": 2.0, "y1": 3.0 },
                       "primitives": [] }"#;
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("a flat box frames nothing");
        assert!(
            err.to_string().contains("not a positive box"),
            "the answer names the box: {err}"
        );
        let wide = doc.replace("\"x1\": 2.0", "\"x1\": 9.0");
        assert!(
            Scene::from_bytes(wide.as_bytes()).is_ok(),
            "a box wider than the plane still frames"
        );
        // Two corners far enough apart overflow to an infinity, and a box of
        // infinite width would fit at zero and paint nothing.
        let huge = doc
            .replace("\"x0\": 2.0", "\"x0\": -3.0e38")
            .replace("\"x1\": 2.0", "\"x1\": 3.0e38");
        let err = Scene::from_bytes(huge.as_bytes()).expect_err("an infinite box frames nothing");
        assert!(err.to_string().contains("not a positive box"), "{err}");
    }

    /// A picture leaf stays off the wire: a document that states one is refused
    /// instead of painting every other primitive of the list.
    #[test]
    fn an_image_leaf_is_refused() {
        let doc = DOC.replace(
            r#"{ "Rect": { "rect": { "x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0 },
                    "paint": { "Solid": { "r": 10, "g": 20, "b": 30, "a": 255 } } } }"#,
            r#"{ "Image": { "pixels": {}, "transform": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] } }"#,
        );
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("an image leaf is not read");
        assert!(
            err.to_string().contains("unknown variant"),
            "a picture names itself: {err}"
        );
    }

    /// An unknown field is an error, so a producer and this reader agree on the
    /// whole shape and never on the part they both know.
    #[test]
    fn an_unknown_field_is_refused() {
        let doc = DOC.replace(
            "\"plane\": { \"width\": 800.0, \"height\": 480.0 }",
            "\"plane\": { \"width\": 800.0, \"height\": 480.0, \"dpi\": 96.0 }",
        );
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("dpi is not a field here");
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    /// The same guard holds *inside* a primitive, which is where a producer
    /// invents a field: dropping it silently would paint a picture nobody asked
    /// for.
    #[test]
    fn an_unknown_field_inside_a_primitive_is_refused() {
        let doc = r#"{ "version": 1, "plane": { "width": 4.0, "height": 2.0 },
                       "primitives": [ { "Rect": {
                           "rect": { "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0 },
                           "paint": { "Solid": { "r": 1, "g": 2, "b": 3, "a": 255 } },
                           "clip": { "unknown": true } } } ] }"#;
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("a rect carries no clip");
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    /// A plane that no fit can frame is refused before a window opens over it.
    #[test]
    fn a_plane_of_no_extent_is_refused() {
        let doc = DOC.replace("\"width\": 800.0", "\"width\": 0.0");
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("no plane, no picture");
        assert!(err.to_string().contains("not a positive extent"), "{err}");
    }

    /// An empty list is a plane with nothing on it, which is a legal document:
    /// a producer whose window shows nothing still states the plane it covers.
    #[test]
    fn an_empty_list_is_a_plane() {
        let doc = r#"{ "version": 1, "plane": { "width": 4.0, "height": 2.0 },
                       "primitives": [] }"#;
        let scene = Scene::from_bytes(doc.as_bytes()).expect("an empty plane reads");
        assert!(scene.primitives.is_empty());
        assert_eq!(scene.plane, Size { w: 4.0, h: 2.0 });
    }

    /// A picture leaf is refused on the way out as well as on the way in, so a
    /// list this tree writes never holds one and a reader never has to guess
    /// what a half-written document means.
    #[test]
    fn a_list_that_holds_a_picture_leaf_does_not_write() {
        use crate::{
            DecodedImage,
            dim::{ImgH, ImgW},
        };

        let list = [Primitive::Image {
            pixels: DecodedImage::from_exact(ImgW::new(1), ImgH::new(1), vec![0; 4]),
            transform: kurbo::Affine::IDENTITY,
        }];
        let err = serde_json::to_string(&list).expect_err("a picture is not a document");
        assert!(err.to_string().contains("Image"), "{err}");
    }

    /// A stroke style is kurbo's, so a document states the dash pattern the
    /// renderer draws and no second shape of the same style exists here.
    #[test]
    fn a_dash_pattern_crosses_the_wire() {
        let style = kurbo::Stroke::new(1.5).with_dashes(0.0, [3.0, 2.0]);
        let text = serde_json::to_string(&style).expect("a stroke style writes");
        let back: kurbo::Stroke = serde_json::from_str(&text).expect("and reads back");
        assert_eq!(back, style);
        assert_eq!(back.dash_pattern.len(), 2);
    }

    /// The type is field-named, so a fixture an agent reads names the same
    /// fields this tree does.
    #[test]
    fn a_written_document_reads_back() {
        let doc = SceneDoc {
            version: VERSION,
            plane: Extent {
                width: 12.0,
                height: 6.0,
            },
            viewport: None,
            primitives: vec![Primitive::PushLayer { alpha: 0.25 }],
        };
        let text = serde_json::to_string(&doc).expect("a document writes");
        assert!(
            !text.contains("viewport"),
            "a document that states no box writes none: {text}"
        );
        let scene = Scene::from_bytes(text.as_bytes()).expect("and reads back");
        assert_eq!(scene.primitives, Arc::from(doc.primitives));
        assert_eq!(scene.plane, Size { w: 12.0, h: 6.0 });
        assert_eq!(scene.viewport, None);
    }

    /// The stroke style's own serde is kurbo's, so a field this tree never
    /// writes (a join kind) still reads.
    #[test]
    fn a_stroke_reads_from_the_documents_shape() {
        let stroke = Stroke {
            style: kurbo::Stroke::new(3.0),
            paint: Paint::Solid(Color::new(1, 1, 1, 255)),
        };
        let text = serde_json::to_string(&stroke).expect("a stroke writes");
        let back: Stroke = serde_json::from_str(&text).expect("and reads back");
        assert_eq!(back, stroke);
    }
}
