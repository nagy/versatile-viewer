//! A scene document that a producer outside this process wrote.
//!
//! `vv --scene <file>` reads one: a version, the plane extent and the display
//! list, so a program that owns the layout of a picture hands over the shapes
//! and this viewer draws them at any zoom.
//! The list is the picture and there is no raster behind it, so a scene
//! document is not a [`Surface`](crate::surface::Surface): the two entrances
//! here answer a plane and a list, and nothing in this module can rasterize.
//!
//! # The document
//!
//! ```json
//! {
//!   "version": 1,
//!   "plane": { "width": 800.0, "height": 480.0 },
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

use crate::{scene::Primitive, surface::Size};

/// The one document version this tree reads.
///
/// A producer that moves to another shape bumps this, and an older viewer
/// refuses the document by name instead of reading the fields it recognizes.
pub const VERSION: u32 = 1;

/// One scene document as a producer states it.
///
/// The extent is the plane the list was written for, in the producer's own
/// units: the list is in those units and carries no scale, so the viewer frames
/// the whole extent and a zoom is a change of the transform alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDoc {
    /// The document version, which must be [`VERSION`].
    pub version: u32,
    /// The plane the list describes.
    pub plane: Extent,
    /// The primitives, in paint order.
    pub primitives: Vec<Primitive>,
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
    /// The document version, which must be [`VERSION`].
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
    #[display("scene document version {found}, and this tree reads {wanted}")]
    Version {
        /// The version the document states.
        found: u32,
        /// The version this tree reads.
        wanted: u32,
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
    /// shape, [`SceneError::Version`] when they state another version, and
    /// [`SceneError::Plane`] when the extent it names cannot be framed.
    pub fn from_bytes(data: &[u8]) -> Result<Self, SceneError> {
        // The version first, on its own: a document of another shape is refused
        // by version and not by whichever field this tree happens to read
        // before it.
        let header: Header = serde_json::from_slice(data)?;
        if header.version != VERSION {
            return Err(SceneError::Version {
                found: header.version,
                wanted: VERSION,
            });
        }
        let doc: SceneDoc = serde_json::from_slice(data)?;
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
        Ok(Self {
            plane,
            primitives: Arc::from(doc.primitives),
        })
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

    /// A version this tree does not read is refused by name, so a producer that
    /// moved first never half-paints.
    #[test]
    fn another_version_is_refused_by_name() {
        let doc = DOC.replace("\"version\": 1", "\"version\": 2");
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("version 2 is not read here");
        assert_eq!(
            err.to_string(),
            format!("scene document version 2, and this tree reads {VERSION}")
        );
    }

    /// The version is read before the shape, so a producer that moved first
    /// reads why, and not whichever field this tree stopped at.
    #[test]
    fn another_version_is_refused_before_the_shape_is_read() {
        let doc = r#"{ "version": 2, "plane": { "width": 1.0, "height": 1.0 },
                       "primitives": [ { "FancyShape": { "nothing": true } } ] }"#;
        let err = Scene::from_bytes(doc.as_bytes()).expect_err("version 2 is not read here");
        assert!(
            err.to_string().contains("version 2"),
            "the answer names the version: {err}"
        );
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
            primitives: vec![Primitive::PushLayer { alpha: 0.25 }],
        };
        let text = serde_json::to_string(&doc).expect("a document writes");
        let scene = Scene::from_bytes(text.as_bytes()).expect("and reads back");
        assert_eq!(scene.primitives, Arc::from(doc.primitives));
        assert_eq!(scene.plane, Size { w: 12.0, h: 6.0 });
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
