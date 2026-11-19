//! The answer vocabulary of a pushed picture.
//!
//! A caller outside the process asks the view to draw something, then reads one
//! of two answers: a [`Shown`] once the picture lands, or a [`Status`] of what
//! the view holds now.
//! Two crates that never name each other describe that answer, because
//! `vv-view` builds it and `vv-mcp` reports it to a client, so the pair lives
//! here with the rest of the plain data.
//! `vv-view` re-exports the pair as `push::Shown` and `push::Status` beside the
//! request it answers, and `vv-mcp` re-exports them beside its tools.
//!
//! Neither type names a texture, a plane buffer or a window, so a caller builds
//! one with no device.

/// A picture the viewer opened from pushed content.
///
/// The extent measures plane 0, in surface units (pixels for a raster or a
/// vector base raster, points for a PDF page).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shown {
    /// How many planes the source holds (1 for a picture, one per page for a
    /// document).
    pub planes: usize,
    /// The width of plane 0, in surface units.
    pub width: f32,
    /// The height of plane 0, in surface units.
    pub height: f32,
}

/// What the view holds at the moment of the call.
#[derive(Debug, Clone, PartialEq)]
pub struct Status {
    /// True while the grid draws, which a blank view always does.
    pub grid: bool,
    /// The planes of the open source, or 0 when no source shows.
    pub planes: usize,
    /// The label of the open source: the path, or `mcp` for pushed bytes.
    pub source: String,
    /// The width of the open plane, in surface units, or 0 when nothing shows.
    pub width: f32,
    /// The height of the open plane, in surface units, or 0 when nothing
    /// shows.
    pub height: f32,
    /// The output pixels per surface unit, or None before the first frame
    /// computes the fit.
    pub scale: Option<f32>,
}
