//! What a caller outside the process asks the view to draw.
//!
//! The MCP server, an editor plugin or a network front end all reach the
//! picture through one seam: the caller builds a [`Push`] and hands it to
//! [`Viewer::accept`](crate::viewer::Viewer::accept).
//! The binary drains its channel into that call once per frame, so the view
//! stays host-agnostic and no tool code touches a texture.
//!
//! A picture arrives as bytes or as a path, and the surface opens on a worker:
//! the main thread never decodes, and the tool call blocks on the reply the
//! worker sends once the plane exists and its first texture lands.
//! A decode failure travels back as the reply's `Err`, so a caller reads the
//! decoder's reason instead of a picture that never came.
//!
//! The view keeps no history here: a [`Push::Bytes`] replaces what shows, and
//! [`Push::Clear`] returns to the blank state the launch started from.

use std::{path::PathBuf, sync::mpsc::Sender};

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

/// The label of content that came from memory instead of a path.
pub const MEMORY_LABEL: &str = "mcp";

/// One request from a caller outside the process.
///
/// Every variant carries the channel its answer rides, so a caller blocks on
/// the result it asked for and the view never guesses where to report.
pub enum Push {
    /// Show a picture that exists only in memory.
    Bytes {
        /// The bytes as the caller holds them.
        data: Vec<u8>,
        /// Where the answer goes.
        reply: Sender<Result<Shown, String>>,
    },
    /// Show the file at this path, as one picture.
    ///
    /// A multi-page document opens as a book, exactly like a launch of the
    /// same file, and the grid stays as it was.
    Path {
        /// The file to open.
        path: PathBuf,
        /// Where the answer goes.
        reply: Sender<Result<Shown, String>>,
    },
    /// Return to the blank view the launch started from.
    Clear {
        /// Where the answer goes.
        reply: Sender<Status>,
    },
    /// Report what the view holds.
    Status {
        /// Where the answer goes.
        reply: Sender<Status>,
    },
}
