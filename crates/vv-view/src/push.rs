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
//!
//! The two answers of this seam, [`Shown`] and [`Status`], live in `vv-core`, so
//! this layer and the MCP server describe one shape and the binary moves an
//! answer through without rewriting a field.

use std::{path::PathBuf, sync::mpsc::Sender};

// The answer vocabulary of a pushed picture, which this module shares with
// `vv-mcp`: the pair lives in `vv-core`, so the view and the protocol layer
// describe one shape.
pub use vv_core::push::{Shown, Status};

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
