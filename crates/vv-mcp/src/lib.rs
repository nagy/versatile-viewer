//! The Model Context Protocol server of versatile-viewer.
//!
//! It speaks the stream transport over stdin and stdout, so the binary starts
//! it beside the window and never in place of it.
//! The crate reads no viewer state: a tool call that needs the picture reaches
//! it through a later seam, and this layer stays the protocol husk until then.

pub mod mcp;
