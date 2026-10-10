//! The MCP (Model Context Protocol) server.
//!
//! The binary starts this server when the user passes `--mcp`.
//! It runs on a background thread and owns stdin and stdout.
//! The viewer window runs as usual on the main thread.
//!
//! The server implements one protocol revision: `2026-07-28`.
//! That revision is stateless.
//! There is no `initialize` handshake and no session state.
//! Each request carries its protocol version and client capabilities in the
//! `io.modelcontextprotocol/*` `_meta` fields, and `server/discover` replaces
//! the handshake.
//! The server advertises that single revision, so an older client fails fast
//! on `UnsupportedProtocolVersion`.
//!
//! The server is a husk.
//! It advertises tools, but the tool list is empty.
//! It advertises no resources and no prompts.
//! Deprecated `logging` is not advertised.
//!
//! ## Adding a tool
//!
//! Add one entry to [`TOOLS`] and one arm in `call_tool`.
//! Two rules come from the community MCP server checklist.
//!
//! Name the tool `vv_<action>` in snake case.
//! An agent session often runs several MCP servers side by side, and a bare
//! `show_image` collides with another server and confuses tool ownership.
//! Renaming a tool later breaks the agent configs that name it.
//!
//! State the four hints on every tool.
//! `read_only_hint` is true only for a getter.
//! `destructive_hint` is true only when the tool removes data.
//! `idempotent_hint` is true when the same arguments repeat the same state.
//! `open_world_hint` is true only when the tool reaches outside the process.
//!
//! Return a decode or I/O failure as a tool result and not as a protocol
//! error, because the caller reads a tool result.
//! Reserve the protocol error for a request this server cannot route.
//!
//! ## Planned tools
//!
//! - `vv_show_image(path)` decodes and shows a local file.
//!   Hints: read-only false, destructive false, idempotent true, open-world
//!   false.
//! - `vv_show_image_bytes(data_base64, hint_ext)` decodes from memory.
//!   Same hints.
//! - `vv_clear()` returns to the launch view.
//!   Same hints.
//! - `vv_get_status()` reports the mode and the current image.
//!   Hints: read-only true, destructive false, idempotent true, open-world
//!   false.
//!
//! Every diagnostic goes to stderr, because stdout carries only protocol
//! frames.
//! `main.rs` also redirects the raylib trace log to stderr for the same
//! reason.
//!
//! Client configuration (stdio transport):
//!
//! ```json
//! {
//!   "command": "vv",
//!   "args": ["--mcp", "/path/to/images"]
//! }
//! ```
//!
//! The server runs a tokio current-thread runtime.
//! A failure to start or run the server never stops the viewer; the module
//! writes the problem to stderr and returns.

use std::{borrow::Cow, io::Write as _};

use derive_more::{Display, Error, From};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt as _,
    model::{Implementation, ProtocolVersion, ServerCapabilities, ServerConfig},
    service::RequestContext,
    transport::stdio,
};

/// The only protocol revision this server implements.
///
/// The revision dropped the `initialize` handshake (SEP-2567) and deprecated
/// the `logging` feature (SEP-2577).
const PROTOCOL: ProtocolVersion = ProtocolVersion::V_2026_07_28;

/// The tools the server advertises, as name and description.
///
/// The husk advertises none.
/// The module docstring states the naming and annotation rules for the first
/// entry, and `call_tool` carries the matching arm.
const TOOLS: &[(&str, &str)] = &[];

/// Instructions the server sends to a client.
const INSTRUCTIONS: &str = "Displays images. The tool surface is empty.";

/// Why the MCP server thread could not start.
#[derive(Debug, Display, Error, From)]
pub enum SpawnError {
    /// The operating system refused the thread.
    #[display("failed to spawn the MCP server thread")]
    Thread(std::io::Error),
}

/// Build one tool that takes no arguments.
///
/// The husk advertises no tools, so this only feeds the tests until the first
/// real tool lands.
/// A real tool replaces the empty schema with its own properties and adds its
/// four hints through [`rmcp::model::ToolAnnotations`].
fn tool(name: &str, description: &str) -> rmcp::model::Tool {
    let mut schema = rmcp::model::JsonObject::new();
    schema.insert("type".to_owned(), serde_json::Value::from("object"));
    rmcp::model::Tool::new(name.to_owned(), description.to_owned(), schema)
}

/// The MCP server.
///
/// The husk keeps no state.
struct ViewerMcpServer;

// The rmcp trait declares these methods as async fns, and the husk has
// nothing to await yet.
// The lint fires on the trait shape, not on the code.
#[allow(clippy::unused_async_trait_impl)]
impl ServerHandler for ViewerMcpServer {
    /// Advertise an empty tool surface at the sole supported revision.
    ///
    /// The server advertises no resources and no prompts.
    /// A client can still discover the server and see its identity.
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("vv", env!("CARGO_PKG_VERSION")).with_title("versatile-viewer"),
            )
            .with_instructions(INSTRUCTIONS)
            .with_protocol_version(PROTOCOL)
    }

    /// Only `2026-07-28` is implemented.
    ///
    /// An older revision fails fast on `UnsupportedProtocolVersion` instead of
    /// negotiating a legacy session this server does not implement.
    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        Cow::Borrowed(std::slice::from_ref(&PROTOCOL))
    }

    /// Return the tool list, which the husk leaves empty.
    ///
    /// The result states the cache hints that the revision defines, because
    /// the schema marks them as required there.
    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, McpError> {
        let cache_hints = context.protocol_version().is_some_and(|v| v >= PROTOCOL);
        Ok(rmcp::model::ListToolsResult {
            result_type: Some(rmcp::model::ResultType::COMPLETE),
            tools: TOOLS
                .iter()
                .map(|(name, description)| tool(name, description))
                .collect(),
            next_cursor: None,
            ttl_ms: cache_hints.then_some(0),
            cache_scope: cache_hints.then_some(rmcp::model::CacheScope::Public),
            meta: None,
        })
    }

    /// Reject every tool call, because the server advertises no tools.
    ///
    /// A client sees a protocol error and not an internal failure, because the
    /// request names a tool that cannot exist here.
    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, McpError> {
        Err(McpError::invalid_params(
            format!("unknown tool: {}", request.name),
            None,
        ))
    }
}

/// Start the MCP stdio server on a background thread.
///
/// The function returns as soon as the thread runs.
///
/// # Errors
///
/// The function fails when the operating system refuses the thread.
pub fn spawn() -> Result<(), SpawnError> {
    let server = ViewerMcpServer;
    std::thread::Builder::new()
        .name("mcp".to_owned())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    write_stderr(&format!("vv: mcp: no tokio runtime: {e}"));
                    return;
                }
            };
            rt.block_on(async move {
                match server.serve(stdio()).await {
                    Ok(running) => {
                        write_stderr("vv: mcp: serving 2026-07-28 on stdio");
                        if let Err(e) = running.waiting().await {
                            write_stderr(&format!("vv: mcp: server stopped: {e}"));
                        }
                    }
                    Err(e) => write_stderr(&format!("vv: mcp: server failed to start: {e}")),
                }
            });
        })
        .map(|_| ())
        .map_err(SpawnError::from)
}

/// Write one diagnostic line to stderr.
///
/// Stdout carries protocol frames only, so the module never writes diagnostics
/// there.
fn write_stderr(line: &str) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{line}");
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader, DuplexStream};

    use super::*;

    /// The server under test.
    fn server() -> ViewerMcpServer {
        ViewerMcpServer
    }

    /// A tokio runtime for the duplex tests.
    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
    }

    /// The protocol version and capabilities a stateless client sends in the
    /// `_meta` fields of every request.
    fn client_meta() -> serde_json::Value {
        serde_json::json!({
            "io.modelcontextprotocol/protocolVersion": PROTOCOL.as_str(),
            "io.modelcontextprotocol/clientInfo": { "name": "test", "version": "0" },
            "io.modelcontextprotocol/clientCapabilities": {},
        })
    }

    /// Send one request line and read the response line.
    ///
    /// The reader skips notifications, which have no `id`.
    async fn round_trip(
        stream: &mut DuplexStream,
        request: serde_json::Value,
    ) -> serde_json::Value {
        let mut line = serde_json::to_vec(&request).expect("encode request");
        line.push(b'\n');
        stream.write_all(&line).await.expect("write request");
        let mut reader = BufReader::new(&mut *stream);
        loop {
            let mut buf = String::new();
            let n = reader.read_line(&mut buf).await.expect("read response");
            assert!(n > 0, "server closed the connection");
            let value: serde_json::Value =
                serde_json::from_str(buf.trim()).expect("response is JSON");
            if value.get("id").is_some_and(|id| !id.is_null()) {
                return value;
            }
        }
    }

    /// Start the server on one end of a duplex pipe.
    fn serve(rt: &tokio::runtime::Runtime) -> DuplexStream {
        let (server_side, client_side) = tokio::io::duplex(8192);
        let srv = server();
        rt.block_on(async move {
            tokio::spawn(async move {
                if let Ok(running) = srv.serve(server_side).await {
                    let _ = running.waiting().await;
                }
            });
        });
        client_side
    }

    /// `server/discover` names the sole revision and the server identity.
    #[test]
    fn discover_names_the_sole_revision() {
        let rt = runtime();
        let mut stream = serve(&rt);
        let response = rt.block_on(round_trip(
            &mut stream,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": { "_meta": client_meta() },
            }),
        ));
        let result = &response["result"];
        assert_eq!(result["resultType"], "complete");
        assert_eq!(
            result["supportedVersions"],
            serde_json::json!([PROTOCOL.as_str()])
        );
        assert!(result["capabilities"]["tools"].is_object());
        assert!(result["capabilities"].get("resources").is_none());
        assert!(result["capabilities"].get("prompts").is_none());
        assert!(result["capabilities"].get("logging").is_none());
        assert_eq!(
            result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
            "vv"
        );
    }

    /// The husk advertises an empty tool list.
    #[test]
    fn tools_list_is_empty() {
        let rt = runtime();
        let mut stream = serve(&rt);
        let response = rt.block_on(round_trip(
            &mut stream,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": { "_meta": client_meta() },
            }),
        ));
        assert_eq!(response["result"]["tools"], serde_json::json!([]));
        assert_eq!(response["result"]["resultType"], "complete");
    }

    /// A call to a tool the server does not advertise fails as a client error.
    #[test]
    fn unknown_tool_is_a_client_error() {
        let rt = runtime();
        let mut stream = serve(&rt);
        let response = rt.block_on(round_trip(
            &mut stream,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "nope",
                    "arguments": {},
                    "_meta": client_meta(),
                },
            }),
        ));
        assert_eq!(response["error"]["code"], -32602);
        assert!(
            response["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("nope")),
            "the error names the tool"
        );
    }

    /// A request for another revision fails fast, because the server implements
    /// only `2026-07-28`.
    #[test]
    fn a_foreign_revision_is_refused() {
        let rt = runtime();
        let mut stream = serve(&rt);
        let mut meta = client_meta();
        meta["io.modelcontextprotocol/protocolVersion"] = serde_json::json!("2025-06-18");
        let response = rt.block_on(round_trip(
            &mut stream,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": { "_meta": meta },
            }),
        ));
        assert_eq!(response["error"]["code"], -32022);
        assert_eq!(response["error"]["data"]["requested"], "2025-06-18");
        assert_eq!(
            response["error"]["data"]["supported"],
            serde_json::json!([PROTOCOL.as_str()])
        );
    }

    /// `get_info` pins the revision and the identity.
    #[test]
    fn info_pins_the_revision() {
        let info = server().get_info();
        assert_eq!(info.protocol_version, PROTOCOL);
        assert_eq!(info.server_info.name, "vv");
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(info.server_info.title.as_deref(), Some("versatile-viewer"));
        assert!(info.capabilities.tools.is_some());
        assert!(info.capabilities.resources.is_none());
        assert!(info.capabilities.prompts.is_none());
        assert!(info.capabilities.logging.is_none());
    }

    /// The server supports exactly one revision.
    #[test]
    fn one_revision_only() {
        assert_eq!(server().supported_protocol_versions().as_ref(), &[PROTOCOL]);
        assert_eq!(PROTOCOL.as_str(), "2026-07-28");
        assert!(
            !PROTOCOL.has_initialize(),
            "the revision has no initialize handshake"
        );
    }
}
