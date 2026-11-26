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
//! The server advertises four tools, and every one of them reaches the picture
//! through the [`Sink`] the binary installs.
//! `vv_show_image_bytes` shows a picture that never touches the disk, so
//! `vv --mcp` with no path runs entirely on pushed content.
//!
//! ## Adding a tool
//!
//! Add one entry to [`tools`] and one arm in `call_tool`.
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
//! ## The tools
//!
//! - `vv_show_image_bytes(data_base64)` decodes a picture from memory and
//!   shows it.
//!   The content decides the format, so the call carries no extension: a JPEG
//!   XL codestream, a PNG, an SVG document and a PDF page all route on their
//!   own bytes.
//!   The call answers once the plane exists, and the answer names the plane
//!   count and the size of plane 0.
//! - `vv_show_image(path)` opens a local file the same way, and a document
//!   opens as a book.
//! - `vv_clear()` returns to the blank view.
//! - `vv_get_status()` reports the mode, the plane count, the source label,
//!   the plane size and the scale.
//!
//! Every diagnostic goes to stderr, because stdout carries only protocol
//! frames.
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

use std::{borrow::Cow, io::Write as _, sync::Arc};

use base64::Engine as _;
use derive_more::{Display, Error, From};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler, ServiceExt as _,
    model::{Implementation, ProtocolVersion, ServerCapabilities, ServerConfig},
    service::RequestContext,
    transport::stdio,
};
// The answer vocabulary of a pushed picture, which this crate shares with the
// view layer: one shape for the reply a tool returns.
pub use vv_core::push::{Shown, Status};

/// The only protocol revision this server implements.
///
/// The revision dropped the `initialize` handshake (SEP-2567) and deprecated
/// the `logging` feature (SEP-2577).
const PROTOCOL: ProtocolVersion = ProtocolVersion::V_2026_07_28;

/// Instructions the server sends to a client.
const INSTRUCTIONS: &str = "Shows images. Push a picture from memory with `vv_show_image_bytes`, \
                            open a file with `vv_show_image`, read what shows with \
                            `vv_get_status`, and return to the blank view with `vv_clear`.";

/// The largest byte payload one `vv_show_image_bytes` call accepts.
///
/// The cap matches `convert::max_output`, so one push fits a photo-grade PNG
/// or a JPEG XL image and a runaway client stops before it eats the pipe.
pub const MAX_PUSH_BYTES: usize = 64 * 1024 * 1024;

/// Why the MCP server thread could not start.
#[derive(Debug, Display, Error, From)]
pub enum SpawnError {
    /// The operating system refused the thread.
    #[display("failed to spawn the MCP server thread")]
    Thread(std::io::Error),
}

/// What the server asks of the viewer.
///
/// The binary implements this over the viewer's own push seam, so this crate
/// names no GPU type, no texture and no window: a test drives the tools with a
/// sink of its own, and the protocol layer stays testable without a device.
pub trait Sink: Send + Sync + 'static {
    /// Show a picture that exists only in memory.
    ///
    /// # Errors
    ///
    /// The function answers the decoder's reason when the bytes hold no
    /// picture this viewer reads.
    fn show_bytes(&self, data: Vec<u8>) -> Result<Shown, String>;

    /// Show the file at `path` as one picture, or as a book when it holds
    /// pages.
    ///
    /// # Errors
    ///
    /// The function answers the open reason when the file cannot be read or
    /// decoded.
    fn show_path(&self, path: std::path::PathBuf) -> Result<Shown, String>;

    /// Return to the blank view.
    fn clear(&self) -> Status;

    /// Report what the view holds now.
    fn status(&self) -> Status;
}

/// The schema of one tool: a JSON object with the named properties.
fn schema(properties: serde_json::Value, required: &[&str]) -> rmcp::model::JsonObject {
    let mut schema = rmcp::model::JsonObject::new();
    schema.insert("type".to_owned(), serde_json::Value::from("object"));
    schema.insert("properties".to_owned(), properties);
    schema.insert("required".to_owned(), serde_json::json!(required));
    schema
}

/// The four hints a read-write tool states.
fn writes() -> rmcp::model::ToolAnnotations {
    rmcp::model::ToolAnnotations::new()
        .read_only(false)
        .destructive(false)
        .idempotent(true)
        .open_world(false)
}

/// The four hints a getter states.
fn reads() -> rmcp::model::ToolAnnotations {
    rmcp::model::ToolAnnotations::new()
        .read_only(true)
        .destructive(false)
        .idempotent(true)
        .open_world(false)
}

/// The tool surface the server advertises.
///
/// The list and the arms of `call_tool` move together, so a name here always
/// has one arm there.
fn tools() -> Vec<rmcp::model::Tool> {
    vec![
        rmcp::model::Tool::new(
            "vv_show_image_bytes",
            "Decode a picture from base64 bytes and show it. The content decides the format, so \
             the call carries no extension or file name. The window shows the picture as soon as \
             the decode finishes.",
            schema(
                serde_json::json!({
                    "data_base64": {
                        "type": "string",
                        "description": "The picture bytes, base64 encoded."
                    }
                }),
                &["data_base64"],
            ),
        )
        .with_annotations(writes()),
        rmcp::model::Tool::new(
            "vv_show_image",
            "Show the file at a local path. A multi-page document opens as a book.",
            schema(
                serde_json::json!({
                    "path": {
                        "type": "string",
                        "description": "The image, document or directory to show."
                    }
                }),
                &["path"],
            ),
        )
        .with_annotations(writes()),
        rmcp::model::Tool::new(
            "vv_clear",
            "Return to the blank view and forget the open picture.",
            schema(serde_json::json!({}), &[]),
        )
        .with_annotations(writes()),
        rmcp::model::Tool::new(
            "vv_get_status",
            "Report the open view: the mode, the plane count, the source label, the plane size \
             and the scale on each axis.",
            schema(serde_json::json!({}), &[]),
        )
        .with_annotations(reads()),
    ]
}

/// The MCP server.
///
/// The server keeps the sink the binary installed, so every tool call has one
/// route to the picture.
struct ViewerMcpServer<S: Sink> {
    sink: Arc<S>,
}

// The rmcp trait declares these methods as async fns, and a tool call answers
// without awaiting here: the sink blocks on its own reply.
// The lint fires on the trait shape, not on the code.
#[allow(clippy::unused_async_trait_impl)]
impl<S: Sink> ServerHandler for ViewerMcpServer<S> {
    /// Advertise the four tools at the sole supported revision.
    ///
    /// The server advertises no resources and no prompts.
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

    /// Return the tool list.
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
            tools: tools(),
            next_cursor: None,
            ttl_ms: cache_hints.then_some(0),
            cache_scope: cache_hints.then_some(rmcp::model::CacheScope::Public),
            meta: None,
        })
    }

    /// Route one tool call to the sink.
    ///
    /// A tool that this server does not advertise is a protocol error, because
    /// the request names something that cannot exist here.
    /// A decode, an open or a bad argument is a tool result, because the
    /// caller reads it.
    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, McpError> {
        let arguments = request.arguments.unwrap_or_default();
        let result = match request.name.as_ref() {
            "vv_show_image_bytes" => show_bytes(&*self.sink, &arguments),
            "vv_show_image" => show_path(&*self.sink, &arguments),
            "vv_clear" => Ok(shown_status(&self.sink.clear())),
            "vv_get_status" => Ok(shown_status(&self.sink.status())),
            other => {
                return Err(McpError::invalid_params(
                    format!("unknown tool: {other}"),
                    None,
                ));
            }
        };
        Ok(match result {
            Ok(value) => rmcp::model::CallToolResult::structured(value).into(),
            Err(reason) => rmcp::model::CallToolResult::structured_error(serde_json::json!({
                "error": reason
            }))
            .into(),
        })
    }
}

/// Decode the `data_base64` argument and push its bytes.
///
/// # Errors
///
/// The function answers a missing argument, a payload past
/// [`MAX_PUSH_BYTES`], bytes that are not base64, or the decode failure the
/// sink reports.
fn show_bytes(
    sink: &dyn Sink,
    arguments: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let text = string_argument(arguments, "data_base64")?;
    let encoded = text.trim();
    // The cap reads on the encoded text first, so a flood stops before the
    // decode allocates: base64 spends 4 bytes on every 3.
    if encoded.len() / 4 * 3 > MAX_PUSH_BYTES {
        return Err(format!("the payload passes the {MAX_PUSH_BYTES} byte cap"));
    }
    let data = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|err| format!("data_base64 holds no base64: {err}"))?;
    if data.len() > MAX_PUSH_BYTES {
        return Err(format!("the payload passes the {MAX_PUSH_BYTES} byte cap"));
    }
    let shown = sink.show_bytes(data)?;
    Ok(serde_json::json!({
        "planes": shown.planes,
        "width": shown.width,
        "height": shown.height,
    }))
}

/// Read the `path` argument and push that file.
///
/// # Errors
///
/// The function answers a missing argument or the open failure the sink
/// reports.
fn show_path(
    sink: &dyn Sink,
    arguments: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let path = string_argument(arguments, "path")?;
    let shown = sink.show_path(std::path::PathBuf::from(path))?;
    Ok(serde_json::json!({
        "planes": shown.planes,
        "width": shown.width,
        "height": shown.height,
    }))
}

/// Read one string argument.
///
/// # Errors
///
/// The function answers a missing or non-string value, naming the key.
fn string_argument(
    arguments: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<String, String> {
    arguments
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("the {key} argument holds a string"))
}

/// The JSON answer of a status report.
fn shown_status(status: &Status) -> serde_json::Value {
    serde_json::json!({
        "mode": if status.grid { "grid" } else { "image" },
        "planes": status.planes,
        "source": status.source,
        "width": status.width,
        "height": status.height,
        // The two axes carry their own factor, because a view that states the
        // box its window shows fills it on each axis: one number here would
        // name neither the picture nor the box.
        "scale": status.scale.map(|scale| serde_json::json!({
            "x": scale.x,
            "y": scale.y,
        })),
    })
}

/// Start the MCP stdio server on a background thread.
///
/// The function returns as soon as the thread runs.
/// `sink` is the route to the picture, which the binary implements over the
/// viewer's push seam.
///
/// # Errors
///
/// The function fails when the operating system refuses the thread.
pub fn spawn<S: Sink>(sink: S) -> Result<(), SpawnError> {
    let server = ViewerMcpServer {
        sink: Arc::new(sink),
    };
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
    use std::sync::Mutex;

    use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader, DuplexStream};

    use super::*;

    /// A sink that records what the tools asked for, so a test reads the route
    /// without a device or a window.
    #[derive(Default)]
    struct Fake {
        calls: Mutex<Vec<String>>,
        /// The bytes the last `show_bytes` call carried.
        bytes: Mutex<Vec<u8>>,
        /// The defect `show_bytes` and `show_path` answer, if any.
        failure: Option<String>,
    }

    impl Fake {
        fn failing(reason: &str) -> Self {
            Self {
                failure: Some(reason.to_owned()),
                ..Self::default()
            }
        }
    }

    impl Sink for Fake {
        fn show_bytes(&self, data: Vec<u8>) -> Result<Shown, String> {
            self.calls.lock().unwrap().push("bytes".to_owned());
            *self.bytes.lock().unwrap() = data;
            match &self.failure {
                Some(reason) => Err(reason.clone()),
                None => Ok(Shown {
                    planes: 1,
                    width: 3.0,
                    height: 2.0,
                }),
            }
        }

        fn show_path(&self, path: std::path::PathBuf) -> Result<Shown, String> {
            self.calls.lock().unwrap().push(path.display().to_string());
            match &self.failure {
                Some(reason) => Err(reason.clone()),
                None => Ok(Shown {
                    planes: 2,
                    width: 595.0,
                    height: 842.0,
                }),
            }
        }

        fn clear(&self) -> Status {
            self.calls.lock().unwrap().push("clear".to_owned());
            Status {
                grid: true,
                planes: 0,
                source: "mcp".to_owned(),
                width: 0.0,
                height: 0.0,
                scale: None,
            }
        }

        fn status(&self) -> Status {
            self.calls.lock().unwrap().push("status".to_owned());
            Status {
                grid: false,
                planes: 1,
                source: "mcp".to_owned(),
                width: 3.0,
                height: 2.0,
                scale: Some(vv_core::region::Scale::uniform(2.0)),
            }
        }
    }

    /// The server under test.
    fn server() -> ViewerMcpServer<Fake> {
        ViewerMcpServer {
            sink: Arc::new(Fake::default()),
        }
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
    ///
    /// The sink comes back with the stream, so a test reads what the tools
    /// asked the viewer for.
    fn serve(rt: &tokio::runtime::Runtime) -> (DuplexStream, Arc<Fake>) {
        let (server_side, client_side) = tokio::io::duplex(8192);
        let sink = Arc::new(Fake::default());
        let srv = ViewerMcpServer {
            sink: Arc::clone(&sink),
        };
        rt.block_on(async move {
            tokio::spawn(async move {
                if let Ok(running) = srv.serve(server_side).await {
                    let _ = running.waiting().await;
                }
            });
        });
        (client_side, sink)
    }

    /// Call one tool and read the response.
    fn call(
        rt: &tokio::runtime::Runtime,
        name: &str,
        arguments: &serde_json::Value,
    ) -> serde_json::Value {
        let (mut stream, _) = serve(rt);
        rt.block_on(round_trip(
            &mut stream,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": { "name": name, "arguments": arguments, "_meta": client_meta() },
            }),
        ))
    }

    /// `server/discover` names the sole revision and the server identity.
    #[test]
    fn discover_names_the_sole_revision() {
        let rt = runtime();
        let (mut stream, _) = serve(&rt);
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

    /// The server advertises the four tools, each with the four hints.
    #[test]
    fn tools_list_names_the_four_tools() {
        let rt = runtime();
        let (mut stream, _) = serve(&rt);
        let response = rt.block_on(round_trip(
            &mut stream,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": { "_meta": client_meta() },
            }),
        ));
        assert_eq!(response["result"]["resultType"], "complete");
        let tools = response["result"]["tools"].as_array().expect("a list");
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|tool| tool["name"].as_str())
            .collect();
        assert_eq!(
            names,
            [
                "vv_show_image_bytes",
                "vv_show_image",
                "vv_clear",
                "vv_get_status"
            ]
        );
        for tool in tools {
            let hints = &tool["annotations"];
            assert!(hints["readOnlyHint"].is_boolean(), "{tool}");
            assert!(hints["destructiveHint"].is_boolean(), "{tool}");
            assert!(hints["idempotentHint"].is_boolean(), "{tool}");
            assert!(hints["openWorldHint"].is_boolean(), "{tool}");
            assert_eq!(tool["inputSchema"]["type"], "object");
        }
        assert_eq!(tools[3]["annotations"]["readOnlyHint"], true);
        assert_eq!(tools[0]["annotations"]["readOnlyHint"], false);
    }

    /// The bytes of a push reach the sink, and the answer names the plane.
    #[test]
    fn show_image_bytes_pushes_the_bytes() {
        let rt = runtime();
        let (mut stream, sink) = serve(&rt);
        let payload = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3]);
        let response = rt.block_on(round_trip(
            &mut stream,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "vv_show_image_bytes",
                    "arguments": { "data_base64": payload },
                    "_meta": client_meta(),
                },
            }),
        ));
        assert_eq!(response["result"]["isError"], false, "{response}");
        assert_eq!(response["result"]["structuredContent"]["planes"], 1);
        assert_eq!(sink.bytes.lock().unwrap().as_slice(), &[1, 2, 3]);
    }

    /// A decode failure reaches the caller as a tool error, not as a protocol
    /// error.
    #[test]
    fn a_decode_failure_is_a_tool_result() {
        let rt = runtime();
        let (server_side, client_side) = tokio::io::duplex(8192);
        let srv = ViewerMcpServer {
            sink: Arc::new(Fake::failing("unknown image format")),
        };
        rt.block_on(async move {
            tokio::spawn(async move {
                if let Ok(running) = srv.serve(server_side).await {
                    let _ = running.waiting().await;
                }
            });
        });
        let mut stream = client_side;
        let response = rt.block_on(round_trip(
            &mut stream,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "vv_show_image",
                    "arguments": { "path": "/nowhere.png" },
                    "_meta": client_meta(),
                },
            }),
        ));
        assert!(response.get("error").is_none(), "{response}");
        assert_eq!(response["result"]["isError"], true);
        assert_eq!(
            response["result"]["structuredContent"]["error"],
            "unknown image format"
        );
    }

    /// Bytes that are not base64 fail the call before the sink sees them.
    #[test]
    fn a_bad_payload_is_a_tool_error() {
        let rt = runtime();
        let response = call(
            &rt,
            "vv_show_image_bytes",
            &serde_json::json!({ "data_base64": "not base64!" }),
        );
        assert_eq!(response["result"]["isError"], true, "{response}");
    }

    /// A status call reports the mode and the plane.
    #[test]
    fn status_reports_the_view() {
        let rt = runtime();
        let response = call(&rt, "vv_get_status", &serde_json::json!({}));
        let status = &response["result"]["structuredContent"];
        assert_eq!(status["mode"], "image");
        assert_eq!(status["planes"], 1);
        assert_eq!(status["source"], "mcp");
        assert_eq!(status["scale"]["x"], 2.0);
        assert_eq!(status["scale"]["y"], 2.0);
    }

    /// A call to a tool the server does not advertise fails as a client error.
    #[test]
    fn unknown_tool_is_a_client_error() {
        let rt = runtime();
        let (mut stream, _) = serve(&rt);
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
        let (mut stream, _) = serve(&rt);
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
