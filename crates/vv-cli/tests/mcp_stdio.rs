//! End-to-end test for the MCP stdio transport.
//!
//! The test re-runs its own binary as a child in child mode.
//! The child calls [`vv_mcp::mcp::spawn`] and parks, so the real
//! stdin and stdout wiring is exercised without a window.
//! The parent writes protocol frames to the child's stdin and reads the
//! responses from its stdout.
//!
//! The harness prints its own lines to the child's stdout, so the reader
//! skips every line that is not a JSON object.

use std::{
    env,
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
    time::Duration,
};

/// The protocol revision under test.
const PROTOCOL: &str = "2026-07-28";

/// The child mode marker.
const CHILD_ENV: &str = "VV_MCP_STDIO_CHILD";

/// The test that runs in the child process.
const TEST_NAME: &str = "stdio_serves_discover_and_the_tool_list";

/// The sink of the child process.
///
/// No window exists in this test, so the sink reports a blank view and refuses
/// every picture: the transport is what this test proves.
struct NoView;

impl vv_mcp::mcp::Sink for NoView {
    fn show_bytes(&self, _data: Vec<u8>) -> Result<vv_mcp::mcp::Shown, String> {
        Err("no window in this test".to_owned())
    }

    fn show_path(&self, _path: std::path::PathBuf) -> Result<vv_mcp::mcp::Shown, String> {
        Err("no window in this test".to_owned())
    }

    fn show_scene(&self, _document: serde_json::Value) -> Result<vv_mcp::mcp::Shown, String> {
        Err("no window in this test".to_owned())
    }

    fn clear(&self) -> vv_mcp::mcp::Status {
        blank()
    }

    fn status(&self) -> vv_mcp::mcp::Status {
        blank()
    }
}

/// The status of a view that shows nothing.
fn blank() -> vv_mcp::mcp::Status {
    vv_mcp::mcp::Status {
        grid: true,
        planes: 0,
        source: "mcp".to_owned(),
        width: 0.0,
        height: 0.0,
        scale: None,
    }
}

/// A request `_meta` block with the keys the revision requires.
fn meta() -> serde_json::Value {
    serde_json::json!({
        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
        "io.modelcontextprotocol/clientInfo": { "name": "stdio-test", "version": "0" },
        "io.modelcontextprotocol/clientCapabilities": {},
    })
}

/// Read the child's stdout until one line carries a JSON object.
///
/// The test harness prefixes its own progress text to the same line, so the
/// reader takes the text from the first `{` to the last `}`.
fn read_frame(reader: &mut impl BufRead) -> serde_json::Value {
    loop {
        let mut buf = String::new();
        let n = reader.read_line(&mut buf).expect("read child stdout");
        assert!(n > 0, "the child closed its stdout");
        let Some(start) = buf.find('{') else {
            continue;
        };
        let Some(end) = buf.rfind('}') else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&buf[start..=end]) else {
            continue;
        };
        if value.is_object() {
            return value;
        }
    }
}

/// Run one request and return the response with a matching `id`.
fn round_trip(
    writer: &mut impl Write,
    reader: &mut impl BufRead,
    request: &serde_json::Value,
) -> serde_json::Value {
    let mut line = serde_json::to_vec(request).expect("encode request");
    line.push(b'\n');
    writer.write_all(&line).expect("write request");
    writer.flush().expect("flush request");
    loop {
        let value = read_frame(reader);
        if value.get("id").is_some_and(|id| !id.is_null()) {
            return value;
        }
    }
}

/// Serve real stdio when the test runs in child mode.
fn maybe_run_as_child() -> bool {
    if env::var_os(CHILD_ENV).is_none() {
        return false;
    }
    vv_mcp::mcp::spawn(NoView).expect("spawn the MCP server");
    // The server thread owns stdin.
    // Parking keeps the process alive until the parent kills it.
    std::thread::sleep(Duration::from_secs(30));
    true
}

#[test]
fn stdio_serves_discover_and_the_tool_list() {
    if maybe_run_as_child() {
        return;
    }

    let exe = env::current_exe().expect("test binary path");
    let mut child = Command::new(exe)
        // `--nocapture` lets the server's stdout writes reach the pipe.
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the child");

    let mut writer = child.stdin.take().expect("child stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("child stdout"));

    let discover = round_trip(
        &mut writer,
        &mut reader,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": { "_meta": meta() },
        }),
    );
    assert_eq!(
        discover["result"]["supportedVersions"],
        serde_json::json!([PROTOCOL]),
        "the server advertises one revision"
    );
    assert_eq!(
        discover["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"],
        "vv"
    );

    let list = round_trip(
        &mut writer,
        &mut reader,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": { "_meta": meta() },
        }),
    );
    let names: Vec<&str> = list["result"]["tools"]
        .as_array()
        .expect("a list")
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect();
    assert_eq!(
        names,
        [
            "vv_show_image_bytes",
            "vv_show_image",
            "vv_show_scene",
            "vv_clear",
            "vv_get_status"
        ]
    );

    // A display list crosses the real pipe: the fields of the document reach
    // the sink, and the sink's refusal comes back as a tool result rather than
    // a protocol error.
    let scene = round_trip(
        &mut writer,
        &mut reader,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "vv_show_scene",
                "arguments": {
                    "document": {
                        "version": 2,
                        "plane": { "width": 4.0, "height": 2.0 },
                        "primitives": []
                    }
                },
                "_meta": meta(),
            },
        }),
    );
    assert_eq!(scene["result"]["isError"], true, "{scene}");
    assert_eq!(
        scene["result"]["structuredContent"]["error"],
        "no window in this test"
    );

    let call = round_trip(
        &mut writer,
        &mut reader,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": { "name": "nope", "arguments": {}, "_meta": meta() },
        }),
    );
    assert_eq!(call["error"]["code"], -32602);

    drop(writer);
    let _ = child.kill();
    let _ = child.wait();
}
