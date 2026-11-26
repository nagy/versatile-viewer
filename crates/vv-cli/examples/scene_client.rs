//! A program that draws its own picture and hands it to `vv`.
//!
//! ```text
//! scene_client      builds a document, speaks the protocol over stdio
//! └── vv --mcp      owns the window, the input and the paint
//! ```
//!
//! Run it with `cargo run --example scene_client`, after `cargo build` has made
//! the binary this example spawns (`VV_BIN` names another one).
//! `cargo run --example scene_client -- --out scene.json` writes the same
//! document instead and starts no viewer, so `vv --scene scene.json` paints it.
//!
//! Nothing of this tree is linked below: the document is JSON, the call is one
//! request line, and the answer names the plane. A program in another language
//! writes exactly this, which is why the JSON is spelled out by hand rather than
//! built from `scene::Primitive` — a producer that linked this tree would be a
//! second copy of it in its own build.
//!
//! The four steps a client takes:
//!
//! 1. spawn `vv --mcp`, whose stdout carries protocol frames and nothing else;
//! 2. write a `tools/call` line naming `vv_show_scene` and its `document`;
//! 3. read the answer line and check `isError`;
//! 4. keep speaking, or wait, while the window stays interactive.
//!
//! Revision `2026-07-28` is stateless: there is no `initialize` handshake and no
//! session, and every request carries the protocol version and the client's own
//! name in its `_meta` fields.

use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};

use anyhow::{Context, Result, bail};

/// The only protocol revision `vv` serves.
const PROTOCOL: &str = "2026-07-28";

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let out = match arguments.next().as_deref() {
        None => None,
        Some("--out") => Some(PathBuf::from(
            arguments.next().context("--out names a path")?,
        )),
        Some(other) => bail!("usage: scene_client [--out <path>], and {other} is neither"),
    };
    let document = document();
    match out {
        // The file route: the same document, painted by a launch instead of a
        // tool call.
        Some(path) => {
            std::fs::write(&path, serde_json::to_vec_pretty(&document)?)
                .with_context(|| format!("cannot write {}", path.display()))?;
            println!("wrote {}", path.display());
            println!("`vv --scene {}` paints it", path.display());
            Ok(())
        }
        None => show(&document),
    }
}

/// The document this producer draws.
///
/// A plane of its own units, a paper, a bar with a ramp, a polyline and a mark.
/// `version` is the shape of the document, and a viewer refuses a number it does
/// not know before it reads a field.
///
/// The list is stated in plane units and carries no scale, so the viewer decides
/// where those units land: a zoom is a change of the transform and never a new
/// document. A producer whose two axes are independent (a time axis and a price
/// axis, a map's two meridians) states a `viewport` box of the plane instead and
/// each axis takes its own ratio. This drawing is one picture with one aspect,
/// so it states no box and the viewer frames the whole plane uniformly.
fn document() -> serde_json::Value {
    serde_json::json!({
        "version": 2,
        "plane": { "width": 320.0, "height": 200.0 },
        "primitives": [
            // The paper. A producer paints what it drew on: a list that paints
            // none shows the window background through the gaps.
            { "Rect": {
                "rect": { "x0": 0.0, "y0": 0.0, "x1": 320.0, "y1": 200.0 },
                "paint": { "Solid": { "r": 250, "g": 248, "b": 244, "a": 255 } } } },
            // A bar with a ramp. `from` and `to` are in surface units, and the
            // stops are the colors along them.
            { "Path": {
                "path": [
                    { "MoveTo": { "x": 40.0, "y": 150.0 } },
                    { "LineTo": { "x": 280.0, "y": 150.0 } },
                    { "LineTo": { "x": 280.0, "y": 175.0 } },
                    { "LineTo": { "x": 40.0, "y": 175.0 } },
                    { "LineTo": { "x": 40.0, "y": 150.0 } }
                ],
                "transform": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "fill": { "Linear": {
                    "from": [40.0, 0.0],
                    "to": [280.0, 0.0],
                    "stops": [
                        { "offset": 0.0, "color": { "r": 40, "g": 90, "b": 200, "a": 255 } },
                        { "offset": 1.0, "color": { "r": 200, "g": 60, "b": 40, "a": 255 } }
                    ],
                    "spread": "Pad" } },
                "fill_rule": "NonZero",
                "stroke": null } },
            // The data: a polyline with a stroke and no fill, so its width, its
            // caps, its joins and its dashes travel with the shape.
            { "Path": {
                "path": [
                    { "MoveTo": { "x": 40.0, "y": 120.0 } },
                    { "LineTo": { "x": 90.0, "y": 95.0 } },
                    { "LineTo": { "x": 140.0, "y": 105.0 } },
                    { "LineTo": { "x": 190.0, "y": 60.0 } },
                    { "LineTo": { "x": 240.0, "y": 70.0 } },
                    { "LineTo": { "x": 280.0, "y": 30.0 } }
                ],
                "transform": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "fill": null,
                "fill_rule": "NonZero",
                "stroke": {
                    "style": {
                        "width": 2.0,
                        "join": "Round",
                        "miter_limit": 4.0,
                        "start_cap": "Round",
                        "end_cap": "Round",
                        "dash_offset": 0.0,
                        "dash_pattern": []
                    },
                    "paint": { "Solid": { "r": 30, "g": 30, "b": 34, "a": 255 } } } } },
            // The last point of the polyline, marked.
            { "Rect": {
                "rect": { "x0": 274.0, "y0": 24.0, "x1": 286.0, "y1": 36.0 },
                "paint": { "Solid": { "r": 200, "g": 60, "b": 40, "a": 255 } } } }
        ]
    })
}

/// Spawn `vv --mcp`, hand it the document and wait for the window to close.
///
/// The tool is asked for first, because a `vv` that does not serve it answers a
/// protocol error and no picture at all: a binary built before the tool existed
/// is a stale build far more often than it is a missing feature, so the message
/// names the binary and the command that rebuilds it.
fn show(document: &serde_json::Value) -> Result<()> {
    let binary = executable();
    let mut child = spawn(&binary)?;
    let mut stdin = child.stdin.take().context("vv closed its stdin")?;
    let mut reader = BufReader::new(child.stdout.take().context("vv closed its stdout")?);

    let listed = round_trip(
        &mut stdin,
        &mut reader,
        1,
        "tools/list",
        serde_json::json!({}),
    )?;
    if !advertises(&listed, "vv_show_scene") {
        let _ = child.kill();
        bail!(
            "{} serves no vv_show_scene, so it is older than this example: run `cargo build`, or \
             name a current one with VV_BIN",
            binary.display()
        );
    }

    let answer = round_trip(
        &mut stdin,
        &mut reader,
        2,
        "tools/call",
        serde_json::json!({
            "name": "vv_show_scene",
            "arguments": { "document": document },
        }),
    )?;
    // A request the server cannot route answers a protocol error and no result,
    // and a document it refuses answers a tool result with `isError` set.
    if let Some(error) = answer.get("error") {
        let _ = child.kill();
        bail!("{} answered an error: {error}", binary.display());
    }
    let result = &answer["result"];
    if result["isError"] == serde_json::json!(true) {
        let reason = &result["structuredContent"]["error"];
        let _ = child.kill();
        bail!("{} refused the document: {reason}", binary.display());
    }
    println!(
        "{} shows {} by {} plane units",
        binary.display(),
        result["structuredContent"]["width"],
        result["structuredContent"]["height"]
    );
    println!("pan and zoom there, and press q to quit");
    let status = child.wait().context("cannot wait for vv")?;
    if status.success() {
        Ok(())
    } else {
        bail!("vv stopped with {status}")
    }
}

/// Write one request and read the frame that answers it.
fn round_trip(
    stdin: &mut impl Write,
    reader: &mut impl BufRead,
    id: u64,
    method: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value> {
    // Every request carries the revision and the client's own name, because the
    // revision is stateless and there is no handshake to put them in.
    let mut params = params;
    params["_meta"] = serde_json::json!({
        "io.modelcontextprotocol/protocolVersion": PROTOCOL,
        "io.modelcontextprotocol/clientInfo": { "name": "scene_client", "version": "0" },
        "io.modelcontextprotocol/clientCapabilities": {},
    });
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    });
    writeln!(stdin, "{request}").context("cannot write the request")?;
    stdin.flush().context("cannot flush the request")?;
    read_answer(reader, id)
}

/// Read frames until one carries `id`.
///
/// A server may write notifications on the same stream, and a notification has
/// no `id`, so the reader skips every frame that names none.
fn read_answer(reader: &mut impl BufRead, id: u64) -> Result<serde_json::Value> {
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .context("cannot read an answer")?;
        if read == 0 {
            bail!("vv stopped before it answered");
        }
        let Ok(frame) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
            continue;
        };
        if frame.get("id") == Some(&serde_json::json!(id)) {
            return Ok(frame);
        }
    }
}

/// Does a `tools/list` answer name this tool?
fn advertises(listed: &serde_json::Value, tool: &str) -> bool {
    listed["result"]["tools"].as_array().is_some_and(|tools| {
        tools
            .iter()
            .any(|entry| entry["name"] == serde_json::json!(tool))
    })
}

/// The `vv` this example spawns.
///
/// `VV_BIN` names one, and otherwise the binary `cargo build` writes beside the
/// example's own directory is used, then `vv` from `PATH`.
fn executable() -> PathBuf {
    if let Some(named) = std::env::var_os("VV_BIN") {
        return PathBuf::from(named);
    }
    let beside = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().and_then(Path::parent).map(Path::to_path_buf))
        .map(|dir| dir.join("vv"));
    match beside {
        Some(path) if path.is_file() => path,
        _ => PathBuf::from("vv"),
    }
}

/// Start `vv --mcp` with its own pipes.
///
/// Its stdin and stdout belong to this conversation, so they are piped.
/// Its stderr is inherited: a diagnostic from the viewer belongs on the
/// terminal that ran the producer.
fn spawn(binary: &Path) -> Result<Child> {
    Command::new(binary)
        .arg("--mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| {
            format!(
                "cannot start {} (run `cargo build` first)",
                binary.display()
            )
        })
}
