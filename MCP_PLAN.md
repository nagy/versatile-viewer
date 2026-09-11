# MCP server plan — vv as an agent-driven display

Goal: start `vv` by hand, place it anywhere in the WM, leave it blank, and
let an AI agent connect and push images (later: video) into the window.

## Design decisions

### Transport: localhost HTTP (MCP Streamable HTTP), not stdio

stdio MCP servers are spawned *by the client* (the agent). That inverts our
control flow: we want the *user* to own the window process and the *agent*
to attach to it. So vv listens on `127.0.0.1:<port>` and speaks MCP
Streamable HTTP (JSON-RPC over HTTP POST; SSE optional later).

- Default port: 43077 (`VV_MCP_PORT` / `--mcp-port` to override; `0` =
  pick a free port, actual port printed to stderr so the agent config can
  point at it).
- Bind loopback only. Optional shared-secret: vv writes a per-run token to
  `${XDG_RUNTIME_DIR}/vv-$PID.token` (0600); agent sends it as
  `Authorization: Bearer`. Off by default (single-user desktop, loopback).

### SDK: `rmcp` (official Rust MCP SDK)

Tool surface is tiny; rmcp gives us schema/protocol handling for free.
It pulls tokio + axum — acceptable; they stay off the hot render path.
If the dependency weight becomes a problem, fallback is a hand-rolled
~200-line JSON-RPC handler (initialize / tools/list / tools/call only);
structure the code so the transport sits behind one trait-ish seam.

### Threading: listener thread → channel → raylib main loop

raylib wants the main thread. MCP handling is fully decoupled:

1. `main()` spawns a listener thread running the rmcp server (tokio
   runtime confined to that thread).
2. Tool calls enqueue a `McpCommand` on an `std::sync::mpsc::Sender`
   (clone into the server state) and block on a reply channel until the
   render loop consumes the command — tool result reports real
   success/failure (decode errors etc.), not "queued".
3. The raylib loop drains the mpsc each frame (non-blocking `try_recv`),
   applies commands, then answers the reply channel.

No locks around render state; the channel is the only crossing.

### New mode: `Mode::Idle`

`vv --mcp` (or `--mcp` alone) starts with an empty, blank window — no
directory scan, no grid. Status text hints: "vv: waiting for MCP agent
(port 43077)". While idle, normal keys still work (q quits, resize
refits nothing). First `show_image` jumps straight to image view; ESC
from there returns to Idle, not the grid.

Without `--mcp`, vv behaves exactly as today — zero change to the
interactive workflow.

## Tools (v1)

- `show_image(path: string) -> {width, height, ok}`
  Decode + display a local file through the existing `loader`
  (JXL/common/magic sniffing all reused). Fits all sides
  (Shift+W behavior). Full-res decode via the rayon pool, error text
  returned to the agent on failure.
- `show_image_bytes(data_base64: string, hint_ext?: string)`
  Decode from memory (image crate + jxl-oxide both decode from bytes
  already); agent can push screenshots or generated images without a
  temp file. Size cap ~64 MiB.
- `clear()` — back to Idle blank.
- `get_status() -> {mode, current_path?, image_size?, mcp_port}` — lets
  the agent verify it is talking to the right window and whether its
  last push landed.

Later (not v1): `show_url` (fetch + stream, rides milestone-7
progressive decoding), `set_zoom/pan`, video.

## Milestones

10. [ ] Idle mode + `--mcp` flag + listener thread + command channel +
       drain in the main loop (no tools yet beyond `get_status`).
11. [ ] `show_image` + `show_image_bytes` + `clear`; token file; port
       printing; VV_DEBUG traces MCP commands like grid events.
12. [ ] Polish: tool result ergonomics, error strings, README/AGENTS
       docs, agent-usage example (Claude Code / generic MCP config).
13. [ ] Later: `show_url` with progressive streaming; video once
       milestone 9 lands.

## Testing

- Unit: command-channel round-trip, base64 decode cap, port parsing.
- Integration (no GUI): run the JSON-RPC layer headless against a fake
  command sink; decode-from-bytes paths reuse existing loader tests.
- Manual: two-terminal smoke test — `vv --mcp` + curl JSON-RPC, then a
  real agent config.
