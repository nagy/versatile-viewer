# Split one crate into a workspace of layers

Status: accepted. Implemented on the tree that carries this record.

## Context

The viewer was one package of about twenty thousand lines with twenty-nine
modules. Every module of that package reached every other: the image view
named the grid, the grid named the renderer, the renderer named the pixel
types, and the matrix of `crate::` edges carried mutual pairs: `dim` and `draw`,
`sign` and `svg`, and `surface` with each of its backends. Rust allows those
cycles inside one crate, and forbids exactly one of them across two.

Three costs came with the single crate.

**A rebuild took the whole tree.** An edit in `view.rs` (1685 lines) recompiled
all twenty-nine modules, because a crate is one compilation unit. An edit in
`grid.rs` (2203 lines) did the same.

**The seam of the library was not a seam.** `viewer::Viewer` exists so a host
can build the picture on its own device (ADR 0003, stage 2), and
`tests/gpu_offscreen.rs` proves it headless. But a headless consumer of the
decode stack could not take the decode stack alone: `use versatile_viewer::`
brought wgpu, winit, resvg, hayro and the MCP server with it, whatever the
caller asked for.

**Nothing stopped a layer from naming the layer above it.** The mutual pairs
were legal, so a module could reach in any direction and nobody noticed.

## Decision

Make the tree a cargo workspace of six libraries and one binary, with one
directed edge set.

```
vv-core    (no dependency)
vv-pixels  -> vv-core
vv-gpu     -> vv-core vv-pixels
vv-source  -> vv-core vv-pixels
vv-view    -> vv-core vv-pixels vv-gpu vv-source
vv-mcp     (no dependency)
vv-cli     -> vv-core vv-gpu vv-mcp vv-view (+ vv-source in tests)
```

The grouping answers the cycles instead of breaking them. Every mutual pair of
the old crate lands inside one new crate, so no trait moved and no call site
learned a new shape:

| Cycle | Home |
|---|---|
| `dim` and `draw` | `vv-core` |
| `sign` and `svg` | `vv-source` |
| `surface` and `book`, `pdf`, `convert`, `svg`, `thumbcache` | `vv-source` |

The root `Cargo.toml` holds no package. `crates/vv-cli` owns the `vv` binary,
its two integration tests and its two examples, and it holds no library
target, so the tests and the binary import each layer by its own crate name.
There is no facade crate: a facade would carry its own version, its own
changelog and its own build.

A crate root re-exports the lower modules its own API names
(`pub use vv_core::{dim, draw}`), so a module writes `crate::dim::ImgW` and an
intra-crate path never spells a crate name. The re-export line is short and
sits at the top of each `lib.rs`.

No `[workspace.dependencies]` table exists. Each member declares its own
dependencies beside the comment that says why the dependency is there, so the
comment and the crate that reads it never drift apart.

## Consequences

A rebuild of `vv-view` no longer compiles `vv-source`, and a rebuild of
`vv-core` still compiles everything, which is the reason `vv-core` stays small
and dependency-free.

`vv-source` holds no wgpu and no winit, so the whole `Surface` API runs
headless: a converter, a thumbnail server or a test can decode without a
device. The loader moved up into `vv-view` for that reason alone, because it
uploads through `texture::Uploader`.

Private fields stay private across the workspace. `DecodedImage` was read field
by field from `surface.rs` while both shared a crate root; that code now calls
`width()`, `height()` and `as_rgba()`, so the invariant of the type got
stricter, not looser.

`view` stays `pub(crate)` inside `vv-view`, so no item of the image view became
public. A `vv-view` and `vv-viewer` pair would have forced every one of them
`pub`, and that is why the two live together.

One fixture had to move. `pdf::MINIMAL_PDF` and `pdf::TWO_PAGE_PDF` served the
`pdf` tests and, from another crate, the `grid` and `loader` tests. A
`#[cfg(test)]` item belongs to the crate that compiles it, so the two choices
were a `#[doc(hidden)] pub` fixture in the library API or a copy in the test
build of `vv-view`. The copy won: a test byte string is not a contract of the
library, and `crates/vv-view/src/fixtures.rs` names the pdf copy it follows.

`DecodeError` lost its `Cache` variant: it named `thumbcache::CacheError` from
the layer above, and no call site ever built it. The cache reports its own
error.

Cross-crate inlining is the price of the split. Every member builds with the
same profile, so a hot small function that crosses a crate boundary wants
`lto` in the release profile; the checks and the binary are not hot paths of
one another.

The flake passes `--workspace --locked` to every cargo invocation, so one
derivation covers the whole graph, cargo never re-fingerprints it between the
deps drv and the package drv, and a lockfile that drifted fails the build
instead of fetching. `checks.gpu` names `-p vv-cli` alone, because the headless
paints live in the binary crate and the deps drv already holds every artifact
that crate needs.

## Alternatives

**Keep one crate and split modules only.** This is what the tree had. It fixes
nothing about rebuild cost, and the cycles stay legal, so the layering stays a
convention that no compiler enforces.

**Split by feature area (a crate per backend).** A crate for `svg`, one for
`pdf`, one for `convert`, one for `thumbcache`. Each of those would have to
break a cycle with `surface` through a trait crate, and each adds a manifest,
a public API and a version. The measured gain is small: an edit in `svg.rs`
recompiles `surface` either way, because `surface` calls it. The gain becomes
real only when a codec turns optional, and then the trait crate is the first
step of that work.

**A facade crate that re-exports every layer.** One import root for the binary,
and the old `versatile_viewer::` paths keep working unchanged. The cost is a
crate whose whole body is `pub use`, a version to bump, and a reader who cannot
tell which layer a path belongs to. The binary names its layers instead.

## Verification

`cargo test --workspace`, `cargo clippy --workspace --all-targets -- --deny
warnings`, `cargo doc --workspace --no-deps` with no warning, `nix fmt` and
`nix flake check`, which runs every member under one artifact set and the
headless paints on lavapipe.
