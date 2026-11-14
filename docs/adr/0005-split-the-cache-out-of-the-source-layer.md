# Split the on-disk cache out of the source layer

Status: accepted. Implemented on the tree that carries this record.

## Context

ADR 0004 split the single crate into layers and left `crates/vv-source/src/thumbcache.rs`
inside `vv-source`. That module grew to 1129 lines and carried three jobs that
share no code with a decoder:

- the fixed entry header and its `FORMAT` version,
- a JPEG XL encoder (`jixel`) and a second decoder setup (`jxl-oxide`),
- the cache home (`xdg`), the entry name (`sha2`) and the write budget.

Two crates read that module, and they read it for different reasons. `vv-source`
stores the answer of a converter rule under `$XDG_CACHE_HOME/vv/converted/`, and
`vv-view` stores the grid thumbnails under `$XDG_CACHE_HOME/vv/thumbnails/`. The
consumer that names it most often is the one that owns it least.

The split also carried a dependency leak in the other direction. `vv-source`
declared `jixel`, `xdg` and `sha2` for the cache alone. A reader of the decoder
manifest therefore found an encoder it never calls, and a change to the entry
format recompiled every decoder module.

## Decision

Move the module to `crates/vv-cache/src/lib.rs` as the crate `vv-cache`, over
`vv-core` and `vv-pixels`.

```
vv-core    (no dependency)
vv-pixels  -> vv-core
vv-cache   -> vv-core vv-pixels
vv-gpu     -> vv-core vv-pixels
vv-source  -> vv-core vv-pixels vv-cache
vv-view    -> vv-core vv-pixels vv-gpu vv-source vv-cache
vv-mcp     (no dependency)
vv-cli     -> vv-core vv-gpu vv-mcp vv-view (+ vv-source in tests)
```

Both entries move: `Cache::load_converted` and `Cache::store_converted` stay
methods of the same type, so no cache method lives apart from its directory
layout. The module keeps every constant, every test and its `CacheError`.

`jixel`, `xdg` and `sha2` move to the new manifest. `sha2` also stays in
`vv-source`, because `convert.rs` hashes its own cache key there and `vv-cache`
reads a key as an opaque string.

The two consumers re-export the crate as a module (`pub use vv_cache as cache;`),
so `crate::cache::Cache` reads the same in `vv-view/src/grid.rs` and
`vv-source/src/convert.rs` as `crate::dim` does.

One helper had to move down. `thumbcache.rs` converted a jxl-oxide framebuffer
to RGBA8 through `decode::fb_to_rgba`, which lives in the layer above. A cache
entry holds a JPEG XL codestream, so the entry decode needs the same conversion
the file decode needs. That conversion now lives in `vv-pixels` as
`rgba_from_samples(width, height, channels, samples)`, over plain f32 samples
and no decoder type, and `decode::fb_to_rgba` is a thin wrapper that names the
framebuffer fields. `vv-pixels` gains no dependency, and `vv-source` does not
hand a decoder to the layer below.

## Consequences

A change to the entry format compiles `vv-cache` and the two consumers, and it
compiles no decoder module that the change does not touch.

`vv-source` loses one module and three dependencies, so its manifest names
decoders and the converter seam alone.

The cache is now the one place that knows where a derived byte lives on disk,
which is what a future in-memory or network entry kind needs.

The edge count grows by two. `vv-cache` is small (1129 lines, mostly tests), so
the cost is two manifests and one more node in the graph, and the gain is a
crate whose whole job is one disk format.

## Alternatives

**Keep the module in `vv-source`.** That is what the tree had. It forces the
decoder layer to own an encoder, and it makes `vv-view` reach a module it does
not define through a crate that holds decoders it never calls.

**Split `convert` out as well.** `convert.rs` holds the rule table, process
supervision and the cache key, and it holds no decoder either. The change would
add a second crate and a `Surface` cycle for `open` that answers a
`Box<dyn Surface>`, so this record leaves it alone until the converter seam
grows a second consumer.

**A `vv-jxl` crate for the sample conversion.** One function does not pay for a
manifest. `vv-pixels` already owns every pixel conversion of the tree, and the
function takes plain samples, so the decoder crates stay out of it.
