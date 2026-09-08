# versatile-viewer (`vv`)

Fast image viewer (video planned). nsxiv meets mpv, with JPEG XL as a
first-class citizen.

JPEG XL decodes through [jxl-oxide], pure Rust — no libjxl C dependency —
with progressive streaming: a blurry full-size preview appears long before
all bytes arrive. Common formats (PNG, JPEG, WebP, GIF, BMP, TIFF, TGA)
decode through the [image] crate. Thumbnails render in a directory grid
that fills the window, decoded in parallel on a background pool.

[jxl-oxide]: https://crates.io/crates/jxl-oxide
[image]: https://crates.io/crates/image

## Run

```sh
nix run . -- <image-path-or-directory>   # single image or directory grid
```

Requires the usual X11/GL runtime libraries; the Nix build bakes them in
via rpath, so the result is self-contained. Outside Nix:
`cargo build --release` against a system raylib works too.

## Keys

| key                 | grid view                          | image view                          |
|---------------------|------------------------------------|-------------------------------------|
| `q`                 | quit                               | quit                                |
| `Enter` / `Esc`     | open selection / —                 | return to grid / (inert alone)      |
| `h j k l` / arrows  | move selection (auto-repeat)       | pan                                 |
| `Space` / `Backspace` | —                                | next / previous image               |
| `n` / `p`           | —                                  | next / previous image               |
| `g` / `G`           | first / last entry                 | first / last image                  |
| `W` / `Shift+W`     | —                                  | fit-down / fit-all                  |
| `e` / `Shift+E`     | —                                  | fit width / fit height              |
| `t`                 | —                                  | toggle fit-all / fill               |
| `+` / `-`           | zoom thumbnails                    | free zoom (25% steps, eased)        |
| mouse wheel         | scroll (when zoomed in)            | —                                   |

`Esc` never quits the program.

## Environment variables

| variable         | effect                                                                       |
|------------------|------------------------------------------------------------------------------|
| `VV_DEBUG=1`     | trace grid open/return events and raw key events to stderr                   |
| `VV_SLOW_STREAM=1` | dribble 4 KB chunks with 1 s pauses until the first preview renders        |
| `VV_BLUR_BG=1`   | image-view background is a tiny blurred copy of the viewed image             |
| `VV_BG_DIM=0..1` | brightness of the blurred background (default 0.6)                           |
| `VV_BLUR_PX=<n>` | long side of the blur texture, 8..=1024 (default 128; fewer = blurrier)      |

## Development

```sh
nix build                     # package
nix build .#checks.default    # tests
nix fmt                       # treefmt: rustfmt, taplo, nixfmt
nix develop                   # shell with LD_LIBRARY_PATH + rpath RUSTFLAGS
```

Project notes and the milestone plan live in [AGENTS.md](AGENTS.md);
recommendations in [RECOMMENDATIONS.org](RECOMMENDATIONS.org).

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE).
