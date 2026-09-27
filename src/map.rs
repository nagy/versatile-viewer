//! OpenStreetMap slippy-map view (phase 1: tiles, pan, animated integer
//! zoom). Web-Mercator math is hand-rolled — the whole projection fits in
//! a handful of pure functions (see osm.el for the same formulas in elisp),
//! so no projection crate is needed.
//!
//! Tile model (slippy map tiles): one tile is 256 px; zoom level z has
//! 2^z x 2^z tiles; global pixel coordinates at level z map a point to
//! `tile = pixel / 256`, with (0,0) at the top-left of the (Mercator-
//! flattened) world. The view keeps the map center in *fractional* global
//! pixel coordinates (like osm.el's wx/wy): panning adds pixels, zooming
//! rescales them by 2^(delta z).
//!
//! Zoom is integer-level (tile servers serve 2^z×2^z tiles), but the
//! transition animates: the current level's tiles are GPU-scaled by 2^±f
//! with an exponential ease (~150 ms, same feel as the image-view free
//! zoom), then the view snaps to the new level and its tiles load in.
//!
//! Downloads run on dedicated worker threads (ureq, blocking, rustls) that
//! pull from a shared priority queue: nearest-to-center tiles first, no
//! duplicates, no hammering. PNG decode happens on the worker; the main
//! thread only drains a channel and uploads textures — it never blocks
//! (this is exactly where osm.el stutters: elisp decodes on the UI thread).
//!
//! Phase 1 keeps tiles in an in-memory LRU texture cache. Disk cache,
//! parent-tile underzoom fill, server switching: phase 2.

// Pixel/coordinate math mixes f64 (global pixels, degrees) and f32
// (raylib's window units); the casts are deliberate and every value here is
// far below precision limits, so the pedantic cast lints are noise.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use std::{
    collections::{HashMap, HashSet, VecDeque},
    io::Read,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender, channel},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use raylib::{color::Color, consts::KeyboardKey, prelude::*};

use crate::upload_rgba;

/// Tile edge length in pixels (standard slippy-map size).
pub const TILE: f64 = 256.0;
/// Highest zoom level served by tile.openstreetmap.org.
pub const MAX_ZOOM: u8 = 19;
/// Lowest zoom worth showing (osm.el default min-zoom; whole world at z=2).
pub const MIN_ZOOM: u8 = 2;
/// Web-Mercator latitude limit: the projection is unbounded in y toward the
/// poles; tiles are cut at lat = ±atan(sinh(pi)) = ±85.0511°.
pub const MAX_LAT: f64 = 85.051_128_779_806_6;

/// Tile usage policy requires an identifying User-Agent; browser UAs are
/// explicitly frowned upon. Single personal user, so volume is negligible.
const USER_AGENT: &str = "vv/0.1 (personal single-user image/map viewer)";
/// Hard cap on one tile response, so a misbehaving server cannot balloon
/// memory (a real 256x256 PNG is ~10–100 KB).
const MAX_TILE_BYTES: u64 = 8 << 20;
/// Connect timeout for tile requests.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Parallel tile download threads. Small on purpose: the OSM tile usage
/// policy asks for modest concurrency; three threads keep up with panning.
const WORKERS: usize = 3;
/// Upper bound on queued requests (stale ones sink in priority and get cut).
const MAX_QUEUED: usize = 256;
/// Tile texture cache cap (GPU): 512 * 256 * 256 * 4 B = 128 MiB. Evicted
/// down to `EVICT_TO` when exceeded.
const TILE_CACHE_CAP: usize = 512;
const EVICT_TO: usize = TILE_CACHE_CAP - 64;
/// A failed tile is not retried for this long (transient network hiccups
/// must not turn into per-frame hammering).
const RETRY_AFTER: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------- slippy math

/// Global pixel x for `lon` (degrees) at zoom level `zoom`.
pub fn lon_to_x(lon: f64, zoom: u8) -> f64 {
    (lon + 180.0) / 360.0 * TILE * (1u32 << zoom) as f64
}

/// Global pixel y for `lat` (degrees) at zoom level `zoom`. Latitude is
/// clamped to the Mercator limit.
pub fn lat_to_y(lat: f64, zoom: u8) -> f64 {
    let lat = lat.clamp(-MAX_LAT, MAX_LAT).to_radians();
    (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / std::f64::consts::PI) / 2.0
        * TILE
        * (1u32 << zoom) as f64
}

/// Longitude (degrees) for global pixel `x` at zoom level `zoom`.
pub fn x_to_lon(x: f64, zoom: u8) -> f64 {
    x / (TILE * (1u32 << zoom) as f64) * 360.0 - 180.0
}

/// Latitude (degrees) for global pixel `y` at zoom level `zoom`.
pub fn y_to_lat(y: f64, zoom: u8) -> f64 {
    let n = y / (TILE * (1u32 << zoom) as f64);
    let lat = (std::f64::consts::PI * (1.0 - 2.0 * n)).sinh().atan();
    lat.to_degrees()
}

/// Slippy URL for a tile (`{z}/{x}/{y}.png` path; osm.el's %z/%x/%y format).
pub fn tile_url(z: u8, x: u32, y: u32) -> String {
    format!("https://tile.openstreetmap.org/{z}/{x}/{y}.png")
}

/// A tile in the world grid: zoom level + integer tile indices.
pub type TileKey = (u8, u32, u32);

/// Visible tile index ranges for a viewport. `(cx, cy)` is the center in
/// global pixels at `level`; the world is drawn at `scale` (1.0 normally,
/// fractional during a zoom animation). Clamped to the world; nothing
/// exists outside it.
pub fn visible_tiles(
    level: u8,
    cx: f64,
    cy: f64,
    win_w: f32,
    win_h: f32,
    scale: f64,
) -> (i64, i64, i64, i64) {
    let n = (1u32 << level) as i64;
    let half_w = win_w as f64 / (2.0 * scale);
    let half_h = win_h as f64 / (2.0 * scale);
    let tx0 = (((cx - half_w) / TILE).floor() as i64).clamp(0, n - 1);
    let tx1 = (((cx + half_w) / TILE).floor() as i64).clamp(0, n - 1);
    let ty0 = (((cy - half_h) / TILE).floor() as i64).clamp(0, n - 1);
    let ty1 = (((cy + half_h) / TILE).floor() as i64).clamp(0, n - 1);
    (tx0, tx1, ty0, ty1)
}

/// Parse the optional `--osm [z/lat/lon]` argument (e.g. `15/52.52/13.405`).
/// Latitude is clamped to the Mercator limit, longitude to ±180.
pub fn parse_pos(s: &str) -> Result<(u8, f64, f64)> {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 3 {
        bail!("--osm position must be z/lat/lon (e.g. 15/52.52/13.405), got {s:?}");
    }
    let zoom: u8 = parts[0]
        .parse()
        .with_context(|| format!("bad zoom in {s:?}"))?;
    if zoom > MAX_ZOOM {
        bail!("zoom {zoom} exceeds the tile servers' maximum ({MAX_ZOOM})");
    }
    let lat: f64 = parts[1]
        .parse()
        .with_context(|| format!("bad latitude in {s:?}"))?;
    let lon: f64 = parts[2]
        .parse()
        .with_context(|| format!("bad longitude in {s:?}"))?;
    if !(-MAX_LAT..=MAX_LAT).contains(&lat) {
        bail!("latitude {lat} outside the Mercator range (±{MAX_LAT})");
    }
    if !(-180.0..=180.0).contains(&lon) {
        bail!("longitude {lon} outside ±180");
    }
    Ok((zoom, lat, lon))
}

// ---------------------------------------------------------------- tile fetcher

/// One finished tile fetch: decoded pixels (or a failure), delivered to the
/// main thread which owns the GL context (texture upload happens there).
pub enum TileMsg {
    Done {
        key: TileKey,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
    Failed {
        key: TileKey,
    },
}

struct FetchShared {
    queue: VecDeque<TileKey>,
    in_flight: HashSet<TileKey>,
}

/// Threaded tile downloader. Workers pull from a priority queue (sorted by
/// the main thread's per-frame visibility order), fetch over a shared agent,
/// decode the PNG off-thread and ship RGBA over the channel. Dropping the
/// fetcher signals the workers to exit.
pub struct TileFetcher {
    shared: Arc<Mutex<FetchShared>>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<TileMsg>,
}

impl TileFetcher {
    fn new() -> TileFetcher {
        let (tx, rx) = channel();
        let shared = Arc::new(Mutex::new(FetchShared {
            queue: VecDeque::new(),
            in_flight: HashSet::new(),
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        for _ in 0..WORKERS {
            let tx = tx.clone();
            let shared = shared.clone();
            let cancel = cancel.clone();
            // Detached: exits on cancel or when the receiver is gone.
            std::thread::spawn(move || fetch_worker(&tx, &shared, &cancel));
        }
        TileFetcher { shared, cancel, rx }
    }

    /// Make sure these keys (in priority order, best first) are queued for
    /// download. Already-queued/in-flight keys keep their earlier place;
    /// unknown keys sink; the queue is capped so stale requests cannot pile
    /// up during fast panning.
    fn ensure(&self, keys: &[TileKey]) {
        let mut s = self.shared.lock().unwrap();
        for k in keys {
            if !s.in_flight.contains(k) && !s.queue.contains(k) {
                s.queue.push_back(*k);
            }
        }
        let rank: HashMap<TileKey, usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
        s.queue
            .make_contiguous()
            .sort_by_key(|k| rank.get(k).copied().unwrap_or(usize::MAX));
        while s.queue.len() > MAX_QUEUED {
            s.queue.pop_back();
        }
    }

    /// Poll the next finished tile without blocking (main thread).
    fn try_recv(&self) -> Option<TileMsg> {
        self.rx.try_recv().ok()
    }
}

impl Drop for TileFetcher {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

fn fetch_worker(tx: &Sender<TileMsg>, shared: &Mutex<FetchShared>, cancel: &AtomicBool) {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(CONNECT_TIMEOUT)
        .build();
    loop {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        let key = {
            let mut s = shared.lock().unwrap();
            match s.queue.pop_front() {
                Some(k) => {
                    s.in_flight.insert(k);
                    Some(k)
                }
                None => None,
            }
        };
        let Some(key) = key else {
            // Idle: short poll. Cheap, and keeps startup latency tiny.
            std::thread::sleep(Duration::from_millis(15));
            continue;
        };
        let msg = match fetch_tile(&agent, key) {
            Ok((rgba, width, height)) => TileMsg::Done {
                key,
                rgba,
                width,
                height,
            },
            Err(err) => {
                // One stderr line per failure is fine for a single-user tool.
                eprintln!("vv: tile {key:?}: {err:#}");
                TileMsg::Failed { key }
            }
        };
        let _ = tx.send(msg);
        shared.lock().unwrap().in_flight.remove(&key);
        if cancel.load(Ordering::Relaxed) {
            return;
        }
    }
}

fn fetch_tile(agent: &ureq::Agent, key: TileKey) -> Result<(Vec<u8>, u32, u32)> {
    let (z, x, y) = key;
    let resp = agent
        .get(&tile_url(z, x, y))
        .set("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("GET {}", tile_url(z, x, y)))?;
    let mut bytes = Vec::new();
    resp.into_reader()
        .take(MAX_TILE_BYTES)
        .read_to_end(&mut bytes)
        .context("reading tile body")?;
    let img = image::load_from_memory(&bytes)
        .with_context(|| {
            format!(
                "decoding tile {} PNG ({} bytes)",
                tile_url(z, x, y),
                bytes.len()
            )
        })?
        .into_rgba8();
    let (width, height) = img.dimensions();
    Ok((img.into_raw(), width, height))
}

// ---------------------------------------------------------------- map state

/// One tile in the view's cache: GPU texture (uploaded on the main thread)
/// or a failure marker with a retry cooldown timestamp.
struct Tile {
    tex: Option<Texture2D>,
    failed_at: Option<Instant>,
    /// Frame of last use; eviction takes the oldest.
    last_used: u64,
}

/// Zoom transition animation: from-level tiles scale by 2^(dir*f) while f
/// eases 0→1; at f=1 the view snaps to level `from + dir` (center pixels
/// rescale by the same factor).
struct Anim {
    from: u8,
    dir: i8,
    f: f64,
    /// Center in from-level pixel coordinates (fixed during the anim: the
    /// geographic center stays under the window center).
    cx: f64,
    cy: f64,
}

/// Full slippy-map view state. Owns its fetcher and tile cache.
pub struct MapState {
    zoom: u8,
    /// Map center in global pixel coordinates at `zoom` (fractional).
    cx: f64,
    cy: f64,
    target_cx: f64,
    target_cy: f64,
    /// Launch position; `g` returns here.
    home: (u8, f64, f64), // zoom, lat, lon
    anim: Option<Anim>,
    tiles: HashMap<TileKey, Tile>,
    fetcher: TileFetcher,
    frame: u64,
}

impl MapState {
    fn new(pos: Option<(u8, f64, f64)>) -> MapState {
        let (zoom, lat, lon) = pos.unwrap_or((
            // osm.el's default zoom: 15 (street level).
            15,
            // Default home: Berlin center (arbitrary but recognizable; osm.el
            // starts at a similar default).
            52.5195, 13.4037,
        ));
        let cx = lon_to_x(lon, zoom);
        let cy = lat_to_y(lat, zoom);
        MapState {
            zoom,
            cx,
            cy,
            target_cx: cx,
            target_cy: cy,
            home: (zoom, lat, lon),
            anim: None,
            tiles: HashMap::new(),
            fetcher: TileFetcher::new(),
            frame: 0,
        }
    }

    /// The center as (lat, lon), valid in every frame (during an animation
    /// the from-level coordinates denote the same geographic point).
    fn center_latlon(&self) -> (f64, f64) {
        match &self.anim {
            Some(a) => (y_to_lat(a.cy, a.from), x_to_lon(a.cx, a.from)),
            None => (y_to_lat(self.cy, self.zoom), x_to_lon(self.cx, self.zoom)),
        }
    }

    /// Queue every visible tile (plus, during an animation, the target
    /// level's tiles so they are ready at settle) unless cached or recently
    /// failed. Priority: nearest-to-center first.
    fn request_tiles(&self, win_w: f32, win_h: f32) {
        let mut keys: Vec<TileKey> = Vec::new();
        let push = |keys: &mut Vec<TileKey>, level: u8, cx: f64, cy: f64, scale: f64| {
            let (tx0, tx1, ty0, ty1) = visible_tiles(level, cx, cy, win_w, win_h, scale);
            let n = (1u32 << level) as i64;
            // Center tile first, then outward: Chebyshev distance from
            // the center tile.
            let ctx = (cx / TILE).floor().clamp(0.0, (n - 1) as f64) as i64;
            let cty = (cy / TILE).floor().clamp(0.0, (n - 1) as f64) as i64;
            let mut v: Vec<TileKey> = (ty0..=ty1)
                .flat_map(|ty| (tx0..=tx1).map(move |tx| (level, tx as u32, ty as u32)))
                .collect();
            v.sort_by_key(|k| {
                let (_, tx, ty) = *k;
                (tx as i64 - ctx).abs().max((ty as i64 - cty).abs())
            });
            keys.extend(v);
        };
        match &self.anim {
            Some(a) => {
                // Target level first (needed at settle), base level behind
                // (mostly cached already; covers stragglers).
                let m = 2f64.powi(a.dir as i32);
                push(
                    &mut keys,
                    (a.from as i8 + a.dir) as u8,
                    a.cx * m,
                    a.cy * m,
                    1.0,
                );
                push(&mut keys, a.from, a.cx, a.cy, 2f64.powf(a.dir as f64 * a.f));
            }
            None => push(&mut keys, self.zoom, self.cx, self.cy, 1.0),
        }
        // Skip cached and cooldown-fresh failed tiles.
        let mut fresh: Vec<TileKey> = Vec::with_capacity(keys.len());
        for k in keys {
            match self.tiles.get(&k) {
                Some(t) if t.tex.is_some() => {}
                Some(t) if t.failed_at.is_some_and(|at| at.elapsed() < RETRY_AFTER) => {}
                _ => fresh.push(k),
            }
        }
        self.fetcher.ensure(&fresh);
    }

    /// Drain finished downloads (texture upload needs the main thread / GL
    /// context), ease pan, advance the zoom animation, evict old tiles.
    fn update(&mut self, rl: &mut RaylibHandle, thread: &RaylibThread, win_w: f32, win_h: f32) {
        self.frame += 1;

        while let Some(msg) = self.fetcher.try_recv() {
            match msg {
                TileMsg::Done {
                    key,
                    rgba,
                    width,
                    height,
                } => match upload_rgba(rl, thread, &rgba, width, height) {
                    Ok(tex) => {
                        self.tiles.insert(
                            key,
                            Tile {
                                tex: Some(tex),
                                failed_at: None,
                                last_used: self.frame,
                            },
                        );
                    }
                    Err(e) => {
                        eprintln!("vv: tile {key:?}: texture upload failed: {e}");
                        self.tiles.entry(key).and_modify(|t| {
                            t.failed_at = Some(Instant::now());
                            t.last_used = self.frame;
                        });
                    }
                },
                TileMsg::Failed { key } => {
                    self.tiles
                        .entry(key)
                        .and_modify(|t| {
                            t.failed_at = Some(Instant::now());
                            t.last_used = self.frame;
                        })
                        .or_insert(Tile {
                            tex: None,
                            failed_at: Some(Instant::now()),
                            last_used: self.frame,
                        });
                }
            }
        }

        // Pan easing (same exponential ease as the image view; taps glide,
        // holds scroll smoothly).
        let alpha = 1.0 - (-rl.get_frame_time() as f64 / 0.05).exp();
        self.cx += (self.target_cx - self.cx) * alpha;
        self.cy += (self.target_cy - self.cy) * alpha;
        if (self.target_cx - self.cx).abs() < 0.25 {
            self.cx = self.target_cx;
        }
        if (self.target_cy - self.cy).abs() < 0.25 {
            self.cy = self.target_cy;
        }
        self.clamp_center();

        // Zoom animation progress: exponential ease, snap when done.
        let settled = match &mut self.anim {
            Some(a) => {
                a.f += (1.0 - a.f) * alpha;
                1.0 - a.f < 0.001
            }
            None => false,
        };
        if settled {
            let a = self.anim.take().unwrap();
            self.zoom = (a.from as i8 + a.dir) as u8;
            let m = 2f64.powi(a.dir as i32);
            self.cx = a.cx * m;
            self.cy = a.cy * m;
            self.target_cx = self.cx;
            self.target_cy = self.cy;
            self.clamp_center();
        }

        // LRU eviction: drop the oldest unused tiles when the cache exceeds
        // its cap. Dropping a Texture2D unloads it from the GPU; this runs
        // on the main thread, where the GL context lives.
        if self.tiles.len() > TILE_CACHE_CAP {
            let mut by_age: Vec<(TileKey, u64)> =
                self.tiles.iter().map(|(k, t)| (*k, t.last_used)).collect();
            by_age.sort_unstable_by_key(|(_, u)| *u);
            let excess = self.tiles.len() - EVICT_TO;
            for (k, _) in by_age.into_iter().take(excess) {
                self.tiles.remove(&k); // Texture2D drop = UnloadTexture
            }
        }

        self.request_tiles(win_w, win_h);
    }

    /// Keep the center inside the (level-dependent) world bounds.
    fn clamp_center(&mut self) {
        let world = TILE * (1u32 << self.zoom) as f64;
        self.target_cx = self.target_cx.clamp(0.0, world);
        self.target_cy = self.target_cy.clamp(0.0, world);
        self.cx = self.cx.clamp(0.0, world);
        self.cy = self.cy.clamp(0.0, world);
    }

    fn zoom_step(&mut self, dir: i8) {
        if self.anim.is_some() {
            return; // a 150 ms animation is already running
        }
        let new = self.zoom as i8 + dir;
        if !(MIN_ZOOM as i8..=MAX_ZOOM as i8).contains(&new) {
            return;
        }
        self.target_cx = self.cx;
        self.target_cy = self.cy;
        self.anim = Some(Anim {
            from: self.zoom,
            dir,
            f: 0.0,
            cx: self.cx,
            cy: self.cy,
        });
    }

    /// `g`: back to the launch position (cancels any animation).
    fn go_home(&mut self) {
        let (zoom, lat, lon) = self.home;
        self.anim = None;
        self.zoom = zoom;
        self.cx = lon_to_x(lon, zoom);
        self.cy = lat_to_y(lat, zoom);
        self.target_cx = self.cx;
        self.target_cy = self.cy;
    }

    /// Key handling (before `begin_drawing`, which borrows the handle).
    /// q quits; ESC is inert (ESC never quits — map mode has no grid to
    /// return to). h/j/k/l + arrows pan; +/-/Space/wheel change zoom level;
    /// g returns home.
    fn handle_input(&mut self, rl: &mut RaylibHandle, win_w: f32, win_h: f32) -> bool {
        let mut quit = false;
        while let Some(k) = rl.get_key_pressed() {
            match k {
                KeyboardKey::KEY_Q => quit = true,
                KeyboardKey::KEY_G => self.go_home(),
                KeyboardKey::KEY_KP_ADD => self.zoom_step(1),
                KeyboardKey::KEY_KP_SUBTRACT => self.zoom_step(-1),
                KeyboardKey::KEY_SPACE => {
                    // osm.el: SPC zooms in, S-SPC zooms out.
                    let shift = rl.is_key_down(KeyboardKey::KEY_LEFT_SHIFT)
                        || rl.is_key_down(KeyboardKey::KEY_RIGHT_SHIFT);
                    self.zoom_step(if shift { -1 } else { 1 });
                }
                _ => {}
            }
        }
        // +/- via typed characters too (layout-independent, same as the
        // image view's free zoom).
        while let Some(c) = rl.get_char_pressed() {
            match c {
                '+' => self.zoom_step(1),
                '-' => self.zoom_step(-1),
                _ => {}
            }
        }
        if rl.is_key_pressed(KeyboardKey::KEY_EQUAL) {
            self.zoom_step(1);
        }
        if rl.is_key_pressed(KeyboardKey::KEY_MINUS) {
            self.zoom_step(-1);
        }
        let wheel = rl.get_mouse_wheel_move();
        if wheel != 0.0 {
            self.zoom_step(if wheel > 0.0 { 1 } else { -1 });
        }

        // Panning (vim keys + arrows, unrestricted; pixel-space speed).
        if self.anim.is_none() {
            let speed = win_w.max(win_h) as f64 * 1.8 * rl.get_frame_time() as f64;
            let left = rl.is_key_down(KeyboardKey::KEY_H) || rl.is_key_down(KeyboardKey::KEY_LEFT);
            let right =
                rl.is_key_down(KeyboardKey::KEY_L) || rl.is_key_down(KeyboardKey::KEY_RIGHT);
            let up = rl.is_key_down(KeyboardKey::KEY_K) || rl.is_key_down(KeyboardKey::KEY_UP);
            let down = rl.is_key_down(KeyboardKey::KEY_J) || rl.is_key_down(KeyboardKey::KEY_DOWN);
            if left {
                self.target_cx -= speed;
            }
            if right {
                self.target_cx += speed;
            }
            if up {
                self.target_cy -= speed;
            }
            if down {
                self.target_cy += speed;
            }
        }
        quit
    }

    /// Draw the world: visible tiles (cached textures, gray placeholders
    /// otherwise), then a small status line. Must run inside `begin_drawing`.
    fn draw(&mut self, d: &mut RaylibDrawHandle, win_w: f32, win_h: f32) {
        let (level, scale, cx, cy) = match &self.anim {
            Some(a) => (a.from, 2f64.powf(a.dir as f64 * a.f), a.cx, a.cy),
            None => (self.zoom, 1.0, self.cx, self.cy),
        };
        let (tx0, tx1, ty0, ty1) = visible_tiles(level, cx, cy, win_w, win_h, scale);
        let x0 = cx - win_w as f64 / (2.0 * scale);
        let y0 = cy - win_h as f64 / (2.0 * scale);

        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let key: TileKey = (level, tx as u32, ty as u32);
                let dx = (tx as f64 * TILE - x0) * scale;
                let dy = (ty as f64 * TILE - y0) * scale;
                let size = (TILE * scale) as f32;
                let dest = Rectangle {
                    x: dx as f32,
                    y: dy as f32,
                    width: size,
                    height: size,
                };
                match self.tiles.get_mut(&key) {
                    Some(t) => {
                        t.last_used = self.frame;
                        if let Some(tex) = &t.tex {
                            let src = Rectangle {
                                x: 0.0,
                                y: 0.0,
                                width: tex.width() as f32,
                                height: tex.height() as f32,
                            };
                            d.draw_texture_pro(tex, src, dest, Vector2::ZERO, 0.0, Color::WHITE);
                        } else {
                            // Still loading or failed: dark placeholder.
                            d.draw_rectangle(
                                dest.x as i32,
                                dest.y as i32,
                                dest.width as i32,
                                dest.height as i32,
                                Color::new(26, 26, 26, 255),
                            );
                        }
                    }
                    None => {
                        d.draw_rectangle(
                            dest.x as i32,
                            dest.y as i32,
                            dest.width as i32,
                            dest.height as i32,
                            Color::new(26, 26, 26, 255),
                        );
                    }
                }
            }
        }

        // Status line: zoom level + center coordinates.
        let (lat, lon) = self.center_latlon();
        let text = format!("z{}  {:.4}, {:.4}", self.zoom, lat, lon);
        d.draw_text(&text, 8, win_h as i32 - 24, 16, Color::GRAY);
    }
}

// ---------------------------------------------------------------- entry point

/// `--osm` mode: own window + run loop (the file viewer's loop knows
/// nothing about maps, and vice versa — merging the two view loops can
/// happen when documents and maps share one window).
// Takes `Option<String>` so the main() dispatch can pass `args.next()`.
#[allow(clippy::needless_pass_by_value)]
pub fn run(pos: Option<String>) -> Result<()> {
    let pos = pos.as_deref().map(parse_pos).transpose()?;
    let (mut rl, thread) = raylib::init()
        .size(1024, 768)
        .title("versatile-viewer — osm map")
        .resizable()
        .build();
    // q quits via our own key handling; ESC must not close the window.
    rl.set_exit_key(None);
    rl.set_target_fps(60);

    // Declared after `rl`: the tile textures are GPU objects and must drop
    // while the window still exists.
    let mut map = MapState::new(pos);

    while !rl.window_should_close() {
        let win_w = rl.get_screen_width() as f32;
        let win_h = rl.get_screen_height() as f32;
        if map.handle_input(&mut rl, win_w, win_h) {
            break;
        }
        map.update(&mut rl, &thread, win_w, win_h);

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);
        map.draw(&mut d, win_w, win_h);
    }
    Ok(())
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
#[allow(clippy::float_cmp)] // the asserted values are exact by construction
mod tests {
    use super::*;

    /// Berlin center (52.52, 13.405). Global pixel coordinates and tile
    /// index at z15; reference values computed independently from the
    /// slippy-map formulas.
    #[test]
    fn berlin_pixel_coordinates_at_z15() {
        let x = lon_to_x(13.405, 15);
        let y = lat_to_y(52.52, 15);
        assert!((x - 4_506_663.14).abs() < 0.01, "x = {x}");
        assert!((y - 2_751_087.29).abs() < 0.01, "y = {y}");
        // Tile index: floor(pixel / 256).
        assert_eq!((x / TILE).floor() as u32, 17_604);
        assert_eq!((y / TILE).floor() as u32, 10_746);
    }

    #[test]
    fn known_tile_indices_other_zooms() {
        // Sydney z11, New York z6 (independent reference values).
        let (x, y) = (lon_to_x(151.2093, 11), lat_to_y(-33.8688, 11));
        assert_eq!((x / TILE).floor() as u32, 1_884);
        assert_eq!((y / TILE).floor() as u32, 1_228);
        let (x, y) = (lon_to_x(-74.006, 6), lat_to_y(40.7128, 6));
        assert_eq!((x / TILE).floor() as u32, 18);
        assert_eq!((y / TILE).floor() as u32, 24);
    }

    #[test]
    fn projection_round_trips() {
        for zoom in [2u8, 7, 15, 19] {
            for lon in [-180.0, -74.006, -0.1, 0.0, 13.405, 151.2093, 179.9] {
                assert!((x_to_lon(lon_to_x(lon, zoom), zoom) - lon).abs() < 1e-9);
            }
            for lat in [-85.0, -33.8688, 0.0, 40.7128, 52.52, 85.0] {
                assert!((y_to_lat(lat_to_y(lat, zoom), zoom) - lat).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn latitude_clamps_at_mercator_limit() {
        // Beyond the limit the projection would diverge; everything past
        // ±MAX_LAT must clamp to the same pixel row.
        // 85.06° lies beyond the limit; it must clamp to the same pixel
        // row as MAX_LAT itself (a row that is ~0: the world's top edge).
        assert_eq!(lat_to_y(85.06, 8), lat_to_y(MAX_LAT, 8));
        assert_eq!(lat_to_y(-89.0, 8), lat_to_y(-MAX_LAT, 8));
        // 85.0° is still INSIDE the range, so it must not be clamped:
        // y(85°, z8) is ~107 px, distinctly below the top edge.
        assert!(lat_to_y(85.0, 8) > 100.0);
    }

    #[test]
    fn y_grows_downward() {
        // Mercator: north = small y. 60° N must sit above (smaller y) 0°
        // at every zoom level.
        for zoom in [2u8, 12, 19] {
            assert!(lat_to_y(60.0, zoom) < lat_to_y(0.0, zoom));
        }
    }

    #[test]
    fn zoom_rescales_pixels_by_two() {
        // One level up doubles global pixel coordinates — the exact
        // property the pan state relies on when switching levels.
        for zoom in [2u8, 9, 18] {
            assert_eq!(lon_to_x(13.405, zoom + 1), lon_to_x(13.405, zoom) * 2.0);
            assert_eq!(lat_to_y(52.52, zoom + 1), lat_to_y(52.52, zoom) * 2.0);
        }
    }

    #[test]
    fn visible_tiles_clamp_to_world() {
        // Center at the world's top-left corner: out-of-world tile indices
        // must clamp to 0.
        let (tx0, tx1, ty0, ty1) = visible_tiles(3, 0.0, 0.0, 1024.0, 768.0, 1.0);
        // half window = 512 px right of the center → x reaches into tile 2;
        // 384 px down → y reaches into tile 1 (floor((0+512)/256) = 2).
        assert_eq!((tx0, tx1, ty0, ty1), (0, 2, 0, 1));
        // A 1024x768 window at scale 1 covers 4-5 tiles per axis (boundary
        // dependent; also depends on where the center lands inside a tile).
        let (tx0, tx1, ty0, ty1) =
            visible_tiles(15, 4_506_663.14, 2_751_087.29, 1024.0, 768.0, 1.0);
        assert_eq!(tx1 - tx0, 4);
        assert_eq!(ty1 - ty0, 3);
        // Halving the scale (zoom-out anim) doubles the coverage.
        let (tx0, tx1, ..) = visible_tiles(15, 4_506_663.14, 2_751_087.29, 1024.0, 768.0, 0.5);
        assert_eq!(tx1 - tx0, 8);
    }

    #[test]
    fn tile_url_matches_slippy_format() {
        assert_eq!(
            tile_url(15, 17_604, 10_746),
            "https://tile.openstreetmap.org/15/17604/10746.png"
        );
    }

    #[test]
    fn parse_pos_accepts_and_rejects() {
        let (z, lat, lon) = parse_pos("15/52.52/13.405").unwrap();
        assert_eq!((z, lat, lon), (15, 52.52, 13.405));
        assert!(parse_pos("0/0/0").is_ok());
        assert!(parse_pos("19/85.05/180").is_ok());
        // Zoom beyond the server max, out-of-range coordinates, junk.
        assert!(parse_pos("20/0/0").is_err());
        assert!(parse_pos("10/86/0").is_err());
        assert!(parse_pos("10/0/181").is_err());
        assert!(parse_pos("10/52").is_err());
        assert!(parse_pos("a/b/c").is_err());
    }
}
