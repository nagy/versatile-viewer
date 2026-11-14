//! Converters: an external program turns a file into an SVG document or a PDF.
//!
//! The viewer parses a few formats by itself (see
//! [`crate::surface::open_surface`]).
//! Every other format arrives through a converter: a program that reads the
//! source bytes on stdin and writes an SVG document or a PDF on stdout.
//! A rule names the extensions it claims and the argv it runs, so the viewer
//! never parses an unknown format, never links a heavy engine, and never
//! needs a rebuild for one.
//!
//! The table comes from `$XDG_CONFIG_HOME/vv/converters.toml`, which
//! `VV_CONVERTERS` can point at another file.
//! A missing file leaves an empty table, so a format arrives only through a
//! rule the user writes: the viewer names no program and ships no rule.
//!
//! ```toml
//! [defaults]
//! slots = 2                # conversions that run at one time
//! timeout_ms = 15000       # wall clock of one conversion
//! max_output = 67108864    # bytes a converter may write to stdout
//!
//! [[convert]]
//! ext = ["mmd", "mermaid"]
//! command = ["merman-cli", "render", "-"]
//! background = "#ffffff"   # paper under the paints, `#rrggbb` or `#rrggbbaa`
//!
//! [[convert]]
//! ext = ["typ"]
//! command = ["typst", "compile", "--format", "pdf", "-", "/dev/stdout"]
//! timeout_ms = 30000
//! ```
//!
//! The first rule previews a Mermaid diagram with `merman-cli`, a browserless
//! renderer: it reads the source on stdin and writes the SVG document on
//! stdout.
//! A converter that starts a browser serves the same shape with `mmdc`.
//!
//! The second compiles a Typst document: the `typst` CLI reads the source on
//! stdin and writes a PDF on stdout, which keeps one plane per page, so the
//! document shows as the book a PDF file shows.
//!
//! The `[defaults]` table reads before every rule, so a rule inherits its
//! values whatever the order of the file.
//!
//! The command runs without a shell, so every argument stays one argument and
//! no config file reaches an interpolation.
//! It runs with the directory of the source as its working directory, so a tool
//! that resolves a relative path reads it beside the file: a Typst `#include`,
//! `#image` or `#read` works, and the CLI roots the project there.
//! The answer decides the backend: `%PDF-` bytes open as a
//! [`crate::pdf::PdfSurface`], and any other answer must parse as SVG, which the
//! vector backend paints: one parsed tree then serves the grid thumbnail, the
//! base raster and the window patch, exactly like a `.svg` file.
//! The converter runs at open time on a decode worker, so the picture costs one
//! process start and no main-thread stall.
//!
//! A browser-backed converter costs a second and hundreds of megabytes, so two
//! rules run at one time by default and the rest queue on
//! [`Slots`].
//! Every conversion has a wall clock and a stdout cap, and the child dies on
//! either, so a stuck converter can never hold a worker.
//! The conversion runs behind detached reader threads, so a grandchild that
//! keeps a pipe open can never stall the worker either: the worker takes the
//! bytes that arrived under one grace period, shared by both pipes, and it
//! drains stderr to its end so a chatty converter never meets a closed pipe.
//! A process that outlives its clock dies as a whole process group, so the
//! browser a converter started dies with it.
//!
//! The converted answer caches under the XDG cache home, keyed by the source
//! bytes and the argv, because a grid decode and an image view open the same
//! file twice (see [`crate::thumbcache`]).
//! A rule change therefore invalidates every entry of that rule.
//!
//! The table lives in a process-wide [`LazyLock`], because a decode runs deep
//! inside a surface open on a worker and no owner passes a table down that
//! path.
//! A test builds a [`Converters`] of its own, so no test reads the table of the
//! user.

use std::{
    fs,
    io::{Read, Write as _},
    os::unix::process::CommandExt as _,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

use derive_more::{Display, Error, From};

use crate::{
    draw::Color,
    surface::{Surface, SurfaceError, VectorSurface},
};

/// The most bytes this backend reads from a source file.
///
/// A converter reads text and small binary formats, so a real file stays far
/// below this.
const MAX_SOURCE: usize = 8 * 1024 * 1024;

/// The default wall clock of one conversion.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(15);

/// The default cap of the bytes a converter writes to stdout.
const DEFAULT_OUTPUT: usize = 64 * 1024 * 1024;

/// The default number of conversions that run at one time.
const DEFAULT_SLOTS: usize = 2;

/// The most stderr bytes kept for an error message.
const STDERR_HEAD: usize = 4096;

/// The bytes one read of a child pipe takes at a time.
const CHUNK: usize = 16 * 1024;

/// The quiet period between two polls of a converter.
const POLL: Duration = Duration::from_millis(10);

/// The most time both reader threads get after their process is gone.
///
/// One deadline covers the two pipes, so a grandchild that keeps one open
/// costs at most this period and never one period per pipe.
const GRACE: Duration = Duration::from_secs(2);

/// The starts tried while the program answers `ETXTBSY`.
const SPAWN_TRIES: u32 = 20;

/// The config file under the XDG config home.
const CONFIG: &str = "vv/converters.toml";

/// The variable that names another config file.
const CONFIG_ENV: &str = "VV_CONVERTERS";

/// The key prefix of a cached conversion.
///
/// Bump it when the payload of an entry changes, so no old entry loads.
#[cfg(feature = "cache")]
const KEY_VERSION: &str = "vv-converted-2";

/// The magic of a PDF answer.
const PDF_MAGIC: &[u8] = b"%PDF-";

/// Why a converter table did not load.
///
/// A bad table never stops the viewer: the boundary prints this and keeps the
/// empty table.
#[derive(Debug, Display, Error, From)]
pub enum ConvertError {
    /// The config file cannot be read.
    #[display("failed to read the converter config")]
    Io(#[from] std::io::Error),
    /// The config file is not valid TOML.
    #[display("converter config is not TOML")]
    Toml(#[from] toml::de::Error),
    /// The config file holds a key or a value this version does not know.
    #[display("converter config: {_0}")]
    Config(#[error(ignore)] String),
}

/// One converter rule.
#[derive(Debug, Clone)]
pub struct Rule {
    /// The extensions the rule claims, in lower case and without the dot.
    extensions: Vec<String>,
    /// The argv of the program, where the first element is the program.
    command: Vec<String>,
    /// The wall clock of one conversion.
    timeout: Duration,
    /// The most bytes the program may write to stdout.
    max_output: usize,
    /// The paper under the paints, or `None` for a tree that paints its own
    /// page.
    background: Option<Color>,
}

impl Rule {
    /// Whether this rule claims `ext`, which arrives in any letter case.
    fn claims(&self, ext: &str) -> bool {
        self.extensions
            .iter()
            .any(|own| own.eq_ignore_ascii_case(ext))
    }

    /// The program name, for an error message.
    fn program_name(&self) -> &str {
        self.command.first().map_or("converter", String::as_str)
    }
}

/// The conversions that run at one time.
///
/// The counter and the condvar say how many slots stay free, so a decode worker
/// waits for a free one instead of starting a browser of its own.
#[derive(Debug)]
struct Slots {
    /// The free slots.
    free: Mutex<usize>,
    /// Signals a released slot.
    ready: Condvar,
}

impl Slots {
    /// `slots` conversions at one time, and at least one.
    fn new(slots: usize) -> Self {
        Self {
            free: Mutex::new(slots.max(1)),
            ready: Condvar::new(),
        }
    }

    /// Wait for a free slot.
    fn acquire(&self) -> Slot<'_> {
        let mut free = self.lock();
        while *free == 0 {
            free = self
                .ready
                .wait(free)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *free -= 1;
        Slot { slots: self }
    }

    /// The counter, with a poisoned lock taken as it stands.
    fn lock(&self) -> MutexGuard<'_, usize> {
        self.free.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One held conversion slot, released on drop.
#[derive(Debug)]
struct Slot<'a> {
    /// The pool this slot came from.
    slots: &'a Slots,
}

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        let mut free = self.slots.lock();
        *free += 1;
        self.slots.ready.notify_one();
    }
}

/// The converter table.
#[derive(Debug)]
pub struct Converters {
    /// The rules, in config order.
    rules: Vec<Rule>,
    /// The conversions that run at one time.
    slots: Slots,
    /// The converted SVG entries.
    #[cfg(feature = "cache")]
    cache: crate::thumbcache::Cache,
}

impl Default for Converters {
    fn default() -> Self {
        Self::empty()
    }
}

impl Converters {
    /// A table with no rule, over the real cache.
    ///
    /// A missing config file answers it, and so does a config file that does
    /// not parse, so the viewer never guesses a program.
    #[must_use]
    pub fn empty() -> Self {
        Self::from_rules(Vec::new(), DEFAULT_SLOTS)
    }

    /// A table over `rules`, at `slots` conversions at one time.
    fn from_rules(rules: Vec<Rule>, slots: usize) -> Self {
        Self {
            rules,
            slots: Slots::new(slots),
            #[cfg(feature = "cache")]
            cache: crate::thumbcache::Cache::new(),
        }
    }

    /// A table over `rules` with its entries inside `dir`, for a test.
    ///
    /// The test never touches the cache of the user.
    #[cfg(test)]
    fn in_dir(rules: Vec<Rule>, dir: &Path) -> Self {
        Self::from_rules(rules, DEFAULT_SLOTS).with_cache_dir(dir)
    }

    /// Put the converted entries of this table under `dir`.
    ///
    /// A host or a test names its own directory, so a throwaway tree holds the
    /// entries and the cache of the user stays untouched.
    /// A build without the `cache` feature keeps no entries, so it answers the
    /// table unchanged.
    #[must_use]
    pub fn with_cache_dir(self, dir: &Path) -> Self {
        #[cfg(feature = "cache")]
        {
            Self {
                cache: crate::thumbcache::Cache::in_dir(dir),
                ..self
            }
        }
        #[cfg(not(feature = "cache"))]
        {
            let _ = dir;
            self
        }
    }

    /// Parse a config file.
    ///
    /// Every rule comes from the text: the table of this process holds no rule
    /// of its own.
    ///
    /// # Errors
    ///
    /// The function errors when the text is not TOML, holds an unknown key, or
    /// holds a rule without an extension or without a command.
    pub fn parse(text: &str) -> Result<Self, ConvertError> {
        let root: toml::Table = toml::from_str(text)?;
        // `[defaults]` first, because every rule inherits its values and a
        // table reads in key order rather than in file order.
        let defaults = match root.get("defaults") {
            Some(value) => Defaults::parse(value)?,
            None => Defaults::default(),
        };
        let mut rules = Vec::new();
        for (key, value) in &root {
            match key.as_str() {
                "defaults" => {}
                "convert" => {
                    for rule in array(value, "convert")? {
                        rules.push(Rule::parse(rule, &defaults)?);
                    }
                }
                other => return Err(unknown(other, "defaults, convert")),
            }
        }
        Ok(Self::from_rules(rules, defaults.slots))
    }

    /// Load the table from the config file.
    ///
    /// A missing file answers an empty table, so no format converts until the
    /// user writes a rule.
    ///
    /// # Errors
    ///
    /// The function errors when the file cannot be read or does not parse.
    pub fn load() -> Result<Self, ConvertError> {
        let Some(path) = config_path() else {
            return Ok(Self::empty());
        };
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::empty());
            }
            Err(err) => return Err(ConvertError::Io(err)),
        };
        Self::parse(&text).map_err(|err| ConvertError::Config(format!("{}: {err}", path.display())))
    }

    /// Whether a rule claims `path`.
    #[must_use]
    pub fn claims(&self, path: &Path) -> bool {
        self.rule(path).is_some()
    }

    /// The rule that claims `path`, where the first match wins.
    fn rule(&self, path: &Path) -> Option<&Rule> {
        let ext = path.extension()?.to_str()?;
        self.rules.iter().find(|rule| rule.claims(ext))
    }

    /// Convert `path` and open the answer as a surface.
    ///
    /// The conversion runs here, once, and the answer serves every later
    /// paint: a PDF keeps one plane per page, and an SVG document keeps one
    /// parsed tree.
    ///
    /// # Errors
    ///
    /// The function errors when no rule claims the file, the file cannot be
    /// read, the program fails to start, exits with a failure, writes more than
    /// its cap, or answers neither a PDF nor an SVG document.
    pub fn open(&self, path: &Path) -> Result<Box<dyn Surface>, SurfaceError> {
        let Some(rule) = self.rule(path) else {
            return Err(SurfaceError::Unsupported(format!(
                "no converter claims `{}`",
                path.display()
            )));
        };
        let source = read_source(path)?;
        let answer = self.converted(rule, &source, path.parent())?;
        surface(rule, answer)
    }

    /// The answer of `source` under `rule`, from the cache or the program.
    ///
    /// # Errors
    ///
    /// The function errors when the program fails, exits with a failure,
    /// writes more than its cap, or runs past its wall clock.
    fn converted(
        &self,
        rule: &Rule,
        source: &[u8],
        dir: Option<&Path>,
    ) -> Result<Vec<u8>, SurfaceError> {
        #[cfg(feature = "cache")]
        {
            let key = key(rule, source);
            if let Some(bytes) = self.cache.load_converted(&key) {
                return Ok(bytes);
            }
            let bytes = self.run(rule, source, dir)?;
            // A cache failure stays silent, like `grid::store_thumb`: the
            // picture is already in hand, and a switched-off cache, a read-only
            // cache home or a spent budget is a normal setup.
            let _ = self.cache.store_converted(&key, &bytes);
            Ok(bytes)
        }
        #[cfg(not(feature = "cache"))]
        {
            self.run(rule, source, dir)
        }
    }

    /// Run one conversion and answer its stdout.
    ///
    /// The call holds one slot, feeds the source, bounds the output, bounds the
    /// wall clock, and reaps the child.
    ///
    /// # Errors
    ///
    /// The function errors when the program cannot start, fails, writes more
    /// than its cap, runs past its wall clock, or keeps its output open past
    /// the grace period without writing a byte.
    fn run(&self, rule: &Rule, source: &[u8], dir: Option<&Path>) -> Result<Vec<u8>, SurfaceError> {
        let _slot = self.slots.acquire();
        let mut child = spawn(rule, dir)?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        // The readers share their buffer with this worker, so an answer that
        // already arrived survives a pipe that a grandchild keeps open.
        let out = Arc::new(Mutex::new(PipeBytes::default()));
        let err = Arc::new(Mutex::new(PipeBytes::default()));
        let max_output = rule.max_output;
        // Detached readers, so a grandchild that holds a pipe open can never
        // stall the decode worker: the worker takes the buffer under a grace
        // period and walks on.
        {
            let out = Arc::clone(&out);
            std::thread::spawn(move || drain_capped(stdout, max_output, &out));
        }
        {
            let err = Arc::clone(&err);
            std::thread::spawn(move || drain_head(stderr, STDERR_HEAD, &err));
        }
        let input = source.to_vec();
        std::thread::spawn(move || {
            if let Some(mut stdin) = stdin {
                let _ = stdin.write_all(&input);
            }
        });
        let status = wait(rule, &mut child)?;
        // One grace period covers both pipes, so a lingering helper costs the
        // worker one period and not two.
        let deadline = Instant::now() + GRACE;
        let stderr_text = String::from_utf8_lossy(&bytes_until(&err, deadline).bytes).into_owned();
        let answer = bytes_until(&out, deadline);
        if answer.over_cap {
            return Err(SurfaceError::Render(format!(
                "the converter `{}` wrote more than the {max_output} byte output cap",
                rule.program_name()
            )));
        }
        if !status.success() {
            return Err(SurfaceError::Render(format!(
                "{} {}: {}",
                diagnose(rule, "failed"),
                exit_note(status),
                stderr_text.trim()
            )));
        }
        // A pipe that stays open past the grace period still carries the bytes
        // that arrived, because a converter often leaves a helper behind. An
        // answer cut short then fails the SVG parse, which names the cause.
        if !answer.finished && answer.bytes.is_empty() {
            return Err(SurfaceError::Render(diagnose(
                rule,
                "kept its output open past the grace period and wrote nothing",
            )));
        }
        Ok(answer.bytes)
    }
}

/// Read the source bytes of `path`, and refuse a file past [`MAX_SOURCE`].
///
/// The size check runs before the read, so a large file costs no allocation,
/// and the read itself stops one byte past the cap, so a file that grows
/// between the two still fails.
///
/// # Errors
///
/// The function errors when the file cannot be read, or is larger than
/// [`MAX_SOURCE`].
fn read_source(path: &Path) -> Result<Vec<u8>, SurfaceError> {
    let cap = u64::try_from(MAX_SOURCE).unwrap_or(u64::MAX);
    let too_large = || {
        SurfaceError::Unsupported(format!(
            "`{}` is larger than {MAX_SOURCE} bytes, the converter source cap",
            path.display()
        ))
    };
    let len = fs::metadata(path)?.len();
    if len > cap {
        return Err(too_large());
    }
    let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    fs::File::open(path)?
        .take(cap.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_SOURCE {
        return Err(too_large());
    }
    Ok(bytes)
}

/// Wrap the answer of `rule` in the surface its content calls for.
///
/// `%PDF-` bytes open as a [`crate::pdf::PdfSurface`], so a document keeps its
/// pages and the book view wraps it like a PDF file.
/// Every other answer must parse as SVG, which becomes the one plane of a
/// [`VectorSurface`], with the paper of the rule under its paints.
/// A PDF answer ignores `background`, because a page paints its own paper.
///
/// # Errors
///
/// The function errors when the answer is neither a readable PDF nor an SVG
/// document.
fn surface(rule: &Rule, answer: Vec<u8>) -> Result<Box<dyn Surface>, SurfaceError> {
    if answer.starts_with(PDF_MAGIC) {
        return Ok(Box::new(crate::pdf::PdfSurface::from_bytes(answer)?));
    }
    let text = std::str::from_utf8(&answer)
        .map_err(|_| SurfaceError::Render("the converter answered no UTF-8 text".into()))?;
    let vector = crate::svg::from_str(text).map_err(|err| {
        SurfaceError::Render(format!(
            "the converter answered neither a PDF nor an SVG document: {err}"
        ))
    })?;
    let mut surface = VectorSurface::from_vector(vector);
    if let Some(background) = rule.background {
        surface = surface.with_background(background);
    }
    Ok(Box::new(surface))
}

/// Start the program of `rule` with piped stdin, stdout and stderr.
///
/// `dir` becomes the working directory, so the program resolves a relative path
/// beside the source it converts.
///
/// # Errors
///
/// The function errors when the program cannot be started.
fn spawn(rule: &Rule, dir: Option<&Path>) -> Result<Child, SurfaceError> {
    let Some((program, args)) = rule.command.split_first() else {
        return Err(SurfaceError::Unsupported(
            "the converter rule holds no command".into(),
        ));
    };
    let dir = dir.unwrap_or_else(|| Path::new("."));
    let mut tries = 0;
    loop {
        let started = Command::new(program)
            .args(args)
            .current_dir(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // The child leads a process group of its own, so the kill of
            // `wait` reaches every helper it started (a converter often
            // drives a browser).
            .process_group(0)
            .spawn();
        match started {
            // A concurrent fork in this process holds a write handle of the
            // program for a moment, and exec then answers `ETXTBSY`. The
            // window closes at once, so a short retry wins.
            Err(err)
                if err.kind() == std::io::ErrorKind::ExecutableFileBusy && tries < SPAWN_TRIES =>
            {
                tries += 1;
                std::thread::sleep(POLL);
            }
            Err(err) => {
                return Err(SurfaceError::Unsupported(format!(
                    "cannot start the converter `{program}`: {err}"
                )));
            }
            Ok(child) => return Ok(child),
        }
    }
}

/// Wait for `child` under the wall clock of `rule`.
///
/// A process that outlives its clock dies as a whole process group ([`kill_group`])
/// and is reaped, so no zombie stays, no worker waits forever, and no helper of
/// the converter holds a browser open.
///
/// # Errors
///
/// The function errors when the wait fails or the clock runs out.
fn wait(rule: &Rule, child: &mut Child) -> Result<ExitStatus, SurfaceError> {
    let deadline = Instant::now() + rule.timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if Instant::now() >= deadline => {
                kill_group(child);
                let _ = child.wait();
                return Err(SurfaceError::Render(format!(
                    "the converter `{}` ran longer than {:?}",
                    rule.program_name(),
                    rule.timeout
                )));
            }
            Ok(None) => std::thread::sleep(POLL),
            Err(err) => {
                return Err(SurfaceError::Render(format!(
                    "{}: {err}",
                    diagnose(rule, "cannot be waited for")
                )));
            }
        }
    }
}

/// Kill `child` and every process of the group its spawn created.
///
/// The spawn puts the child in a group of its own, so the negative pid of the
/// leader names the whole conversion.
/// The wait already answered `None` for this child, so its pid stays allocated
/// and the group number cannot name another process.
fn kill_group(child: &mut Child) {
    let killed = i32::try_from(child.id()).is_ok_and(|pid| {
        // SAFETY: `kill` reads no memory. The negative pid names the process
        // group that `spawn` created, which no other process holds.
        let sent = unsafe { libc::kill(-pid, libc::SIGKILL) };
        sent == 0
    });
    if !killed {
        // The group already ended, or the call failed: the direct child stays
        // the second chance, and the wait reaps it either way.
        let _ = child.kill();
    }
}

/// The message of `reason` for `rule`, with the program name.
fn diagnose(rule: &Rule, reason: &str) -> String {
    format!("the converter `{}` {reason}", rule.program_name())
}

/// The exit note of `status`, for an error message.
fn exit_note(status: ExitStatus) -> String {
    match status.code() {
        Some(code) => format!("exited with code {code}"),
        None => "was killed by a signal".to_owned(),
    }
}

/// The bytes a reader thread took from one child pipe.
///
/// The reader and the worker share one value, so a buffer that already holds an
/// answer survives a pipe that never reaches its end.
#[derive(Debug, Default, Clone)]
struct PipeBytes {
    /// The bytes read so far.
    bytes: Vec<u8>,
    /// The reader reached the end of the pipe.
    finished: bool,
    /// More bytes arrived than the reader keeps.
    over_cap: bool,
}

/// Read `reader` into `shared`, keep at most `limit` bytes, and stop at the
/// first byte past it.
///
/// The call closes the pipe on a flood, so the child meets `SIGPIPE` and the
/// conversion ends instead of filling the memory.
fn drain_capped(reader: Option<impl Read>, limit: usize, shared: &Mutex<PipeBytes>) {
    let Some(mut reader) = reader else {
        lock_ignoring_poison(shared).finished = true;
        return;
    };
    let mut buf = [0u8; CHUNK];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let mut pending = lock_ignoring_poison(shared);
                let room = limit.saturating_sub(pending.bytes.len());
                let take = n.min(room);
                pending.bytes.extend_from_slice(&buf[..take]);
                if n > take {
                    pending.over_cap = true;
                    break;
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    lock_ignoring_poison(shared).finished = true;
}

/// Read `reader` to its end, and keep the first `limit` bytes in `shared`.
///
/// The call drains the whole pipe, because a closed read end kills a chatty
/// child with `SIGPIPE` before it writes its answer: a converter that logs more
/// than `limit` bytes would fail for no reason of its own.
fn drain_head(reader: Option<impl Read>, limit: usize, shared: &Mutex<PipeBytes>) {
    let Some(mut reader) = reader else {
        lock_ignoring_poison(shared).finished = true;
        return;
    };
    let mut buf = [0u8; CHUNK];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let mut pending = lock_ignoring_poison(shared);
                let room = limit.saturating_sub(pending.bytes.len());
                pending.bytes.extend_from_slice(&buf[..n.min(room)]);
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    lock_ignoring_poison(shared).finished = true;
}

/// The buffer of `shared`, with a poisoned lock taken as it stands.
fn lock_ignoring_poison(shared: &Mutex<PipeBytes>) -> MutexGuard<'_, PipeBytes> {
    shared.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The bytes `shared` holds once its pipe ends or `deadline` passes.
fn bytes_until(shared: &Mutex<PipeBytes>, deadline: Instant) -> PipeBytes {
    loop {
        let pending = lock_ignoring_poison(shared);
        if pending.finished || Instant::now() >= deadline {
            return pending.clone();
        }
        drop(pending);
        std::thread::sleep(POLL);
    }
}

/// The name of the config file, or `None` without a config home.
fn config_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(CONFIG_ENV) {
        return Some(PathBuf::from(path));
    }
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME").filter(|dir| !dir.is_empty()) {
        return Some(PathBuf::from(dir).join(CONFIG));
    }
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".config").join(CONFIG))
}

/// The cache key of one conversion: the argv and the source bytes.
#[cfg(feature = "cache")]
fn key(rule: &Rule, source: &[u8]) -> String {
    use sha2::{Digest as _, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(KEY_VERSION);
    for arg in &rule.command {
        hasher.update(arg.as_bytes());
        hasher.update([0x1f]);
    }
    hasher.update([0x1e]);
    hasher.update(source);
    let mut key = String::with_capacity(64);
    for byte in hasher.finalize() {
        use std::fmt::Write as _;
        let _ = write!(key, "{byte:02x}");
    }
    key
}

/// The value of `[defaults]`, which every rule inherits.
#[derive(Debug, Clone, Copy)]
struct Defaults {
    /// The conversions that run at one time.
    slots: usize,
    /// The wall clock of one conversion.
    timeout: Duration,
    /// The most bytes a converter may write to stdout.
    max_output: usize,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            slots: DEFAULT_SLOTS,
            timeout: DEFAULT_TIMEOUT,
            max_output: DEFAULT_OUTPUT,
        }
    }
}

impl Defaults {
    /// Read the `[defaults]` table.
    ///
    /// # Errors
    ///
    /// The function errors when the value is no table or holds an unknown key.
    fn parse(value: &toml::Value) -> Result<Self, ConvertError> {
        let mut defaults = Self::default();
        let table = table(value, "defaults")?;
        for (key, value) in table {
            match key.as_str() {
                "slots" => defaults.slots = positive(value, "slots")?,
                "timeout_ms" => defaults.timeout = millis(value, "timeout_ms")?,
                "max_output" => defaults.max_output = positive(value, "max_output")?,
                other => return Err(unknown(other, "slots, timeout_ms, max_output")),
            }
        }
        Ok(defaults)
    }
}

impl Rule {
    /// Read one `[[convert]]` table over `defaults`.
    ///
    /// # Errors
    ///
    /// The function errors when the table holds an unknown key, no extension,
    /// or no command.
    fn parse(value: &toml::Value, defaults: &Defaults) -> Result<Self, ConvertError> {
        let mut rule = Self {
            extensions: Vec::new(),
            command: Vec::new(),
            timeout: defaults.timeout,
            max_output: defaults.max_output,
            background: None,
        };
        let table = table(value, "convert")?;
        for (key, value) in table {
            match key.as_str() {
                "ext" => {
                    for ext in strings(value, "ext")? {
                        let ext = ext.trim_start_matches('.').trim().to_lowercase();
                        if ext.is_empty() {
                            return Err(ConvertError::Config(
                                "`ext` holds an empty extension".into(),
                            ));
                        }
                        rule.extensions.push(ext);
                    }
                }
                "command" => rule.command = strings(value, "command")?,
                "timeout_ms" => rule.timeout = millis(value, "timeout_ms")?,
                "max_output" => rule.max_output = positive(value, "max_output")?,
                "background" => rule.background = Some(color(value)?),
                other => {
                    return Err(unknown(
                        other,
                        "ext, command, timeout_ms, max_output, background",
                    ));
                }
            }
        }
        if rule.extensions.is_empty() {
            return Err(ConvertError::Config(
                "a `[[convert]]` rule names no extension".into(),
            ));
        }
        if rule.command.is_empty() {
            return Err(ConvertError::Config(format!(
                "the rule of `{}` names no command",
                rule.extensions.join(", ")
            )));
        }
        Ok(rule)
    }
}

/// A table value, named `key`.
///
/// # Errors
///
/// The function errors when the value is no table.
fn table<'a>(value: &'a toml::Value, key: &str) -> Result<&'a toml::Table, ConvertError> {
    value
        .as_table()
        .ok_or_else(|| ConvertError::Config(format!("`{key}` is no table")))
}

/// An array value, named `key`.
///
/// # Errors
///
/// The function errors when the value is no array.
fn array<'a>(value: &'a toml::Value, key: &str) -> Result<&'a Vec<toml::Value>, ConvertError> {
    value
        .as_array()
        .ok_or_else(|| ConvertError::Config(format!("`{key}` is no array")))
}

/// The strings of an array value, named `key`.
///
/// # Errors
///
/// The function errors when the value is no array or holds no string.
fn strings(value: &toml::Value, key: &str) -> Result<Vec<String>, ConvertError> {
    let mut out = Vec::new();
    for item in array(value, key)? {
        let text = item
            .as_str()
            .ok_or_else(|| ConvertError::Config(format!("`{key}` holds no string")))?;
        out.push(text.to_owned());
    }
    Ok(out)
}

/// A positive integer value, named `key`.
///
/// # Errors
///
/// The function errors when the value is no positive integer.
fn positive(value: &toml::Value, key: &str) -> Result<usize, ConvertError> {
    let number = value
        .as_integer()
        .ok_or_else(|| ConvertError::Config(format!("`{key}` is no integer")))?;
    usize::try_from(number)
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| ConvertError::Config(format!("`{key}` must be positive")))
}

/// A millisecond count as a duration, named `key`.
///
/// # Errors
///
/// The function errors when the value is no positive integer.
fn millis(value: &toml::Value, key: &str) -> Result<Duration, ConvertError> {
    let ms = positive(value, key)?;
    Ok(Duration::from_millis(u64::try_from(ms).unwrap_or(u64::MAX)))
}

/// A paper color, `#rrggbb` or `#rrggbbaa`.
///
/// # Errors
///
/// The function errors when the value is no hex color of either length.
fn color(value: &toml::Value) -> Result<Color, ConvertError> {
    let text = value
        .as_str()
        .ok_or_else(|| ConvertError::Config("`background` is no string".into()))?;
    let hex = text.strip_prefix('#').unwrap_or(text);
    let bad = || ConvertError::Config("`background` must be `#rrggbb` or `#rrggbbaa`".into());
    // The ASCII test comes first, so every slice below sits on a character
    // boundary and a mistyped multi-byte color never panics.
    if !hex.is_ascii() || !matches!(hex.len(), 6 | 8) {
        return Err(bad());
    }
    let channel = |at: usize| u8::from_str_radix(&hex[at..at + 2], 16).map_err(|_| bad());
    let (r, g, b) = (channel(0)?, channel(2)?, channel(4)?);
    let a = if hex.len() == 8 { channel(6)? } else { 255 };
    Ok(Color::new(r, g, b, a))
}

/// The error of an unknown key.
fn unknown(key: &str, known: &str) -> ConvertError {
    ConvertError::Config(format!("unknown key `{key}`, known: {known}"))
}

/// The converter table of this process.
///
/// A broken config prints one line and leaves an empty table, so a typo runs no
/// program and every other file keeps its own decoder.
static CONVERTERS: LazyLock<Converters> = LazyLock::new(|| match Converters::load() {
    Ok(table) => table,
    Err(err) => {
        eprintln!("vv: {err}");
        Converters::empty()
    }
});

/// The rule table of this process.
///
/// The grid directory filter and a surface open read one table, so a
/// configured format lists exactly where it opens.
pub(crate) fn rules() -> &'static Converters {
    &CONVERTERS
}

/// Whether a converter of this process claims `path`.
#[must_use]
pub fn claims(path: &Path) -> bool {
    rules().claims(path)
}

/// Convert `path` through the table of this process and open the answer.
///
/// # Errors
///
/// The function errors when no rule claims the file, or when the conversion
/// fails (see [`Converters::open`]).
pub fn open(path: &Path) -> Result<Box<dyn Surface>, SurfaceError> {
    CONVERTERS.open(path)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::TempDir;

    use super::*;
    use crate::surface::View;

    /// A red rectangle on a transparent page.
    const SVG: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><rect width="100" height="50" fill="#ff0000"/></svg>"##;

    /// A small red square in the top-left corner of a transparent page.
    const PATCH: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><rect width="10" height="10" fill="#ff0000"/></svg>"##;

    /// Write an executable shell script into `dir`.
    fn script(dir: &TempDir, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt as _;
        let path = dir.path().join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write the script");
        let mut perms = std::fs::metadata(&path).expect("stat").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod the script");
        path
    }

    /// A rule that runs `program` with the default limits.
    fn rule(program: &Path) -> Rule {
        Rule {
            extensions: vec!["mmd".into()],
            command: vec![program.display().to_string()],
            timeout: DEFAULT_TIMEOUT,
            max_output: DEFAULT_OUTPUT,
            background: None,
        }
    }

    #[test]
    fn an_empty_table_claims_nothing() {
        let table = Converters::empty();
        assert!(!table.claims(Path::new("a/flow.mmd")));
        assert!(!table.claims(Path::new("FLOW.MMD")));
        assert!(!table.claims(Path::new("flow.mermaid")));
        assert!(!table.claims(Path::new("notes.txt")));
        assert!(!table.claims(Path::new("noext")));
    }

    #[test]
    fn parse_reads_the_defaults_and_the_rules() {
        let table = Converters::parse(
            "[defaults]\nslots = 1\ntimeout_ms = 250\nmax_output = 4096\n\n[[convert]]\next = \
             [\".exr\"]\ncommand = [\"exr2svg\", \"-\"]\nbackground = \"#102030\"\n",
        )
        .expect("a valid table");
        // The rule of the config claims its own extension, and no other rule
        // of the table claims one.
        assert!(table.claims(Path::new("photo.EXR")));
        assert!(!table.claims(Path::new("flow.mmd")));
        assert!(!table.claims(Path::new("photo.png")));
        let exr = table.rule(Path::new("photo.exr")).expect("the exr rule");
        assert_eq!(exr.command, vec!["exr2svg".to_owned(), "-".to_owned()]);
        assert_eq!(exr.background, Some(Color::new(0x10, 0x20, 0x30, 0xff)));
        // `[defaults]` reads before the rules, so the rule inherits its values
        // even though `convert` sorts before `defaults` in the table.
        assert_eq!(exr.timeout, Duration::from_millis(250));
        assert_eq!(exr.max_output, 4096);
        assert_eq!(table.slots.lock().to_string(), "1");
    }

    #[test]
    fn a_bad_paper_color_is_an_error_not_a_panic() {
        // Six bytes, but the second one is two bytes wide, so a slice at 2
        // would land inside a character.
        assert!(
            Converters::parse(
                "[[convert]]\next = [\"exr\"]\ncommand = [\"x\"]\nbackground = \"aébcd\"\n"
            )
            .is_err()
        );
    }

    #[test]
    fn an_empty_extension_is_an_error() {
        assert!(Converters::parse("[[convert]]\next = [\"\"]\ncommand = [\"x\"]\n").is_err());
    }

    #[test]
    fn the_documented_table_parses() {
        // The table of the README and of the module docstring, verbatim.
        let text = "\
[defaults]
slots = 2
timeout_ms = 15000
max_output = 67108864

[[convert]]
ext = [\"mmd\", \"mermaid\"]
command = [\"merman-cli\", \"render\", \"-\"]
background = \"#ffffff\"

[[convert]]
ext = [\"typ\"]
command = [\"typst\", \"compile\", \"--format\", \"pdf\", \"-\", \"/dev/stdout\"]
timeout_ms = 30000
";
        let table = Converters::parse(text).expect("the documented table parses");
        assert!(table.claims(Path::new("flow.mermaid")));
        let rule = table.rule(Path::new("flow.mmd")).expect("the mmd rule");
        assert_eq!(rule.background, Some(Color::WHITE));
        assert_eq!(
            rule.command,
            vec!["merman-cli".to_owned(), "render".into(), "-".into()]
        );
        assert_eq!(rule.timeout, DEFAULT_TIMEOUT);
        // The typst rule holds twice the clock of a diagram, because a large
        // document takes a while.
        let typst = table.rule(Path::new("doc.typ")).expect("the typ rule");
        assert_eq!(typst.timeout, Duration::from_secs(30));
        assert_eq!(typst.background, None, "a page paints its own paper");
        assert_eq!(table.slots.lock().to_string(), "2");
    }

    #[test]
    fn a_rule_without_paper_leaves_the_page_transparent() {
        let table = Converters::parse(
            "[[convert]]\next = [\"mmd\"]\ncommand = [\"merman-cli\", \"render\", \"-\"]\n",
        )
        .expect("a valid table");
        let mmd = table.rule(Path::new("flow.mmd")).expect("the mmd rule");
        assert_eq!(mmd.background, None, "the rule states no paper");
    }

    #[test]
    fn parse_rejects_a_bad_table() {
        // An unknown key.
        assert!(Converters::parse("[wat]\nx = 1\n").is_err());
        // A rule without a command.
        assert!(Converters::parse("[[convert]]\next = [\"exr\"]\n").is_err());
        // A rule without an extension.
        assert!(Converters::parse("[[convert]]\ncommand = [\"x\"]\n").is_err());
        // A bad paper color.
        assert!(
            Converters::parse(
                "[[convert]]\next = [\"exr\"]\ncommand = [\"x\"]\nbackground = \"white\"\n"
            )
            .is_err()
        );
    }

    #[test]
    fn a_converter_opens_an_svg_surface() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(
            &dir,
            "cat.sh",
            &format!("cat > /dev/null\nprintf '%s' '{SVG}'"),
        );
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "flowchart LR\n").expect("write the source");
        let surface = table.open(&path).expect("the converted surface");
        assert_eq!(surface.plane_count(), 1);
        // The declared 100 x 50 bakes at 1024 x 512, like a `.svg` file.
        let base = surface
            .render(View::whole(0, 1.0))
            .expect("the base raster");
        assert_eq!(base.width(), crate::dim::ImgW::new(1024));
        assert_eq!(base.height(), crate::dim::ImgH::new(512));
        assert_eq!(&base.as_rgba()[..4], &[255, 0, 0, 255]);
    }

    #[test]
    fn a_rule_paints_its_own_paper() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(
            &dir,
            "patch.sh",
            &format!("cat > /dev/null\nprintf '%s' '{PATCH}'"),
        );
        let mut rule = rule(&program);
        rule.background = Some(Color::WHITE);
        let table = Converters::in_dir(vec![rule], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "x").expect("write the source");
        let base = table
            .open(&path)
            .expect("the converted surface")
            .render(View::whole(0, 1.0))
            .expect("the base raster");
        // The page is transparent below and right of the red square, so the
        // paper `#ffffff` fills it opaque.
        let last = base.as_rgba().len() - 4;
        assert_eq!(&base.as_rgba()[last..], &[255, 255, 255, 255]);
    }

    #[test]
    fn a_converter_opens_a_pdf_answer_as_a_document() {
        let dir = tempfile::tempdir().expect("a temp dir");
        // The stand-in converter answers the bytes of a two-page PDF.
        let pdf = dir.path().join("answer.pdf");
        std::fs::write(&pdf, crate::pdf::TWO_PAGE_PDF).expect("write the PDF");
        let program = script(
            &dir,
            "doc.sh",
            &format!("cat > /dev/null\ncat '{}'", pdf.display()),
        );
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("doc.mmd");
        std::fs::write(&path, "a source the viewer does not parse").expect("write the source");
        let surface = table.open(&path).expect("the converted document");
        // A PDF answer keeps one plane per page, so the document keeps its
        // pages and the book view wraps it.
        assert_eq!(surface.plane_count(), 2);
        assert_eq!(
            surface.bounds(1).expect("the second page"),
            crate::surface::Size { w: 200.0, h: 100.0 }
        );
        assert!(surface.previews());
        assert!(surface.can_refine());
    }

    #[test]
    fn a_pdf_answer_becomes_a_book_over_the_source() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let pdf = dir.path().join("answer.pdf");
        std::fs::write(&pdf, crate::pdf::TWO_PAGE_PDF).expect("write the PDF");
        let program = script(
            &dir,
            "book.sh",
            &format!("cat > /dev/null\ncat '{}'", pdf.display()),
        );
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("doc.mmd");
        std::fs::write(&path, "x").expect("write the source");
        let plain = table.open(&path).expect("the converted document");
        assert_eq!(plain.plane_count(), 2);
        // The book view stacks every spread of a multi-plane source.
        let inner: std::sync::Arc<dyn Surface> = plain.into();
        let book = crate::book::BookSurface::new(inner).expect("the book");
        assert_eq!(book.plane_count(), 1, "a book is one plane");
    }

    #[test]
    fn a_rule_runs_in_the_directory_of_the_source() {
        let dir = tempfile::tempdir().expect("a temp dir");
        // The converter answers a relative path, so the answer arrives only
        // when the working directory holds it.
        std::fs::write(
            dir.path().join("part.svg"),
            "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'><rect width='10' \
             height='10' fill='#ff0000'/></svg>",
        )
        .expect("write the relative part");
        let program = script(&dir, "rel.sh", "cat > /dev/null\ncat part.svg");
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("doc.mmd");
        std::fs::write(&path, "x").expect("write the source");
        let surface = table.open(&path).expect("the answer beside the source");
        assert_eq!(surface.plane_count(), 1);
    }

    #[test]
    fn a_failing_converter_reports_its_code_and_stderr() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(&dir, "fail.sh", "echo 'no diagram here' >&2\nexit 3");
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "x").expect("write the source");
        let Err(err) = table.open(&path) else {
            panic!("the conversion fails");
        };
        let text = err.to_string();
        assert!(text.contains("code 3"), "{text}");
        assert!(text.contains("no diagram here"), "{text}");
    }

    #[test]
    fn a_slow_converter_is_killed() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(&dir, "slow.sh", "cat > /dev/null\nsleep 30");
        let mut rule = rule(&program);
        rule.timeout = Duration::from_millis(100);
        let table = Converters::in_dir(vec![rule], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "x").expect("write the source");
        let started = Instant::now();
        let Err(err) = table.open(&path) else {
            panic!("the conversion times out");
        };
        assert!(err.to_string().contains("ran longer"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(5), "the kill lands");
    }

    #[test]
    fn a_flood_of_output_hits_the_cap() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(&dir, "flood.sh", "cat > /dev/null\nyes '<svg/>'");
        let mut rule = rule(&program);
        rule.max_output = 1024;
        let table = Converters::in_dir(vec![rule], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "x").expect("write the source");
        let Err(err) = table.open(&path) else {
            panic!("the cap stops the read");
        };
        assert!(err.to_string().contains("output cap"), "{err}");
    }

    #[test]
    fn a_source_past_the_cap_is_refused() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(
            &dir,
            "cat.sh",
            &format!("cat > /dev/null\nprintf '%s' '{SVG}'"),
        );
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, vec![b'x'; MAX_SOURCE + 1]).expect("write the source");
        let Err(err) = table.open(&path) else {
            panic!("the cap refuses the source");
        };
        assert!(err.to_string().contains("source cap"), "{err}");
    }

    #[test]
    fn an_answer_that_is_no_svg_fails_readably() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(&dir, "html.sh", "cat > /dev/null\necho '<html></html>'");
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "x").expect("write the source");
        let Err(err) = table.open(&path) else {
            panic!("html is no svg");
        };
        assert!(
            err.to_string()
                .contains("neither a PDF nor an SVG document"),
            "{err}"
        );
    }

    #[cfg(feature = "cache")]
    #[test]
    fn a_second_open_reads_the_cached_svg() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(
            &dir,
            "cache.sh",
            &format!("cat > /dev/null\nprintf '%s' '{SVG}'"),
        );
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "flowchart LR\n").expect("write the source");
        table.open(&path).expect("the first conversion");
        // The program is gone, so only the cache can answer now.
        std::fs::remove_file(&program).expect("remove the script");
        table.open(&path).expect("the cached conversion");
    }

    #[test]
    fn a_chatty_stderr_keeps_the_answer() {
        let dir = tempfile::tempdir().expect("a temp dir");
        // 2000 lines, far past the 4 KiB the error message keeps.
        let body = format!(
            r#"cat > /dev/null
i=0
while [ $i -lt 2000 ]; do
  echo "a warning line $i" >&2
  i=$((i+1))
done
printf '%s' '{SVG}'
"#
        );
        let program = script(&dir, "noisy.sh", &body);
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "x").expect("write the source");
        // A closed stderr would kill the child with `SIGPIPE` before it writes
        // its answer, so the surface must open.
        table
            .open(&path)
            .expect("the chatty converter still answers");
    }

    #[test]
    fn a_grandchild_that_holds_stdout_keeps_the_answer() {
        let dir = tempfile::tempdir().expect("a temp dir");
        let program = script(
            &dir,
            "fork.sh",
            &format!("cat > /dev/null\n(sleep 30) &\nprintf '%s' '{SVG}'"),
        );
        let table = Converters::in_dir(vec![rule(&program)], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "x").expect("write the source");
        // The pipe never reaches its end, so the worker takes what arrived
        // once the grace period runs out: one period for both pipes, never
        // one apiece.
        let started = Instant::now();
        let base = table
            .open(&path)
            .expect("the answer survives the open pipe")
            .render(View::whole(0, 1.0))
            .expect("the base raster");
        assert_eq!(&base.as_rgba()[..4], &[255, 0, 0, 255]);
        assert!(
            started.elapsed() < GRACE + Duration::from_millis(1500),
            "one grace period covers the two pipes, took {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn a_timed_out_converter_takes_its_helpers_with_it() {
        let dir = tempfile::tempdir().expect("a temp dir");
        // The helper writes its own pid, so the test can follow it after the
        // clock kills the converter that started it.
        let pidfile = dir.path().join("helper.pid");
        let program = script(
            &dir,
            "fork.sh",
            &format!(
                "cat > /dev/null\nsleep 30 &\necho $! > '{}'\nsleep 30",
                pidfile.display()
            ),
        );
        let mut rule = rule(&program);
        rule.timeout = Duration::from_millis(500);
        let table = Converters::in_dir(vec![rule], dir.path());
        let path = dir.path().join("flow.mmd");
        std::fs::write(&path, "x").expect("write the source");
        let Err(err) = table.open(&path) else {
            panic!("the conversion times out");
        };
        assert!(err.to_string().contains("ran longer"), "{err}");
        let pid = std::fs::read_to_string(&pidfile)
            .expect("the helper wrote its pid")
            .trim()
            .to_owned();
        // The clock kills the group the spawn created, so the helper leaves
        // with the converter instead of holding a browser open.
        let alive = || Path::new(&format!("/proc/{pid}")).exists();
        let deadline = Instant::now() + Duration::from_secs(5);
        while alive() && Instant::now() < deadline {
            std::thread::sleep(POLL);
        }
        assert!(!alive(), "the helper of the timed-out converter died");
    }

    #[test]
    fn one_slot_serializes_the_conversions() {
        use std::sync::mpsc;
        let slots = Arc::new(Slots::new(1));
        let held = slots.acquire();
        let (tx, rx) = mpsc::channel();
        let worker = {
            let slots = Arc::clone(&slots);
            std::thread::spawn(move || {
                let _second = slots.acquire();
                let _ = tx.send(());
            })
        };
        // The second worker waits while the one slot stays held.
        assert!(
            rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "the second conversion must wait"
        );
        drop(held);
        rx.recv_timeout(Duration::from_secs(5))
            .expect("the released slot lets the second conversion start");
        worker.join().expect("the worker ends");
    }
}
