//! On-disk thumbnail cache under `$XDG_CACHE_HOME`.
//!
//! The cache keeps the grid thumbnails of a whole directory on disk, so a
//! second launch of the same directory uploads textures instead of decoding
//! files.
//!
//! The file name is the SHA-256 of the canonical image path plus the
//! thumbnail long side, hex encoded, with a `.thumb` extension.
//! The content carries the source stamp (length and modification time) and
//! the image dimensions in a fixed header, so a changed source file never
//! shows a stale thumbnail.
//!
//! The cache lives in `$XDG_CACHE_HOME/vv/thumbnails`, which is `~/.cache`
//! when the variable is unset.
//! Without a home directory the cache is off, and the grid decodes every file
//! like before.
//! `VV_CACHE=0` turns the cache off too.
//! The cache directory is created on the first write, so a viewer that only
//! reads never creates it.
//! The cache keeps every entry past the exit of the viewer, so the next launch
//! starts warm.
//! A stale entry answers with [`CacheError::Stale`] and re-decodes, and an
//! entry whose source is gone stays unread.
//! The cache directory is user-private and the source path is user-private, so
//! the entries hold no data from another user.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use derive_more::{Display, Error, From};
use sha2::{Digest, Sha256};

use crate::{
    DecodedImage,
    dim::{ImgH, ImgW, LongSide},
};

/// The directory under the XDG cache home.
const DIR: &str = "vv/thumbnails";
/// File extension of a cached thumbnail.
const EXT: &str = "thumb";
/// `b"VVTH"`: the cache file magic.
const MAGIC: [u8; 4] = *b"VVTH";
/// The header layout version.
/// Change it when the header or the pixel layout changes, so old files never
/// load.
/// A file of another format never matches the magic, so a change of the pixel
/// layout needs a version bump only for a build that reads the same magic.
const FORMAT: u32 = 1;
/// Header bytes before the RGBA8 pixels.
/// magic, format, thumb width, thumb height, image width, image height,
/// source length, source mtime seconds, source mtime nanoseconds, source
/// path length.
const HEADER: usize = 4 + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 4 + 4;
/// The longest source path the header stores.
const MAX_PATH: usize = 4096;
/// Writes beyond this budget stop for the rest of the run, so a burst of
/// directories never fills the disk in one launch.
/// A later run starts the budget over.
const WRITE_BUDGET: u64 = 256 * 1024 * 1024;
/// A half-written temporary file older than this is dead, and the next write
/// of its key removes it.
const TMP_MAX_AGE: Duration = Duration::from_secs(3600);

/// Why a cache file did not load or store.
///
/// A miss is normal, and the grid then decodes the file.
/// Every variant means the grid decodes the file.
#[derive(Debug, Display, Error, From)]
pub enum CacheError {
    /// The cache is off: no cache home, or `VV_CACHE=0`.
    #[display("cache is off")]
    Off,
    /// The cache file does not exist, so this is a normal miss.
    #[display("cache file does not exist")]
    #[from(skip)]
    NotFound(#[error(ignore)] std::io::Error),
    /// The cache file or directory cannot be read or written.
    #[display("cache I/O failed")]
    Io(#[from] std::io::Error),
    /// The cache file holds another path, size or mtime, so the source
    /// changed.
    #[display("cache entry is stale")]
    Stale,
    /// The cache file bytes do not match the format.
    #[display("cache file is corrupt")]
    Corrupt,
    /// This run already wrote [`WRITE_BUDGET`] bytes, so the cache stops
    /// storing. A later run starts the budget over.
    #[display("cache write budget is spent")]
    Full,
}

/// A thumbnail from the cache, with the source dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedThumb {
    /// The cached RGBA8 thumbnail.
    pub thumb: DecodedImage,
    /// The source dimensions, as a decode capped at
    /// [`crate::MAX_TEXTURE_SIDE`] would report them.
    pub image: (ImgW, ImgH),
}

/// The thumbnail cache.
///
/// The code shares one `&Cache` with every decode worker.
/// The write budget is atomic, so no lock guards it.
#[derive(Debug)]
pub struct Cache {
    /// The cache directory, or None when the cache is off.
    dir: Option<PathBuf>,
    /// The bytes this run wrote so far, against [`WRITE_BUDGET`].
    written: AtomicU64,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

impl Cache {
    /// Open the cache under `$XDG_CACHE_HOME/vv/thumbnails`.
    ///
    /// The function never fails and never creates a directory.
    /// A missing home directory turns the cache off.
    #[must_use]
    pub fn new() -> Self {
        let off = std::env::var_os("VV_CACHE").is_some_and(|v| v == "0");
        let dir = (!off)
            .then(|| xdg::BaseDirectories::new().get_cache_home())
            .flatten()
            .map(|dir| dir.join(DIR));
        Self {
            dir,
            written: AtomicU64::new(0),
        }
    }

    /// A cache with no directory, for a viewer without a home directory and
    /// for tests.
    #[must_use]
    pub fn off() -> Self {
        Self {
            dir: None,
            written: AtomicU64::new(0),
        }
    }

    /// A cache over `dir`, which bypasses the cache home lookup.
    ///
    /// The directory is created on the first write, like the real one.
    /// A test uses this to keep its files inside a temporary directory, so it
    /// never fills the cache of the user.
    #[must_use]
    pub fn in_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: Some(dir.into()),
            written: AtomicU64::new(0),
        }
    }

    /// The cache directory, when the cache is on.
    #[must_use]
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// Read the cached thumbnail of `path` at `long_side`.
    ///
    /// The file must match the source length and modification time, so a
    /// rewritten image never shows the old thumbnail.
    /// Run this on a decode worker, never on the main thread.
    ///
    /// # Errors
    ///
    /// The function errors on a miss, a changed source, or a corrupt file.
    pub fn load(&self, path: &Path, long_side: LongSide) -> Result<CachedThumb, CacheError> {
        let (file, canonical) = self.entry(path, long_side)?;
        let bytes = fs::read(&file).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CacheError::NotFound(e)
            } else {
                CacheError::Io(e)
            }
        })?;
        decode_entry(&bytes, &Stamp::of(&canonical)?, &canonical)
    }

    /// Store the thumbnail of `path` at `long_side`.
    ///
    /// The function writes a temporary file first and renames it, so a reader
    /// never sees a half-written file.
    /// Two writers of one key write the same bytes, so the last rename wins
    /// with no harm.
    /// Run this on a decode worker.
    ///
    /// # Errors
    ///
    /// The function errors when the cache is off, the source cannot be
    /// stamped, the buffer does not match its dimensions, or the write fails.
    pub fn store(
        &self,
        path: &Path,
        long_side: LongSide,
        thumb: &DecodedImage,
        image: (ImgW, ImgH),
    ) -> Result<(), CacheError> {
        let (file, canonical) = self.entry(path, long_side)?;
        let bytes = encode_entry(&Stamp::of(&canonical)?, &canonical, thumb, image)?;
        let len = bytes.len() as u64;
        // One atomic add owns the budget, so two workers never overshoot it.
        let mut used = self.written.fetch_add(len, Ordering::Relaxed);
        if used.saturating_add(len) > WRITE_BUDGET {
            self.written.fetch_sub(len, Ordering::Relaxed);
            return Err(CacheError::Full);
        }
        used += len;
        let dir = file.parent().ok_or(CacheError::Off)?;
        let write = fs::create_dir_all(dir).and_then(|()| {
            sweep_dead_tmp(dir);
            let tmp = file.with_extension(format!("{EXT}.{}.{used}.tmp", std::process::id()));
            fs::write(&tmp, &bytes)
                .and_then(|()| fs::rename(&tmp, &file))
                .inspect_err(|_| {
                    let _ = fs::remove_file(&tmp);
                })
        });
        if let Err(err) = write {
            self.written.fetch_sub(len, Ordering::Relaxed);
            return Err(CacheError::Io(err));
        }
        Ok(())
    }

    /// The cache file for `path` at `long_side`, or None when the cache is
    /// off or the path does not resolve.
    #[must_use]
    pub fn file(&self, path: &Path, long_side: LongSide) -> Option<PathBuf> {
        self.entry(path, long_side).ok().map(|(file, _)| file)
    }

    /// The cache file and the canonical source path for `path` at
    /// `long_side`.
    ///
    /// The code hashes the canonical path, so the name is independent of the
    /// launch directory, and the file header stores the same path.
    ///
    /// # Errors
    ///
    /// The function errors when the cache is off or the path does not
    /// resolve.
    fn entry(&self, path: &Path, long_side: LongSide) -> Result<(PathBuf, PathBuf), CacheError> {
        let dir = self.dir.as_ref().ok_or(CacheError::Off)?;
        let canonical = fs::canonicalize(path)?;
        let file = dir.join(format!("{}.{EXT}", key(&canonical, long_side)));
        Ok((file, canonical))
    }
}

/// Remove the half-written temporary files of killed workers from `dir`.
///
/// A crash leaves its temporary file behind, and no sweep runs on exit.
/// The next store drops the dead ones, so they never fill the cache home.
/// A `*.thumb` entry is never removed: a stale entry answers
/// [`CacheError::Stale`] and the source re-decodes.
fn sweep_dead_tmp(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    for entry in entries.flatten() {
        let path = entry.path();
        // A temporary file is named `<key>.thumb.<pid>.<n>.tmp`, so only the
        // `.tmp` tail identifies it.
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("tmp"))
        {
            continue;
        }
        let dead = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default() + TMP_MAX_AGE < now);
        if dead {
            let _ = fs::remove_file(&path);
        }
    }
}

/// The hex SHA-256 of the canonical path and the thumbnail long side.
///
/// The path bytes go in as `OsStr` bytes, so a non-UTF-8 name keeps a
/// distinct key.
#[must_use]
pub fn key(path: &Path, long_side: LongSide) -> String {
    use std::os::unix::ffi::OsStrExt;
    let mut h = Sha256::new();
    h.update(path.as_os_str().as_bytes());
    h.update([0]);
    h.update(long_side.get().to_le_bytes());
    let mut s = String::with_capacity(64);
    for b in h.finalize() {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// The source stamp: length and modification time.
///
/// A cached entry carries the stamp of the file it came from.
/// A rewrite changes the length or the mtime, so the entry stops matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Stamp {
    /// The source length in bytes.
    len: u64,
    /// The modification time in whole seconds since the Unix epoch.
    secs: u64,
    /// The sub-second part of the modification time, in nanoseconds.
    nanos: u32,
}

impl Stamp {
    /// The stamp of the file at `path`.
    fn of(path: &Path) -> Result<Self, CacheError> {
        let meta = fs::metadata(path)?;
        let since = meta
            .modified()
            .unwrap_or(UNIX_EPOCH)
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        Ok(Self {
            len: meta.len(),
            secs: since.as_secs(),
            nanos: since.subsec_nanos(),
        })
    }
}

/// The cache file bytes: header and RGBA8 pixels.
fn encode_entry(
    stamp: &Stamp,
    path: &Path,
    thumb: &DecodedImage,
    image: (ImgW, ImgH),
) -> Result<Vec<u8>, CacheError> {
    use std::os::unix::ffi::OsStrExt;
    let src = path.as_os_str().as_bytes();
    if src.len() > MAX_PATH {
        return Err(CacheError::Corrupt);
    }
    let expect = usize::try_from(thumb.width.get())
        .ok()
        .and_then(|w| {
            usize::try_from(thumb.height.get())
                .ok()
                .and_then(|h| w.checked_mul(h))
        })
        .and_then(|px| px.checked_mul(4));
    if expect != Some(thumb.data.len()) || expect == Some(0) {
        return Err(CacheError::Corrupt);
    }
    let mut out = Vec::with_capacity(HEADER + src.len() + thumb.data.len());
    for field in [
        u32::from_le_bytes(MAGIC),
        FORMAT,
        thumb.width.get(),
        thumb.height.get(),
        image.0.get(),
        image.1.get(),
    ] {
        out.extend_from_slice(&field.to_le_bytes());
    }
    out.extend_from_slice(&stamp.len.to_le_bytes());
    out.extend_from_slice(&stamp.secs.to_le_bytes());
    out.extend_from_slice(&stamp.nanos.to_le_bytes());
    out.extend_from_slice(&(src.len() as u32).to_le_bytes());
    out.extend_from_slice(src);
    out.extend_from_slice(&thumb.data);
    Ok(out)
}

/// Parse cache file bytes, with the source stamp and path checked.
fn decode_entry(bytes: &[u8], stamp: &Stamp, path: &Path) -> Result<CachedThumb, CacheError> {
    use std::os::unix::ffi::OsStrExt;
    let mut r = Reader::new(bytes);
    if r.u32()? != u32::from_le_bytes(MAGIC) || r.u32()? != FORMAT {
        return Err(CacheError::Corrupt);
    }
    let thumb_w = ImgW::new(r.u32()?);
    let thumb_h = ImgH::new(r.u32()?);
    let image = (ImgW::new(r.u32()?), ImgH::new(r.u32()?));
    let stored = Stamp {
        len: r.u64()?,
        secs: r.u64()?,
        nanos: r.u32()?,
    };
    let path_len = r.u32()? as usize;
    if path_len > MAX_PATH {
        return Err(CacheError::Corrupt);
    }
    let (stored_path, data) = bytes
        .get(HEADER..HEADER + path_len)
        .zip(bytes.get(HEADER + path_len..))
        .ok_or(CacheError::Corrupt)?;
    if stored != *stamp || stored_path != path.as_os_str().as_bytes() {
        return Err(CacheError::Stale);
    }
    let bytes = |v: u32| usize::try_from(v).ok();
    let expect = bytes(thumb_w.get())
        .and_then(|w| bytes(thumb_h.get()).and_then(|h| w.checked_mul(h)))
        .and_then(|px| px.checked_mul(4));
    if expect != Some(data.len()) || expect == Some(0) {
        return Err(CacheError::Corrupt);
    }
    Ok(CachedThumb {
        thumb: DecodedImage {
            width: thumb_w,
            height: thumb_h,
            data: data.to_vec(),
        },
        image,
    })
}

/// A cursor over the fixed header, which keeps the reads bounds checked.
struct Reader<'a> {
    bytes: &'a [u8],
    /// The read position.
    at: usize,
}

impl<'a> Reader<'a> {
    /// A cursor at the start of `bytes`.
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    /// The next four bytes as a little-endian `u32`.
    fn u32(&mut self) -> Result<u32, CacheError> {
        let end = self.at + 4;
        let bytes = self
            .bytes
            .get(self.at..end)
            .and_then(|s| <[u8; 4]>::try_from(s).ok())
            .ok_or(CacheError::Corrupt)?;
        self.at = end;
        Ok(u32::from_le_bytes(bytes))
    }

    /// The next eight bytes as a little-endian `u64`.
    fn u64(&mut self) -> Result<u64, CacheError> {
        let end = self.at + 8;
        let bytes = self
            .bytes
            .get(self.at..end)
            .and_then(|s| <[u8; 8]>::try_from(s).ok())
            .ok_or(CacheError::Corrupt)?;
        self.at = end;
        Ok(u64::from_le_bytes(bytes))
    }
}

#[cfg(test)]
mod tests {
    use std::{io::Write as _, os::unix::ffi::OsStrExt as _};

    use super::*;

    /// A cache in a fresh temporary directory, plus that directory.
    fn temp_cache() -> (tempfile::TempDir, Cache) {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = Cache::in_dir(dir.path().join("cache"));
        (dir, cache)
    }

    fn thumb(w: u32, h: u32, v: u8) -> DecodedImage {
        DecodedImage {
            width: ImgW::new(w),
            height: ImgH::new(h),
            data: vec![v; (w * h * 4) as usize],
        }
    }

    #[test]
    fn key_is_hex_and_distinct_per_path_and_size() {
        let a = key(Path::new("/tmp/a.jpg"), LongSide::new(512));
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, key(Path::new("/tmp/b.jpg"), LongSide::new(512)));
        assert_ne!(a, key(Path::new("/tmp/a.jpg"), LongSide::new(1024)));
        assert_eq!(a, key(Path::new("/tmp/a.jpg"), LongSide::new(512)));
    }

    #[test]
    fn store_then_load_round_trips() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"not really a png").expect("write");
        let image = thumb(8, 4, 7);
        cache
            .store(
                &src,
                LongSide::new(512),
                &image,
                (ImgW::new(64), ImgH::new(32)),
            )
            .expect("store");
        let out = cache.load(&src, LongSide::new(512)).expect("load");
        assert_eq!(out.thumb, image);
        assert_eq!(out.image, (ImgW::new(64), ImgH::new(32)));
        // A second key in the same run takes its own temporary name, so the
        // two workers never write over each other's file.
        let other = dir.path().join("other.png");
        fs::write(&other, b"not really a png either").expect("write");
        for p in [&src, &other] {
            cache
                .store(
                    p,
                    LongSide::new(512),
                    &image,
                    (ImgW::new(64), ImgH::new(32)),
                )
                .expect("store");
            assert!(cache.load(p, LongSide::new(512)).is_ok(), "{p:?}");
        }
    }

    #[test]
    fn another_long_side_holds_another_entry() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        cache
            .store(
                &src,
                LongSide::new(512),
                &thumb(2, 2, 1),
                (ImgW::new(2), ImgH::new(2)),
            )
            .expect("store");
        assert!(matches!(
            cache.load(&src, LongSide::new(1024)),
            Err(CacheError::NotFound(_))
        ));
    }

    #[test]
    fn a_rewritten_source_is_stale() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"one").expect("write");
        cache
            .store(
                &src,
                LongSide::new(512),
                &thumb(2, 2, 1),
                (ImgW::new(2), ImgH::new(2)),
            )
            .expect("store");
        // A longer file changes the stamp even inside the same mtime tick.
        let mut f = fs::File::create(&src).expect("create");
        f.write_all(b"much longer content").expect("write");
        drop(f);
        assert!(matches!(
            cache.load(&src, LongSide::new(512)),
            Err(CacheError::Stale)
        ));
    }

    #[test]
    fn a_missing_file_is_not_found() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        assert!(matches!(
            cache.load(&src, LongSide::new(512)),
            Err(CacheError::NotFound(_))
        ));
    }

    #[test]
    fn a_corrupt_file_does_not_load() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        let file = cache.file(&src, LongSide::new(512)).expect("cache on");
        fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
        fs::write(&file, b"VVTH").expect("write");
        assert!(matches!(
            cache.load(&src, LongSide::new(512)),
            Err(CacheError::Corrupt)
        ));
        // A truncated pixel block is corrupt too.
        let mut bad = Vec::new();
        bad.extend_from_slice(&MAGIC);
        bad.extend_from_slice(&FORMAT.to_le_bytes());
        for field in [4u32, 4, 8, 8] {
            bad.extend_from_slice(&field.to_le_bytes());
        }
        let stamp = Stamp::of(&src).expect("stamp");
        bad.extend_from_slice(&stamp.len.to_le_bytes());
        bad.extend_from_slice(&stamp.secs.to_le_bytes());
        bad.extend_from_slice(&stamp.nanos.to_le_bytes());
        // A matching path, so the pixel length decides.
        let src_bytes = src.as_os_str().as_bytes();
        bad.extend_from_slice(&(src_bytes.len() as u32).to_le_bytes());
        bad.extend_from_slice(src_bytes);
        bad.extend_from_slice(&[0; 3]);
        fs::write(&file, &bad).expect("write");
        assert!(matches!(
            cache.load(&src, LongSide::new(512)),
            Err(CacheError::Corrupt)
        ));
    }

    #[test]
    fn another_path_in_the_file_is_stale() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        let other = dir.path().join("other.png");
        fs::write(&other, b"x").expect("write");
        let bytes = encode_entry(
            &Stamp::of(&other).expect("stamp"),
            &other,
            &thumb(2, 2, 1),
            (ImgW::new(2), ImgH::new(2)),
        )
        .expect("encode");
        let file = cache.file(&src, LongSide::new(512)).expect("cache on");
        fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
        fs::write(&file, &bytes).expect("write");
        assert!(matches!(
            cache.load(&src, LongSide::new(512)),
            Err(CacheError::Stale)
        ));
    }

    #[test]
    fn a_mismatched_buffer_does_not_store() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        let bad = DecodedImage {
            width: ImgW::new(4),
            height: ImgH::new(4),
            data: vec![0; 3],
        };
        assert!(matches!(
            cache.store(&src, LongSide::new(512), &bad, (ImgW::new(4), ImgH::new(4))),
            Err(CacheError::Corrupt)
        ));
    }

    #[test]
    fn a_missing_source_does_not_store() {
        let (_dir, cache) = temp_cache();
        assert!(matches!(
            cache.store(
                Path::new("/nonexistent"),
                LongSide::new(512),
                &thumb(1, 1, 0),
                (ImgW::new(1), ImgH::new(1))
            ),
            Err(CacheError::Io(_))
        ));
    }

    #[test]
    fn an_off_cache_does_nothing() {
        let cache = Cache::off();
        let src = Path::new("/nonexistent");
        assert!(cache.file(src, LongSide::new(512)).is_none());
        assert!(cache.load(src, LongSide::new(512)).is_err());
        assert!(
            cache
                .store(
                    src,
                    LongSide::new(512),
                    &thumb(1, 1, 0),
                    (ImgW::new(1), ImgH::new(1))
                )
                .is_err()
        );
    }

    #[test]
    fn the_write_budget_stops_the_writes() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        cache.written.store(WRITE_BUDGET, Ordering::Relaxed);
        assert!(matches!(
            cache.store(
                &src,
                LongSide::new(512),
                &thumb(2, 2, 1),
                (ImgW::new(2), ImgH::new(2))
            ),
            Err(CacheError::Full)
        ));
        // A refused write leaves the budget alone.
        assert_eq!(cache.written.load(Ordering::Relaxed), WRITE_BUDGET);
    }

    #[test]
    fn a_stored_entry_outlives_the_cache() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        cache
            .store(
                &src,
                LongSide::new(512),
                &thumb(2, 2, 1),
                (ImgW::new(2), ImgH::new(2)),
            )
            .expect("store");
        let file = cache.file(&src, LongSide::new(512)).expect("cache on");
        drop(cache);
        assert!(file.exists(), "the exit keeps the cache entries");
    }

    #[test]
    fn a_store_sweeps_a_dead_temporary_file() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        let cache_dir = cache.dir().expect("cache on").to_path_buf();
        fs::create_dir_all(&cache_dir).expect("mkdir");
        let tmp = cache_dir.join("dead.thumb.123.tmp");
        fs::write(&tmp, b"x").expect("write");
        let old = filetime_ago(&tmp, Duration::from_secs(7200));
        assert!(old, "could not age the temporary file");
        cache
            .store(
                &src,
                LongSide::new(512),
                &thumb(2, 2, 1),
                (ImgW::new(2), ImgH::new(2)),
            )
            .expect("store");
        assert!(!tmp.exists(), "the store drops the dead temporary file");
        assert!(cache.load(&src, LongSide::new(512)).is_ok());
        drop(dir);
    }

    /// Set the mtime of `path` this far into the past, when the platform
    /// allows it.
    fn filetime_ago(path: &Path, ago: Duration) -> bool {
        let past = SystemTime::now() - ago;
        let Ok(f) = fs::File::options().write(true).open(path) else {
            return false;
        };
        f.set_modified(past).is_ok()
    }
}
