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
//! A vector source stores its base raster size there (see [`crate::svg`]),
//! because the file declares no pixel size of its own.
//! The header also carries the [`Placeholder`] of the thumbnail: a `ThumbHash`
//! of at most 32 bytes.
//! A launch reads that header only (see [`Cache::load_placeholder`]), so a
//! warm grid paints every cell from a 200 byte read per file, before any
//! JPEG XL decode.
//! A JPEG XL codestream follows the header: lossy at [`QUALITY`], with an
//! exact alpha channel.
//! A 512 px entry therefore costs tens of kilobytes, not the megabyte of raw
//! RGBA8.
//! The encoder is `jixel`, pure Rust, and the decoder is the same `jxl-oxide`
//! the viewer uses for a JPEG XL file.
//! Both run on a decode worker, so the main thread never encodes.
//!
//! The cache lives in `$XDG_CACHE_HOME/vv/thumbnails`, which is `~/.cache`
//! when the variable is unset.
//! Without a home directory the cache is off, and the grid decodes every file
//! like before.
//! `VV_CACHE=0` turns the cache off too.
//! A build without the `cache` feature answers [`CacheError::Off`] from every
//! call, and `src/thumbcache_off.rs` holds that stub.
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
    placeholder::Placeholder,
};

/// The directory under the XDG cache home.
const DIR: &str = "vv/thumbnails";
/// File extension of a cached thumbnail.
const EXT: &str = "thumb";
/// `b"VVTH"`: the cache file magic.
const MAGIC: [u8; 4] = *b"VVTH";
/// The header layout version.
/// Change it when the header or the entry payload changes, so old files never
/// load.
/// A file of another format never matches the magic, so a change of the
/// payload needs a version bump only for a build that reads the same magic.
/// Version 2 holds a lossy JPEG XL codestream where version 1 held RGBA8
/// pixels, so every version 1 entry stays unread and the source re-decodes.
/// Version 3 adds the [`Placeholder`] payload after the source stamp.
const FORMAT: u32 = 3;
/// The JPEG XL quality of an entry, on the `jixel` 0..100 scale
/// (`distance_from_quality`).
/// 95 is distance 0.1.
/// The grid draws a 512 px thumb smaller than 512 px in a normal cell, so the
/// loss stays invisible, and a warm cell differs from a cold one only by that
/// loss.
const QUALITY: f32 = 95.0;
/// The alpha distance of an entry. Zero keeps the alpha channel exact, while
/// the color channels stay lossy.
/// A default would follow [`QUALITY`] and band a transparent gradient.
const ALPHA_DISTANCE: f32 = 0.0;
/// Header bytes before the source path.
/// magic, format, thumb width, thumb height, image width, image height,
/// source length, source mtime seconds, source mtime nanoseconds, source
/// path length, placeholder length, placeholder bytes.
const HEADER: usize = 4 + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 4 + 4 + 1 + crate::placeholder::MAX_BYTES;
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
    /// The cache file holds no placeholder payload, so a launch shows the
    /// plain grey cell until the decode lands.
    #[display("cache entry holds no placeholder")]
    #[from(skip)]
    NoPlaceholder,
    /// The encoder rejected the thumbnail, so this entry never reaches the
    /// disk. The grid already holds the pixels, so the only loss is the warm
    /// launch.
    #[display("cache encode failed")]
    Encode(#[from] jixel::EncodeError),
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

    /// Read the placeholder of `path` at `long_side`, plus the source
    /// dimensions.
    ///
    /// The function reads the fixed header and the source path only, so it
    /// never touches the codestream and never decodes an image.
    /// That makes it the cheap first pass of a launch: one open per file,
    /// about 200 bytes of reads, and a DCT render of 32 px on the worker.
    /// The file must match the source length and modification time, like
    /// [`Self::load`].
    /// Run this on a decode worker, never on the main thread.
    ///
    /// # Errors
    ///
    /// The function errors on a miss, a changed source, a corrupt header, or
    /// an entry without a placeholder payload.
    pub fn load_placeholder(
        &self,
        path: &Path,
        long_side: LongSide,
    ) -> Result<(Placeholder, (ImgW, ImgH)), CacheError> {
        use std::{io::Read as _, os::unix::ffi::OsStrExt as _};
        let (file, canonical) = self.entry(path, long_side)?;
        let mut file = fs::File::open(&file).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CacheError::NotFound(e)
            } else {
                CacheError::Io(e)
            }
        })?;
        let mut head = [0u8; HEADER];
        file.read_exact(&mut head)
            .map_err(|_| CacheError::Corrupt)?;
        let head = parse_head(&head)?;
        let mut stored_path = vec![0u8; head.path_len];
        file.read_exact(&mut stored_path)
            .map_err(|_| CacheError::Corrupt)?;
        if head.stamp != Stamp::of(&canonical)? || stored_path != canonical.as_os_str().as_bytes() {
            return Err(CacheError::Stale);
        }
        let hash = head.hash.ok_or(CacheError::NoPlaceholder)?;
        Ok((hash, head.image))
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

/// The cache file bytes: header and a lossy JPEG XL codestream.
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
    let codestream = jixel::encode_image_with_alpha(
        &thumb.data,
        usize::try_from(thumb.width.get()).map_err(|_| CacheError::Corrupt)?,
        usize::try_from(thumb.height.get()).map_err(|_| CacheError::Corrupt)?,
        &jixel::EncodeConfig::default()
            .with_quality(QUALITY)
            .with_alpha_distance(ALPHA_DISTANCE),
    )?;
    let mut out = Vec::with_capacity(HEADER + src.len() + codestream.len());
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
    // The placeholder is a hint, so a payload the encoder cannot fit writes a
    // zero length and the entry still loads.
    let hash = Placeholder::encode(thumb);
    out.push(hash.map_or(0, |h| h.len() as u8));
    let mut padded = [0u8; crate::placeholder::MAX_BYTES];
    if let Some(hash) = hash {
        let bytes = hash.as_bytes();
        padded
            .get_mut(..bytes.len())
            .ok_or(CacheError::Corrupt)?
            .copy_from_slice(bytes);
    }
    out.extend_from_slice(&padded);
    out.extend_from_slice(src);
    out.extend_from_slice(&codestream);
    Ok(out)
}

/// The fixed header of an entry, parsed.
struct Head {
    /// The thumbnail dimensions.
    thumb: (ImgW, ImgH),
    /// The source dimensions.
    image: (ImgW, ImgH),
    /// The stamp the entry was written with.
    stamp: Stamp,
    /// The length of the source path that follows the header.
    path_len: usize,
    /// The placeholder payload, or None when the entry carries none.
    hash: Option<Placeholder>,
}

/// Parse the fixed [`HEADER`] bytes of an entry.
///
/// The function reads no bytes beyond the header, so a caller that needs only
/// the placeholder reads that much and stops.
fn parse_head(bytes: &[u8]) -> Result<Head, CacheError> {
    let mut r = Reader::new(bytes);
    if r.u32()? != u32::from_le_bytes(MAGIC) || r.u32()? != FORMAT {
        return Err(CacheError::Corrupt);
    }
    let thumb = (ImgW::new(r.u32()?), ImgH::new(r.u32()?));
    let image = (ImgW::new(r.u32()?), ImgH::new(r.u32()?));
    let stamp = Stamp {
        len: r.u64()?,
        secs: r.u64()?,
        nanos: r.u32()?,
    };
    let path_len = r.u32()? as usize;
    if path_len > MAX_PATH {
        return Err(CacheError::Corrupt);
    }
    let hash_len = usize::from(r.u8()?);
    let payload = r.slice(crate::placeholder::MAX_BYTES)?;
    if hash_len > crate::placeholder::MAX_BYTES {
        return Err(CacheError::Corrupt);
    }
    let hash = payload
        .get(..hash_len)
        .filter(|_| hash_len > 0)
        .and_then(Placeholder::from_bytes);
    Ok(Head {
        thumb,
        image,
        stamp,
        path_len,
        hash,
    })
}

/// Parse cache file bytes, with the source stamp and path checked.
fn decode_entry(bytes: &[u8], stamp: &Stamp, path: &Path) -> Result<CachedThumb, CacheError> {
    use std::os::unix::ffi::OsStrExt;
    let head = parse_head(bytes)?;
    let (stored_path, codestream) = bytes
        .get(HEADER..HEADER + head.path_len)
        .zip(bytes.get(HEADER + head.path_len..))
        .ok_or(CacheError::Corrupt)?;
    if head.stamp != *stamp || stored_path != path.as_os_str().as_bytes() {
        return Err(CacheError::Stale);
    }
    let thumb = decode_codestream(codestream)?;
    if (thumb.width, thumb.height) != head.thumb {
        return Err(CacheError::Corrupt);
    }
    Ok(CachedThumb {
        thumb,
        image: head.image,
    })
}

/// Decode the JPEG XL codestream of an entry into RGBA8.
///
/// A truncated or damaged codestream answers [`CacheError::Corrupt`], so the
/// grid decodes the source instead.
/// The decode runs on a decode worker, never on the main thread.
fn decode_codestream(codestream: &[u8]) -> Result<DecodedImage, CacheError> {
    let image = jxl_oxide::JxlImage::builder()
        .read(codestream)
        .map_err(|_| CacheError::Corrupt)?;
    let render = image.render_frame(0).map_err(|_| CacheError::Corrupt)?;
    crate::fb_to_rgba(&render.image_all_channels()).map_err(|_| CacheError::Corrupt)
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

    /// The next byte.
    fn u8(&mut self) -> Result<u8, CacheError> {
        let byte = *self.bytes.get(self.at).ok_or(CacheError::Corrupt)?;
        self.at += 1;
        Ok(byte)
    }

    /// The next `len` bytes.
    fn slice(&mut self, len: usize) -> Result<&'a [u8], CacheError> {
        let end = self.at + len;
        let bytes = self.bytes.get(self.at..end).ok_or(CacheError::Corrupt)?;
        self.at = end;
        Ok(bytes)
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

    /// A thumbnail with a smooth color ramp and a transparent gradient, so a
    /// lost channel or a broken alpha plane shows up in a comparison.
    /// The ramp stays gentle: a harsh synthetic edge drives the lossy encoder
    /// far past what a real 512 px thumbnail shows.
    fn varied_thumb(w: u32, h: u32) -> DecodedImage {
        let mut data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&[
                    (x * 40 / w.max(1)) as u8,
                    (y * 40 / h.max(1)) as u8,
                    ((x + y) * 20 / w.max(1)) as u8,
                    (255 - (x * 255 / w.max(1)) as u8),
                ]);
            }
        }
        DecodedImage {
            width: ImgW::new(w),
            height: ImgH::new(h),
            data,
        }
    }

    /// The largest channel difference a lossy entry may show against its
    /// source, per channel sample.
    /// A 512 px grid cell draws the cached thumb below its own size, so this
    /// bound never reads as a visible artifact.
    const MAX_DELTA: u8 = 40;

    /// Assert that a loaded thumb keeps the shape, the alpha plane and a
    /// bounded color error.
    fn assert_lossy_match(out: &DecodedImage, src: &DecodedImage) {
        assert_eq!((out.width, out.height), (src.width, src.height));
        assert_eq!(out.data.len(), src.data.len());
        for (a, b) in out
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .zip(src.data.as_chunks::<4>().0)
        {
            assert_eq!(a[3], b[3], "the alpha channel stays exact");
            assert!(a[0].abs_diff(b[0]) <= MAX_DELTA, "red: {a:?} against {b:?}");
            assert!(
                a[1].abs_diff(b[1]) <= MAX_DELTA,
                "green: {a:?} against {b:?}"
            );
            assert!(
                a[2].abs_diff(b[2]) <= MAX_DELTA,
                "blue: {a:?} against {b:?}"
            );
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
        let image = varied_thumb(8, 4);
        cache
            .store(
                &src,
                LongSide::new(512),
                &image,
                (ImgW::new(64), ImgH::new(32)),
            )
            .expect("store");
        let out = cache.load(&src, LongSide::new(512)).expect("load");
        assert_lossy_match(&out.thumb, &image);
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
    fn the_placeholder_reads_before_the_decode() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"not really a png").expect("write");
        let image = varied_thumb(64, 32);
        let want = (ImgW::new(640), ImgH::new(320));
        cache
            .store(&src, LongSide::new(512), &image, want)
            .expect("store");
        // The header alone answers the source dimensions and a payload, so a
        // launch paints the cell without reading the codestream.
        let (hash, got) = cache
            .load_placeholder(&src, LongSide::new(512))
            .expect("placeholder");
        assert_eq!(got, want);
        assert!(!hash.is_empty());
        assert!(hash.len() <= crate::placeholder::MAX_BYTES);
        let rendered = hash.render().expect("render");
        assert!(rendered.width.get() > 0 && rendered.height.get() > 0);
        // The payload of the same entry is stable, so two reads agree.
        let (again, _) = cache
            .load_placeholder(&src, LongSide::new(512))
            .expect("placeholder");
        assert_eq!(again, hash);
    }

    #[test]
    fn a_rewritten_source_has_no_placeholder() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"one").expect("write");
        cache
            .store(
                &src,
                LongSide::new(512),
                &thumb(4, 4, 7),
                (ImgW::new(4), ImgH::new(4)),
            )
            .expect("store");
        let mut f = fs::File::create(&src).expect("create");
        f.write_all(b"much longer content").expect("write");
        drop(f);
        assert!(matches!(
            cache.load_placeholder(&src, LongSide::new(512)),
            Err(CacheError::Stale)
        ));
    }

    #[test]
    fn a_truncated_entry_has_no_placeholder() {
        let (dir, cache) = temp_cache();
        let src = dir.path().join("img.png");
        fs::write(&src, b"x").expect("write");
        let file = cache.file(&src, LongSide::new(512)).expect("cache on");
        fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
        // A header cut in the middle of the placeholder field.
        fs::write(&file, b"VVTH").expect("write");
        assert!(matches!(
            cache.load_placeholder(&src, LongSide::new(512)),
            Err(CacheError::Corrupt)
        ));
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
        // A truncated codestream is corrupt too.
        let valid = encode_entry(
            &Stamp::of(&src).expect("stamp"),
            &src,
            &varied_thumb(8, 4),
            (ImgW::new(8), ImgH::new(4)),
        )
        .expect("encode");
        fs::write(&file, &valid[..valid.len() - 16]).expect("write");
        assert!(matches!(
            cache.load(&src, LongSide::new(512)),
            Err(CacheError::Corrupt)
        ));
        // An empty payload is corrupt too.
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
        // A matching path, so the header-only payload reaches the decoder.
        let src_bytes = src.as_os_str().as_bytes();
        bad.extend_from_slice(&(src_bytes.len() as u32).to_le_bytes());
        // No placeholder payload, then the path and a stub codestream.
        bad.push(0);
        bad.extend_from_slice(&[0; crate::placeholder::MAX_BYTES]);
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
