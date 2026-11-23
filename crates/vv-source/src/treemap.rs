//! Treemaps: a weighted tree drawn as nested rectangles.
//!
//! A `.treemap` path opens this backend, and the bytes of that file say
//! nothing: the module builds its own sample tree, so the extension asks for a
//! way to *show* a map rather than for a document to read.
//!
//! The plane takes the shape of the window (see [`Surface::frame_window`]), so
//! a fit covers the window edge to edge and the window color never shows
//! between two rectangles.
//! [`layout`] is the pure half: it turns a tree and a frame into [`Placed`]
//! rects with the squarified algorithm of Bruls, Huizing and van Wijk, so a
//! weight reads as an area and a rect stays as square as the grid allows.
//! The gutters are not drawn: every rect deflates before it lays its children
//! out, so [`PAPER`] shows through the gaps and the border.
//! [`svg`] writes one layout as an SVG document, and [`crate::svg`] parses and
//! paints it, so a label rides the host fonts through the path a `.svg` file
//! takes, and a region paint pays only for the pixels the window shows.
//!
//! A future zoomable treemap keeps [`layout`] as it stands: the frame is an
//! argument, so a zoom re-lays-out the sub-frame under the viewport with the
//! same call and no change to [`TreemapSurface`].

use std::{
    fmt::Write as _,
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{
    DecodedImage,
    dim::LongSide,
    draw::Color,
    region::{self, Crop},
    scene::{self, Paint, Primitive},
    surface::{ChunkSource, Size, Surface, SurfaceError, VectorSurface, View},
};

/// The plane a treemap takes before a window names its own frame, in pixels.
///
/// A single-file launch renders its plane before the window exists, so the
/// window then opens at this size and the first frame measures the real one.
const DEFAULT_FRAME: Size = Size {
    w: 1280.0,
    h: 800.0,
};

/// The gutter between two top-level groups, in frame pixels.
const GUTTER_TOP: f32 = 5.0;
/// The gutter between two inner groups, in frame pixels.
const GUTTER_MID: f32 = 3.0;
/// The gutter between two leaves, in frame pixels.
const GUTTER_LEAF: f32 = 1.4;

/// The label size of a top-level group, in frame pixels.
const FONT_TOP: f32 = 13.0;
/// The label size of an inner group, in frame pixels.
const FONT_MID: f32 = 10.5;
/// The label size inside one leaf.
const FONT_LEAF: f32 = 11.0;
/// How much larger than its framed box a label rasterizes.
///
/// A label is an [`Primitive::Image`] leaf until the text path lands, and a leaf
/// is a picture: it blurs past its own pixels exactly like a photo does.
/// Three times the framed size keeps the label sharp through the first zoom
/// steps, and the crate pays that bill once per plane, when the list is built.
const LABEL_OVERSAMPLE: f32 = 3.0;

/// How tall a label's box is per unit of label size.
///
/// The glyph body of a label sits under three quarters of its size, so the box
/// leaves the descender room and no clip eats a comma.
const LABEL_BOX: f32 = 1.35;

/// The height of a group band per unit of label size.
const BAND: f32 = 1.5;
/// The slack a band leaves around its label, in frame pixels.
const BAND_PAD: f32 = 5.0;

/// The background, which shows through every gutter.
const PAPER: Color = Color::new(16, 16, 18, 255);
/// The label color, which reads against every fill.
const INK: Color = Color::new(240, 240, 240, 255);

/// The hues of the top-level groups, in degrees.
///
/// Six hues keep two neighbours apart, and a seventh wraps to a neighbour of
/// the first, which reads as one more family beside it.
const HUES: [f32; 6] = [122.0, 218.0, 330.0, 30.0, 265.0, 186.0];

/// The sectors of the sample tree, with the industries each one holds.
const SECTORS: [(&str, [&str; 5]); 6] = [
    (
        "Technology",
        [
            "Semiconductors",
            "Software",
            "Hardware",
            "Cloud",
            "Networks",
        ],
    ),
    (
        "Financials",
        [
            "Banks",
            "Insurance",
            "Capital Markets",
            "Payments",
            "Asset Management",
        ],
    ),
    (
        "Healthcare",
        [
            "Drug Makers",
            "Devices",
            "Biotechnology",
            "Health Plans",
            "Distribution",
        ],
    ),
    (
        "Energy",
        [
            "Oil & Gas",
            "Equipment",
            "Renewables",
            "Pipelines",
            "Chemicals",
        ],
    ),
    (
        "Consumer",
        ["Retail", "Food", "Apparel", "Travel", "Household"],
    ),
    (
        "Industrials",
        [
            "Machinery",
            "Aerospace",
            "Railroads",
            "Transport",
            "Building",
        ],
    ),
];

/// The fewest leaves one industry of the sample tree holds.
const LEAVES_MIN: usize = 45;
/// The most leaves one industry of the sample tree holds.
const LEAVES_MAX: usize = 70;

/// One node of a treemap: a group with children, or a leaf with a weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// The name the rect carries as its label.
    pub label: String,
    /// The area the node takes, in the units of the tree.
    ///
    /// A group carries the sum of its children, so one weight rules the whole
    /// map.
    pub weight: f32,
    /// The children of a group, empty for a leaf.
    pub children: Vec<Node>,
}

impl Node {
    /// A leaf of `weight`.
    pub fn leaf(label: impl Into<String>, weight: f32) -> Self {
        Node {
            label: label.into(),
            weight,
            children: Vec::new(),
        }
    }

    /// A group of `children`, whose weight is their sum.
    pub fn group(label: impl Into<String>, children: Vec<Node>) -> Self {
        let weight = children.iter().map(|child| child.weight).sum();
        Node {
            label: label.into(),
            weight,
            children,
        }
    }

    /// Whether the node holds no children.
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Whether the extension of `path` asks for a treemap.
///
/// The format carries no magic, so the extension alone decides and the content
/// of the file never does: a `.treemap` file is a way to ask for the sample
/// tree.
#[must_use]
pub fn claims(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("treemap"))
}

/// The tree a `.treemap` file shows.
///
/// The generator is seeded, so every run and every test draws the same map, and
/// the weights come from a skewed draw: the cube of a uniform one clusters
/// them, so a few rects carry most of the area the way a real map does.
#[must_use]
pub fn sample_tree() -> Node {
    let mut rand = Rand::new(0x5EED_5EED_0000_0001);
    let sectors = SECTORS
        .into_iter()
        .map(|(sector, industries)| {
            let industries = industries
                .into_iter()
                .map(|industry| Node::group(industry, sample_leaves(&mut rand)))
                .collect();
            Node::group(sector, industries)
        })
        .collect();
    Node::group("versatile-viewer", sectors)
}

/// The leaves of one industry, each a ticker-shaped label and a skewed weight.
fn sample_leaves(rand: &mut Rand) -> Vec<Node> {
    let count = rand.between(LEAVES_MIN as f32, LEAVES_MAX as f32).round() as usize;
    (0..count.max(1))
        .map(|index| {
            let draw = rand.unit();
            let weight = 0.1 + 5.0 * draw * draw * draw;
            Node::leaf(format!("{}{index:02}", ticker(rand)), weight)
        })
        .collect()
}

/// Four uppercase letters, the shape a map like this labels its leaves with.
fn ticker(rand: &mut Rand) -> String {
    (0..4)
        .map(|_| char::from(b'A' + (rand.next() % 26) as u8))
        .collect()
}

/// What a placed rect puts on the paper.
#[derive(Debug, Clone, Copy)]
pub enum Ink {
    /// A leaf: the whole rect takes the fill.
    Leaf,
    /// A group: the band at the top of the rect carries the label, and the
    /// children fill the rest.
    Group {
        /// The strip at the top of the rect that carries the label.
        band: Crop,
    },
    /// A group too small for a band: it draws nothing, because its children
    /// fill the rect.
    Bare,
}

/// One rect of a laid-out tree.
#[derive(Debug, Clone, Copy)]
pub struct Placed<'a> {
    /// The rect the node takes, in frame units, its gutter already cut out.
    pub rect: Crop,
    /// The node the rect shows.
    pub node: &'a Node,
    /// The depth of the node: 0 for a top-level group.
    pub depth: u8,
    /// The fill of the rect, and the shade its band takes.
    pub fill: Color,
    /// What the rect puts on the paper.
    pub ink: Ink,
}

/// Lay `root` out over `frame`, gutters and label bands included.
///
/// The frame edge carries half a top gutter, so the map's own border is the gap
/// between two groups and the window edge needs no second line.
/// A group sorts its children by weight before they lay out, so the layout
/// never depends on the order of the tree.
#[must_use]
pub fn layout(root: &Node, frame: Crop) -> Vec<Placed<'_>> {
    let mut out = Vec::new();
    let inner = deflate(frame, GUTTER_TOP / 2.0);
    if root.is_leaf() {
        place(root, inner, 0, HUES[0], &mut out);
        return out;
    }
    let mut kids: Vec<&Node> = root.children.iter().collect();
    kids.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let rects = squarify(&weights(&kids), inner);
    for (index, (kid, rect)) in kids.iter().zip(rects).enumerate() {
        let inset = deflate(rect, GUTTER_TOP / 2.0);
        if inset.is_empty() {
            continue;
        }
        place(kid, inset, 0, HUES[index % HUES.len()], &mut out);
    }
    out
}

/// Lay one node out inside `rect`, and everything below it.
///
/// `hue` is the hue of the top-level group the node belongs to, so a sector's
/// leaves stay in one family and the shade of a rect comes from its own label.
fn place<'a>(node: &'a Node, rect: Crop, depth: u8, hue: f32, out: &mut Vec<Placed<'a>>) {
    let fill = shade(hue, &node.label);
    if node.is_leaf() {
        out.push(Placed {
            rect,
            node,
            depth,
            fill,
            ink: Ink::Leaf,
        });
        return;
    }
    let font = font_for(depth);
    let band = band_for(rect, font, label_width(&node.label, font));
    let inner = match band {
        Some(band) => region::crop(
            rect.min_x(),
            band.max_y(),
            rect.width(),
            rect.max_y() - band.max_y(),
        ),
        None => rect,
    };
    out.push(Placed {
        rect,
        node,
        depth,
        fill,
        ink: band.map_or(Ink::Bare, |band| Ink::Group { band }),
    });
    let mut kids: Vec<&Node> = node.children.iter().collect();
    kids.sort_by(|a, b| b.weight.total_cmp(&a.weight));
    let rects = squarify(&weights(&kids), inner);
    let gutter = gutter_for(depth);
    for (kid, rect) in kids.iter().zip(rects) {
        let inset = deflate(rect, gutter / 2.0);
        if inset.is_empty() {
            continue;
        }
        place(kid, inset, depth + 1, hue, out);
    }
}

/// The weights of `nodes`, in their order.
fn weights(nodes: &[&Node]) -> Vec<f32> {
    nodes.iter().map(|node| node.weight).collect()
}

/// The squarified rects of `weights` inside `rect`.
///
/// The algorithm of Bruls, Huizing and van Wijk: a row grows one weight at a
/// time while its worst aspect ratio improves, and the row lands as a strip
/// along the short side of what is left.
/// A weight that is zero, negative or not a number takes no area, so the rect
/// it answers is empty.
#[must_use]
pub fn squarify(weights: &[f32], rect: Crop) -> Vec<Crop> {
    let mut out = vec![EMPTY; weights.len()];
    let live: Vec<(usize, f32)> = weights
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, weight)| weight.is_finite() && *weight > 0.0)
        .collect();
    let total: f32 = live.iter().map(|(_, weight)| weight).sum();
    if total <= 0.0 || rect.is_empty() {
        return out;
    }
    let area = rect.width() * rect.height();
    let areas: Vec<f32> = live
        .iter()
        .map(|(_, weight)| weight / total * area)
        .collect();
    let mut rest = rect;
    let mut start = 0;
    while start < areas.len() {
        let short = rest.width().min(rest.height());
        if short <= 0.0 {
            break;
        }
        // Grow the row while the worst aspect in it does not get worse.
        let mut sum = areas[start];
        let mut len = 1;
        let mut best = worst_aspect(&areas[start..=start], sum, short);
        while start + len < areas.len() {
            let grown = sum + areas[start + len];
            let worst = worst_aspect(&areas[start..=start + len], grown, short);
            if worst > best {
                break;
            }
            sum = grown;
            best = worst;
            len += 1;
        }
        // The row takes a strip of `sum / short` along the short side, and its
        // items share that strip by area.
        // The edges of the items come from one run of cumulative sums, so two
        // neighbours land on the same coordinate and never take a pixel from each
        // other, and the last edge clamps onto the strip.
        let strip = sum / short;
        let mut edges = Vec::with_capacity(len + 1);
        edges.push(0.0);
        let mut acc = 0.0;
        for k in 0..len {
            acc += areas[start + k] / strip;
            edges.push(acc.min(short));
        }
        if rest.width() >= rest.height() {
            let row = clip(
                region::crop(rest.min_x(), rest.min_y(), strip, rest.height()),
                rect,
            );
            for k in 0..len {
                let (from, to) = (edges[k], edges[k + 1]);
                out[live[start + k].0] = clip(
                    region::crop(row.min_x(), row.min_y() + from, row.width(), to - from),
                    row,
                );
            }
            rest = region::crop(
                (rest.min_x() + strip).max(rest.min_x()),
                rest.min_y(),
                (rest.width() - strip).max(0.0),
                rest.height(),
            );
        } else {
            let row = clip(
                region::crop(rest.min_x(), rest.min_y(), rest.width(), strip),
                rect,
            );
            for k in 0..len {
                let (from, to) = (edges[k], edges[k + 1]);
                out[live[start + k].0] = clip(
                    region::crop(row.min_x() + from, row.min_y(), to - from, row.height()),
                    row,
                );
            }
            rest = region::crop(
                rest.min_x(),
                (rest.min_y() + strip).max(rest.min_y()),
                rest.width(),
                (rest.height() - strip).max(0.0),
            );
        }
        start += len;
    }
    out
}

/// The worst aspect ratio of `areas` laid along a side of length `short`, when
/// the row's total area is `sum`.
fn worst_aspect(areas: &[f32], sum: f32, short: f32) -> f32 {
    if sum <= 0.0 || short <= 0.0 {
        return f32::INFINITY;
    }
    let thickness = sum / short;
    let mut worst = 0.0_f32;
    for area in areas {
        let side = area / thickness;
        if side <= 0.0 {
            continue;
        }
        worst = worst.max((thickness / side).max(side / thickness));
    }
    worst
}

/// An empty rect, which paints nothing and holds no child.
const EMPTY: Crop = region::crop(0.0, 0.0, 0.0, 0.0);

/// `rect` clipped to `limit`.
///
/// The last strip of a squarified row can overshoot its frame by a fraction of a
/// pixel, and a rect that leaves the plane would paint outside the map, so every
/// rect lands inside the frame it was laid out in.
fn clip(rect: Crop, limit: Crop) -> Crop {
    rect.intersection(&limit).unwrap_or(EMPTY)
}

/// `rect` pulled in by `pad` on every side, or [`EMPTY`] when it cannot take
/// it.
fn deflate(rect: Crop, pad: f32) -> Crop {
    let (width, height) = (rect.width() - pad * 2.0, rect.height() - pad * 2.0);
    if !(width > 0.0 && height > 0.0) {
        return EMPTY;
    }
    region::crop(rect.min_x() + pad, rect.min_y() + pad, width, height)
}

/// The label size of a rect at `depth`.
fn font_for(depth: u8) -> f32 {
    match depth {
        0 => FONT_TOP,
        1 => FONT_MID,
        _ => FONT_MID * 0.9,
    }
}

/// The gutter between the children of a node at `depth`.
fn gutter_for(depth: u8) -> f32 {
    match depth {
        0 => GUTTER_TOP,
        1 => GUTTER_MID,
        _ => GUTTER_LEAF,
    }
}

/// The label band at the top of `rect`, when the rect leaves the room for a
/// label of `font` size that `text` pixels wide.
///
/// A band takes a sixth of the rect at the most, so a small group keeps its
/// area for its children, and a group whose label would not fit vertically
/// keeps no band at all.
fn band_for(rect: Crop, font: f32, text: f32) -> Option<Crop> {
    let height = font * BAND;
    if text + BAND_PAD * 2.0 > rect.width() || rect.height() < height * 3.0 {
        return None;
    }
    Some(region::crop(
        rect.min_x(),
        rect.min_y(),
        rect.width(),
        height,
    ))
}

/// The width a label of `font` size takes, near enough for a fit test.
///
/// usvg measures the real run at parse time, and a band that clips its label
/// reads worse than no band, so the estimate stays a little generous.
fn label_width(text: &str, font: f32) -> f32 {
    text.chars().count() as f32 * font * 0.58 + font * 0.4
}

/// The fill of a node of the group hue `hue`, shaded by its own label.
///
/// The label seeds the shade, so a rect keeps its color across resizes and the
/// leaves of one group stay a family around the hue of their sector.
fn shade(hue: f32, label: &str) -> Color {
    let seed = stable_hash(label);
    hsv(
        hue + 16.0 * spread(seed, 1),
        0.54 + 0.10 * spread(seed, 2),
        0.62 + 0.14 * spread(seed, 3),
    )
}

/// The RGB color of a hue in degrees, a saturation and a value in `0..=1`.
fn hsv(hue: f32, saturation: f32, value: f32) -> Color {
    let clamped = |v: f32| {
        if v.is_finite() {
            v.clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let (saturation, value) = (clamped(saturation), clamped(value));
    let turn = if hue.is_finite() {
        hue.rem_euclid(360.0)
    } else {
        0.0
    } / 60.0;
    let sector = turn.floor();
    let frac = turn - sector;
    let (p, q, t) = (
        value * (1.0 - saturation),
        value * (1.0 - saturation * frac),
        value * (1.0 - saturation * (1.0 - frac)),
    );
    let (red, green, blue) = match sector as u32 % 6 {
        0 => (value, t, p),
        1 => (q, value, p),
        2 => (p, value, t),
        3 => (p, q, value),
        4 => (t, p, value),
        _ => (value, p, q),
    };
    let byte = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    Color::new(byte(red), byte(green), byte(blue), 255)
}

/// An FNV-1a hash of `text`.
///
/// A node's shade must not move when the window does, so it comes from the
/// label and never from a draw or an address.
fn stable_hash(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A draw in `-1..1` from `seed`, so one seed gives one shade per `round`.
fn spread(seed: u64, round: u32) -> f32 {
    let mut mixed = seed ^ (u64::from(round) << 40);
    mixed = (mixed ^ (mixed >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    mixed = (mixed ^ (mixed >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    mixed ^= mixed >> 33;
    // The top 24 bits give a value a float holds exactly.
    (mixed >> 40) as f32 / (1_u32 << 23) as f32 - 1.0
}

/// A splitmix64 stream, so the sample tree is the same on every run.
struct Rand(u64);

impl Rand {
    /// A stream from `seed`.
    const fn new(seed: u64) -> Self {
        Rand(seed)
    }

    /// The next word of the stream.
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut word = self.0;
        word = (word ^ (word >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        word = (word ^ (word >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        word ^ (word >> 31)
    }

    /// A draw in `0..1`.
    ///
    /// The top 24 bits give a value an `f32` holds exactly.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1_u32 << 24) as f32
    }

    /// A draw in `lo..hi`.
    fn between(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

/// The SVG document of one layout, `frame` units wide.
///
/// The document is what [`crate::svg`] parses, so a label rides the host fonts
/// through the path a `.svg` file takes.
/// The band of a group takes a darker shade of its fill, so the label reads
/// against it without a second color table.
#[must_use]
pub fn svg(layout: &[Placed], frame: Size) -> String {
    let mut text = String::with_capacity(layout.len() * 96 + 256);
    let (w, h) = (frame.w, frame.h);
    let _ = write!(
        text,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">"#
    );
    // The paper fills first, so a gutter and the outer border show it and never
    // the window color behind the plane.
    let _ = write!(
        text,
        r##"<rect x="0" y="0" width="{w}" height="{h}" fill="#{r:02x}{g:02x}{b:02x}"/>"##,
        r = PAPER.r,
        g = PAPER.g,
        b = PAPER.b
    );
    for placed in layout {
        match placed.ink {
            Ink::Leaf => {
                let rect = placed.rect;
                let (x, y, rw, rh) = (rect.min_x(), rect.min_y(), rect.width(), rect.height());
                let fill = placed.fill;
                let _ = write!(
                    text,
                    r##"<rect x="{x}" y="{y}" width="{rw}" height="{rh}" fill="#{r:02x}{g:02x}{b:02x}"/>"##,
                    r = fill.r,
                    g = fill.g,
                    b = fill.b
                );
                if let Some((x, baseline, font)) = leaf_label(placed) {
                    let label = escape(&placed.node.label);
                    let _ = write!(
                        text,
                        r##"<text x="{x}" y="{baseline}" font-family="sans-serif" font-size="{font}" fill="#{ir:02x}{ig:02x}{ib:02x}" text-anchor="middle">{label}</text>"##,
                        ir = INK.r,
                        ig = INK.g,
                        ib = INK.b
                    );
                }
            }
            Ink::Group { band } => {
                let dark = dim(placed.fill);
                let (x, y, rw, rh) = (band.min_x(), band.min_y(), band.width(), band.height());
                let _ = write!(
                    text,
                    r##"<rect x="{x}" y="{y}" width="{rw}" height="{rh}" fill="#{r:02x}{g:02x}{b:02x}"/>"##,
                    r = dark.r,
                    g = dark.g,
                    b = dark.b
                );
                let font = font_for(placed.depth);
                let label = escape(&placed.node.label);
                let baseline = band.min_y() + band.height() / 2.0 + font * 0.35;
                let _ = write!(
                    text,
                    r##"<text x="{x}" y="{baseline}" font-family="sans-serif" font-size="{font}" fill="#{ir:02x}{ig:02x}{ib:02x}">{label}</text>"##,
                    x = band.min_x() + BAND_PAD,
                    ir = INK.r,
                    ig = INK.g,
                    ib = INK.b
                );
            }
            Ink::Bare => {}
        }
    }
    text.push_str("</svg>");
    text
}

/// Where the label of a leaf sits, and at what size, or `None` when the rect
/// leaves no room for it.
///
/// The label centers in the rect, which reads as one caption per tile, and a
/// rect too small for its own text carries none: a clipped ticker reads worse
/// than a plain tile.
fn leaf_label(placed: &Placed) -> Option<(f32, f32, f32)> {
    let font = FONT_LEAF.min(placed.rect.height() / 2.4);
    if font < 7.0 {
        return None;
    }
    let room = placed.rect.width() - BAND_PAD;
    if label_width(&placed.node.label, font) > room {
        return None;
    }
    let baseline = placed.rect.min_y() + placed.rect.height() / 2.0 + font * 0.35;
    let x = placed.rect.min_x() + placed.rect.width() / 2.0;
    Some((x, baseline, font))
}

/// `color` at three quarters of its channels, for a group band under its label.
fn dim(color: Color) -> Color {
    let ch = |value: u8| value / 4 * 3;
    Color::new(ch(color.r), ch(color.g), ch(color.b), color.a)
}

/// `text` with the three characters XML reserves replaced by their escapes.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            other => out.push(other),
        }
    }
    out
}

/// A treemap: one plane that takes the shape of the window that shows it.
#[derive(Debug)]
pub struct TreemapSurface {
    tree: Node,
    canvas: Mutex<Canvas>,
}

/// The frame a treemap paints, and the parsed tree of that frame.
#[derive(Debug)]
struct Canvas {
    /// The plane size the view frames, which follows the window.
    frame: Size,
    /// The painted tree of `frame`, built by the first paint that needs it.
    ///
    /// The frame rides beside it, so a resize frame only records a size and the
    /// settled repaint pays for the parse and the rasterize.
    painted: Option<(Size, VectorSurface)>,
    /// The primitives of `frame`, built by the first draw that asks for them.
    ///
    /// The list holds no scale and no region, so it survives every zoom and pan
    /// of the same frame and only a resize rebuilds it.
    scene: Option<(Size, Arc<[Primitive]>)>,
}

impl TreemapSurface {
    /// Open `path` as a treemap.
    ///
    /// The file is opened and never read: a `.treemap` path is a way to ask for
    /// the sample tree, so its bytes cannot change the picture, and a path that
    /// vanished still fails here like every other source.
    ///
    /// # Errors
    ///
    /// The function errors when the file cannot be opened.
    pub fn open(path: &Path) -> Result<Self, SurfaceError> {
        Ok(Self::from_tree(open_file(path)?))
    }

    /// A treemap of `tree`, framed for the default window.
    #[must_use]
    pub fn from_tree(tree: Node) -> Self {
        TreemapSurface {
            tree,
            canvas: Mutex::new(Canvas {
                frame: Self::frame_for(DEFAULT_FRAME),
                painted: None,
                scene: None,
            }),
        }
    }

    /// The plane a treemap takes for a window of `win` pixels.
    ///
    /// The shape is the window's, so a fit covers every edge, and the long side
    /// clamps into the range a vector base raster takes: the declared size and
    /// the base raster then agree, and a region arrives in frame units.
    fn frame_for(win: Size) -> Size {
        let sane = |value: f32| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                1.0
            }
        };
        let (w, h) = (sane(win.w), sane(win.h));
        let long = w.max(h);
        let side = crate::svg::base_side(long).raw().max(1) as f32;
        let scale = side / long;
        Size {
            w: (w * scale).round().max(1.0),
            h: (h * scale).round().max(1.0),
        }
    }

    /// Run `read` against the surface of the current frame, laying the tree out
    /// and parsing it when the frame moved.
    ///
    /// The paint is the expensive half, so it happens on the caller's worker and
    /// never at [`Surface::bounds`]: a resize frame only records the new frame.
    fn with_surface<T>(
        &self,
        read: impl FnOnce(&VectorSurface) -> Result<T, SurfaceError>,
    ) -> Result<T, SurfaceError> {
        let mut canvas = self.canvas.lock().map_err(|_| locked())?;
        let stale = canvas
            .painted
            .as_ref()
            .is_none_or(|(frame, _)| *frame != canvas.frame);
        if stale {
            let placed = layout(&self.tree, canvas.frame.whole());
            let document = svg(&placed, canvas.frame);
            let vector = crate::svg::from_str(&document)?;
            canvas.painted = Some((canvas.frame, VectorSurface::from_vector(vector)));
        }
        match canvas.painted.as_ref() {
            Some((_, surface)) => read(surface),
            None => Err(SurfaceError::Render("the treemap painted no layout".into())),
        }
    }
}

/// The primitives of one layout, `frame` units wide.
///
/// This is what [`svg`] writes as a document and [`crate::svg`] then parses: the
/// paper, one rect per tile, the darker band of a group, and one picture per
/// label.
/// The rects are answered directly, so the document round trip survives only for
/// the labels, which ride the host fonts through it until the glyph path lands.
///
/// # Errors
///
/// The function errors when a label rasterizes to nothing.
fn primitives(layout: &[Placed], frame: Size) -> Result<Vec<Primitive>, SurfaceError> {
    let mut out = Vec::with_capacity(layout.len() * 2 + 1);
    // The paper fills first, so a gutter and the outer border show it and never
    // the window color behind the plane.
    out.push(Primitive::Rect {
        rect: scene::rect_of(frame.whole()),
        paint: Paint::Solid(PAPER),
    });
    for placed in layout {
        match placed.ink {
            Ink::Leaf => {
                out.push(Primitive::Rect {
                    rect: scene::rect_of(placed.rect),
                    paint: Paint::Solid(placed.fill),
                });
                if let Some((center, baseline, font)) = leaf_label(placed)
                    && let Some(leaf) = label_leaf(&placed.node.label, center, baseline, font)?
                {
                    out.push(leaf);
                }
            }
            Ink::Group { band } => {
                out.push(Primitive::Rect {
                    rect: scene::rect_of(band),
                    paint: Paint::Solid(dim(placed.fill)),
                });
                let font = font_for(placed.depth);
                let baseline = band.min_y() + band.height() / 2.0 + font * 0.35;
                let center = band.min_x() + BAND_PAD + label_width(&placed.node.label, font) / 2.0;
                if let Some(leaf) = label_leaf(&placed.node.label, center, baseline, font)? {
                    out.push(leaf);
                }
            }
            Ink::Bare => {}
        }
    }
    Ok(out)
}

/// The picture of one label, centered on `center` and sitting on `baseline`.
///
/// A label rasterizes through [`crate::svg`], the painter the document path
/// already used, at [`LABEL_OVERSAMPLE`] times its framed box, so it stays sharp
/// through the first zoom steps of the map.
///
/// # Errors
///
/// The function errors when the label's document fails to parse or rasterize.
fn label_leaf(
    label: &str,
    center: f32,
    baseline: f32,
    font: f32,
) -> Result<Option<Primitive>, SurfaceError> {
    let (box_w, box_h) = (label_width(label, font), font * LABEL_BOX);
    let usable = font.is_finite() && font > 0.0 && box_w > 0.0 && box_h > 0.0;
    if !usable {
        return Ok(None);
    }
    // The document declares the pixel size and draws in frame units, so one
    // frame unit maps onto `LABEL_OVERSAMPLE` pixels.
    let (px_w, px_h) = (
        (box_w * LABEL_OVERSAMPLE).round().max(1.0) as u32,
        (box_h * LABEL_OVERSAMPLE).round().max(1.0) as u32,
    );
    // The baseline sits this far into the box, measured from its top.
    let descent = font * 1.05;
    let document = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{px_w}" height="{px_h}" viewBox="0 0 {box_w} {box_h}"><text x="{cx}" y="{descent}" font-family="sans-serif" font-size="{font}" fill="#{r:02x}{g:02x}{b:02x}" text-anchor="middle">{label}</text></svg>"##,
        cx = box_w / 2.0,
        r = INK.r,
        g = INK.g,
        b = INK.b,
        label = escape(label),
    );
    let vector = crate::svg::from_str(&document)?;
    let pixels = vector.rasterize(LongSide::new(px_w.max(px_h)))?;
    let box_ = region::crop(center - box_w / 2.0, baseline - descent, box_w, box_h);
    Ok(Some(scene::image_of(box_, pixels)))
}

/// The file, opened and dropped, so a path that vanished fails at open.
///
/// The tree is the sample tree either way: a `.treemap` file carries no data
/// this backend reads.
fn open_file(path: &Path) -> Result<Node, SurfaceError> {
    std::fs::File::open(path)?;
    Ok(sample_tree())
}

/// The error a poisoned canvass lock answers.
fn locked() -> SurfaceError {
    SurfaceError::Render("the treemap canvas lock is poisoned".into())
}

impl Surface for TreemapSurface {
    fn plane_count(&self) -> usize {
        1
    }

    fn bounds(&self, plane: usize) -> Result<Size, SurfaceError> {
        if plane != 0 {
            return Err(SurfaceError::Plane(plane));
        }
        Ok(self.canvas.lock().map_err(|_| locked())?.frame)
    }

    fn render_whole(&self, plane: usize, scale: f32) -> Result<DecodedImage, SurfaceError> {
        // The plane is the frame, so a whole render is a region render of all of
        // it.
        self.render(View::whole(plane, scale))
    }

    fn render(&self, view: View) -> Result<DecodedImage, SurfaceError> {
        // The layout anchors a region, so a patch paints only the rect the
        // window shows, at the scale it shows it.
        self.with_surface(|surface| surface.render(view))
    }

    fn display(&self, plane: usize) -> Result<Option<Arc<[Primitive]>>, SurfaceError> {
        // The layout *is* the list: a rect per tile, the darker band of a group,
        // and one picture per label. No scale and no region, so a zoom reuses it
        // and only a resize of the window rebuilds it.
        if plane != 0 {
            return Err(SurfaceError::Plane(plane));
        }
        let mut canvas = self.canvas.lock().map_err(|_| locked())?;
        let frame = canvas.frame;
        if let Some((built, list)) = canvas.scene.as_ref()
            && *built == frame
        {
            return Ok(Some(Arc::clone(list)));
        }
        let placed = layout(&self.tree, frame.whole());
        let list: Arc<[Primitive]> = Arc::from(primitives(&placed, frame)?);
        canvas.scene = Some((frame, Arc::clone(&list)));
        Ok(Some(list))
    }

    fn progressive(&self) -> Result<Option<Box<dyn ChunkSource>>, SurfaceError> {
        // The layout lays out and paints in one step.
        Ok(None)
    }

    // The tree paints at any scale, so a settled zoom pays for a sharper patch
    // of the same map.
    fn can_refine(&self) -> bool {
        true
    }

    fn frame_window(&self, win: Size) -> bool {
        let frame = Self::frame_for(win);
        let Ok(mut canvas) = self.canvas.lock() else {
            // A poisoned lock keeps the frame it had: a treemap that cannot
            // paint again has no new frame to answer.
            return false;
        };
        if canvas.frame == frame {
            return false;
        }
        canvas.frame = frame;
        true
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// A frame of the default window.
    fn frame() -> Crop {
        DEFAULT_FRAME.whole()
    }

    /// The leaf rects of a layout.
    fn leaves(placed: &[Placed]) -> Vec<Crop> {
        placed
            .iter()
            .filter(|p| matches!(p.ink, Ink::Leaf))
            .map(|p| p.rect)
            .collect()
    }

    /// Whether `inner` lies inside `outer` to within a thousandth of a pixel.
    ///
    /// A rect stores its corner and its size, so a far edge that lands exactly on
    /// the boundary rounds apart in the last digit of an `f32`.
    fn inside(inner: Crop, outer: Crop) -> bool {
        const EPSILON: f32 = 1e-3;
        inner.is_empty()
            || (inner.min_x() >= outer.min_x() - EPSILON
                && inner.min_y() >= outer.min_y() - EPSILON
                && inner.max_x() <= outer.max_x() + EPSILON
                && inner.max_y() <= outer.max_y() + EPSILON)
    }

    /// Whether two rects share more than a thousandth of a pixel.
    ///
    /// A rect stores its corner and its size, so the far edge of one rect and
    /// the near edge of its neighbour round apart in the last digit of an
    /// `f32` when the two neighbours were summed in different orders.
    /// A raster rounds to whole pixels, so that share paints nothing.
    fn overlaps(a: Crop, b: Crop) -> bool {
        const EPSILON: f32 = 1e-3;
        a.min_x() + EPSILON < b.max_x()
            && b.min_x() + EPSILON < a.max_x()
            && a.min_y() + EPSILON < b.max_y()
            && b.min_y() + EPSILON < a.max_y()
    }

    #[test]
    fn the_sample_tree_is_the_same_on_every_run() {
        assert_eq!(sample_tree(), sample_tree());
    }

    #[test]
    fn the_sample_tree_reaches_three_levels_and_about_two_thousand_leaves() {
        let tree = sample_tree();
        assert_eq!(tree.children.len(), SECTORS.len());
        assert_eq!(tree.children[0].children.len(), 5);
        let count = leaves(&layout(&tree, frame())).len();
        assert!(
            (1000..3000).contains(&count),
            "the sample tree holds {count} leaves"
        );
    }

    #[test]
    fn every_leaf_lands_inside_the_frame() {
        let tree = sample_tree();
        let outer = frame();
        for rect in leaves(&layout(&tree, outer)) {
            assert!(inside(rect, outer), "{rect:?} leaves the frame {outer:?}");
        }
    }

    #[test]
    fn two_leaves_never_share_a_pixel() {
        let rects = leaves(&layout(&sample_tree(), frame()));
        for (i, a) in rects.iter().enumerate() {
            for b in &rects[i + 1..] {
                assert!(!overlaps(*a, *b), "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn the_leaves_cover_most_of_the_frame_and_leave_the_gutters_to_the_paper() {
        let outer = frame();
        let covered: f32 = leaves(&layout(&sample_tree(), outer))
            .iter()
            .map(|rect| rect.width() * rect.height())
            .sum();
        let share = covered / (outer.width() * outer.height());
        assert!((0.6..0.95).contains(&share), "the leaves cover {share}");
    }

    #[test]
    fn a_group_carries_a_band_exactly_when_its_label_fits() {
        let tree = sample_tree();
        let placed = layout(&tree, frame());
        let banded: Vec<&Placed> = placed
            .iter()
            .filter(|p| matches!(p.ink, Ink::Group { .. }))
            .collect();
        assert!(!banded.is_empty(), "no group carries a label");
        for group in banded {
            let Ink::Group { band } = group.ink else {
                unreachable!("the filter holds a band")
            };
            assert!(group.rect.contains_rect(&band));
            assert!(band.height() < group.rect.height() / 2.0);
        }
    }

    #[test]
    fn a_squarified_area_follows_its_weight() {
        let outer = region::crop(0.0, 0.0, 400.0, 300.0);
        let weights = [4.0, 2.0, 1.0, 1.0];
        let rects = squarify(&weights, outer);
        let area = outer.width() * outer.height();
        for (weight, rect) in weights.iter().zip(&rects) {
            let want = weight / 8.0 * area;
            let got = rect.width() * rect.height();
            assert!(
                (got - want).abs() < want * 0.02,
                "weight {weight} took {got} of {want}"
            );
        }
    }

    #[test]
    fn a_weight_of_zero_takes_an_empty_rect_and_the_rest_keeps_its_share() {
        let outer = region::crop(0.0, 0.0, 200.0, 100.0);
        let rects = squarify(&[1.0, 0.0, -2.0, 1.0], outer);
        assert!(rects[1].is_empty() && rects[2].is_empty());
        assert!((rects[0].width() * rects[0].height() - 10_000.0).abs() < 200.0);
        assert!((rects[3].width() * rects[3].height() - 10_000.0).abs() < 200.0);
    }

    #[test]
    fn a_frame_keeps_the_shape_of_its_window() {
        for (w, h) in [
            (1920.0, 1080.0),
            (800.0, 600.0),
            (600.0, 1000.0),
            (37.0, 91.0),
        ] {
            let frame = TreemapSurface::frame_for(Size { w, h });
            let want = w / h;
            let got = frame.w / frame.h;
            assert!((want - got).abs() < 0.01, "{w}x{h} took {frame:?}");
            assert!(frame.w.max(frame.h) >= 1024.0 && frame.w.max(frame.h) <= 8192.0);
        }
    }

    #[test]
    fn a_nonsense_window_still_answers_a_plane() {
        let frame = TreemapSurface::frame_for(Size {
            w: 0.0,
            h: f32::NAN,
        });
        assert!(frame.w >= 1.0 && frame.h >= 1.0);
    }

    #[test]
    fn a_treemap_takes_the_frame_its_window_names() {
        let surface = TreemapSurface::from_tree(sample_tree());
        let before = surface.bounds(0).unwrap();
        assert_eq!(before, TreemapSurface::frame_for(DEFAULT_FRAME));
        assert!(surface.frame_window(Size {
            w: 1920.0,
            h: 1080.0
        }));
        let after = surface.bounds(0).unwrap();
        assert_eq!(
            after,
            TreemapSurface::frame_for(Size {
                w: 1920.0,
                h: 1080.0
            })
        );
        // The same window twice changes nothing, so the view reads no new
        // plane and repaints nothing.
        assert!(!surface.frame_window(Size {
            w: 1920.0,
            h: 1080.0
        }));
    }

    #[test]
    fn a_rendered_treemap_measures_the_frame_and_its_corner_is_the_paper() {
        let surface = TreemapSurface::from_tree(sample_tree());
        let plane = surface.bounds(0).unwrap();
        let image = surface.render_whole(0, 1.0).unwrap();
        assert_eq!(image.width().raw(), plane.w as u32);
        assert_eq!(image.height().raw(), plane.h as u32);
        // The frame edge carries half a gutter, so the first pixel is paper and
        // never a fill: the window color has nowhere to land.
        let corner = &image.as_rgba()[..4];
        assert_eq!(corner, [PAPER.r, PAPER.g, PAPER.b, PAPER.a]);
    }

    #[test]
    fn a_fresh_window_paints_its_own_shape() {
        let surface = TreemapSurface::from_tree(sample_tree());
        let win = Size {
            w: 900.0,
            h: 1600.0,
        };
        surface.frame_window(win);
        let plane = surface.bounds(0).unwrap();
        let image = surface
            .render(View::whole(0, plane.h / win.h))
            .expect("the paint of the new frame");
        assert_eq!(image.width().raw(), 900);
        assert_eq!(image.height().raw(), 1600);
    }

    #[test]
    fn a_region_paint_measures_the_region_it_was_asked_for() {
        let surface = TreemapSurface::from_tree(sample_tree());
        let region = region::crop(100.0, 50.0, 200.0, 120.0);
        let image = surface.render(View::region(0, region, 0.5)).unwrap();
        assert_eq!(image.width().raw(), 100);
        assert_eq!(image.height().raw(), 60);
    }

    #[test]
    fn a_plane_out_of_range_is_an_error() {
        let surface = TreemapSurface::from_tree(sample_tree());
        assert!(matches!(surface.bounds(1), Err(SurfaceError::Plane(1))));
        assert!(matches!(
            surface.render_whole(3, 1.0),
            Err(SurfaceError::Plane(3))
        ));
        assert!(matches!(surface.display(1), Err(SurfaceError::Plane(1))));
    }

    #[test]
    fn the_display_list_is_the_layout_as_rects() {
        let surface = TreemapSurface::from_tree(sample_tree());
        let list = surface.display(0).expect("a list").expect("some");
        let frame = surface.bounds(0).expect("a frame");
        let tree = sample_tree();
        let placed = layout(&tree, frame.whole());
        // The paper, then one rect per tile: every placed node fills one.
        let rects = list
            .iter()
            .filter(|primitive| matches!(primitive, Primitive::Rect { .. }))
            .count();
        assert_eq!(rects, placed.len() + 1);
        // The map carries labels, so the list carries pictures.
        assert!(
            list.iter()
                .any(|primitive| matches!(primitive, Primitive::Image { .. })),
            "a labelled map answers at least one label leaf"
        );
    }

    #[test]
    fn one_list_serves_every_zoom_of_a_frame() {
        let surface = TreemapSurface::from_tree(sample_tree());
        let first = surface.display(0).expect("a list").expect("some");
        let again = surface.display(0).expect("a list").expect("some");
        assert!(
            Arc::ptr_eq(&first, &again),
            "the frame's list is built once"
        );
        // A window of another shape is another frame, so it rebuilds.
        assert!(surface.frame_window(Size { w: 640.0, h: 480.0 }));
        let resized = surface.display(0).expect("a list").expect("some");
        assert!(!Arc::ptr_eq(&first, &resized));
    }

    #[test]
    fn the_thumbnail_scale_paints_a_smaller_plane() {
        let surface = TreemapSurface::from_tree(sample_tree());
        let image = surface.render_whole(0, 0.4).unwrap();
        assert_eq!(image.width().raw(), (DEFAULT_FRAME.w * 0.4) as u32);
    }

    #[test]
    fn the_document_names_the_paper_and_a_group_label() {
        let tree = sample_tree();
        let placed = layout(&tree, frame());
        let document = svg(&placed, DEFAULT_FRAME);
        assert!(document.starts_with("<svg "));
        assert!(document.ends_with("</svg>"));
        assert!(document.contains(&format!("#{:02x}{:02x}{:02x}", PAPER.r, PAPER.g, PAPER.b)));
        assert!(document.contains("Technology"));
        assert!(document.contains("<text "));
    }

    #[test]
    fn a_label_with_xml_characters_is_escaped() {
        let tree = Node::group(
            "Oil & <Gas>",
            vec![Node::leaf("a&b", 1.0), Node::leaf("c<d", 1.0)],
        );
        let document = svg(&layout(&tree, frame()), DEFAULT_FRAME);
        assert!(document.contains("a&amp;b"));
        assert!(document.contains("c&lt;d"));
        assert!(!document.contains("a&b"));
    }

    #[test]
    fn the_shade_of_a_label_never_moves() {
        let tree = sample_tree();
        let first = layout(&tree, frame());
        let second = layout(&tree, frame());
        let colors: Vec<Color> = first.iter().map(|p| p.fill).collect();
        assert_eq!(colors, second.iter().map(|p| p.fill).collect::<Vec<_>>());
    }

    proptest! {
        // No seed file: a saved case becomes a plain `#[test]` beside these
        // strategies, so the suite reads its regressions in one place.
        #![proptest_config(ProptestConfig {
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        /// Every squarified rect stays inside the frame, no two share a pixel,
        /// and the rects never take more area than the frame holds.
        #[test]
        fn squarify_keeps_the_rects_inside_and_apart(
            weights in prop::collection::vec(0.01_f32..100.0, 1..40),
            width in 20.0_f32..800.0,
            height in 20.0_f32..800.0,
        ) {
            let outer = region::crop(0.0, 0.0, width, height);
            let rects = squarify(&weights, outer);
            let mut sum = 0.0;
            for (i, a) in rects.iter().enumerate() {
                prop_assert!(a.width() >= 0.0 && a.height() >= 0.0);
                prop_assert!(inside(*a, outer));
                sum += a.width() * a.height();
                for b in &rects[i + 1..] {
                    prop_assert!(!overlaps(*a, *b));
                }
            }
            prop_assert!(sum <= width * height * 1.001);
            prop_assert!(sum >= width * height * 0.999);
        }
    }
}
