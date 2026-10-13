//! Pack glyphs for the board (see `measure::fontpack` for the blob's layout).
//!
//! Usage: `fontpack <font.ttf> <px> <out.bin> [characters]` — px is the font size in pixels; characters default to the twelve dozenal digits Zil..Stelor (U+0010..U+001B).

use std::fs;
use std::path::Path;

use measure::fontpack::{dozenal, pack};
use measure::model::PanelModel;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../measurements");

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: fontpack <font.ttf> <px> <out.bin> [characters]";
    let ttf = fs::read(args.next().expect(usage)).expect("read font");
    let px: f32 = args.next().expect(usage).parse().expect("px");
    let out = args.next().expect(usage);
    let chars: Vec<char> = args.next().map(|s| s.chars().collect()).unwrap_or_else(dozenal);
    let model = PanelModel::load(&Path::new(ROOT).join("i1pro-2026-10-11"), &Path::new(ROOT).join("i1pro-2026-10-11-toe"));
    let (blob, raw) = pack(&ttf, px, &chars, &model);
    fs::write(&out, &blob).expect("write blob");
    println!("{out}: {} glyphs at {px} px, {} bytes ({raw} bytes of 4-bit coverage before run-length coding)", chars.len(), blob.len());
}
