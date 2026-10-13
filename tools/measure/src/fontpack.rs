//! Pack glyphs for the board: anti-aliased 4-bit coverage, run-length coded, plus the panel's linear-light table for each RGB565 level (from the measured model), so the board blends text in linear light with integer maths only.
//!
//! Blob, little-endian:
//! - u32 magic `FNT1`, u32 total length
//! - u16 × 128: linear light (0..=65535) of each red (32), green (64) and blue (32) RGB565 level, after the panel's bit replication
//! - u16 glyph count, u16 ascent (px), u16 descent (px), u16 0
//! - per glyph, 20 bytes: u32 code point, u16 width, u16 height, i16 left (from the pen), i16 top (from the baseline, up is negative), u16 advance, u16 0, u32 data offset (from the blob's start)
//! - glyph data, rows top to bottom: a byte `1kLLLLLL` is a run of L+1 pixels of coverage 0 (k = 0) or 15 (k = 1); `0NNNNNNN` is N+1 literal 4-bit coverages that follow, two per byte, high nibble first

use ab_glyph::{Font, FontRef, PxScale, ScaleFont, point};

use crate::model::PanelModel;

/// The twelve dozenal digits Zil..Stelor (U+0010..U+001B), the default characters.
pub fn dozenal() -> Vec<char> {
    (0x10u32..0x1C).filter_map(char::from_u32).collect()
}

fn rle(coverage: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < coverage.len() {
        let v = coverage[i];
        let mut j = i;
        if v == 0 || v == 15 {
            while j < coverage.len() && coverage[j] == v && j - i < 64 {
                j += 1;
            }
            out.push(0x80 | if v == 15 { 0x40 } else { 0 } | (j - i - 1) as u8);
        } else {
            while j < coverage.len() && coverage[j] != 0 && coverage[j] != 15 && j - i < 128 {
                j += 1;
            }
            out.push((j - i - 1) as u8);
            for pair in coverage[i..j].chunks(2) {
                out.push(pair[0] << 4 | pair.get(1).copied().unwrap_or(0));
            }
        }
        i = j;
    }
    out
}

/// Pack `chars` from the font at `px` (the font size in pixels) into a blob, with `model`'s light table; returns the blob and the bytes of 4-bit coverage before run-length coding.
pub fn pack(ttf: &[u8], px: f32, chars: &[char], model: &PanelModel) -> (Vec<u8>, usize) {
    let font = FontRef::try_from_slice(ttf).expect("parse font");
    let scaled = font.as_scaled(PxScale::from(px));
    let mut blob = Vec::new();
    blob.extend_from_slice(b"FNT1");
    blob.extend_from_slice(&0u32.to_le_bytes());
    for (c, bits) in [(0, 5), (1, 6), (2, 5)] {
        for v in 0u32..1 << bits {
            let code = (v << (8 - bits)) | (v >> (2 * bits - 8));
            let lin = (model.k[c][code as usize].clamp(0.0, 1.0) * 65535.0).round() as u16;
            blob.extend_from_slice(&lin.to_le_bytes());
        }
    }
    let (ascent, descent) = (scaled.ascent().ceil() as u16, (-scaled.descent()).ceil() as u16);
    blob.extend_from_slice(&(chars.len() as u16).to_le_bytes());
    blob.extend_from_slice(&ascent.to_le_bytes());
    blob.extend_from_slice(&descent.to_le_bytes());
    blob.extend_from_slice(&0u16.to_le_bytes());

    let table_at = blob.len();
    blob.resize(table_at + 20 * chars.len(), 0);
    let mut raw = 0;
    for (n, &ch) in chars.iter().enumerate() {
        let id = font.glyph_id(ch);
        assert!(id.0 != 0, "the font has no glyph for {ch:?} (U+{:04X})", ch as u32);
        let glyph = id.with_scale_and_position(PxScale::from(px), point(0.0, 0.0));
        let advance = scaled.h_advance(id).round() as u16;
        let (w, h, left, top, coverage) = match font.outline_glyph(glyph) {
            Some(outline) => {
                let b = outline.px_bounds();
                let (w, h) = (b.width() as usize, b.height() as usize);
                let mut cov = vec![0u8; w * h];
                outline.draw(|x, y, c| cov[y as usize * w + x as usize] = (c.clamp(0.0, 1.0) * 15.0).round() as u8);
                (w as u16, h as u16, b.min.x as i16, b.min.y as i16, cov)
            }
            None => (0, 0, 0, 0, Vec::new()),
        };
        raw += coverage.len().div_ceil(2);
        let offset = blob.len() as u32;
        blob.extend_from_slice(&rle(&coverage));
        let e = &mut blob[table_at + 20 * n..table_at + 20 * (n + 1)];
        e[0..4].copy_from_slice(&(ch as u32).to_le_bytes());
        e[4..6].copy_from_slice(&w.to_le_bytes());
        e[6..8].copy_from_slice(&h.to_le_bytes());
        e[8..10].copy_from_slice(&left.to_le_bytes());
        e[10..12].copy_from_slice(&top.to_le_bytes());
        e[12..14].copy_from_slice(&advance.to_le_bytes());
        e[16..20].copy_from_slice(&offset.to_le_bytes());
    }
    while blob.len() % 4 != 0 {
        blob.push(0);
    }
    let len = blob.len() as u32;
    blob[4..8].copy_from_slice(&len.to_le_bytes());
    (blob, raw)
}
