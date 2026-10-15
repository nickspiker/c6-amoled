//! The coastline overlay: strips of anti-aliased coverage along the coast (built and uploaded by `goes coast`), flickered ±16 codes in gamma space, one flip per frame period, so the outline shows on any cloud while the time-average stays the true image.
//!
//! The blob: "CST1", its whole length (u32, little-endian, a whole number of words), then strips over row pairs (the panel's windows must start and span even rows and columns): even row, even first column, even width (u16 each, little-endian), then each pixel's coverage 0..=15 as a nibble, high first, the first row then the second (width bytes in all). Strips are in row order. Ends with row 0xFFFF.

use esp_storage::FlashStorage;

/// Where it's kept: the `nvs` partition after the font.
pub const ADDR: u32 = 0xC000;
pub const SIZE: u32 = 0x3000;
pub const MAGIC: &[u8; 4] = b"CST1";

/// Word-aligned for the ROM flash routines.
#[repr(C, align(4))]
pub struct Blob(pub [u8; SIZE as usize]);

/// The blob's length if `head` (its first 8 bytes) starts a valid one.
pub fn length(head: &[u8]) -> Option<usize> {
    let len = u32::from_le_bytes(head[4..8].try_into().ok()?);
    (&head[..4] == MAGIC && len >= 10 && len <= SIZE && len % 4 == 0).then_some(len as usize)
}

/// Read the kept blob into `blob`: its length, or 0 if there's none.
pub fn load(flash: &mut FlashStorage, blob: &mut [u8; SIZE as usize]) -> usize {
    if flash.read_nor(ADDR, &mut blob[..8]).is_err() {
        return 0;
    }
    match length(&blob[..8]) {
        Some(len) if flash.read_nor(ADDR, &mut blob[..len]).is_ok() => len,
        _ => 0,
    }
}

/// The strips: (row, first column, width, coverage nibbles).
pub fn strips(blob: &[u8]) -> impl Iterator<Item = (u16, u16, u16, &[u8])> {
    let mut at = 8;
    core::iter::from_fn(move || {
        let word = |i: usize| -> Option<u16> { Some(u16::from_le_bytes([*blob.get(i)?, *blob.get(i + 1)?])) };
        let y = word(at).filter(|&y| y != 0xFFFF)?;
        let (x, w) = (word(at + 2)?, word(at + 4)?);
        let data = blob.get(at + 6..at + 6 + w as usize)?;
        at += 6 + w as usize;
        Some((y, x, w, data))
    })
}

/// Coverage of a strip's pixel k on its row r (0 or 1).
pub fn coverage(data: &[u8], w: u16, r: u16, k: u16) -> u8 {
    let i = (r * w + k) as usize;
    if i % 2 == 0 { data[i / 2] >> 4 } else { data[i / 2] & 15 }
}

/// One RGB565 pixel (high byte first, in place) nudged by `sign` × 16 codes in gamma space, scaled by coverage `c` (0..=15): red and blue move 2c/15 of their 5-bit levels, green 4c/15 of its 6-bit ones, floored, clamped at the ends. The same in both phases, so they average out wherever nothing clamps.
pub fn nudge(px: &mut [u8], c: u8, sign: i32) {
    let p = u16::from_be_bytes([px[0], px[1]]);
    let c = c as i32;
    let ch = |shift: u32, max: i32, step: i32| (((p >> shift) as i32 & max) + sign * (step * c / 15)).clamp(0, max) as u16;
    let q = ch(11, 31, 2) << 11 | ch(5, 63, 4) << 5 | ch(0, 31, 2);
    px.copy_from_slice(&q.to_be_bytes());
}
