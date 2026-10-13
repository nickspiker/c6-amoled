//! The touch screen and the brightness readout: drag up or down anywhere to step through 144 brightness levels, shown while you adjust as two dozenal digits (Oxanium with the Zil..Stelor glyphs) on a soft dark disc over the loop. Text is blended in linear light through the panel's measured response, integers only.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicI32, AtomicU8, Ordering};

use embassy_time::{Duration, Instant, Timer};
use esp_hal::Blocking;
use esp_hal::i2c::master::I2c;
use esp_println::println;
use static_cell::ConstStaticCell;

use crate::{HEIGHT, OVERLAY, Shared, WIDTH};

/// Brightness register value for each of the 144 levels, evenly spaced in perceived lightness (about the cube root of light) over the panel's measured response, light ∝ value^2.29: value = 255 · ((level + 1) / 144)^(3 / 2.29), at least 1.
pub const LEVELS: [u8; 144] = [
    1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 33, 34, 36, 37, 38, 40, 41, 43, 45, 46, 48, 49, 51, 52, 54, 56, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 85, 86, 88, 90, 92, 94, 95, 97, 99, 101, 103, 105, 107, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 169, 171, 173, 175, 177, 179, 181, 183, 186, 188, 190, 192, 194, 196, 199, 201, 203, 205, 207, 210, 212, 214, 216, 219, 221, 223, 225, 228, 230, 232, 234, 237, 239, 241, 243, 246, 248, 250, 253, 255,
];
pub const TOP_LEVEL: u8 = 143;

/// Where the packed glyphs live: the `nvs` partition past the settings sector (written with `F`).
pub const FONT_ADDR: u32 = 0xA000;
pub const FONT_SIZE: u32 = 0x5000;
const FONT_MAGIC: &[u8; 4] = b"FNT1";

/// A glyph blob from tools/measure `fontpack` (its docs give the layout), held in RAM.
pub struct Font {
    blob: Vec<u8>,
}

struct Glyph {
    w: usize,
    h: usize,
    left: i32,
    top: i32,
    advance: i32,
    at: usize,
}

impl Font {
    pub fn bytes(&self) -> &[u8] {
        &self.blob
    }

    pub fn parse(blob: Vec<u8>) -> Option<Font> {
        let ok = blob.len() >= 272 && &blob[..4] == FONT_MAGIC && u32::from_le_bytes(blob[4..8].try_into().ok()?) as usize == blob.len();
        ok.then_some(Font { blob })
    }

    fn u16(&self, at: usize) -> u16 {
        u16::from_le_bytes([self.blob[at], self.blob[at + 1]])
    }

    /// Linear light (0..=65535) of RGB565 level `v` on channel `c` (0 red, 1 green, 2 blue).
    fn lin(&self, c: usize, v: usize) -> u32 {
        self.u16(8 + 2 * ([0, 32, 96][c] + v)) as u32
    }

    /// The level on channel `c` whose light is nearest `target`.
    fn nearest(&self, c: usize, target: u32) -> u16 {
        let n = if c == 1 { 64 } else { 32 };
        let (mut lo, mut hi) = (0, n - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.lin(c, mid) < target { lo = mid + 1 } else { hi = mid }
        }
        if lo > 0 && target - self.lin(c, lo - 1).min(target) < self.lin(c, lo).saturating_sub(target) {
            lo -= 1;
        }
        lo as u16
    }

    fn glyph(&self, cp: u32) -> Option<Glyph> {
        let count = self.u16(264) as usize;
        (0..count).map(|n| 272 + 20 * n).find(|&e| u32::from_le_bytes(self.blob[e..e + 4].try_into().unwrap()) == cp).map(|e| Glyph {
            w: self.u16(e + 4) as usize,
            h: self.u16(e + 6) as usize,
            left: self.u16(e + 8) as i16 as i32,
            top: self.u16(e + 10) as i16 as i32,
            advance: self.u16(e + 12) as i32,
            at: u32::from_le_bytes(self.blob[e + 16..e + 20].try_into().unwrap()) as usize,
        })
    }

    /// Unpack a glyph's run-length-coded coverage, handing each nonzero pixel to `put(x, y, coverage)`.
    fn draw(&self, g: &Glyph, mut put: impl FnMut(usize, usize, u8)) {
        let (mut i, mut p) = (g.at, 0);
        while p < g.w * g.h {
            let b = self.blob[i];
            i += 1;
            if b & 0x80 != 0 {
                let n = (b & 0x3F) as usize + 1;
                if b & 0x40 != 0 {
                    (p..p + n).for_each(|q| put(q % g.w, q / g.w, 15));
                }
                p += n;
            } else {
                for k in 0..(b as usize + 1) {
                    let v = if k % 2 == 0 { self.blob[i + k / 2] >> 4 } else { self.blob[i + k / 2] & 15 };
                    if v != 0 {
                        put((p + k) % g.w, (p + k) / g.w, v);
                    }
                }
                p += b as usize + 1;
                i += (b as usize + 2) / 2;
            }
        }
    }
}

/// The readout: digit coverage on a canvas around the panel's centre, a dark disc behind it, and how far it has faded in (0 hidden .. 15 full). Its buffers are static: the heap is small and mostly the Wi-Fi driver's.
pub struct Overlay {
    pub fade: u8,
    value: Option<u8>,
    coverage: &'static mut [u8; OW * OH],
    /// For each backdrop strength 0..16, the dimmed level of every red, green and blue RGB565 level; built from the font's light table on first use.
    dim: &'static mut [[u16; 128]; 16],
    dim_ready: bool,
}

static COVERAGE: ConstStaticCell<[u8; OW * OH]> = ConstStaticCell::new([0; OW * OH]);
static DIM: ConstStaticCell<[[u16; 128]; 16]> = ConstStaticCell::new([[0; 128]; 16]);

const OW: usize = 240;
const OH: usize = 120;
const OX: i32 = (WIDTH as i32 - OW as i32) / 2;
const OY: i32 = (HEIGHT as i32 - OH as i32) / 2;
/// The backdrop disc: radius and soft edge (px).
const DISC: i32 = 120;
const EDGE: i32 = 14;
/// How dark the backdrop gets at full strength: light × (256 − 15 · DIM_STEP) / 256.
const DIM_STEP: u32 = 10;

impl Overlay {
    /// Call once: takes the static buffers.
    pub fn new() -> Overlay {
        Overlay { fade: 0, value: None, coverage: COVERAGE.take(), dim: DIM.take(), dim_ready: false }
    }

    /// Forget the laid-out digits and tables, after the font changes.
    pub fn reset(&mut self) {
        self.value = None;
        self.dim_ready = false;
    }

    /// Lay out `value` as two dozenal digits, centred.
    pub fn set(&mut self, font: &Font, value: u8) {
        if !self.dim_ready {
            for (s, t) in self.dim.iter_mut().enumerate() {
                for (c, n, base) in [(0, 32, 0), (1, 64, 32), (2, 32, 96)] {
                    for v in 0..n {
                        t[base + v] = font.nearest(c, font.lin(c, v) * (256 - s as u32 * DIM_STEP) / 256);
                    }
                }
            }
            self.dim_ready = true;
        }
        if self.value == Some(value) {
            return;
        }
        self.value = Some(value);
        self.coverage.fill(0);
        let digits = [font.glyph(0x10 + (value / 12) as u32), font.glyph(0x10 + (value % 12) as u32)];
        let (Some(a), Some(b)) = (&digits[0], &digits[1]) else { return };
        let cap = (-a.top).max(-b.top);
        let baseline = (OH as i32 + cap) / 2;
        let mut pen = (OW as i32 - a.advance - b.advance) / 2;
        for g in [a, b] {
            let (gx, gy) = (pen + g.left, baseline + g.top);
            font.draw(g, |x, y, v| {
                let (cx, cy) = (gx + x as i32, gy + y as i32);
                if (0..OW as i32).contains(&cx) && (0..OH as i32).contains(&cy) {
                    self.coverage[cy as usize * OW + cx as usize] = v;
                }
            });
            pen += g.advance;
        }
    }

    /// Blend the readout into one strip in the band buffer (RGB565, high byte first): the strip covers columns `x0..x0 + w` of rows `y0..y0 + rows`.
    pub fn apply(&self, font: &Font, band: &mut [u8], x0: u16, y0: u16, w: u16, rows: u16) {
        let c = WIDTH as i32;
        let (r2, inner) = ((2 * DISC) * (2 * DISC), (2 * (DISC - EDGE)) * (2 * (DISC - EDGE)));
        for y in y0 as i32..(y0 + rows) as i32 {
            let dy = 2 * y + 1 - c;
            if dy * dy >= r2 {
                continue;
            }
            for x in (c / 2 - DISC).max(x0 as i32)..(c / 2 + DISC).min((x0 + w) as i32) {
                let dx = 2 * x + 1 - c;
                let d2 = dx * dx + dy * dy;
                if d2 >= r2 {
                    continue;
                }
                let strength = if d2 <= inner { 15 } else { 15 * (r2 - d2) / (r2 - inner) } as u32;
                let s = (strength * self.fade as u32 / 15) as usize;
                let i = 2 * ((y - y0 as i32) as usize * w as usize + (x - x0 as i32) as usize);
                let p = (band[i] as u16) << 8 | band[i + 1] as u16;
                let t = &self.dim[s];
                let mut v = [t[(p >> 11) as usize], t[32 + (p >> 5 & 63) as usize], t[96 + (p & 31) as usize]];
                let (cx, cy) = (x - OX, y - OY);
                if (0..OW as i32).contains(&cx) && (0..OH as i32).contains(&cy) {
                    let a = self.coverage[cy as usize * OW + cx as usize] as u32 * self.fade as u32 / 15;
                    if a > 0 {
                        for (ch, level) in v.iter_mut().enumerate() {
                            let l = font.lin(ch, *level as usize);
                            *level = font.nearest(ch, (l * (15 - a) + 65535 * a) / 15);
                        }
                    }
                }
                let q = v[0] << 11 | v[1] << 5 | v[2];
                band[i] = (q >> 8) as u8;
                band[i + 1] = q as u8;
            }
        }
    }
}

/// The touch controller (CST816-family at 0x15) and the last reading, for `info`.
const TOUCH: u8 = 0x15;
pub static TOUCH_ID: AtomicU8 = AtomicU8::new(0);
pub static TOUCH_Y: AtomicI32 = AtomicI32::new(-1);
/// Drag distance per level, and which way is up: the panel is mounted rotated 180°, so moving up the screen raises the controller's y.
const PX_PER_LEVEL: i32 = 3;
const UP: i32 = 1;
/// The readout stays this long after the finger lifts, then fades over 15 polls.
const LINGER_MS: u64 = 1200;
const POLL_MS: u64 = 20;

/// Poll the touch controller: a drag sets the brightness and shows the readout; after lifting, the readout lingers, fades, and the new level is saved.
#[embassy_executor::task]
pub async fn touch(mut i2c: I2c<'static, Blocking>, board: &'static Shared) {
    let mut id = [0u8];
    match i2c.write_read(TOUCH, &[0xA7], &mut id) {
        Ok(()) => println!("touch: controller id {:#04x}", id[0]),
        Err(e) => println!("touch: no answer from {TOUCH:#04x} ({e:?})"),
    }
    TOUCH_ID.store(id[0], Ordering::Relaxed);
    // Keep it awake between touches (CST816 DisAutoSleep), so polling always gets an answer.
    let _ = i2c.write(TOUCH, &[0xFE, 1]);
    let (mut down, mut y0, mut from) = (false, 0, 0u8);
    let mut lifted: Option<Instant> = None;
    loop {
        let mut d = [0u8; 6];
        // Blocking is fine: a poll is 7 bytes at 400 kHz, well under a millisecond, and the blocking driver is much smaller than the async one.
        let fingers = match i2c.write_read(TOUCH, &[0x01], &mut d) {
            Ok(()) => d[1] & 0x0F,
            Err(_) => 0,
        };
        if fingers > 0 {
            let y = ((d[4] as i32 & 0x0F) << 8) | d[5] as i32;
            TOUCH_Y.store(y, Ordering::Relaxed);
            let mut b = board.lock().await;
            if !down {
                (down, y0, from) = (true, y, b.level);
            }
            let level = (from as i32 + UP * (y - y0) / PX_PER_LEVEL).clamp(0, TOP_LEVEL as i32) as u8;
            if level != b.level || b.overlay.fade != 15 {
                b.set_level(level);
                b.overlay.fade = 15;
                OVERLAY.signal(());
            }
            lifted = None;
        } else if down {
            down = false;
            lifted = Some(Instant::now());
        }
        if let Some(t) = lifted {
            let since = t.elapsed().as_millis();
            if since >= LINGER_MS {
                let fade = 15u64.saturating_sub((since - LINGER_MS) / POLL_MS) as u8;
                let mut b = board.lock().await;
                b.overlay.fade = fade;
                OVERLAY.signal(());
                if fade == 0 {
                    lifted = None;
                    b.save_level();
                }
            }
        }
        Timer::after(Duration::from_millis(POLL_MS)).await;
    }
}
