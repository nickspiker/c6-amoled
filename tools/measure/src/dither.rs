//! Ordered dithering to RGB565 in linear light, against the panel's measured levels, with a blue-noise threshold map.
//!
//! Each pixel's target light falls between two levels the panel can emit at 565 (5- or 6-bit values expanded to 8 bits by bit replication, then through the measured response, toe included); the threshold map picks the upper one in proportion to how far up the gap the target sits, so area-averaged light is right everywhere. Ordered rather than error-diffused: the same input always gives the same dots and a local change only flips local pixels, so animated loops don't boil.

use crate::model::PanelModel;

/// Side of the tiling threshold map.
pub const TILE: usize = 64;

/// A TILE×TILE blue-noise threshold map, values in (0, 1), by Ulichney's void-and-cluster method: Gaussian energy (σ 1.5) on the torus, relax the initial pattern, then rank pixels by removing tightest clusters and filling largest voids.
pub fn blue_noise() -> Vec<f64> {
    let n = TILE * TILE;
    let sigma = 1.5f64;
    let mut kern = vec![0.0; n];
    for dy in 0..TILE {
        for dx in 0..TILE {
            let (x, y) = (dx.min(TILE - dx) as f64, dy.min(TILE - dy) as f64);
            kern[dy * TILE + dx] = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
        }
    }
    let splat = |e: &mut [f64], p: usize, s: f64| {
        let (px, py) = (p % TILE, p / TILE);
        for y in 0..TILE {
            let ky = (y + TILE - py) % TILE * TILE;
            for x in 0..TILE {
                e[y * TILE + x] += s * kern[ky + (x + TILE - px) % TILE];
            }
        }
    };
    let tightest = |on: &[bool], e: &[f64]| (0..n).filter(|&p| on[p]).max_by(|&a, &b| e[a].total_cmp(&e[b])).unwrap();
    let voidest = |on: &[bool], e: &[f64]| (0..n).filter(|&p| !on[p]).min_by(|&a, &b| e[a].total_cmp(&e[b])).unwrap();

    // Initial pattern: a tenth of the pixels, placed by xorshift.
    let (mut on, mut energy) = (vec![false; n], vec![0.0; n]);
    let (mut s, mut count) = (0x9E37_79B9_7F4A_7C15u64, 0);
    while count < n / 10 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let p = (s % n as u64) as usize;
        if !on[p] {
            on[p] = true;
            splat(&mut energy, p, 1.0);
            count += 1;
        }
    }
    // Relax: move the tightest cluster into the largest void until that's a no-op.
    for _ in 0..100 * n {
        let c = tightest(&on, &energy);
        on[c] = false;
        splat(&mut energy, c, -1.0);
        let v = voidest(&on, &energy);
        on[v] = true;
        splat(&mut energy, v, 1.0);
        if v == c {
            break;
        }
    }
    let mut rank = vec![0usize; n];
    let (mut o, mut e) = (on.clone(), energy.clone());
    for r in (0..count).rev() {
        let c = tightest(&o, &e);
        o[c] = false;
        splat(&mut e, c, -1.0);
        rank[c] = r;
    }
    // On the torus every pixel's total kernel weight is the same, so the tightest cluster of zeros is the largest void of ones: one rule fills both halves.
    for r in count..n {
        let v = voidest(&on, &energy);
        on[v] = true;
        splat(&mut energy, v, 1.0);
        rank[v] = r;
    }
    rank.iter().map(|&r| (r as f64 + 0.5) / n as f64).collect()
}

/// One channel's distinct 565 levels: the light each emits (fraction of full drive, increasing) and its 8-bit code after bit replication.
pub struct Levels {
    light: Vec<f64>,
    code: Vec<u8>,
}

impl Levels {
    /// Channel `c` of `model`: 5 bits for red and blue, 6 for green. Values the toe maps to the same light collapse to the lowest.
    pub fn new(model: &PanelModel, c: usize) -> Levels {
        let bits = if c == 1 { 6 } else { 5 };
        let (mut light, mut code) = (Vec::new(), Vec::new());
        for v in 0u32..1 << bits {
            let c8 = ((v << (8 - bits)) | (v >> (2 * bits - 8))) as u8;
            let l = model.k[c][c8 as usize];
            if light.last().is_none_or(|&last| l > last) {
                light.push(l);
                code.push(c8);
            }
        }
        Levels { light, code }
    }

    /// The 8-bit code for target light `t` (fraction of full drive) at a pixel whose threshold is `th`.
    pub fn pick(&self, t: f64, th: f64) -> u8 {
        let i = self.light.partition_point(|&l| l <= t);
        if i == 0 {
            return self.code[0];
        }
        if i == self.light.len() {
            return *self.code.last().unwrap();
        }
        let (lo, hi) = (self.light[i - 1], self.light[i]);
        if (t - lo) / (hi - lo) > th { self.code[i] } else { self.code[i - 1] }
    }
}
