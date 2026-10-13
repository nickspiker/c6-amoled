//! The panel's colour model in VSF LMS, built from the i1Pro sweep: the cone responses of its three primaries at full drive (the matrix to and from linear panel RGB) and each channel's measured response curve (code → fraction of full drive), toe included.
//!
//! Assumes the state the sweep measured: brightness register 255, 24-bit interface.

use std::path::Path;

use vsf::colour::convert::{apply_matrix_3x3_f32, invert_matrix_3x3_f32};

use crate::spectra::{Spectrum, read_sp};

pub struct PanelModel {
    /// Cone responses of the R, G, B primaries at full drive, column-major (VSF's matrix layout): linear panel RGB → LMS.
    pub prim: [f32; 9],
    /// LMS → linear panel RGB.
    pub inv: [f32; 9],
    /// Per channel: fraction of full drive emitted at each code (monotone; 0 through the toe).
    pub k: [[f64; 256]; 3],
    /// Cone response of the light the panel emits at all-black drive: zero for the AMOLED, the backlight's leak for an LCD. It adds to every colour and can't be removed.
    pub black: [f64; 3],
}

impl PanelModel {
    /// Build from a full sweep directory (black, ramps) and a toe sweep directory (white codes 1–16, full field).
    pub fn load(sweep: &Path, toe: &Path) -> PanelModel {
        let sp = |dir: &Path, n: &str| read_sp(&dir.join(format!("{n}.sp")));
        let black = sp(sweep, "black");
        let channels = ["r", "g", "b"];
        let full: Vec<Spectrum> = channels.iter().map(|c| sp(sweep, &format!("ramp_{c}_255")).minus(&black)).collect();

        let mut prim = [0f32; 9];
        for (c, s) in full.iter().enumerate() {
            let l = s.lms();
            for i in 0..3 {
                prim[c * 3 + i] = l[i] as f32;
            }
        }

        // Toe shape from the white low-end pass: each code's emission relative to code 16 (floor = mean of the pass's black readings).
        let toe_floor = Spectrum::mean(&(0..3).map(|i| sp(toe, &format!("toe_black_{i}"))).collect::<Vec<_>>());
        let toe16 = sp(toe, "toe_w_016").minus(&toe_floor);
        let toe_ratio = |code: usize| sp(toe, &format!("toe_w_{code:03}")).minus(&toe_floor).fit(&toe16).0.max(0.0);

        let measured: Vec<usize> = (16..=248).step_by(8).chain([255]).collect();
        let mut k = [[0f64; 256]; 3];
        for (c, ch) in channels.iter().enumerate() {
            let pts: Vec<(usize, f64)> = measured.iter().map(|&code| (code, sp(sweep, &format!("ramp_{ch}_{code:03}")).minus(&black).fit(&full[c]).0)).collect();
            // Codes 16–255: interpolate between measured codes in log–log (piecewise power law).
            for w in pts.windows(2) {
                let ((c0, k0), (c1, k1)) = (w[0], w[1]);
                for code in c0..=c1 {
                    let t = ((code as f64).ln() - (c0 as f64).ln()) / ((c1 as f64).ln() - (c0 as f64).ln());
                    k[c][code] = (k0.ln() * (1.0 - t) + k1.ln() * t).exp();
                }
            }
            // Codes 6–15: the measured toe, scaled onto this channel's code-16 output. Codes 0–5 emit nothing.
            for code in 6..16 {
                k[c][code] = k[c][16] * toe_ratio(code);
            }
            // Enforce monotonicity so the inverse lookup is well defined.
            for code in 1..256 {
                k[c][code] = k[c][code].max(k[c][code - 1]);
            }
        }
        PanelModel { prim, inv: invert_matrix_3x3_f32(&prim), k, black: [0.0; 3] }
    }

    /// Build an LCD's model from its sweep (tools/measure plan `lcd`): every reading minus the black reading (the backlight's leak, kept as `black`), primaries from each channel at 255, and each channel's response from its ramp at codes 4, 8, 16 … 248, 255 — log–log between measured codes from 8 up, linear from code 0 to the first, where readings sit near the leak and logs are unstable.
    pub fn load_lcd(sweep: &Path) -> PanelModel {
        let sp = |n: &str| read_sp(&sweep.join(format!("{n}.sp")));
        let black = sp("black");
        let channels = ["r", "g", "b"];
        let full: Vec<Spectrum> = channels.iter().map(|c| sp(&format!("ramp_{c}_255")).minus(&black)).collect();
        let mut prim = [0f32; 9];
        for (c, s) in full.iter().enumerate() {
            let l = s.lms();
            for i in 0..3 {
                prim[c * 3 + i] = l[i] as f32;
            }
        }
        let measured: Vec<usize> = [4].into_iter().chain((8..=248).step_by(8)).chain([255]).collect();
        let mut k = [[0f64; 256]; 3];
        for (c, ch) in channels.iter().enumerate() {
            let pts: Vec<(usize, f64)> = core::iter::once((0, 0.0)).chain(measured.iter().map(|&code| (code, sp(&format!("ramp_{ch}_{code:03}")).minus(&black).fit(&full[c]).0.max(0.0)))).collect();
            for w in pts.windows(2) {
                let ((c0, k0), (c1, k1)) = (w[0], w[1]);
                for code in c0..=c1 {
                    k[c][code] = if c0 >= 8 && k0 > 0.0 && k1 > 0.0 {
                        let t = ((code as f64).ln() - (c0 as f64).ln()) / ((c1 as f64).ln() - (c0 as f64).ln());
                        (k0.ln() * (1.0 - t) + k1.ln() * t).exp()
                    } else {
                        k0 + (k1 - k0) * (code - c0) as f64 / (c1 - c0) as f64
                    };
                }
            }
            for code in 1..256 {
                k[c][code] = k[c][code].max(k[c][code - 1]);
            }
        }
        PanelModel { prim, inv: invert_matrix_3x3_f32(&prim), k, black: black.lms() }
    }

    /// Replace the primary matrix with full-drive R, G, B spectra (floor-subtracted) captured now — e.g. after the instrument was re-placed, which changes the absolute level reaching it. The response curves are relative and carry over.
    pub fn rebase(&mut self, full: [&Spectrum; 3]) {
        for (c, s) in full.iter().enumerate() {
            let l = s.lms();
            for i in 0..3 {
                self.prim[c * 3 + i] = l[i] as f32;
            }
        }
        self.inv = invert_matrix_3x3_f32(&self.prim);
    }

    /// LMS multiplier that puts Illuminant E white (LMS 1, 1, 1) at `fraction` of the brightest E white the panel can emit.
    pub fn white_scale(&self, fraction: f64) -> f64 {
        let rgb = apply_matrix_3x3_f32(&self.inv, &[1.0, 1.0, 1.0]);
        fraction / rgb.iter().cloned().fold(f32::MIN, f32::max) as f64
    }

    /// Linear panel RGB (fractions of full drive) for an LMS target; may fall outside [0, 1] when the target is out of gamut.
    pub fn rgb_of(&self, lms: [f64; 3]) -> [f64; 3] {
        apply_matrix_3x3_f32(&self.inv, &[lms[0] as f32, lms[1] as f32, lms[2] as f32]).map(|v| v as f64)
    }

    /// The LMS the panel actually emits for linear panel RGB.
    pub fn lms_of(&self, rgb: [f64; 3]) -> [f64; 3] {
        apply_matrix_3x3_f32(&self.prim, &[rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]).map(|v| v as f64)
    }

    /// Code whose output is nearest `target` (fraction of full drive) on channel `c`.
    pub fn code_for(&self, c: usize, target: f64) -> u8 {
        let k = &self.k[c];
        if target <= k[6] / 2.0 {
            return 0;
        }
        let i = k.partition_point(|&v| v < target).min(255);
        let below = i.saturating_sub(1);
        if (target - k[below]).abs() <= (k[i] - target).abs() { below as u8 } else { i as u8 }
    }

    /// Encode an LMS target to panel codes, clipping to the gamut per channel. Returns the codes and whether clipping happened.
    pub fn encode(&self, lms: [f64; 3]) -> ([u8; 3], bool) {
        let rgb = self.rgb_of(lms);
        let clipped = rgb.iter().any(|&v| !(-1e-4..=1.0 + 1e-4).contains(&v));
        let mut codes = [0u8; 3];
        for c in 0..3 {
            codes[c] = self.code_for(c, rgb[c].clamp(0.0, 1.0));
        }
        (codes, clipped)
    }
}
