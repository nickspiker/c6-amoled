//! Emissive spectra (ArgyllCMS `.sp` files, mW/(m²·sr·nm)) and their cone responses under VSF's Stockman & Sharpe 2000 10° fundamentals.

use std::fs;
use std::path::Path;

use vsf::colour::spectral::constants::{LMS2PHOTOPIC, LMS_2000_10DEG_1NM};

/// A sampled spectrum on a regular grid; zero outside it.
#[derive(Clone, Debug)]
pub struct Spectrum {
    pub start: f64,
    pub step: f64,
    pub v: Vec<f64>,
}

impl Spectrum {
    /// Linear interpolation at `nm`; zero outside the sampled range.
    pub fn at(&self, nm: f64) -> f64 {
        let f = (nm - self.start) / self.step;
        if f < 0.0 || f > (self.v.len() - 1) as f64 {
            return 0.0;
        }
        let i = (f.floor() as usize).min(self.v.len() - 2);
        let t = f - i as f64;
        self.v[i] * (1.0 - t) + self.v[i + 1] * t
    }

    /// Sample-wise difference; both spectra must share a grid.
    pub fn minus(&self, other: &Spectrum) -> Spectrum {
        assert!(self.start == other.start && self.step == other.step && self.v.len() == other.v.len(), "spectra on different grids");
        Spectrum { start: self.start, step: self.step, v: self.v.iter().zip(&other.v).map(|(a, b)| a - b).collect() }
    }

    /// Sample-wise mean of spectra sharing a grid.
    pub fn mean(all: &[Spectrum]) -> Spectrum {
        let mut m = all[0].clone();
        for s in &all[1..] {
            for (a, b) in m.v.iter_mut().zip(&s.v) {
                *a += b;
            }
        }
        m.v.iter_mut().for_each(|a| *a /= all.len() as f64);
        m
    }

    /// Least-squares scale of this spectrum onto `reference` (same grid) and the relative RMS left over — the observer-free "how much of that emission is this".
    pub fn fit(&self, reference: &Spectrum) -> (f64, f64) {
        let rr: f64 = reference.v.iter().map(|r| r * r).sum();
        let k = self.v.iter().zip(&reference.v).map(|(s, r)| s * r).sum::<f64>() / rr;
        let resid: f64 = self.v.iter().zip(&reference.v).map(|(s, r)| (s - k * r).powi(2)).sum::<f64>().sqrt();
        (k, resid / (k.abs() * rr.sqrt()).max(f64::MIN_POSITIVE))
    }

    /// Cone responses: Σ spectrum(λ) · fundamentals(λ) over the fundamentals' 1 nm grid (390–830 nm). Sum-normalised fundamentals make an equal-energy spectrum of 1 per nm come out as [1, 1, 1].
    pub fn lms(&self) -> [f64; 3] {
        let t = &LMS_2000_10DEG_1NM;
        let mut out = [0.0; 3];
        for i in 0..t.num_samples() {
            let v = self.at(t.wavelength_at(i) as f64);
            for (c, o) in out.iter_mut().enumerate() {
                *o += v * t.data[i * 3 + c] as f64;
            }
        }
        out
    }
}

/// Read an ArgyllCMS CGATS `.sp` file.
pub fn read_sp(path: &Path) -> Spectrum {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let field = |key: &str| -> f64 { text.lines().find_map(|l| l.strip_prefix(key)).unwrap_or_else(|| panic!("{key} missing in {}", path.display())).trim().trim_matches('"').parse().unwrap() };
    let (start, end, bands) = (field("SPECTRAL_START_NM"), field("SPECTRAL_END_NM"), field("SPECTRAL_BANDS") as usize);
    let data = text.split("BEGIN_DATA\n").nth(1).unwrap_or_else(|| panic!("BEGIN_DATA missing in {}", path.display()));
    let v: Vec<f64> = data.split_whitespace().take(bands).map(|t| t.parse().unwrap()).collect();
    assert_eq!(v.len(), bands, "{}: short data", path.display());
    Spectrum { start, step: (end - start) / (bands - 1) as f64, v }
}

/// Photopic luminance in VSF's weights (L and M shares; only ratios are meaningful).
pub fn photopic(lms: [f64; 3]) -> f64 {
    (0..3).map(|i| LMS2PHOTOPIC[i] as f64 * lms[i]).sum()
}

/// VSF's AGB ratios: α = L/(L+M), β = S/(S+M), γ = M/(L+S).
pub fn agb(lms: [f64; 3]) -> [f64; 3] {
    let [l, m, s] = lms;
    [l / (l + m), s / (s + m), m / (l + s)]
}
