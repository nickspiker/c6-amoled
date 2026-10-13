//! Report on an LCD sweep (plan `lcd`) in VSF colorimetry: primaries, the backlight's leak at black, each channel's response, additivity, and the backlight's duty → light curve. Prints and writes `report.txt` next to the sweep, plus `response.json` (each channel's light at every 8-bit code) for renderers.
//!
//! Usage: `lcd_report <sweep-dir>`

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use measure::model::PanelModel;
use measure::spectra::{Spectrum, agb, photopic, read_sp};
use vsf::colour::convert::{apply_matrix_3x3_f32, invert_matrix_3x3_f32};
use vsf::colour::rec2020::constants::REC2020_2VSF_RGB;
use vsf::colour::spectral::constants::VSF_RGB2LMS;

/// Wavelength of the peak and the full width at half maximum, nm.
fn peak_fwhm(s: &Spectrum) -> (f64, f64) {
    let (i, &m) = s.v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
    let half = m / 2.0;
    let (mut lo, mut hi) = (i, i);
    while lo > 0 && s.v[lo] > half {
        lo -= 1;
    }
    while hi + 1 < s.v.len() && s.v[hi] > half {
        hi += 1;
    }
    (s.start + i as f64 * s.step, (hi - lo) as f64 * s.step)
}

/// Least-squares slope of ln y against ln x: the exponent of a power law.
fn exponent(pts: &[(f64, f64)]) -> f64 {
    let p: Vec<(f64, f64)> = pts.iter().filter(|(x, y)| *x > 0.0 && *y > 0.0).map(|(x, y)| (x.ln(), y.ln())).collect();
    let n = p.len() as f64;
    let (mx, my) = (p.iter().map(|q| q.0).sum::<f64>() / n, p.iter().map(|q| q.1).sum::<f64>() / n);
    p.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum::<f64>() / p.iter().map(|q| (q.0 - mx).powi(2)).sum::<f64>()
}

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("usage: lcd_report <sweep-dir>"));
    let sp = |n: &str| read_sp(&dir.join(format!("{n}.sp")));
    let model = PanelModel::load_lcd(&dir);
    let black = sp("black");
    let mut r = String::new();

    let _ = writeln!(r, "LCD characterisation, VSF colorimetry (SS2000 10° cones, Illuminant E)  —  {}\n", dir.display());
    let _ = writeln!(r, "PRIMARIES (full drive, minus the black reading)");
    let names = ["red", "green", "blue"];
    let full: Vec<Spectrum> = ["r", "g", "b"].iter().map(|c| sp(&format!("ramp_{c}_255")).minus(&black)).collect();
    let white = sp("ramp_w_255").minus(&black);
    let vw = photopic(white.lms());
    for (c, s) in full.iter().enumerate() {
        let lms = s.lms();
        let (pk, fw) = peak_fwhm(s);
        let [a, b, g] = agb(lms);
        let _ = writeln!(r, "  {:5}  peak {pk:5.0} nm  FWHM {fw:4.0} nm   α {a:.4} β {b:.4} γ {g:.4}   share of white's photopic {:5.1} %", names[c], 100.0 * photopic(lms) / vw);
    }
    let sum: [f64; 3] = core::array::from_fn(|i| full.iter().map(|s| s.lms()[i]).sum());
    let wl = white.lms();
    let _ = writeln!(r, "  additivity: white / (R+G+B) per cone  L {:.3}  M {:.3}  S {:.3}   photopic {:.3}", wl[0] / sum[0], wl[1] / sum[1], wl[2] / sum[2], vw / photopic(sum));

    // Each primary in wavelength-defined Rec.2020 coordinates (VSF): outside the triangle shows as a negative component.
    let rec2lms = {
        let a = VSF_RGB2LMS;
        let b = REC2020_2VSF_RGB;
        // Column-major 3×3 product a·b.
        let mut m = [0f32; 9];
        for col in 0..3 {
            for row in 0..3 {
                m[col * 3 + row] = (0..3).map(|k| a[k * 3 + row] * b[col * 3 + k]).sum();
            }
        }
        m
    };
    let lms2rec = invert_matrix_3x3_f32(&rec2lms);
    let _ = writeln!(r, "\nGAMUT: each primary in Rec.2020 (VSF, wavelength-defined) linear RGB, normalised to its largest component");
    for (c, s) in full.iter().enumerate() {
        let l = s.lms();
        let v = apply_matrix_3x3_f32(&lms2rec, &[l[0] as f32, l[1] as f32, l[2] as f32]);
        let m = v.iter().cloned().fold(f32::MIN, f32::max);
        let _ = writeln!(r, "  {:5}  ({:+.3}, {:+.3}, {:+.3})", names[c], v[0] / m, v[1] / m, v[2] / m);
    }

    let _ = writeln!(r, "\nBLACK (the backlight's leak at all-black drive, backlight full)");
    let bl = black.lms();
    let [a, b, g] = agb(bl);
    let (pk, _) = peak_fwhm(&black);
    let _ = writeln!(r, "  photopic {:.4} of white  (contrast {:.0}:1)   peak {pk:.0} nm   α {a:.4} β {b:.4} γ {g:.4}", photopic(bl) / (vw + photopic(bl)), (vw + photopic(bl)) / photopic(bl));

    let _ = writeln!(r, "\nRESPONSE: fraction of full drive by code (minus black), and the power-law exponent fitted over codes 16–255");
    let codes = [4usize, 8, 16, 32, 64, 96, 128, 160, 192, 224, 255];
    let _ = writeln!(r, "  code   {}", codes.iter().map(|c| format!("{c:>7}")).collect::<String>());
    for (c, ch) in ["red", "green", "blue"].iter().enumerate() {
        let row: String = codes.iter().map(|&code| format!("{:>7.4}", model.k[c][code])).collect();
        let pts: Vec<(f64, f64)> = (16..=255).map(|code| (code as f64 / 255.0, model.k[c][code])).collect();
        let _ = writeln!(r, "  {ch:6} {row}   γ {:.2}", exponent(&pts));
    }
    let wpts: Vec<(f64, f64)> = [16usize, 32, 64, 96, 128, 160, 192, 224, 255].iter().map(|&code| (code as f64 / 255.0, photopic(sp(&format!("ramp_w_{code:03}")).minus(&black).lms()) / vw)).collect();
    let _ = writeln!(r, "  white photopic γ {:.2}", exponent(&wpts));

    let _ = writeln!(r, "\nBACKLIGHT: PWM duty → white and black photopic (relative to duty 255), contrast");
    let mut bpts = Vec::new();
    for duty in [255u32, 191, 127, 63, 31, 15, 7, 3, 1] {
        let (w, k) = if duty == 255 { (sp("ramp_w_255"), black.clone()) } else { (sp(&format!("backlight_b{duty:03}_w")), sp(&format!("backlight_b{duty:03}_black"))) };
        let (vwd, vkd) = (photopic(w.lms()), photopic(k.lms()));
        let rel = vwd / (vw + photopic(bl));
        bpts.push((duty as f64 / 255.0, rel));
        let [a, b, g] = agb(w.lms());
        let _ = writeln!(r, "  duty {duty:3}: white {rel:7.4}  black {:7.5}  contrast {:5.0}:1   white α {a:.4} β {b:.4} γ {g:.4}", vkd / (vw + photopic(bl)), vwd / vkd.max(1e-9));
    }
    let _ = writeln!(r, "  white vs duty, power-law exponent {:.2} (1.00 = light proportional to duty)", exponent(&bpts));

    print!("{r}");
    fs::write(dir.join("report.txt"), &r).unwrap();
    // The response curves for the renderer: per channel, light (fraction of full drive) at each 8-bit code.
    let rows: Vec<String> = model.k.iter().map(|ch| format!("[{}]", ch.iter().map(|v| format!("{v:.6}")).collect::<Vec<_>>().join(","))).collect();
    fs::write(dir.join("response.json"), format!("{{\"k\": [{}]}}\n", rows.join(","))).unwrap();
}
