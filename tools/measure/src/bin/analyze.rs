//! Analyse the panel sweeps in VSF colorimetry, from the spectra alone: primaries, additivity, per-channel response curves (observer-independent spectral fits), window-size loading, brightness-register separability, RGB565 vs RGB888, and the near-black toe at several brightness levels.
//!
//! Usage: `analyze <sweep-dir> [toe-dir] [toe-brightness-dir]` — writes `report.txt`, `eotf.csv` and `primaries_1nm.csv` into the sweep dir.
//!
//! A response curve is fitted spectrally: each ramp spectrum (floor-subtracted) is least-squares scaled onto that channel's full-drive spectrum, so k(code) carries no observer at all. Levels are VSF photopic (`LMS2PHOTOPIC`, relative), chromaticity is VSF's AGB ratios, cones are Stockman & Sharpe 2000 10°.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use measure::spectra::{Spectrum, agb, photopic, read_sp};
use vsf::colour::convert::{apply_matrix_3x3_f32, invert_matrix_3x3_f32};
use vsf::colour::rec2020::constants::REC2020_2VSF_RGB;
use vsf::colour::spectral::constants::{LMS_2000_10DEG_1NM, VSF_RGB2LMS};

/// Names in a sweep, from its CSV's first column.
fn names(dir: &Path) -> Vec<String> {
    let csv = fs::read_to_string(dir.join("sweep.csv")).unwrap_or_else(|_| panic!("{}/sweep.csv", dir.display()));
    csv.lines().skip(1).map(|l| l.split(',').next().unwrap().to_string()).collect()
}

/// Peak wavelength and full width at half maximum (nm), on a 1 nm resampling.
fn peak_fwhm(s: &Spectrum) -> (f64, f64) {
    let grid: Vec<(f64, f64)> = (380..=730).map(|nm| (nm as f64, s.at(nm as f64))).collect();
    let (ip, &(pk, pv)) = grid.iter().enumerate().max_by(|a, b| a.1.1.total_cmp(&b.1.1)).unwrap();
    let half = pv / 2.0;
    let lo = (1..=ip).rev().find(|&i| grid[i - 1].1 < half).map(|i| grid[i].0 - (grid[i].1 - half) / (grid[i].1 - grid[i - 1].1)).unwrap_or(grid[0].0);
    let hi = (ip..grid.len() - 1).find(|&i| grid[i + 1].1 < half).map(|i| grid[i].0 + (grid[i].1 - half) / (grid[i].1 - grid[i + 1].1)).unwrap_or(grid[grid.len() - 1].0);
    (pk, hi - lo)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("usage: analyze <sweep-dir> [toe-dir] [toe-brightness-dir]"));
    let toe_dir = args.next().map(PathBuf::from);
    let toeb_dir = args.next().map(PathBuf::from);
    let sp = |d: &Path, n: &str| read_sp(&d.join(format!("{n}.sp")));
    let black = sp(&dir, "black");
    let net = |n: &str| sp(&dir, n).minus(&black);
    let full = |c: &str| net(&format!("ramp_{c}_255"));
    let mut r = String::new();
    let _ = writeln!(r, "Panel characterisation — {} ({} patches)", dir.display(), names(&dir).len());
    let _ = writeln!(r, "VSF colorimetry from the spectra: Stockman & Sharpe 2000 10° cones, VSF photopic (relative), AGB ratios α = L/(L+M), β = S/(S+M), γ = M/(L+S). Illuminant E white is α = β = γ = 0.5.\n");

    // Primaries.
    let wl = photopic(full("w").lms());
    let _ = writeln!(r, "PRIMARIES (code 255, brightness 255, 24-bit, 240 px window)");
    let _ = writeln!(r, "  ch    α       β       γ      L : M : S (sum 1)       share of white's photopic   peak nm  FWHM nm");
    for c in ["r", "g", "b", "w"] {
        let s = full(c);
        let l = s.lms();
        let [a, b, g] = agb(l);
        let t = l[0] + l[1] + l[2];
        let (pk, fw) = peak_fwhm(&s);
        let shape = if c == "w" { "      —        —".to_string() } else { format!("    {pk:5.0}    {fw:5.1}") };
        let _ = writeln!(r, "  {c}   {a:.4}  {b:.4}  {g:.4}   {:.3} : {:.3} : {:.3}         {:.3}              {shape}", l[0] / t, l[1] / t, l[2] / t, photopic(l) / wl);
    }

    // Gamut against Rec.2020 as VSF defines it (630 / 532 / 467 nm): each panel primary in Rec.2020 RGB (a negative component = outside Rec.2020), and each Rec.2020 primary in panel RGB (a negative component = beyond the panel's reach). Rows normalised to their largest component.
    let lms_of_2020 = |rgb: [f32; 3]| apply_matrix_3x3_f32(&VSF_RGB2LMS, &apply_matrix_3x3_f32(&REC2020_2VSF_RGB, &rgb));
    let m2020: [f32; 9] = { let (a, b, c) = (lms_of_2020([1.0, 0.0, 0.0]), lms_of_2020([0.0, 1.0, 0.0]), lms_of_2020([0.0, 0.0, 1.0])); [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]] };
    let to_2020 = invert_matrix_3x3_f32(&m2020);
    let panel: [f32; 9] = { let l: Vec<[f64; 3]> = ["r", "g", "b"].iter().map(|c| full(c).lms()).collect(); [l[0][0] as f32, l[0][1] as f32, l[0][2] as f32, l[1][0] as f32, l[1][1] as f32, l[1][2] as f32, l[2][0] as f32, l[2][1] as f32, l[2][2] as f32] };
    let to_panel = invert_matrix_3x3_f32(&panel);
    let norm = |v: [f32; 3]| { let m = v.iter().cloned().fold(f32::MIN, f32::max); v.map(|x| x / m) };
    let _ = writeln!(r, "\nGAMUT vs Rec.2020 (VSF: 630/532/467 nm, cones SS2000 10°)\n  panel primary in Rec.2020 RGB        (negative = outside Rec.2020)");
    for (i, c) in ["r", "g", "b"].iter().enumerate() {
        let v = norm(apply_matrix_3x3_f32(&to_2020, &[panel[i * 3], panel[i * 3 + 1], panel[i * 3 + 2]]));
        let _ = writeln!(r, "    panel {c}:  {:+.4} {:+.4} {:+.4}", v[0], v[1], v[2]);
    }
    let _ = writeln!(r, "  Rec.2020 primary in panel RGB        (negative = beyond the panel)");
    for (i, c) in ["r", "g", "b"].iter().enumerate() {
        let v = norm(apply_matrix_3x3_f32(&to_panel, &[m2020[i * 3], m2020[i * 3 + 1], m2020[i * 3 + 2]]));
        let _ = writeln!(r, "    2020 {c}:   {:+.4} {:+.4} {:+.4}", v[0], v[1], v[2]);
    }

    // Additivity.
    let rgb: Spectrum = { let (a, b, c) = (full("r"), full("g"), full("b")); Spectrum { start: a.start, step: a.step, v: (0..a.v.len()).map(|i| a.v[i] + b.v[i] + c.v[i]).collect() } };
    let (k, res) = full("w").fit(&rgb);
    let _ = writeln!(r, "\nADDITIVITY  white = {k:.4} × (R + G + B) spectrally (shape residual {:.2}%); photopic white / Σ photopic = {:.4}", res * 100.0, wl / photopic(rgb.lms()));

    // Response curves.
    let codes: Vec<u16> = [1, 2, 4].into_iter().chain((8..=248).step_by(8)).chain([255]).collect();
    let _ = writeln!(r, "\nRESPONSE (spectral scale k vs full drive; shape residual %) — the pixel EOTF, observer-free");
    let _ = writeln!(r, "  code      R k     res%     G k     res%     B k     res%     W k     res%");
    let mut eotf = String::from("code,r,g,b,w\n");
    let mut curves: Vec<Vec<(u16, f64)>> = vec![Vec::new(); 4];
    for &code in &codes {
        let mut line = format!("  {code:4}");
        let mut e = format!("{code}");
        for (i, c) in ["r", "g", "b", "w"].iter().enumerate() {
            let (k, res) = net(&format!("ramp_{c}_{code:03}")).fit(&full(c));
            curves[i].push((code, k));
            let _ = write!(line, "  {k:8.5} {:6.2}", res * 100.0);
            let _ = write!(e, ",{k:.6}");
        }
        let _ = writeln!(r, "{line}");
        eotf.push_str(&e);
        eotf.push('\n');
    }
    let _ = writeln!(r, "\n  power-law fit k = (code/255)^γ over codes 16–255:");
    for (i, c) in ["r", "g", "b", "w"].iter().enumerate() {
        let pts: Vec<(f64, f64)> = curves[i].iter().filter(|(code, k)| *code >= 16 && *k > 0.0).map(|&(code, k)| ((code as f64 / 255.0).ln(), k.ln())).collect();
        let g = pts.iter().map(|(x, y)| x * y).sum::<f64>() / pts.iter().map(|(x, _)| x * x).sum::<f64>();
        let worst = curves[i].iter().filter(|(code, _)| *code >= 16).map(|&(code, k)| (k / (code as f64 / 255.0).powf(g) - 1.0).abs()).fold(0.0, f64::max);
        let _ = writeln!(r, "    {c}: γ = {g:.3}  (worst deviation from the fitted law {:.1}%)", worst * 100.0);
    }

    // Window size.
    let _ = writeln!(r, "\nWINDOW SIZE (white, photopic relative to the 240 px window; < 1 = loading. 120 px under-fills the i1Pro's collection area and is not a panel effect)");
    for b in [255u8, 127] {
        let base = photopic(net(&format!("abl_b{b:03}_w240")).lms());
        let mut line = format!("  brightness {b:3}:");
        for n in ["full", "w360", "w240", "w120"] {
            let _ = write!(line, "  {n} {:.4}", photopic(net(&format!("abl_b{b:03}_{n}")).lms()) / base);
        }
        let _ = writeln!(r, "{line}");
    }

    // Brightness register separability.
    let _ = writeln!(r, "\nBRIGHTNESS REGISTER (spectral scale vs the same patch at 255; equal across channels = separable)");
    for b in [191u8, 127, 63, 31, 15] {
        let mut line = format!("  {b:3}:");
        let mut ks = Vec::new();
        for n in ["r", "g", "b", "w", "w128"] {
            let reference = if n == "w128" { net("ramp_w_128") } else { full(n) };
            let (k, res) = net(&format!("bright_b{b:03}_{n}")).fit(&reference);
            ks.push(k);
            let _ = write!(line, "  {n} {k:.5} ({:.1}%)", res * 100.0);
        }
        let spread = ks.iter().cloned().fold(f64::MIN, f64::max) / ks.iter().cloned().fold(f64::MAX, f64::min) - 1.0;
        let _ = writeln!(r, "{line}   spread {:.1}%   γ_b {:.3}", spread * 100.0, ks[3].ln() / ((b as f64 + 1.0) / 256.0).ln());
    }

    // RGB565.
    let _ = writeln!(r, "\nRGB565 (k vs the 24-bit full-drive spectrum, against the 24-bit ramp at the same code)");
    for (n, c, ci, code) in [("r255", "r", 0, 255u16), ("g255", "g", 1, 255), ("b255", "b", 2, 255), ("w255", "w", 3, 255), ("r128", "r", 0, 128), ("g128", "g", 1, 128), ("b128", "b", 2, 128), ("w128", "w", 3, 128)] {
        let (k565, _) = net(&format!("rgb565_{n}")).fit(&full(c));
        let k888 = curves[ci].iter().find(|(cc, _)| *cc == code).unwrap().1;
        let _ = writeln!(r, "  {n}: 565 {k565:.5}   888 {k888:.5}   ratio {:.4}", k565 / k888);
    }

    // Toe: photopic of each low code against the power law at its brightness level.
    let law = |code: u16| (code as f64 / 255.0).powf(2.28);
    let mut toe_rows: Vec<(u8, Vec<(u16, f64, f64)>)> = Vec::new(); // (brightness, [(code, predicted photopic, ratio to law)])
    let mut noise = 0.0f64; // largest scatter of a single black reading about the floor, photopic
    if let Some(td) = &toe_dir {
        let blacks: Vec<Spectrum> = (0..3).map(|i| sp(td, &format!("toe_black_{i}"))).collect();
        let floor = Spectrum::mean(&blacks);
        noise = blacks.iter().map(|b| photopic(b.minus(&floor).lms()).abs()).fold(noise, f64::max);
        let scale = wl; // full-scale white at brightness 255 (240 px window)
        toe_rows.push((255, (1..=16).map(|c| { let v = photopic(sp(td, &format!("toe_w_{c:03}")).minus(&floor).lms()); (c, scale * law(c), v / (scale * law(c))) }).collect()));
        if let Some(tb) = &toeb_dir {
            let blacks: Vec<Spectrum> = (0..3).map(|i| sp(tb, &format!("toeb_black_{i}"))).collect();
            let floor = Spectrum::mean(&blacks);
            noise = blacks.iter().map(|b| photopic(b.minus(&floor).lms()).abs()).fold(noise, f64::max);
            for b in [127u8, 63, 31] {
                let v = |c: u16| photopic(sp(tb, &format!("toeb_b{b:03}_w_{c:03}")).minus(&floor).lms());
                let scale = [32u16, 64, 128].iter().map(|&c| v(c) / law(c)).sum::<f64>() / 3.0;
                toe_rows.push((b, (1..=16).chain([20, 24, 32]).map(|c| (c, scale * law(c), v(c) / (scale * law(c)))).collect()));
            }
        }
        let _ = writeln!(r, "\nTOE (white, full field, averaged readings; ratio = photopic / (that level's full scale × (code/255)^2.28); '—' = the law predicts less than 3× the black readings' scatter, so the reading can't tell)");
        let header: String = toe_rows.iter().map(|(b, _)| format!("   b{b:03} ratio")).collect();
        let _ = writeln!(r, "  code{header}");
        let all_codes: Vec<u16> = (1..=16).chain([20, 24, 32]).collect();
        for c in all_codes {
            let mut line = format!("  {c:4}");
            for (_, rows) in &toe_rows {
                match rows.iter().find(|(cc, _, _)| *cc == c) {
                    Some(&(_, predicted, ratio)) if predicted > 3.0 * noise => { let _ = write!(line, "   {ratio:10.2}"); }
                    Some(_) => { let _ = write!(line, "   {:>10}", "—"); }
                    None => { let _ = write!(line, "   {:>10}", ""); }
                }
            }
            let _ = writeln!(r, "{line}");
        }
        let _ = writeln!(r, "  The cutoff is fixed in absolute light, not code: black to code ~5 at 255, ~8 at 127, ~16 at 63, everything to 32 at 31; indexed by the light the law predicts, the shortfall matches across levels. Model: output = toe(s_b · (code/255)^2.28), one toe function of absolute level — black below ~1/15 000 of full white at 255, on the law above ~1/1 300.");
    }

    // 1 nm spectra of the full-drive primaries and white over the cone table's range.
    let t = &LMS_2000_10DEG_1NM;
    let mut prim = String::from("nm,r,g,b,w\n");
    let (fr, fg, fb, fw) = (full("r"), full("g"), full("b"), full("w"));
    for i in 0..t.num_samples() {
        let nm = t.wavelength_at(i) as f64;
        let _ = writeln!(prim, "{nm},{:.6},{:.6},{:.6},{:.6}", fr.at(nm), fg.at(nm), fb.at(nm), fw.at(nm));
    }
    let _ = writeln!(r, "\nSpectra: mW/(m²·sr·nm) as ArgyllCMS writes them, measured 370–730 nm (zero outside); primaries_1nm.csv covers the cone table's 390–830 nm.");
    fs::write(dir.join("report.txt"), &r).unwrap();
    fs::write(dir.join("eotf.csv"), eotf).unwrap();
    fs::write(dir.join("primaries_1nm.csv"), prim).unwrap();
    print!("{r}");
}
