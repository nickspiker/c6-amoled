//! ColorChecker check in VSF colorimetry: render each BabelColor-average patch (its reflectance under Illuminant E, VSF's white) through the panel model, read it back with the i1Pro, and compare cone responses — AGB ratios for chromaticity, VSF photopic for level, per-cone error.
//!
//! Usage: `cc <out-dir> [white-fraction=0.5] [port]` — white-fraction places a perfect E white at that share of the brightest E white the panel can make. The i1Pro must be calibrated and on the panel's centre; the run re-measures the primaries first, so a re-placed instrument (different absolute level) doesn't bias the comparison.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

use measure::link::{Link, capture};
use measure::model::PanelModel;
use measure::spectra::{Spectrum, agb, photopic, read_sp};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../measurements");

/// The BabelColor-average reflectance table: (name, spectrum 380–730 nm at 10 nm, held flat to 830 nm so long-wave cones aren't truncated).
fn colorchecker() -> Vec<(String, Spectrum)> {
    let text = fs::read_to_string(Path::new(ROOT).join("ref/colorchecker_babelcolor_avg_spectral.csv")).expect("ColorChecker table");
    let mut lines = text.lines().skip_while(|l| !l.starts_with("No.,"));
    let header: Vec<f64> = lines.next().unwrap().split(',').skip(2).filter_map(|t| t.parse().ok()).collect();
    assert_eq!((header[0], header[header.len() - 1]), (380.0, 730.0), "unexpected wavelength range");
    let mut out = Vec::new();
    for l in lines {
        let f: Vec<&str> = l.split(',').collect();
        if f[0].is_empty() {
            break;
        }
        let mut v: Vec<f64> = f[2..2 + header.len()].iter().map(|t| t.parse().unwrap()).collect();
        let last = *v.last().unwrap();
        v.extend(std::iter::repeat_n(last, 10)); // 740–830 nm
        out.push((format!("{} {}", f[0], f[1]), Spectrum { start: 380.0, step: 10.0, v }));
    }
    assert_eq!(out.len(), 24, "expected 24 patches");
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().expect("usage: cc <out-dir> [white-fraction] [port]"));
    let fraction: f64 = args.next().map(|v| v.parse().expect("white fraction")).unwrap_or(0.5);
    let port = args.next().unwrap_or_else(|| "/dev/ttyACM0".into());
    fs::create_dir_all(&out).unwrap();

    let mut model = PanelModel::load(&Path::new(ROOT).join("i1pro-2026-10-11"), &Path::new(ROOT).join("i1pro-2026-10-11-toe"));
    let mut patches = vec![("0 perfect white (E)".to_string(), Spectrum { start: 380.0, step: 10.0, v: vec![1.0; 46] })];
    patches.extend(colorchecker());

    let mut link = Link::open(&port);
    link.send("m 24");
    link.send("b 255");
    link.send("w 240 255 255 255");
    println!("warming up 60 s…");
    sleep(Duration::from_secs(60));
    link.send("w 240 0 0 0");
    let floor = Spectrum::mean(&(0..3).map(|i| read_sp(&capture(&out, &format!("black_{i}"), &["-H"]))).collect::<Vec<_>>());
    // Re-measure the primaries in this placement so targets and readings share one absolute scale.
    let mut prim = Vec::new();
    for (n, cmd) in [("prim_r", "w 240 255 0 0"), ("prim_g", "w 240 0 255 0"), ("prim_b", "w 240 0 0 255")] {
        link.send(cmd);
        sleep(Duration::from_millis(300));
        prim.push(read_sp(&capture(&out, n, &["-H"])).minus(&floor));
    }
    assert!(prim.iter().all(|p| p.v.iter().cloned().fold(0.0, f64::max) > 1.0), "the primaries barely register — is the i1Pro on the panel?");
    model.rebase([&prim[0], &prim[1], &prim[2]]);
    let s = model.white_scale(fraction);

    let mut report = String::new();
    let _ = writeln!(report, "ColorChecker on the panel — VSF colorimetry (SS2000 10°, Illuminant E), E white at {fraction} of panel max\n");
    let _ = writeln!(report, "{:26} {:>11}  {:>6} {:>6} {:>6}   {:>6} {:>6} {:>6}   {:>7} {:>7} {:>7}  {:>6}   {:>6} {:>6} {:>6}", "patch", "codes", "α tgt", "β tgt", "γ tgt", "α meas", "β meas", "γ meas", "Δα‰", "Δβ‰", "Δγ‰", "V m/t", "L %", "M %", "S %");
    let mut csv = String::from("patch,r,g,b,clipped,alpha_t,beta_t,gamma_t,alpha_m,beta_m,gamma_m,photopic_ratio,l_err,m_err,s_err\n");
    let mut stats: Vec<[f64; 4]> = Vec::new();
    for (i, (name, refl)) in patches.iter().enumerate() {
        let rel = refl.lms();
        let target = rel.map(|v| v * s);
        let (codes, clipped) = model.encode(target);
        link.send(&format!("w 240 {} {} {}", codes[0], codes[1], codes[2]));
        sleep(Duration::from_millis(300));
        let meas = read_sp(&capture(&out, &format!("cc_{i:02}"), &["-H"])).minus(&floor).lms();
        let (at, am) = (agb(target), agb(meas));
        let d: Vec<f64> = (0..3).map(|j| (am[j] - at[j]) * 1000.0).collect();
        let v = photopic(meas) / photopic(target);
        let cone: Vec<f64> = (0..3).map(|j| (meas[j] / target[j] - 1.0) * 100.0).collect();
        let flag = if clipped { " (clipped)" } else { "" };
        let _ = writeln!(report, "{:26} {:3} {:3} {:3}  {:.4} {:.4} {:.4}   {:.4} {:.4} {:.4}   {:+7.2} {:+7.2} {:+7.2}  {v:.4}   {:+6.1} {:+6.1} {:+6.1}{flag}", name, codes[0], codes[1], codes[2], at[0], at[1], at[2], am[0], am[1], am[2], d[0], d[1], d[2], cone[0], cone[1], cone[2]);
        let _ = writeln!(csv, "{name},{},{},{},{clipped},{},{},{},{},{},{},{v},{},{},{}", codes[0], codes[1], codes[2], at[0], at[1], at[2], am[0], am[1], am[2], cone[0], cone[1], cone[2]);
        if !clipped {
            stats.push([d[0].abs(), d[1].abs(), d[2].abs(), (v - 1.0).abs() * 100.0]);
        }
        println!("{name:26} Δα {:+.2}‰ Δβ {:+.2}‰ Δγ {:+.2}‰  photopic ×{v:.4}{flag}", d[0], d[1], d[2]);
    }
    link.send("w 240 0 0 0");
    let n = stats.len() as f64;
    let mean = |j: usize| stats.iter().map(|s| s[j]).sum::<f64>() / n;
    let max = |j: usize| stats.iter().map(|s| s[j]).fold(0.0, f64::max);
    let _ = writeln!(report, "\nIn-gamut patches ({}): mean |Δα| {:.2}‰ (max {:.2}), |Δβ| {:.2}‰ (max {:.2}), |Δγ| {:.2}‰ (max {:.2}); photopic error mean {:.2}% (max {:.2}%)", stats.len(), mean(0), max(0), mean(1), max(1), mean(2), max(2), mean(3), max(3));
    fs::write(out.join("report.txt"), &report).unwrap();
    fs::write(out.join("cc.csv"), csv).unwrap();
    print!("\n{report}");
}
