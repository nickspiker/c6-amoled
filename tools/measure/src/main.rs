//! Colorimetry sweep for the c6-amoled panel.
//!
//! Drives the firmware's patch server over USB serial (`f`/`w`/`b`/`m` commands, each answered `ok`) and, for every patch, captures the full emissive spectrum (370–730 nm, 3.3 nm) from an i1Pro on the panel via ArgyllCMS `spotread -O <file>.sp`. A CSV row per patch records the command, the panel state and VSF photopic luminance + AGB ratios computed from the spectrum (spotread's own colorimetry is not used).
//!
//! Usage: `measure <out-dir> [port] [plan]` — plan `lcd` for the ESP32-C3 1.47" LCD board (see `plan_lcd`), otherwise for the AMOLED: `full` (default), `toe` (white codes 0–16, full field, averaged readings, for the dark end below the full sweep's noise floor) or `toe-brightness` (the same dark end at brightness 127/63/31). The instrument must already be calibrated (`spotread -e -H -O` on its tile) and placed on the panel's centre.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::{Duration, Instant};

use measure::link::{Link, capture};
use measure::spectra::{agb, photopic, read_sp};

/// One measurement: a name for the files, the patch command, and the brightness + format it runs under.
struct Patch {
    name: String,
    cmd: String,
    brightness: u8,
    bits: u8,
}

fn patch(name: impl Into<String>, cmd: impl Into<String>, brightness: u8, bits: u8) -> Patch {
    Patch { name: name.into(), cmd: cmd.into(), brightness, bits }
}

/// The sweep. Everything uses a 240 px window centred on black unless it is testing window size.
fn plan() -> Vec<Patch> {
    let mut p = vec![patch("black", "w 240 0 0 0", 255, 24)];
    let codes: Vec<u16> = [1, 2, 4].into_iter().chain((8..=248).step_by(8)).chain([255]).collect();
    for (ch, rgb) in [("r", (1, 0, 0)), ("g", (0, 1, 0)), ("b", (0, 0, 1)), ("w", (1, 1, 1))] {
        for &c in &codes {
            p.push(patch(format!("ramp_{ch}_{c:03}"), format!("w 240 {} {} {}", c * rgb.0, c * rgb.1, c * rgb.2), 255, 24));
        }
    }
    // Auto current/brightness limiting: the same white as a full field and as shrinking windows, at two brightness codes.
    for b in [255u8, 127] {
        p.push(patch(format!("abl_b{b:03}_full"), "f 255 255 255", b, 24));
        for s in [360, 240, 120] {
            p.push(patch(format!("abl_b{b:03}_w{s}"), format!("w {s} 255 255 255"), b, 24));
        }
    }
    // Does the brightness register scale every channel alike (separable), or shift colour as it dims?
    for b in [191u8, 127, 63, 31, 15] {
        for (n, cmd) in [("r", "w 240 255 0 0"), ("g", "w 240 0 255 0"), ("b", "w 240 0 0 255"), ("w", "w 240 255 255 255"), ("w128", "w 240 128 128 128")] {
            p.push(patch(format!("bright_b{b:03}_{n}"), cmd, b, 24));
        }
    }
    // How RGB565 codes land on the panel's 24-bit response.
    for (n, cmd) in [("r255", "w 240 255 0 0"), ("g255", "w 240 0 255 0"), ("b255", "w 240 0 0 255"), ("w255", "w 240 255 255 255"), ("r128", "w 240 128 0 0"), ("g128", "w 240 0 128 0"), ("b128", "w 240 0 0 128"), ("w128", "w 240 128 128 128")] {
        p.push(patch(format!("rgb565_{n}"), cmd, 255, 16));
    }
    p
}

/// The dark end: white codes 0–16 as a full field (most light into the aperture; area loading is negligible this dim), three black readings for the floor.
fn plan_toe() -> Vec<Patch> {
    let mut p: Vec<Patch> = (0..3).map(|i| patch(format!("toe_black_{i}"), "f 0 0 0", 255, 24)).collect();
    for c in 1..=16u16 {
        p.push(patch(format!("toe_w_{c:03}"), format!("f {c} {c} {c}"), 255, 24));
    }
    p
}

/// Does the near-black cutoff move with the brightness register? At 127, 63 and 31: white codes 1–16 plus anchors 20–255 (to fit that level's own curve), full field, averaged.
fn plan_toe_brightness() -> Vec<Patch> {
    let mut p: Vec<Patch> = (0..3).map(|i| patch(format!("toeb_black_{i}"), "f 0 0 0", 255, 24)).collect();
    for b in [127u8, 63, 31] {
        for c in (1..=16u16).chain([20, 24, 32, 64, 128, 255]) {
            p.push(patch(format!("toeb_b{b:03}_w_{c:03}"), format!("f {c} {c} {c}"), b, 24));
        }
    }
    p
}

/// The ESP32-C3 1.47" LCD (c3-forecast's patch commands): the panel is tiny, so every patch fills it whole; RGB565 only, so no format switching; `b` is the backlight's PWM duty. Ramps per channel and white, black (the backlight's leakage), and white and black across backlight duties.
fn plan_lcd() -> Vec<Patch> {
    let mut p = vec![patch("black", "f 0 0 0", 255, 16)];
    let codes: Vec<u16> = [4].into_iter().chain((8..=248).step_by(8)).chain([255]).collect();
    for (ch, rgb) in [("r", (1, 0, 0)), ("g", (0, 1, 0)), ("b", (0, 0, 1)), ("w", (1, 1, 1))] {
        for &c in &codes {
            p.push(patch(format!("ramp_{ch}_{c:03}"), format!("f {} {} {}", c * rgb.0, c * rgb.1, c * rgb.2), 255, 16));
        }
    }
    for b in [191u8, 127, 63, 31, 15, 7, 3, 1] {
        p.push(patch(format!("backlight_b{b:03}_w"), "f 255 255 255", b, 16));
        p.push(patch(format!("backlight_b{b:03}_black"), "f 0 0 0", b, 16));
    }
    p
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().expect("usage: measure <out-dir> [port]"));
    let port = args.next().unwrap_or_else(|| "/dev/ttyACM0".into());
    let which = args.next();
    let lcd = which.as_deref() == Some("lcd");
    let (plan, flags, warm): (Vec<Patch>, &[&str], u64) = match which.as_deref() {
        None | Some("full") => (plan(), &["-H"], 120),
        Some("toe") => (plan_toe(), &["-H", "-Y", "aaa"], 30),
        Some("toe-brightness") => (plan_toe_brightness(), &["-H", "-Y", "aaa"], 30),
        Some("lcd") => (plan_lcd(), &["-H"], 120),
        Some(other) => panic!("unknown plan {other} (full | toe | toe-brightness | lcd)"),
    };
    // The LCD board has no format switching, and its patches fill the panel.
    let (on, off) = if lcd { ("f 255 255 255", "f 0 0 0") } else { ("w 240 255 255 255", "w 240 0 0 0") };
    fs::create_dir_all(&out).expect("create out dir");
    let mut panel = Link::open(&port);
    println!("{} patches → {}", plan.len(), out.display());

    if !lcd {
        panel.send("m 24");
    }
    panel.send("b 255");
    panel.send(on);
    println!("warming up {warm} s on white…");
    sleep(Duration::from_secs(warm));

    let mut csv = fs::File::create(out.join("sweep.csv")).expect("create csv");
    writeln!(csv, "name,command,brightness,bits,photopic,alpha,beta,gamma,seconds").unwrap();
    let (mut bits, mut brightness) = (if lcd { 16u8 } else { 24 }, 255u8);
    let start = Instant::now();
    for (i, p) in plan.iter().enumerate() {
        if p.bits != bits {
            panel.send(&format!("m {}", p.bits));
            bits = p.bits;
        }
        if p.brightness != brightness {
            panel.send(&format!("b {}", p.brightness));
            brightness = p.brightness;
        }
        panel.send(&p.cmd);
        sleep(Duration::from_millis(300));
        let t = Instant::now();
        let lms = read_sp(&capture(&out, &p.name, flags)).lms();
        let (v, [a, b, g]) = (photopic(lms), agb(lms));
        let secs = t.elapsed().as_secs_f64();
        writeln!(csv, "{},{},{},{},{v},{a},{b},{g},{secs:.1}", p.name, p.cmd, p.brightness, p.bits).unwrap();
        csv.flush().unwrap();
        println!("[{:3}/{}] {:22} photopic {:12.4}  ({secs:.1} s, {:.0} min elapsed)", i + 1, plan.len(), p.name, v, start.elapsed().as_secs_f64() / 60.0);
    }
    if !lcd {
        panel.send("m 24");
    }
    panel.send("b 255");
    panel.send(off);
    println!("done in {:.1} min", start.elapsed().as_secs_f64() / 60.0);
}
