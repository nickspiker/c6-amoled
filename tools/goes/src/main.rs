//! The satellite face's PC side: keeps the c6-amoled board's flash ring holding the newest GOES-18 frames and playing them.
//!
//! Mapping: band 9 → red, 7 → green, 6 → blue, each linear in absolute radiance between that band's black and white, drive = (L − black) / (white − black), clipped independently; the panel model applies the gamma afterwards. Both ends ratchet on robust per-frame statistics over the disk: white only ever rises, to a frame's 99.99th percentile when that beats it; black only ever falls, to a frame's 0.01th percentile. Whites started from the June 2026 solstice sample (blue at its p99.9, green at 330 K, red at 260 K).
//!
//! Each new frame is fetched (only the chunks under the disk), reprojected, ratcheted, dithered to RGB565 against the panel model, and written into the one ring slot holding the oldest frame over Wi-Fi; the board checks it against its BLAKE3. State lives in the repo (git-ignored): state/white.json (ends and every move), state/ring.json (slots), cache/frames (radiances while the ring holds the frame).
//!
//! Usage:
//!   goes loop [--frames 46] [--fps 15]    keep the board's ring current, polling every minute
//!   goes font [px]                       pack the readout glyphs (default 84 px) and upload them
//!   goes frame YYYYmmdd-HHMM [--encode]  fetch and reproject one frame into the cache, and report its bands (and the BLAKE3 of its encoding)
//!   goes coast [--preview]               build the coastline overlay and upload it (--preview: the newest cached frame with each phase, to /tmp/goes-coast-preview)

mod abi;
mod board;
mod coast;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use abi::{Cal, N};
use board::Board;
use measure::dither::{Levels, TILE, blue_noise};
use measure::model::PanelModel;
use serde_json::{Value, json};

/// Band per panel channel: red, green, blue.
const MAP: [u8; 3] = [9, 7, 6];
const CENTRE: (f64, f64) = (47.6, -124.5);
const SCALE_KM: f64 = 3.9;
const TOP: f64 = 99.99;
const BOTTOM: f64 = 0.01;
const HOLD_MS: u32 = 2000;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn tag(t: i64) -> String {
    chrono::DateTime::from_timestamp(t, 0).unwrap().format("%Y%m%d-%H%M").to_string()
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn stamp() -> String {
    chrono::Utc::now().format("%Y-%m-%d %H:%M UTC").to_string()
}

/// The disk: pixels whose centres lie within 233 px of the panel's centre (the board's test, in its doubled-integer form).
fn inside(x: usize, y: usize) -> bool {
    let (dx, dy) = (2 * x as i64 + 1 - N as i64, 2 * y as i64 + 1 - N as i64);
    dx * dx + dy * dy <= (N * N) as i64
}

/// One frame's three bands (in MAP order) and calibrations.
struct Bands {
    rad: [Vec<f32>; 3],
    cal: [Cal; 3],
}

/// The newest `n` full-disk frames that have all three bands: (scan start, key per MAP band), oldest first.
fn frame_list(n: usize) -> Result<Vec<(i64, [String; 3])>, String> {
    let mut found: BTreeMap<i64, [Option<String>; 3]> = BTreeMap::new();
    for back in 0..(n / 6 + 3) as i64 {
        for key in abi::list_keys(&abi::hour_prefix(now() - back * 3600))? {
            if let Some((band, start)) = abi::parse_key(&key) {
                if let Some(c) = MAP.iter().position(|&b| b == band) {
                    found.entry(start).or_default()[c] = Some(key);
                }
            }
        }
    }
    let complete: Vec<(i64, [String; 3])> = found.into_iter().filter_map(|(t, k)| Some((t, [k[0].clone()?, k[1].clone()?, k[2].clone()?]))).collect();
    Ok(complete[complete.len().saturating_sub(n)..].to_vec())
}

fn frames_dir() -> PathBuf {
    repo().join("cache/frames")
}

/// A frame's bands from the cache, or fetched (the three in parallel) and cached.
fn load(t: i64, keys: &[String; 3]) -> Result<Bands, String> {
    let (bin, meta) = (frames_dir().join(format!("{}.f32", tag(t))), frames_dir().join(format!("{}.json", tag(t))));
    if let (Ok(bytes), Ok(m)) = (fs::read(&bin), fs::read_to_string(&meta)) {
        let floats: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let m: Value = serde_json::from_str(&m).map_err(|e| e.to_string())?;
        let cal = [0, 1, 2].map(|c| cal_from_json(&m[c]));
        if floats.len() == 3 * N * N {
            return Ok(Bands { rad: [0, 1, 2].map(|c| floats[c * N * N..(c + 1) * N * N].to_vec()), cal });
        }
    }
    let results: Vec<Result<(Vec<f32>, Cal), String>> = std::thread::scope(|s| {
        let handles: Vec<_> = keys.iter().map(|k| s.spawn(move || abi::read_band(k, CENTRE.0, CENTRE.1, SCALE_KM))).collect();
        handles.into_iter().map(|h| h.join().unwrap_or(Err("band thread panicked".into()))).collect()
    });
    let mut rad = Vec::new();
    let mut cal = Vec::new();
    for r in results {
        let (v, c) = r?;
        rad.push(v);
        cal.push(c);
    }
    fs::create_dir_all(frames_dir()).map_err(|e| e.to_string())?;
    fs::write(&bin, rad.iter().flatten().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).map_err(|e| e.to_string())?;
    fs::write(&meta, serde_json::to_string(&cal.iter().map(cal_to_json).collect::<Vec<_>>()).unwrap()).map_err(|e| e.to_string())?;
    Ok(Bands { rad: rad.try_into().unwrap(), cal: cal.try_into().unwrap() })
}

fn cal_to_json(c: &Cal) -> Value {
    json!({"band": c.band, "wavelength": c.wavelength, "unit": c.unit, "kappa0": c.kappa0, "planck": c.planck})
}

fn cal_from_json(v: &Value) -> Cal {
    Cal {
        band: v["band"].as_u64().unwrap_or(0) as u8,
        wavelength: v["wavelength"].as_f64().unwrap_or(0.0),
        unit: v["unit"].as_str().unwrap_or("").into(),
        kappa0: v["kappa0"].as_f64(),
        planck: v["planck"].as_array().map(|a| [0, 1, 2, 3].map(|i| a[i].as_f64().unwrap_or(f64::NAN))),
    }
}

/// The q-th percentile of `v` (linear between order statistics, as numpy's default).
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let pos = q / 100.0 * (sorted.len() - 1) as f64;
    let (i, f) = (pos.floor() as usize, pos - pos.floor());
    sorted[i] + (sorted[(i + 1).min(sorted.len() - 1)] - sorted[i]) * f
}

/// The ratchet state (state/white.json): each band's black and white, and every move.
struct State {
    v: Value,
    path: PathBuf,
}

impl State {
    fn load(cal: &[Cal; 3]) -> State {
        let path = repo().join("state/white.json");
        let v = fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_else(|| {
            // Fresh: whites from the solstice sample; black from the first frame.
            json!({"white": {"6": 9.5136, "7": cal[1].planck_radiance(330.0), "9": cal[0].planck_radiance(260.0)}, "black": {}, "moves": []})
        });
        State { v, path }
    }

    fn save(&self) {
        fs::create_dir_all(self.path.parent().unwrap()).ok();
        fs::write(&self.path, serde_json::to_string_pretty(&self.v).unwrap()).ok();
    }

    /// Move the band's white up / black down if this frame's robust extremes pass them; returns (black, white) after.
    fn ratchet(&mut self, t: i64, band: u8, pixels: &[f64]) -> (f64, f64) {
        let mut sorted: Vec<f64> = pixels.iter().copied().filter(|v| v.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);
        let key = band.to_string();
        if !sorted.is_empty() {
            let (lo, hi) = (percentile(&sorted, BOTTOM), percentile(&sorted, TOP));
            for (end, value, beyond) in [("white", hi, hi > self.v["white"][&key].as_f64().unwrap_or(f64::MIN)), ("black", lo, lo < self.v["black"][&key].as_f64().unwrap_or(f64::MAX))] {
                if beyond {
                    let from = self.v[end][&key].clone();
                    self.v["moves"].as_array_mut().unwrap().push(json!({"frame": chrono::DateTime::from_timestamp(t, 0).unwrap().to_rfc3339(), "band": band, "end": end, "from": from, "to": value}));
                    self.v[end][&key] = json!(value);
                }
            }
        }
        (self.v["black"][&key].as_f64().unwrap_or(0.0), self.v["white"][&key].as_f64().unwrap_or(1.0))
    }
}

/// Ratchet the ends on this frame, then dither its linear drive to RGB565 against the panel model: the disk's pixels, row by row, high byte first (the board's stored-frame format).
fn encode(t: i64, b: &Bands, state: &mut State, levels: &[Levels; 3], mask: &[f64]) -> Vec<u8> {
    let mut ends = [(0.0, 1.0); 3];
    for c in 0..3 {
        let disk: Vec<f64> = (0..N * N).filter(|&i| inside(i % N, i / N)).map(|i| b.rad[c][i] as f64).collect();
        let before = state.v["moves"].as_array().map_or(0, |m| m.len());
        ends[c] = state.ratchet(t, MAP[c], &disk);
        for m in &state.v["moves"].as_array().unwrap()[before..] {
            let to = m["to"].as_f64().unwrap_or(0.0);
            println!("  band {}: {} moved {} → {to:.4} ({})", MAP[c], m["end"].as_str().unwrap_or(""), m["from"], b.cal[c].equivalent(to));
        }
    }
    // The same blue-noise map shifted per channel, so the channels' dots don't coincide.
    let shift = [(0, 0), (23, 41), (47, 13)];
    let mut out = Vec::with_capacity(341_160);
    for y in 0..N {
        for x in 0..N {
            if !inside(x, y) {
                continue;
            }
            let i = y * N + x;
            let code = [0, 1, 2].map(|c| {
                let (black, white) = ends[c];
                let drive = ((b.rad[c][i] as f64 - black) / (white - black)).clamp(0.0, 1.0);
                let drive = if drive.is_nan() { 0.0 } else { drive };
                let th = mask[(y + shift[c].1) % TILE * TILE + (x + shift[c].0) % TILE];
                levels[c].pick(drive, th) as u16
            });
            let v = (code[0] >> 3) << 11 | (code[1] >> 2) << 5 | code[2] >> 3;
            out.extend_from_slice(&v.to_be_bytes());
        }
    }
    out
}

/// The board's ring of slots (state/ring.json).
struct Ring {
    v: Value,
    path: PathBuf,
}

impl Ring {
    fn load(count: usize) -> Ring {
        let path = repo().join("state/ring.json");
        let v = fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok()).filter(|v| v["count"].as_u64() == Some(count as u64)).unwrap_or_else(|| json!({"count": count, "next": 0, "slots": {}, "hashes": {}}));
        Ring { v, path }
    }

    fn save(&self) {
        fs::create_dir_all(self.path.parent().unwrap()).ok();
        fs::write(&self.path, serde_json::to_string_pretty(&self.v).unwrap()).ok();
    }

    fn tags(&self) -> Vec<String> {
        self.v["slots"].as_object().map(|m| m.values().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default()
    }

    fn next(&self) -> usize {
        self.v["next"].as_u64().unwrap_or(0) as usize
    }
}

fn play(board: &mut Board, ring: &Ring, count: usize, period: u32) -> Result<(), String> {
    let stored = ring.tags().len();
    if stored > 0 {
        let (start, n) = if stored == count { (ring.next(), count) } else { (0, stored) };
        board.send(&format!("p {start} {n} {period} {HOLD_MS}"))?;
        println!("playing {n} frames at {:.1} fps, newest held {} s", 1000.0 / period as f64, HOLD_MS / 1000);
    }
    Ok(())
}

fn run(count: usize, fps: f64) {
    let model = PanelModel::load(&repo().join("measurements/i1pro-2026-10-11"), &repo().join("measurements/i1pro-2026-10-11-toe"));
    let levels = [0, 1, 2].map(|c| Levels::new(&model, c));
    let mask = blue_noise();
    let period = (1000.0 / fps).round() as u32;
    let mut ring = Ring::load(count);
    let mut started = false;
    loop {
        let attempt = (|| -> Result<(), String> {
            if !started {
                // The board resumes its loop by itself after a power cut; this reasserts the ring as recorded here.
                play(&mut Board::connect()?, &ring, count, period)?;
                started = true;
            }
            let newest = ring.tags().into_iter().max().unwrap_or_default();
            let new: Vec<(i64, [String; 3])> = frame_list(count)?.into_iter().filter(|(t, _)| tag(*t) > newest).collect();
            if new.is_empty() {
                return Ok(());
            }
            println!("fetching {} frames…", new.len());
            let frames: Vec<(i64, Bands)> = new.iter().map(|(t, k)| load(*t, k).map(|b| (*t, b))).collect::<Result<_, _>>()?;
            let mut state = State::load(&frames[0].1.cal);
            // Connect only for the update: the board serves one client at a time.
            let mut board = Board::connect()?;
            for (t, bands) in &frames {
                let slot = ring.next();
                let started_at = std::time::Instant::now();
                let data = encode(*t, bands, &mut state, &levels, &mask);
                state.save();
                board.store(slot, &data)?;
                // The ring only records a slot once the board has confirmed it, so a failed store is simply redone next poll.
                ring.v["slots"][slot.to_string()] = json!(tag(*t));
                ring.v["hashes"][slot.to_string()] = json!(blake3::hash(&data).to_hex().to_string());
                ring.v["next"] = json!((slot + 1) % count);
                ring.save();
                println!("{} → slot {slot} ({:.1} s)", chrono::DateTime::from_timestamp(*t, 0).unwrap().format("%Y-%m-%d %H:%M UTC"), started_at.elapsed().as_secs_f64());
            }
            // Keep only the cached radiances the ring still holds.
            if let Some(oldest) = ring.tags().into_iter().min() {
                for entry in fs::read_dir(frames_dir()).into_iter().flatten().flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.split('.').next().is_some_and(|stem| stem < oldest.as_str()) {
                        fs::remove_file(entry.path()).ok();
                    }
                }
            }
            play(&mut board, &ring, count, period)
        })();
        if let Err(e) = attempt {
            // A network hiccup, an S3 error, a board that stopped answering: log it, keep the ring as it is, and try again next poll.
            println!("{}: poll failed ({e}); retrying in a minute", stamp());
        }
        std::thread::sleep(Duration::from_secs(60));
    }
}

/// Pack the readout glyphs (Oxanium with the dozenal digits) and upload them; the board checks the BLAKE3 and keeps them in flash.
fn font(px: f32) -> Result<(), String> {
    let model = PanelModel::load(&repo().join("measurements/i1pro-2026-10-11"), &repo().join("measurements/i1pro-2026-10-11-toe"));
    let ttf = fs::read(repo().join("assets/font/Oxanium-Regular+glyphs.ttf")).map_err(|e| e.to_string())?;
    let (blob, _) = measure::fontpack::pack(&ttf, px, &measure::fontpack::dozenal(), &model);
    let mut board = Board::connect()?;
    board.write(format!("F {} {}\n", blob.len(), blake3::hash(&blob).to_hex()).as_bytes())?;
    board.write(&blob)?;
    println!("{}", board.reply("F", 30)?);
    println!("{}", board.send("info")?);
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("loop") => {
            let (mut count, mut fps) = (46, 15.0);
            while let Some(a) = args.next() {
                match a.as_str() {
                    "--frames" => count = args.next().and_then(|v| v.parse().ok()).expect("--frames N"),
                    "--fps" => fps = args.next().and_then(|v| v.parse().ok()).expect("--fps F"),
                    other => panic!("unknown option {other}"),
                }
            }
            run(count, fps);
        }
        Some("frame") => {
            let want = args.next().expect("goes frame YYYYmmdd-HHMM");
            let (t, keys) = frame_list(200).expect("listing").into_iter().find(|(t, _)| tag(*t) == want).expect("no such frame in the last day or so");
            let started = std::time::Instant::now();
            let b = load(t, &keys).expect("load");
            for c in 0..3 {
                let v: Vec<f32> = b.rad[c].iter().copied().filter(|v| v.is_finite()).collect();
                let (lo, hi) = (v.iter().copied().fold(f32::MAX, f32::min), v.iter().copied().fold(f32::MIN, f32::max));
                println!("band {:2} ({:.2} µm): {} valid, {lo:.4}–{hi:.4} {} ({} – {})", b.cal[c].band, b.cal[c].wavelength, v.len(), b.cal[c].unit, b.cal[c].equivalent(lo as f64), b.cal[c].equivalent(hi as f64));
            }
            println!("{:.1} s", started.elapsed().as_secs_f64());
            if args.next().as_deref() == Some("--encode") {
                // Encode against a copy of the ratchet state (nothing saved), for comparing encoders.
                let model = PanelModel::load(&repo().join("measurements/i1pro-2026-10-11"), &repo().join("measurements/i1pro-2026-10-11-toe"));
                let levels = [0, 1, 2].map(|c| Levels::new(&model, c));
                let mut state = State::load(&b.cal);
                state.path = std::env::temp_dir().join("goes-state-scratch.json");
                let data = encode(t, &b, &mut state, &levels, &blue_noise());
                println!("encoded {} bytes, blake3 {}", data.len(), blake3::hash(&data).to_hex());
                std::fs::write(std::env::temp_dir().join("goes-encoded.bin"), &data).ok();
            }
        }
        Some("font") => font(args.next().and_then(|v| v.parse().ok()).unwrap_or(84.0)).expect("font upload"),
        Some("coast") => coast_overlay(args.next().as_deref() == Some("--preview")).expect("coast"),
        _ => eprintln!("usage: goes loop [--frames 46] [--fps 15] | goes font [px] | goes coast [--preview]"),
    }
}

/// Build the coastline overlay and upload it (`C len hash`, kept in the board's flash), or with `preview`, write the newest cached frame plain and with each phase of the flicker to /tmp/goes-coast-preview.
fn coast_overlay(preview: bool) -> Result<(), String> {
    let cov = coast::mask()?;
    let blob = coast::pack(&cov);
    println!("coastline: {} pixels, {} bytes packed", cov.iter().filter(|&&c| c != 0).count(), blob.len());
    if !preview {
        let mut board = Board::connect()?;
        board.write(format!("C {} {}\n", blob.len(), blake3::hash(&blob).to_hex()).as_bytes())?;
        board.write(&blob)?;
        println!("{}", board.reply("C", 30)?);
        return Ok(());
    }
    // The newest cached frame, encoded as the loop would encode it (against a scratch copy of the ratchet state).
    let newest = fs::read_dir(frames_dir()).map_err(|e| e.to_string())?.filter_map(|e| e.ok()?.file_name().into_string().ok()?.strip_suffix(".json").map(String::from)).max().ok_or("no cached frame")?;
    let t = chrono::NaiveDateTime::parse_from_str(&newest, "%Y%m%d-%H%M").map_err(|e| e.to_string())?.and_utc().timestamp();
    let b = load(t, &[String::new(), String::new(), String::new()])?;
    let model = PanelModel::load(&repo().join("measurements/i1pro-2026-10-11"), &repo().join("measurements/i1pro-2026-10-11-toe"));
    let levels = [0, 1, 2].map(|c| Levels::new(&model, c));
    let mut state = State::load(&b.cal);
    state.path = std::env::temp_dir().join("goes-state-scratch.json");
    let data = encode(t, &b, &mut state, &levels, &blue_noise());
    let mut frame = vec![0u16; N * N];
    let mut at = 0;
    for (i, p) in frame.iter_mut().enumerate() {
        if inside(i % N, i / N) {
            *p = u16::from_be_bytes([data[at], data[at + 1]]);
            at += 2;
        }
    }
    let dir = Path::new("/tmp/goes-coast-preview");
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    for (name, sign) in [("plain", 0), ("plus", 1), ("minus", -1)] {
        let mut rgb = Vec::with_capacity(N * N * 3);
        for (i, &p) in frame.iter().enumerate() {
            let p = if sign == 0 || cov[i] == 0 { p } else { coast::nudge(p, cov[i], sign) };
            let (r, g, b) = ((p >> 11) as u8, (p >> 5 & 63) as u8, (p & 31) as u8);
            rgb.extend_from_slice(&[r << 3 | r >> 2, g << 2 | g >> 4, b << 3 | b >> 2]);
        }
        let path = dir.join(format!("{newest}-{name}.png"));
        let mut e = png::Encoder::new(std::io::BufWriter::new(fs::File::create(&path).map_err(|e| e.to_string())?), N as u32, N as u32);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().map_err(|e| e.to_string())?.write_image_data(&rgb).map_err(|e| e.to_string())?;
        println!("{}", path.display());
    }
    Ok(())
}
