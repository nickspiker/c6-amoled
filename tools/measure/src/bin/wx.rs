//! Weather face mockup for the round panel: the disc is a sky over the sea, with time running left to right (past 24 h on the left, the next 48 h on the right, now on the centreline) and squeezed toward the rim, where the circle runs out of height anyway.
//!
//! - Sky colour per column from the sun's altitude at that time (day, twilight glow at the horizon, night).
//! - Clouds: a fixed noise field thresholded so the covered share of each column equals the cloud cover.
//! - Rain: streaks whose density follows precipitation, slanted by the wind's east–west component.
//! - Temperature (amber) and dew point (blue) curves; the gap between them is the humidity — where they meet the air is saturated.
//! - Wind on the sea: strokes pointing where the air goes (north up), length by speed, a faint extension to the gust.
//!
//! Host-side and float for now, to iterate on the look; the chip version gets integer tables.
//!
//! Usage: `wx [live|demo] [port|-]` — writes /tmp/wx-preview/wx.png (the panel codes) and pushes the frame to the panel unless the port is `-`.

use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use measure::link::Link;
use measure::model::PanelModel;
use vsf::colour::convert::apply_matrix_3x3_f32;
use vsf::colour::rec2020::constants::REC2020_2VSF_RGB;
use vsf::colour::spectral::constants::VSF_RGB2LMS;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../measurements");
const OUT: &str = "/tmp/wx-preview";
const N: usize = 466;
const R: f64 = 233.0;
const LAT: f64 = 47.5126;
const LON: f64 = -122.5054;
const PAST_H: f64 = 24.0;
const FUTURE_H: f64 = 48.0;
/// Sea surface, as a share of the radius below centre.
const SEA: f64 = -0.58;

/// Hourly series, `t` in unix seconds.
struct Wx {
    t: Vec<f64>,
    temp: Vec<f64>,
    dew: Vec<f64>,
    cloud: Vec<f64>,
    rain: Vec<f64>,
    wind: Vec<f64>,
    gust: Vec<f64>,
    dir: Vec<f64>,
}

impl Wx {
    /// Catmull-Rom through the hourly points, so curves don't kink at every hour.
    fn at(&self, s: &[f64], t: f64) -> f64 {
        let n = self.t.len();
        let i = self.t.partition_point(|&x| x <= t).clamp(1, n - 1);
        let f = ((t - self.t[i - 1]) / (self.t[i] - self.t[i - 1])).clamp(0.0, 1.0);
        let (p0, p1, p2, p3) = (s[i.saturating_sub(2)], s[i - 1], s[i], s[(i + 1).min(n - 1)]);
        let (f2, f3) = (f * f, f * f * f);
        0.5 * (2.0 * p1 + (p2 - p0) * f + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * f3)
    }

    /// Wind direction is circular: interpolate the vector, not the angle.
    fn dir_at(&self, t: f64) -> f64 {
        let i = self.t.partition_point(|&x| x <= t).clamp(1, self.t.len() - 1);
        let f = ((t - self.t[i - 1]) / (self.t[i] - self.t[i - 1])).clamp(0.0, 1.0);
        let (a, b) = (self.dir[i - 1].to_radians(), self.dir[i].to_radians());
        let (x, y) = (a.sin() * (1.0 - f) + b.sin() * f, a.cos() * (1.0 - f) + b.cos() * f);
        x.atan2(y).to_degrees().rem_euclid(360.0)
    }
}

fn fetch() -> Wx {
    let url = format!(
        "https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m,dew_point_2m,cloud_cover,precipitation,wind_speed_10m,wind_gusts_10m,wind_direction_10m&temperature_unit=fahrenheit&wind_speed_unit=mph&timeformat=unixtime&past_days=2&forecast_days=3"
    );
    let out = Command::new("curl").args(["-sf", &url]).output().expect("run curl");
    assert!(out.status.success(), "Open-Meteo fetch failed");
    fs::write(Path::new(OUT).join("data.json"), &out.stdout).unwrap();
    let j: serde_json::Value = serde_json::from_slice(&out.stdout).expect("Open-Meteo JSON");
    let h = &j["hourly"];
    let col = |k: &str| h[k].as_array().unwrap_or_else(|| panic!("no {k}")).iter().map(|v| v.as_f64().unwrap_or(0.0)).collect::<Vec<_>>();
    Wx { t: col("time"), temp: col("temperature_2m"), dew: col("dew_point_2m"), cloud: col("cloud_cover"), rain: col("precipitation"), wind: col("wind_speed_10m"), gust: col("wind_gusts_10m"), dir: col("wind_direction_10m") }
}

/// A made-up front: clear and calm in the past, clouding over, a wet windy southerly blow tomorrow, clearing and backing northerly after.
fn demo(now: f64) -> Wx {
    let t: Vec<f64> = (-30..=54).map(|h| now + h as f64 * 3600.0).collect();
    let hr: Vec<f64> = (-30..=54).map(|h| h as f64).collect();
    let bump = |h: f64, c: f64, w: f64| (-((h - c) / w).powi(2)).exp();
    let diurnal = |h: f64| -(2.0 * PI * (h + 3.0) / 24.0).cos();
    Wx {
        temp: hr.iter().map(|&h| 52.0 + 7.0 * diurnal(h) - 4.0 * bump(h, 20.0, 8.0)).collect(),
        dew: hr.iter().map(|&h| 41.0 + 9.0 * bump(h, 18.0, 12.0) + diurnal(h)).collect(),
        cloud: hr.iter().map(|&h| (100.0 * (bump(h, 18.0, 10.0) + 0.25 * bump(h, -12.0, 4.0))).min(100.0)).collect(),
        rain: hr.iter().map(|&h| 4.0 * bump(h, 17.0, 4.0) + 1.0 * bump(h, 24.0, 2.5)).collect(),
        wind: hr.iter().map(|&h| 4.0 + 18.0 * bump(h, 16.0, 6.0) + 6.0 * bump(h, 36.0, 8.0)).collect(),
        gust: hr.iter().map(|&h| 7.0 + 30.0 * bump(h, 16.0, 6.0) + 9.0 * bump(h, 36.0, 8.0)).collect(),
        dir: hr.iter().map(|&h| if h < 22.0 { 190.0 } else { 190.0 + (h - 22.0).min(10.0) * 17.0 }).collect(),
        t,
    }
}

/// Sun altitude in degrees (Meeus low precision, as the tide widget).
fn sun_alt(unix: f64) -> f64 {
    let d = (unix - 946_728_000.0) / 86_400.0;
    let l = (280.460 + 0.9856474 * d).rem_euclid(360.0);
    let g = (357.528 + 0.9856003 * d).rem_euclid(360.0).to_radians();
    let lam = (l + 1.915 * g.sin() + 0.020 * (2.0 * g).sin()).to_radians();
    let eps = (23.439 - 0.0000004 * d).to_radians();
    let ra = (eps.cos() * lam.sin()).atan2(lam.cos());
    let dec = (eps.sin() * lam.sin()).asin();
    let gmst = (280.46061837 + 360.98564736629 * d).rem_euclid(360.0);
    let ha = (gmst + LON).to_radians() - ra;
    let lat = LAT.to_radians();
    (lat.sin() * dec.sin() + lat.cos() * dec.cos() * ha.cos()).asin().to_degrees()
}

/// Screen x (pixels from centre, ±R) ↔ hours from now. Each side is a quarter sine, so time near now is nearly linear and the far ends crowd into the rim.
#[allow(dead_code)]
fn x_of(h: f64) -> f64 {
    let side = if h < 0.0 { PAST_H } else { FUTURE_H };
    R * (PI / 2.0 * (h / side).clamp(-1.0, 1.0)).sin()
}
fn h_of(x: f64) -> f64 {
    let u = (x / R).clamp(-1.0, 1.0);
    let side = if u < 0.0 { PAST_H } else { FUTURE_H };
    u.asin() / (PI / 2.0) * side
}

fn smooth(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn mix(a: [f64; 3], b: [f64; 3], f: f64) -> [f64; 3] {
    [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}
fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    a.map(|v| v * k)
}

fn hash(x: i64, y: i64, seed: u64) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ seed.wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn value_noise(x: f64, y: f64, seed: u64) -> f64 {
    let (xi, yi) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let a = hash(xi, yi, seed) + (hash(xi + 1, yi, seed) - hash(xi, yi, seed)) * sx;
    let b = hash(xi, yi + 1, seed) + (hash(xi + 1, yi + 1, seed) - hash(xi, yi + 1, seed)) * sx;
    a + (b - a) * sy
}

fn fbm(x: f64, y: f64) -> f64 {
    let mut v = 0.0;
    let mut amp = 0.5;
    let (mut x, mut y) = (x, y);
    for o in 0..5 {
        v += amp * value_noise(x, y, o);
        x *= 2.03;
        y *= 2.03;
        amp *= 0.5;
    }
    v
}

/// Linear Rec.2020 frame (white = 1), composited back to front.
struct Canvas {
    px: Vec<[f64; 3]>,
}

impl Canvas {
    fn over(&mut self, x: usize, y: usize, c: [f64; 3], a: f64) {
        if x < N && y < N && a > 0.0 {
            let p = &mut self.px[y * N + x];
            *p = mix(*p, c, a.min(1.0));
        }
    }

    /// Anti-aliased capsule from a to b, `w` pixels wide.
    fn line(&mut self, a: (f64, f64), b: (f64, f64), w: f64, c: [f64; 3], alpha: f64) {
        let r = w / 2.0;
        let (x0, x1) = ((a.0.min(b.0) - r - 1.0).floor().max(0.0) as usize, (a.0.max(b.0) + r + 1.0).ceil().min(N as f64 - 1.0) as usize);
        let (y0, y1) = ((a.1.min(b.1) - r - 1.0).floor().max(0.0) as usize, (a.1.max(b.1) + r + 1.0).ceil().min(N as f64 - 1.0) as usize);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len2 = (dx * dx + dy * dy).max(1e-9);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let t = (((px - a.0) * dx + (py - a.1) * dy) / len2).clamp(0.0, 1.0);
                let d = ((px - a.0 - t * dx).powi(2) + (py - a.1 - t * dy).powi(2)).sqrt();
                let cov = (r + 0.5 - d).clamp(0.0, 1.0);
                self.over(x, y, c, cov * alpha);
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "live".into());
    let port = args.next().unwrap_or_else(|| "/dev/ttyACM0".into());
    fs::create_dir_all(OUT).unwrap();
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    let wx = if mode == "demo" { demo(now) } else { fetch() };
    let hours = |x: usize| h_of(x as f64 + 0.5 - R);
    let at = |s: &[f64], x: usize| wx.at(s, now + hours(x) * 3600.0);

    let mut cv = Canvas { px: vec![[0.0; 3]; N * N] };

    // Sky and sea, one column (one instant) at a time.
    let zen_day = [0.035, 0.085, 0.30];
    let hor_day = [0.16, 0.22, 0.36];
    let zen_night = [0.0012, 0.0016, 0.0045];
    let hor_night = [0.0025, 0.003, 0.006];
    let glow = [0.50, 0.17, 0.035];
    let noise: Vec<f64> = (0..N * N).map(|i| fbm((i % N) as f64 / 130.0, (i / N) as f64 / 42.0)).collect();
    let texture: Vec<f64> = (0..N * N).map(|i| fbm((i % N) as f64 / 40.0 + 17.0, (i / N) as f64 / 16.0 + 5.0)).collect();
    let mut sorted = noise.clone();
    sorted.sort_by(f64::total_cmp);
    let quantile = |q: f64| sorted[(q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64) as usize];
    let sea_y = R - SEA * R;
    for x in 0..N {
        let t = now + hours(x) * 3600.0;
        let e = sun_alt(t);
        let day = smooth(-9.0, 5.0, e);
        let tw = (-((e + 2.5) / 4.5).powi(2)).exp();
        let cover = (at(&wx.cloud, x) / 100.0).clamp(0.0, 1.0);
        let q = quantile(cover);
        // At night the clouds catch the city's glow, so an overcast night still reads against a clear one.
        let cloud_lit = mix([0.010, 0.007, 0.0045], [0.30, 0.30, 0.31], day);
        let cloud_lit = mix(cloud_lit, [0.40, 0.22, 0.12], tw * 0.6);
        // Overcast dims and greys what clear sky shows between the clouds.
        let veil = 1.0 - 0.5 * cover;
        for y in 0..N {
            let v = (R - (y as f64 + 0.5)) / R;
            let c = if (y as f64 + 0.5) < sea_y {
                let s = ((v - SEA) / (1.0 - SEA)).clamp(0.0, 1.0);
                let zen = mix(zen_night, zen_day, day);
                let hor = mix(hor_night, hor_day, day);
                let mut c = mix(hor, zen, s.powf(0.55));
                c = mix(c, [c[1]; 3], cover * 0.6);
                c = scale(c, veil);
                for k in 0..3 {
                    c[k] += glow[k] * tw * 0.8 * (1.0 - s).powi(7);
                }
                let n = noise[y * N + x];
                let a = ((q - n) / 0.035 + 0.5).clamp(0.0, 1.0);
                // Undersides darker toward the horizon, plus fine texture so even full overcast isn't a flat slab.
                let shade = (0.55 + 0.35 * s) * (0.75 + 0.6 * (texture[y * N + x] - 0.5));
                let lit = mix(scale(cloud_lit, shade), scale(glow, 0.5 * shade), tw * (1.0 - s).powi(2));
                mix(c, lit, a)
            } else {
                // Sea: a dark mirror of the sky above it.
                let sky = mix(hor_night, hor_day, day);
                let d = ((y as f64 + 0.5 - sea_y) / (R * (1.0 + SEA))).clamp(0.0, 1.0);
                let mut c = scale(mix([0.002, 0.010, 0.014], scale(sky, 0.35), 0.5), veil * (1.0 - 0.5 * d));
                for k in 0..3 {
                    c[k] += glow[k] * tw * 0.25 * (1.0 - d).powi(4);
                }
                c
            };
            cv.px[y * N + x] = c;
        }
    }

    // Rain: streaks seeded per pixel, density by precipitation, slanted by the east–west wind.
    for y in 0..sea_y as usize {
        for x in 0..N {
            let p = at(&wx.rain, x);
            if p <= 0.02 || hash(x as i64, y as i64, 99) > 0.006 * p.sqrt() {
                continue;
            }
            let dir = wx.dir_at(now + hours(x) * 3600.0).to_radians();
            let u = -dir.sin() * at(&wx.wind, x); // eastward component (wind blows toward dir+180°)
            let len = 9.0 + 6.0 * hash(x as i64, y as i64, 7);
            let slant = (u * 0.04).clamp(-0.8, 0.8);
            let (x0, y0) = (x as f64 + 0.5, y as f64 + 0.5);
            let k = len / (1.0 + slant * slant).sqrt();
            cv.line((x0, y0), (x0 + slant * k, y0 + k), 1.0, [0.30, 0.40, 0.60], 0.55);
        }
    }

    // Temperature and dew point: one shared scale over the window.
    let (lo, hi) = (0..N).fold((f64::MAX, f64::MIN), |(lo, hi), x| (lo.min(at(&wx.dew, x)), hi.max(at(&wx.temp, x))));
    let (lo, hi) = (lo - 2.0, hi + 2.0);
    let y_of = |f: f64| R - (SEA + 0.12 + (f - lo) / (hi - lo) * 0.78) * R;
    // Faint haze between the curves: thin when the air is dry, the curves touching when it's saturated.
    for x in 0..N {
        let (yt, yd) = (y_of(at(&wx.temp, x)), y_of(at(&wx.dew, x)));
        for y in yt.ceil() as usize..yd.floor().max(yt.ceil()) as usize {
            cv.over(x, y, [0.20, 0.20, 0.22], 0.10);
        }
    }
    for (series, colour, w) in [(&wx.dew, [0.10, 0.45, 1.0], 1.6), (&wx.temp, [1.0, 0.50, 0.10], 2.2)] {
        let mut prev = None;
        for x in 0..N {
            let p = (x as f64 + 0.5, y_of(at(series, x)));
            if let Some(q) = prev {
                cv.line(q, p, w, scale(colour, 0.75), 1.0);
            }
            prev = Some(p);
        }
    }

    // Wind on the sea: strokes every ~18 px in two staggered rows, pointing downwind (north up), 1.6 px per mph, faint out to the gust.
    for row in 0..2 {
        let yc = sea_y + R * (1.0 + SEA) * (0.28 + 0.34 * row as f64);
        let mut xc = 9.0 + 9.0 * row as f64;
        while xc < N as f64 {
            let x = xc as usize;
            let t = now + hours(x) * 3600.0;
            let to = (wx.dir_at(t) + 180.0).to_radians();
            let (ux, uy) = (to.sin(), -to.cos());
            let (sp, gu) = (at(&wx.wind, x), at(&wx.gust, x));
            let (l, lg) = (1.6 * sp.max(1.5), 1.6 * gu.max(sp));
            let tail = (xc - ux * l / 2.0, yc - uy * l / 2.0);
            let head = (xc + ux * l / 2.0, yc + uy * l / 2.0);
            // A wedge: thin upwind tail, fat downwind head, so it reads as moving that way.
            for k in 0..8 {
                let (f0, f1) = (k as f64 / 8.0, (k + 1) as f64 / 8.0);
                let p = |f: f64| (tail.0 + (head.0 - tail.0) * f, tail.1 + (head.1 - tail.1) * f);
                cv.line(p(f0), p(f1), 0.6 + 2.4 * f1, [0.55, 0.62, 0.68], 0.9);
            }
            cv.line(head, (head.0 + ux * (lg - l), head.1 + uy * (lg - l)), 1.0, [0.55, 0.62, 0.68], 0.3);
            xc += 18.0;
        }
    }

    // Now.
    cv.line((R, 0.0), (R, N as f64), 1.0, [0.6, 0.6, 0.6], 0.6);

    // Encode through the panel model; outside the circle is black.
    let model = PanelModel::load(&Path::new(ROOT).join("i1pro-2026-10-11"), &Path::new(ROOT).join("i1pro-2026-10-11-toe"));
    let s = model.white_scale(0.6);
    let mut codes = vec![0u8; N * N * 3];
    for y in 0..N {
        for x in 0..N {
            let (dx, dy) = (x as f64 + 0.5 - R, y as f64 + 0.5 - R);
            if dx * dx + dy * dy > R * R {
                continue;
            }
            let c = cv.px[y * N + x].map(|v| v as f32);
            let l = apply_matrix_3x3_f32(&VSF_RGB2LMS, &apply_matrix_3x3_f32(&REC2020_2VSF_RGB, &c));
            let (k, _) = model.encode([l[0] as f64 * s, l[1] as f64 * s, l[2] as f64 * s]);
            codes[(y * N + x) * 3..(y * N + x) * 3 + 3].copy_from_slice(&k);
        }
    }
    let path = Path::new(OUT).join(format!("wx-{mode}.png"));
    let mut enc = png::Encoder::new(BufWriter::new(File::create(&path).unwrap()), N as u32, N as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(&codes).unwrap();
    println!("{} ({lo:.0}–{hi:.0} °F)", path.display());

    if port != "-" {
        let mut link = Link::open(&port);
        link.send("m 24");
        link.send("b 255");
        link.send_image(&codes);
        println!("on the panel");
    }
}
