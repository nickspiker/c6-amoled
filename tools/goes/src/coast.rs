//! The coastline overlay: Natural Earth's 1:10m coastline projected exactly as the frames are, drawn as a 1.5 px anti-aliased line, and packed into strips for the board, which flickers it ±16 codes at 30 Hz around the true image (see the firmware's `coast`).

use std::fs;

use crate::abi::N;
use crate::{CENTRE, SCALE_KM, inside, repo};

const SOURCE: &str = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson";
/// GRS80, as the ABI files give the ellipsoid the frames are reprojected on.
const A: f64 = 6_378_137.0;
const B: f64 = 6_356_752.314_14;
/// Strips on one row pair closer than this many pixels merge into one: a window's setup costs more than pushing that many unchanged pixels.
const MERGE_GAP: usize = 32;

/// The coastline's polylines as (lon, lat) in degrees, downloaded once into the cache.
fn polylines() -> Result<Vec<Vec<(f64, f64)>>, String> {
    let path = repo().join("cache/coast/ne_10m_coastline.geojson");
    if !path.exists() {
        println!("downloading {SOURCE}");
        let body = ureq::get(SOURCE).call().map_err(|e| e.to_string())?.body_mut().with_config().limit(64 << 20).read_to_vec().map_err(|e| e.to_string())?;
        fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        fs::write(&path, body).map_err(|e| e.to_string())?;
    }
    let j: serde_json::Value = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let point = |p: &serde_json::Value| (p[0].as_f64().unwrap_or(0.0), p[1].as_f64().unwrap_or(0.0));
    let mut lines = Vec::new();
    for f in j["features"].as_array().ok_or("no features")? {
        let g = &f["geometry"];
        let parts: Vec<&serde_json::Value> = match g["type"].as_str() {
            Some("LineString") => vec![&g["coordinates"]],
            Some("MultiLineString") => g["coordinates"].as_array().map(|a| a.iter().collect()).unwrap_or_default(),
            _ => continue,
        };
        for part in parts {
            if let Some(pts) = part.as_array() {
                lines.push(pts.iter().map(point).collect());
            }
        }
    }
    Ok(lines)
}

/// (lon, lat) in degrees → continuous pixel coordinates (x across, y down; pixel c's centre at c + 0.5), the exact inverse of the frames' reprojection: the point on the ellipsoid, then its east and north offsets in the tangent plane at the centre (the frames drop each plane point along the centre's vertical, which leaves east and north unchanged). None beyond 2000 km of the centre: the plane maps the far side of the Earth onto the same pixels (islands half a world away turned up as rings in British Columbia), and the disk only reaches about 910 km.
fn projector() -> impl Fn(f64, f64) -> Option<(f64, f64)> {
    let e2 = 1.0 - (B * B) / (A * A);
    let ecef = move |lat: f64, lon: f64| {
        let (la, lo) = (lat.to_radians(), lon.to_radians());
        let nu = A / (1.0 - e2 * la.sin().powi(2)).sqrt();
        [nu * la.cos() * lo.cos(), nu * la.cos() * lo.sin(), nu * (1.0 - e2) * la.sin()]
    };
    let p0 = ecef(CENTRE.0, CENTRE.1);
    let (la, lo) = (CENTRE.0.to_radians(), CENTRE.1.to_radians());
    let up = [la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()];
    let east = [-lo.sin(), lo.cos(), 0.0];
    let north = [up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0]];
    let m = SCALE_KM * 1000.0;
    move |lon, lat| {
        let p = ecef(lat, lon);
        let d = [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]];
        let dot = |v: [f64; 3]| d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
        (dot(d) < 2.0e6 * 2.0e6).then(|| (dot(east) / m + N as f64 / 2.0, N as f64 / 2.0 - dot(north) / m))
    }
}

/// Coverage per disk pixel, 0..=15: a 1.5 px line, each pixel 1.25 minus its centre's distance to the nearest segment, clamped to 0..1 and floored to sixteenths (at most 15).
pub fn mask() -> Result<Vec<u8>, String> {
    let project = projector();
    let mut cov = vec![0f64; N * N];
    for line in polylines()? {
        let pts: Vec<Option<(f64, f64)>> = line.iter().map(|&(lon, lat)| project(lon, lat)).collect();
        for s in pts.windows(2) {
            let (Some((x0, y0)), Some((x1, y1))) = (s[0], s[1]) else { continue };
            let out = |v: f64| !(-4.0..N as f64 + 4.0).contains(&v);
            if (out(x0) && out(x1)) || (out(y0) && out(y1)) {
                continue;
            }
            let (dx, dy) = (x1 - x0, y1 - y0);
            let len2 = dx * dx + dy * dy;
            let (cx0, cx1) = ((x0.min(x1) - 2.0).max(0.0) as usize, ((x0.max(x1) + 2.0).ceil().max(0.0) as usize).min(N));
            let (cy0, cy1) = ((y0.min(y1) - 2.0).max(0.0) as usize, ((y0.max(y1) + 2.0).ceil().max(0.0) as usize).min(N));
            for y in cy0..cy1 {
                for x in cx0..cx1 {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let t = if len2 > 0.0 { (((px - x0) * dx + (py - y0) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
                    let d = ((px - x0 - t * dx).powi(2) + (py - y0 - t * dy).powi(2)).sqrt();
                    let c = &mut cov[y * N + x];
                    *c = c.max((1.25 - d).clamp(0.0, 1.0));
                }
            }
        }
    }
    Ok((0..N * N).map(|i| if inside(i % N, i / N) { ((cov[i] * 16.0) as u8).min(15) } else { 0 }).collect())
}

/// The board's blob: "CST1", its whole length (u32, little-endian, padded to whole words), then strips over row pairs (the panel's windows must start and span even rows and columns): even row, even first column, even width (u16 each, little-endian), then each pixel's coverage as a nibble, high first, the first row then the second (width bytes in all). Strips on a pair closer than MERGE_GAP merge. Ends with row 0xFFFF.
pub fn pack(cov: &[u8]) -> Vec<u8> {
    let mut out = b"CST1\0\0\0\0".to_vec();
    for y in (0..N).step_by(2) {
        let hit = |x: usize| cov[y * N + x] != 0 || cov[(y + 1) * N + x] != 0;
        let mut x = 0;
        while x < N {
            if !hit(x) {
                x += 1;
                continue;
            }
            let start = x & !1;
            let mut end = x + 1;
            let mut gap = 0;
            while x < N && gap <= MERGE_GAP {
                if hit(x) {
                    end = x + 1;
                    gap = 0;
                } else {
                    gap += 1;
                }
                x += 1;
            }
            let end = (end + 1) & !1;
            for v in [y, start, end - start] {
                out.extend_from_slice(&(v as u16).to_le_bytes());
            }
            for row in [y, y + 1] {
                for pair in (start..end).step_by(2) {
                    out.push(cov[row * N + pair] << 4 | cov[row * N + pair + 1]);
                }
            }
            x = end;
        }
    }
    out.extend_from_slice(&0xFFFFu16.to_le_bytes());
    while out.len() % 4 != 0 {
        out.push(0);
    }
    let len = out.len() as u32;
    out[4..8].copy_from_slice(&len.to_le_bytes());
    out
}

/// One RGB565 pixel nudged by `sign` × 16 codes in gamma space, scaled by coverage `c` (0..=15): red and blue move 2c/15 of their 5-bit levels, green 4c/15 of its 6-bit ones, floored, clamped at the ends. The board does the same.
pub fn nudge(p: u16, c: u8, sign: i32) -> u16 {
    let c = c as i32;
    let ch = |v: u16, shift: u32, mask: i32, step: i32| ((v >> shift) as i32 & mask) + sign * (step * c / 15);
    let r = ch(p, 11, 31, 2).clamp(0, 31) as u16;
    let g = ch(p, 5, 63, 4).clamp(0, 63) as u16;
    let b = ch(p, 0, 31, 2).clamp(0, 31) as u16;
    r << 11 | g << 5 | b
}
