//! GOES-18 ABI full-disk frames from NOAA's public S3 bucket: listing, lazy HDF5 reads over HTTP range requests (only the chunks under the disk, not the 20–430 MB files), and reprojection onto the panel's disk.
//!
//! The disk is a square of ground seen from straight above its centre: the tangent plane there, each point dropped along the centre's vertical onto the GRS80 ellipsoid (the one ABI's fixed grid uses). Each panel pixel's radiance is the mean of a 4×4 grid of sub-points per 2 km source step, each taking its nearest ABI sample: an area average, never interpolated. Radiance stays absolute, in the band's own unit.

use std::sync::Arc;

use hdf5_reader::storage::{BlockCacheStorage, RangeRequestStorage};
use hdf5_reader::{Dataset, Hdf5File, SliceInfo, SliceInfoElem};

/// The dual-stack endpoint (IPv4 and IPv6): the plain one is IPv4-only, and on 2026-10-19 this connection's IPv4 went down while IPv6 kept working.
pub const BUCKET: &str = "https://noaa-goes18.s3.dualstack.us-east-1.amazonaws.com";
pub const N: usize = 466;

/// A band's calibration: `kappa0` turns radiance into reflectance factor (bands 1–6); the Planck coefficients turn it into brightness temperature (7–16).
#[derive(Clone, Debug)]
pub struct Cal {
    pub band: u8,
    pub wavelength: f64,
    pub unit: String,
    pub kappa0: Option<f64>,
    pub planck: Option<[f64; 4]>,
}

impl Cal {
    /// Radiance as reflectance % or brightness temperature K, for logs.
    pub fn equivalent(&self, rad: f64) -> String {
        if let Some(k) = self.kappa0 {
            return format!("{:.1} %", rad * k * 100.0);
        }
        let [fk1, fk2, bc1, bc2] = self.planck.unwrap_or([1.0, 1.0, 0.0, 1.0]);
        if rad <= 0.0 { "0 K".into() } else { format!("{:.1} K", (fk2 / (fk1 / rad + 1.0).ln() - bc1) / bc2) }
    }

    /// The radiance of a blackbody at `kelvin` in this band.
    pub fn planck_radiance(&self, kelvin: f64) -> f64 {
        let [fk1, fk2, bc1, bc2] = self.planck.expect("thermal band");
        fk1 / ((fk2 / (bc1 + bc2 * kelvin)).exp() - 1.0)
    }
}

/// One HTTP agent for every S3 request, with a limit on each: without one, a connection that goes quiet mid-read blocks forever (on 2026-10-17 one hung the loop for 34 hours with the puck replaying the same frames). A 1 MiB block normally takes well under a second; a failed poll is retried a minute later.
fn agent() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(60))).build().into())
}

fn get(url: &str) -> Result<String, String> {
    agent().get(url).call().map_err(|e| e.to_string())?.body_mut().read_to_string().map_err(|e| e.to_string())
}

/// Object keys under a prefix (an hour folder holds well under the listing's 1000-key page).
pub fn list_keys(prefix: &str) -> Result<Vec<String>, String> {
    let xml = get(&format!("{BUCKET}/?list-type=2&prefix={prefix}"))?;
    Ok(xml.split("<Key>").skip(1).filter_map(|s| s.split("</Key>").next()).map(str::to_string).collect())
}

/// A key's band and scan start (unix seconds), from its name: `…M6C13_G18_s20262860350203_…`.
pub fn parse_key(key: &str) -> Option<(u8, i64)> {
    let c = key.find("_G18_s")?;
    let band: u8 = key.get(c - 2..c)?.parse().ok()?;
    let s = key.get(c + 6..c + 19)?;
    let (year, doy): (i32, u32) = (s[0..4].parse().ok()?, s[4..7].parse().ok()?);
    let date = chrono::NaiveDate::from_yo_opt(year, doy)?;
    let time = chrono::NaiveTime::from_hms_opt(s[7..9].parse().ok()?, s[9..11].parse().ok()?, s[11..13].parse().ok()?)?;
    Some((band, date.and_time(time).and_utc().timestamp()))
}

/// The hour folder for a moment: `ABI-L1b-RadF/YYYY/DDD/HH/`.
pub fn hour_prefix(t: i64) -> String {
    let d = chrono::DateTime::from_timestamp(t, 0).unwrap();
    format!("ABI-L1b-RadF/{}/", d.format("%Y/%j/%H"))
}

/// An HDF5 file read lazily over HTTP: 1 MiB blocks, fetched on demand and cached.
fn open(key: &str) -> Result<Hdf5File, String> {
    let url = format!("{BUCKET}/{key}");
    let len: u64 = agent().head(&url).call().map_err(|e| e.to_string())?.headers().get("content-length").and_then(|v| v.to_str().ok()?.parse().ok()).ok_or("no length")?;
    let remote = RangeRequestStorage::new(len, move |off, n| {
        let mut r = agent().get(&url).header("Range", format!("bytes={}-{}", off, off + n as u64 - 1)).call().map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(r.body_mut().with_config().limit(u64::MAX).read_to_vec().map_err(|e| std::io::Error::other(e.to_string()))?)
    });
    Hdf5File::from_storage(Arc::new(BlockCacheStorage::new(Arc::new(remote), 1 << 20, 512))).map_err(|e| e.to_string())
}

fn attr(d: &Dataset, name: &str) -> Option<f64> {
    d.attribute(name).ok()?.read_as_f64().ok()
}

/// A one-value dataset (GOES stores these as 32-bit floats, a few as 64-bit).
fn scalar(f: &Hdf5File, name: &str) -> Option<f64> {
    let d = f.dataset(name).ok()?;
    match d.read_array::<f32>() {
        Ok(a) => a.iter().next().map(|&v| v as f64),
        Err(_) => d.read_array::<f64>().ok()?.iter().next().copied(),
    }
}

/// Ellipsoid and viewpoint: semi-major and semi-minor axes (m), the satellite's distance from Earth's centre, and the sub-satellite longitude (radians).
struct Geometry {
    a: f64,
    b: f64,
    h: f64,
    lon0: f64,
}

impl Geometry {
    /// Earth-centred point (m) under the tangent plane at (clat, clon) at east/north offsets (m): dropped along the centre's vertical onto the ellipsoid.
    fn ground(&self, frame: &Frame, e: f64, n: f64) -> [f64; 3] {
        let p = [0, 1, 2].map(|i| frame.p0[i] + e * frame.east[i] + n * frame.north[i]);
        let d = [1.0 / (self.a * self.a), 1.0 / (self.a * self.a), 1.0 / (self.b * self.b)];
        let up = frame.up;
        let qa: f64 = (0..3).map(|i| up[i] * d[i] * up[i]).sum();
        let qb: f64 = -2.0 * (0..3).map(|i| p[i] * d[i] * up[i]).sum::<f64>();
        let qc: f64 = (0..3).map(|i| p[i] * p[i] * d[i]).sum::<f64>() - 1.0;
        // The root near zero, in the stable form t = c / q.
        let q = (-qb + (qb * qb - 4.0 * qa * qc).sqrt()) / 2.0;
        let t = qc / q;
        [0, 1, 2].map(|i| p[i] - t * up[i])
    }

    /// Earth-centred point → ABI scan angles (x, y), radians (GOES-R PUG §4.2.8).
    fn scan(&self, p: [f64; 3]) -> (f64, f64) {
        let (xr, yr) = (p[0] * self.lon0.cos() + p[1] * self.lon0.sin(), -p[0] * self.lon0.sin() + p[1] * self.lon0.cos());
        let (sx, sy, sz) = (self.h - xr, -yr, p[2]);
        ((-sy / (sx * sx + sy * sy + sz * sz).sqrt()).asin(), (sz / sx).atan())
    }
}

/// The local frame at the disk's centre: its point on the ellipsoid and its east, north and up unit vectors.
struct Frame {
    p0: [f64; 3],
    east: [f64; 3],
    north: [f64; 3],
    up: [f64; 3],
}

impl Frame {
    fn new(g: &Geometry, clat: f64, clon: f64) -> Frame {
        let (la, lo) = (clat.to_radians(), clon.to_radians());
        let e2 = 1.0 - (g.b * g.b) / (g.a * g.a);
        let nu = g.a / (1.0 - e2 * la.sin().powi(2)).sqrt();
        let up = [la.cos() * lo.cos(), la.cos() * lo.sin(), la.sin()];
        let east = [-lo.sin(), lo.cos(), 0.0];
        let north = [up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0]];
        Frame { p0: [nu * la.cos() * lo.cos(), nu * la.cos() * lo.sin(), nu * (1.0 - e2) * la.sin()], east, north, up }
    }
}

/// One band of one frame reprojected onto the N×N disk centred on (clat, clon) at `scale` km per pixel: radiance per pixel (NaN where nothing valid fell), and the band's calibration.
pub fn read_band(key: &str, clat: f64, clon: f64, scale: f64) -> Result<(Vec<f32>, Cal), String> {
    let f = open(key)?;
    fn e(s: &'static str) -> impl Fn(hdf5_reader::error::Error) -> String {
        move |err| format!("{s}: {err}")
    }
    let rad = f.dataset("Rad").map_err(e("Rad"))?;
    let proj = f.dataset("goes_imager_projection").map_err(e("projection"))?;
    let (x, y) = (f.dataset("x").map_err(e("x"))?, f.dataset("y").map_err(e("y"))?);
    let g = Geometry {
        a: attr(&proj, "semi_major_axis").ok_or("no semi_major_axis")?,
        b: attr(&proj, "semi_minor_axis").ok_or("no semi_minor_axis")?,
        h: attr(&proj, "perspective_point_height").ok_or("no perspective_point_height")? + attr(&proj, "semi_major_axis").unwrap(),
        lon0: attr(&proj, "longitude_of_projection_origin").ok_or("no longitude_of_projection_origin")?.to_radians(),
    };
    // Grid: scan angle of index i is offset + i · scale.
    let (x0, dx) = (attr(&x, "add_offset").unwrap_or(0.0), attr(&x, "scale_factor").ok_or("no x scale")?);
    let (y0, dy) = (attr(&y, "add_offset").unwrap_or(0.0), attr(&y, "scale_factor").ok_or("no y scale")?);
    let frame = Frame::new(&g, clat, clon);
    let m = scale * 1000.0;
    let index = |e: f64, n: f64| {
        let (sx, sy) = g.scan(g.ground(&frame, e, n));
        (((sy - y0) / dy).round() as i64, ((sx - x0) / dx).round() as i64)
    };

    // The source rectangle under the whole square, plus a margin.
    let shape = rad.shape().to_vec();
    let (mut r0, mut r1, mut c0, mut c1) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
    for j in 0..=2 * N {
        for i in 0..=2 * N {
            if j % (2 * N) != 0 && i % (2 * N) != 0 {
                continue;
            }
            let (r, c) = index((i as f64 / 2.0 - N as f64 / 2.0) * m, (N as f64 / 2.0 - j as f64 / 2.0) * m);
            (r0, r1, c0, c1) = (r0.min(r), r1.max(r), c0.min(c), c1.max(c));
        }
    }
    let (r0, r1) = ((r0 - 4).max(0), (r1 + 5).min(shape[0] as i64));
    let (c0, c1) = ((c0 - 4).max(0), (c1 + 5).min(shape[1] as i64));
    let sel = SliceInfo { selections: vec![SliceInfoElem::Slice { start: r0 as u64, end: r1 as u64, step: 1 }, SliceInfoElem::Slice { start: c0 as u64, end: c1 as u64, step: 1 }] };
    let raw = rad.read_slice::<i16>(&sel).map_err(e("Rad slice"))?;
    let dqf = f.dataset("DQF").map_err(e("DQF"))?.read_slice::<i8>(&sel).map_err(e("DQF slice"))?;
    let unsigned = rad.attribute("_Unsigned").ok().and_then(|a| a.read_string().ok()).is_some_and(|s| s == "true");
    let (scale_f, offset, fill) = (attr(&rad, "scale_factor").unwrap_or(1.0), attr(&rad, "add_offset").unwrap_or(0.0), attr(&rad, "_FillValue"));
    let width = (c1 - c0) as usize;
    let win: Vec<f32> = raw
        .iter()
        .zip(dqf.iter())
        .map(|(&q, &dq)| {
            let v = if unsigned { q as u16 as f64 } else { q as f64 };
            // DQF 0 good, 1 conditionally usable; 2+ out of range, no value, or focal-plane temperature exceeded.
            if Some(v) == fill || dq as u8 > 1 { f32::NAN } else { (v * scale_f + offset) as f32 }
        })
        .collect();

    // Each pixel: the mean of K×K sub-points, each taking its nearest source sample.
    const K: usize = 4;
    let mut out = vec![f32::NAN; N * N];
    for j in 0..N {
        for i in 0..N {
            let (mut sum, mut n) = (0f64, 0u32);
            for sj in 0..K {
                for si in 0..K {
                    let e = (i as f64 + (si as f64 + 0.5) / K as f64 - N as f64 / 2.0) * m;
                    let nn = (N as f64 / 2.0 - (j as f64 + (sj as f64 + 0.5) / K as f64)) * m;
                    let (r, c) = index(e, nn);
                    if r >= r0 && r < r1 && c >= c0 && c < c1 {
                        let v = win[(r - r0) as usize * width + (c - c0) as usize];
                        if v.is_finite() {
                            sum += v as f64;
                            n += 1;
                        }
                    }
                }
            }
            if n > 0 {
                out[j * N + i] = (sum / n as f64) as f32;
            }
        }
    }

    let band = parse_key(key).map(|(b, _)| b).unwrap_or(0);
    let unit = rad.attribute("units").ok().and_then(|a| a.read_string().ok()).unwrap_or_default();
    let wavelength = scalar(&f, "band_wavelength").unwrap_or(0.0);
    let cal = if band <= 6 {
        Cal { band, wavelength, unit, kappa0: scalar(&f, "kappa0"), planck: None }
    } else {
        let p = ["planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2"].map(|k| scalar(&f, k).unwrap_or(f64::NAN));
        Cal { band, wavelength, unit, kappa0: None, planck: Some(p) }
    };
    Ok((out, cal))
}
