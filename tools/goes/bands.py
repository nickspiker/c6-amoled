"""Preview all sixteen GOES-18 ABI bands over Puget Sound as a top-down disk, to choose an RGB channel assignment, and put a chosen assignment on the panel.

Reads one full-disk L1b frame from NOAA's public S3 bucket over HTTP range requests (only the chunks under the disk, not the 20–430 MB files) and reprojects it: each panel pixel is a square of ground seen from straight above the centre (orthographic onto the GRS80 ellipsoid, the same one ABI's fixed grid uses). Each pixel's radiance is the mean of a 4×4 grid of sub-points per source step (more for the 1 km and 0.5 km bands), each taking its nearest ABI sample — an area average, never interpolated. At the default 3.9 km/px the centre is native north–south resolution; toward the north rim ABI's pixels stretch to ~5 km, so samples repeat there rather than being invented.

Radiance stays absolute throughout, in each band's own unit (W m⁻² sr⁻¹ µm⁻¹ for bands 1–6, mW m⁻² sr⁻¹ (cm⁻¹)⁻¹ for 7–16): NOAA's integer packing is undone exactly, and nothing is stretched per frame. Writes to /tmp/goes-preview:
- picker-<frame>.html: assign any band to R, G, B with a floor and full-scale radiance each (floor 0 by default, so light ∝ radiance). Self-contained, open it in a browser.
- sheet-<frame>.png: all sixteen bands, each from zero radiance to its 98th percentile.
- bands-<frame>.npz: the reprojected radiances (float32) and calibration, which `panel` reads.

Usage:
  .venv/bin/python bands.py [--at 2026-10-12T01:20] [--scale 3.9] [lat lon]    frame nearest the given UTC time (default newest); km per pixel; disk centre (default 47.6 −124.5, Seattle shifted 2° west)
  .venv/bin/python bands.py panel "<assignment line from the picker>" [port]
"""

import base64
import datetime as dt
import io
import json
import re
import subprocess
import sys
import urllib.request
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

BUCKET = "https://noaa-goes18.s3.amazonaws.com"
BANDS = list(range(1, 17))
NAMES = {1: "blue", 2: "red", 3: "veggie", 4: "cirrus", 5: "snow/ice", 6: "cloud particle size", 7: "shortwave window", 8: "upper water vapour", 9: "mid water vapour", 10: "lower water vapour", 11: "cloud-top phase", 12: "ozone", 13: "clean window", 14: "window", 15: "dirty window", 16: "CO₂"}
FINE = {1: 2, 2: 4, 3: 2, 5: 2}  # ABI grid steps per 2 km step
N = 466
OUT = Path("/tmp/goes-preview")
MEASURE = Path(__file__).resolve().parent.parent / "measure"


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of an HTTP object that fetches 1 MiB blocks on demand, so h5py reads only what it touches."""

    BLOCK = 1 << 20

    def __init__(self, url):
        self.url, self.pos, self.blocks, self.fetched = url, 0, {}, 0
        self.size = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30).headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + off
        return self.pos

    def block(self, i):
        if i not in self.blocks:
            lo, hi = i * self.BLOCK, min((i + 1) * self.BLOCK, self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
            self.blocks[i] = urllib.request.urlopen(req, timeout=60).read()
            self.fetched += len(self.blocks[i])
        return self.blocks[i]

    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        done = 0
        while done < n:
            i, o = divmod(self.pos + done, self.BLOCK)
            chunk = self.block(i)[o : o + n - done]
            b[done : done + len(chunk)] = chunk
            done += len(chunk)
        self.pos += n
        return n


def list_keys(prefix):
    xml = urllib.request.urlopen(f"{BUCKET}/?list-type=2&prefix={prefix}", timeout=30).read().decode()
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def find_frame(at):
    """(start stamp, {band: key}) of the complete frame nearest `at` (UTC), or the newest when `at` is None."""
    target = at or dt.datetime.now(dt.timezone.utc)
    frames = {}
    for back in range(-1, 4):
        t = target - dt.timedelta(hours=back)
        for k in list_keys(f"ABI-L1b-RadF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"):
            m = re.search(r"M\dC(\d\d)_G18_s(\d+)_", k)
            if m:
                frames.setdefault(m[2], {})[int(m[1])] = k
    complete = [s for s, b in frames.items() if len(b) == len(BANDS)]
    if not complete:
        sys.exit("no complete frame near that time")
    stamp = lambda s: dt.datetime.strptime(s[:13], "%Y%j%H%M%S").replace(tzinfo=dt.timezone.utc)
    best = max(complete) if at is None else min(complete, key=lambda s: abs(stamp(s) - at))
    return stamp(best), frames[best]


def attr(var, name, default):
    """A numeric attribute, which netCDF stores as a one-element array."""
    return float(np.asarray(var.attrs.get(name, default)).ravel()[0])


def scaled(var, sel=()):
    """A packed variable (or the `sel` slice of it) as float, honouring _Unsigned, _FillValue, scale_factor and add_offset."""
    raw = var[sel]
    if var.attrs.get("_Unsigned", b"") in (b"true", "true"):
        raw = raw.view(raw.dtype.str.replace("i", "u"))
    out = raw.astype(np.float64)
    fill = var.attrs.get("_FillValue")
    if fill is not None:
        out[raw == np.asarray(fill).astype(raw.dtype)] = np.nan
    return out * attr(var, "scale_factor", 1.0) + attr(var, "add_offset", 0.0)


def ellipsoid(proj):
    """(semi-major, semi-minor, satellite distance from Earth's centre, sub-satellite longitude in radians)."""
    a, b = attr(proj, "semi_major_axis", 0), attr(proj, "semi_minor_axis", 0)
    return a, b, attr(proj, "perspective_point_height", 0) + a, np.radians(attr(proj, "longitude_of_projection_origin", 0))


def ground(clat, clon, scale, k, j0, j1, a, b):
    """Earth-centred points (m) under a k×k grid of sub-points in each panel pixel of rows j0..j1, shape (rows·k, N·k, 3): the tangent plane at the centre, dropped straight down (along the centre's vertical) onto the ellipsoid."""
    la, lo = np.radians(clat), np.radians(clon)
    e2 = 1 - (b * b) / (a * a)
    nu = a / np.sqrt(1 - e2 * np.sin(la) ** 2)
    p0 = np.array([nu * np.cos(la) * np.cos(lo), nu * np.cos(la) * np.sin(lo), nu * (1 - e2) * np.sin(la)])
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.cross(up, east)
    sub = (np.arange(k) + 0.5) / k
    e = ((np.arange(N)[:, None] + sub).ravel() - N / 2) * scale * 1000
    n = (N / 2 - (np.arange(j0, j1)[:, None] + sub).ravel()) * scale * 1000
    p = p0 + e[None, :, None] * east + n[:, None, None] * north
    d = np.array([1 / (a * a), 1 / (a * a), 1 / (b * b)])
    # (p − t·up)ᵀ D (p − t·up) = 1, the root near zero, in the stable form t = C / q.
    qa, qb, qc = up @ (d * up), -2 * (p * d) @ up, (p * p * d).sum(-1) - 1
    q = (-qb + np.sqrt(qb * qb - 4 * qa * qc)) / 2
    return p - (qc / q)[..., None] * up


def scan(p, H, lon0):
    """Earth-centred points → ABI scan angles (x, y) in radians (GOES-R PUG §4.2.8)."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    xr, yr = x * np.cos(lon0) + y * np.sin(lon0), -x * np.sin(lon0) + y * np.cos(lon0)
    sx, sy, sz = H - xr, -yr, z
    return np.arcsin(-sy / np.sqrt(sx * sx + sy * sy + sz * sz)), np.arctan(sz / sx)


def read_band(job):
    """One band reprojected onto the disk: (band, radiance N×N float32 in the band's own unit, calibration, bytes read)."""
    band, key, clat, clon, scale = job
    k = 4 * FINE.get(band, 1)
    src = RangeFile(f"{BUCKET}/{key}")
    with h5py.File(src, "r") as f:
        x, y = scaled(f["x"]), scaled(f["y"])
        x0, dx, y0, dy = x[0], x[1] - x[0], y[0], y[1] - y[0]
        a, b, H, lon0 = ellipsoid(f["goes_imager_projection"])
        index = lambda p: (np.rint((scan(p, H, lon0)[1] - y0) / dy).astype(np.int64), np.rint((scan(p, H, lon0)[0] - x0) / dx).astype(np.int64))
        # The source rectangle under the whole square, plus a margin.
        rows, cols = index(ground(clat, clon, scale, 2, 0, N, a, b))
        m = 2 * FINE.get(band, 1) + 2
        r0, r1 = max(rows.min() - m, 0), min(rows.max() + m + 1, f["Rad"].shape[0])
        c0, c1 = max(cols.min() - m, 0), min(cols.max() + m + 1, f["Rad"].shape[1])
        win = scaled(f["Rad"], (slice(r0, r1), slice(c0, c1)))
        win[f["DQF"][r0:r1, c0:c1] > 1] = np.nan  # 0 good, 1 conditionally usable; 2+ out of range, no value, or focal-plane temp exceeded
        out = np.empty((N, N), np.float32)
        strip = max(1, 128 // k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j0 in range(0, N, strip):
                j1 = min(j0 + strip, N)
                r, c = index(ground(clat, clon, scale, k, j0, j1, a, b))
                v = win[np.clip(r - r0, 0, r1 - r0 - 1), np.clip(c - c0, 0, c1 - c0 - 1)]
                out[j0:j1] = np.nanmean(v.reshape(j1 - j0, k, N, k), axis=(1, 3))
        cal = {"wl": float(f["band_wavelength"][()].ravel()[0]), "unit": f["Rad"].attrs["units"].decode()}
        if band <= 6:
            cal["kappa0"] = float(f["kappa0"][()])
        else:
            cal.update({n: float(f[n][()]) for n in ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2")})
    return band, out, cal, src.fetched


def equivalent(m, rad):
    """Radiance as the more familiar figure: reflectance factor % (bands 1–6) or brightness temperature K."""
    if "kappa0" in m:
        return f"{rad * m['kappa0'] * 100:.1f} %"
    if rad <= 0:
        return "0 K"
    return f"{(m['planck_fk2'] / np.log(m['planck_fk1'] / rad + 1) - m['planck_bc1']) / m['planck_bc2']:.1f} K"


def preview(args):
    at, scale = None, 3.9
    while args and args[0].startswith("--"):
        if args[0] == "--at":
            at = dt.datetime.fromisoformat(args[1]).replace(tzinfo=dt.timezone.utc)
        elif args[0] == "--scale":
            scale = float(args[1])
        args = args[2:]
    lat, lon = (float(args[0]), float(args[1])) if len(args) >= 2 else (47.6, -124.5)
    OUT.mkdir(parents=True, exist_ok=True)
    when, keys = find_frame(at)
    tag = f"{when:%Y%m%d-%H%M}"
    print(f"frame {when:%Y-%m-%d %H:%M:%S} UTC; {scale} km/px, {N * scale:.0f} km across")

    rad, meta, total = {}, {}, 0
    with ProcessPoolExecutor(8) as pool:
        for band, img, cal, fetched in pool.map(read_band, [(b, keys[b], lat, lon, scale) for b in BANDS]):
            rad[band] = img
            p98, top = float(np.nanpercentile(img, 98)), float(np.nanmax(img))
            meta[band] = dict(cal, name=NAMES[band], p98=p98, rmax=top)
            total += fetched
            print(f"band {band:2} {cal['wl']:5.2f} µm {NAMES[band]:20} max {top:9.4f} {cal['unit']:22} (= {equivalent(cal, top)})   {fetched / 1e6:5.1f} MB read")
    print(f"{total / 1e6:.0f} MB read in all")

    np.savez_compressed(OUT / f"bands-{tag}.npz", **{f"rad{b}": rad[b] for b in BANDS}, meta=json.dumps(meta), when=when.isoformat(), centre=[lat, lon], scale=scale)
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    outside = (xx - N / 2) ** 2 + (yy - N / 2) ** 2 > (N / 2) ** 2

    # Contact sheet: 4×4, each band from zero radiance (black) to its 98th percentile (white), light ∝ radiance.
    sheet = Image.new("RGB", (4 * N, 4 * (N + 22)), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, band in enumerate(BANDS):
        m = meta[band]
        v = np.nan_to_num(np.clip(rad[band] / m["p98"], 0, 1)) ** (1 / 2.2) if m["p98"] > 0 else np.zeros((N, N))
        v[outside] = 0
        ox, oy = (i % 4) * N, (i // 4) * (N + 22)
        sheet.paste(Image.fromarray((v * 255 + 0.5).astype(np.uint8), "L"), (ox, oy + 22))
        draw.text((ox + 6, oy + 5), f"{band}  {m['wl']:.2f} um  {m['name'].replace('₂', '2')}  white = {m['p98']:.3g} ({equivalent(m, m['p98'])})", fill=(230, 230, 230))
    sheet.save(OUT / f"sheet-{tag}.png")

    data = {b: base64.b64encode(rad[b].astype("<f4").tobytes()).decode() for b in BANDS}
    caption = f"{when:%Y-%m-%d %H:%M} UTC · centre {lat:.2f}° {lon:.2f}° · {scale} km/px, {N * scale:.0f} km across, top-down"
    html = PICKER.replace("/*META*/", json.dumps(meta)).replace("/*DATA*/", json.dumps(data)).replace("/*WHEN*/", caption).replace("/*TAG*/", tag)
    (OUT / f"picker-{tag}.html").write_text(html)
    print(f"{OUT}/picker-{tag}.html, {OUT}/sheet-{tag}.png")


def panel(args):
    """Encode an assignment line (`<frame> R=13:0-95.2 G=10i:0.5-3.1 B=off`, limits in each band's own radiance unit) as linear panel drive and send it through the measure crate's `show`."""
    spec, port = args[0].split(), (args[1] if len(args) > 1 else "/dev/ttyACM0")
    tag = spec[0]
    z = np.load(OUT / f"bands-{tag}.npz")
    drive = np.zeros((N, N, 3))
    for item in spec[1:]:
        m = re.fullmatch(r"([RGB])=(?:off|(\d+)(i?):([-\d.eE+]+)-([-\d.eE+]+))", item)
        assert m, f"can't read {item!r}"
        if m[2] is None:
            continue
        c, lo, hi = "RGB".index(m[1]), float(m[4]), float(m[5])
        f = np.nan_to_num(np.clip((z[f"rad{int(m[2])}"] - lo) / (hi - lo), 0, 1))
        drive[..., c] = 1 - f if m[3] else f
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    drive[(xx - N / 2) ** 2 + (yy - N / 2) ** 2 > (N / 2) ** 2] = 0
    png = OUT / f"panel-{tag}.png"
    write_png16(png, drive)
    subprocess.run(["cargo", "run", "-q", "--release", "--bin", "show", "--", str(png), "panel-linear", "1", "stretch", port], cwd=MEASURE, check=True)


def write_png16(path, rgb):
    """A 16-bit RGB PNG, written by hand since PIL can't write 16-bit colour."""
    import struct
    import zlib

    h, w, _ = rgb.shape
    raw = (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(">u2")
    rows = b"".join(b"\x00" + raw[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


PICKER = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GOES Band Picker</title>
<style>
:root { --bg:#0b0b0c; --fg:#ddd; --dim:#888; --line:#2a2a2e; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px system-ui, sans-serif; }
main { display:flex; flex-wrap:wrap; gap:24px; padding:16px; align-items:flex-start; }
#view { border-radius:50%; image-rendering:pixelated; background:#000; }
.ch { border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-bottom:10px; min-width:360px; }
.ch h3 { margin:0 0 6px; font-size:14px; }
.row { display:flex; gap:8px; align-items:center; margin:4px 0; }
.row label { width:34px; color:var(--dim); }
input[type=range] { flex:1; }
.val { width:150px; text-align:right; font-variant-numeric:tabular-nums; }
.unit { color:var(--dim); font-size:12px; }
#thumbs { display:grid; grid-template-columns:repeat(4, 120px); gap:6px; }
#thumbs figure { margin:0; cursor:pointer; } #thumbs canvas { width:120px; height:120px; border-radius:50%; display:block; }
#thumbs figcaption { font-size:11px; color:var(--dim); text-align:center; }
code { background:#18181b; padding:2px 6px; border-radius:4px; user-select:all; }
small { color:var(--dim); }
</style></head><body><main>
<div><canvas id="view" width="466" height="466"></canvas><p><small>/*WHEN*/</small></p>
<p><label>Zoom <select id="zoom"><option value="1">1 panel pixel = 1 screen pixel</option><option value="2">2×</option><option value="3">3×</option></select></label></p>
<p>Assignment: <code id="spec"></code></p></div>
<div id="controls"></div>
<div><div id="thumbs"></div><p><small>Click a band, then press r / g / b to assign it. Thumbnails: zero radiance → black, the band's 98th percentile → white.</small></p></div>
</main><script>
const META = /*META*/, DATA = /*DATA*/, TAG = "/*TAG*/", N = 466;
const bands = Object.keys(META).map(Number);
const rad = {}; for (const b of bands) { const s = atob(DATA[b]), u8 = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u8[i] = s.charCodeAt(i); rad[b] = new Float32Array(u8.buffer); }
const eq = (m, L) => m.kappa0 ? (L * m.kappa0 * 100).toFixed(1) + " %" : L <= 0 ? "0 K" : ((m.planck_fk2 / Math.log(m.planck_fk1 / L + 1) - m.planck_bc1) / m.planck_bc2).toFixed(1) + " K";
const fmt = L => (+L.toPrecision(5)).toString();
const fresh = (name, band) => ({ name, band, lo: 0, hi: META[band].p98, inv: false });
const ch = [fresh("R", 13), fresh("G", 10), fresh("B", 8)];
let gammaOn = true;
const ctl = document.getElementById("controls");
function controls() {
  ctl.innerHTML = "";
  ch.forEach(c => {
    const m = META[c.band], d = document.createElement("div"); d.className = "ch";
    const top = m ? m.rmax * 1.1 : 1, step = top / 4000;
    d.innerHTML = `<h3>${c.name}: <select>${bands.map(b => `<option value="${b}" ${b === c.band ? "selected" : ""}>${b} · ${META[b].wl.toFixed(2)} µm · ${META[b].name}</option>`).join("")}<option value="0" ${c.band === 0 ? "selected" : ""}>off</option></select></h3>`
      + (c.band ? `<div class="unit">radiance, ${m.unit}</div>`
      + `<div class="row"><label>floor</label><input type="range" min="0" max="${top}" step="${step}" value="${c.lo}"><span class="val"></span></div>`
      + `<div class="row"><label>full</label><input type="range" min="0" max="${top}" step="${step}" value="${c.hi}"><span class="val"></span></div>`
      + `<div class="row"><label></label><label style="width:auto"><input type="checkbox" ${c.inv ? "checked" : ""}> invert</label></div>` : "");
    d.querySelector("select").onchange = e => { const b = +e.target.value; Object.assign(c, b ? fresh(c.name, b) : { band: 0 }); controls(); draw(); };
    const r = d.querySelectorAll("input[type=range]"), v = d.querySelectorAll(".val");
    const show = () => { v[0].textContent = `${fmt(c.lo)} (${eq(m, c.lo)})`; v[1].textContent = `${fmt(c.hi)} (${eq(m, c.hi)})`; };
    r.forEach((el, k) => el.oninput = () => { c[k ? "hi" : "lo"] = +el.value; show(); draw(); });
    if (c.band) show();
    const inv = d.querySelector("input[type=checkbox]"); if (inv) inv.onchange = () => { c.inv = inv.checked; draw(); };
    ctl.appendChild(d);
  });
  const g = document.createElement("div"); g.className = "ch";
  g.innerHTML = `<label><input type="checkbox" ${gammaOn ? "checked" : ""}> screen light ∝ radiance (γ 2.2 display encoding)</label>`;
  g.querySelector("input").onchange = e => { gammaOn = e.target.checked; draw(); };
  ctl.appendChild(g);
}
const view = document.getElementById("view"), vctx = view.getContext("2d"), img = vctx.createImageData(N, N);
// Size the canvas in device pixels, so one panel pixel lands on exactly one (or 2, 3) physical screen pixels whatever the display scaling.
function size() { const z = +document.getElementById("zoom").value, css = N * z / devicePixelRatio; view.style.width = view.style.height = css + "px"; }
document.getElementById("zoom").onchange = size; size();
const lut = new Uint8Array(4097); function buildLut() { for (let i = 0; i <= 4096; i++) lut[i] = Math.round(255 * (gammaOn ? Math.pow(i / 4096, 1 / 2.2) : i / 4096)); }
function draw() {
  buildLut();
  const px = img.data;
  for (let k = 0; k < 3; k++) {
    const c = ch[k];
    if (!c.band) { for (let i = 0; i < N * N; i++) px[i * 4 + k] = 0; continue; }
    const s = 4096 / (c.hi - c.lo), src = rad[c.band];
    for (let i = 0; i < N * N; i++) { let f = (src[i] - c.lo) * s; f = f > 0 ? (f > 4096 ? 4096 : f) : 0; if (c.inv) f = 4096 - f; px[i * 4 + k] = lut[f | 0]; }
  }
  for (let i = 0; i < N * N; i++) px[i * 4 + 3] = 255;
  vctx.putImageData(img, 0, 0);
  document.getElementById("spec").textContent = TAG + " " + ch.map(c => c.band ? `${c.name}=${c.band}${c.inv ? "i" : ""}:${fmt(c.lo)}-${fmt(c.hi)}` : `${c.name}=off`).join(" ");
}
let picked = null;
const th = document.getElementById("thumbs");
for (const b of bands) {
  const m = META[b], fig = document.createElement("figure"), cv = document.createElement("canvas"); cv.width = cv.height = N;
  const cx = cv.getContext("2d"), im = cx.createImageData(N, N), s = m.p98 > 0 ? 1 / m.p98 : 0;
  for (let i = 0; i < N * N; i++) { const f = Math.min(1, Math.max(0, rad[b][i] * s || 0)), v = Math.round(255 * Math.pow(f, 1 / 2.2)); im.data.set([v, v, v, 255], i * 4); }
  cx.putImageData(im, 0, 0);
  fig.appendChild(cv); fig.insertAdjacentHTML("beforeend", `<figcaption>${b} · ${m.wl.toFixed(2)} µm · ${m.name}</figcaption>`);
  fig.onclick = () => { picked = b; th.querySelectorAll("figure").forEach(f => f.style.outline = ""); fig.style.outline = "2px solid #888"; };
  th.appendChild(fig);
}
document.addEventListener("keydown", e => { const k = "rgb".indexOf(e.key.toLowerCase()); if (picked && k >= 0) { Object.assign(ch[k], fresh(ch[k].name, picked)); controls(); draw(); } });
controls(); draw();
</script></body></html>
"""

if __name__ == "__main__":
    if sys.argv[1:2] == ["panel"]:
        panel(sys.argv[2:])
    else:
        preview(sys.argv[1:])
