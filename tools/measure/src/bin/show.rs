//! Show a Rec.2020 image on the panel through its measured colour model, in VSF colorimetry: Rec.2020 (wavelength-defined) → VSF RGB → LMS (SS2000 10°, Illuminant E white) → panel RGB → per-channel codes. All of it runs here on the PC; the board only receives final codes over the `i` command.
//!
//! Usage: `show <image> <rec2020-linear|rec2020-1886|panel-linear> [white] [crop|stretch] [port|-] [888|565]`
//! - panel-linear: values are already linear drive per panel primary (light ∝ value); no colour matrices, no white scaling — each channel goes straight to its code
//! - white: E white's share of the panel's brightest E white (default 0.5), or `auto` to scale the image's brightest pixel to just fit
//! - 565 (panel-linear only): blue-noise ordered dither to the levels RGB565 can make, sent as their bit-replicated 8-bit codes (the same light true 565 would give)
//! - crop (default): centre square, area-averaged down to 466×466 in linear light; stretch: the whole image squeezed to 466×466 (for strips and charts)
//!
//! Also writes the codes it sent as `<image name>.codes.png` in the current directory, for inspection.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use measure::dither::{Levels, TILE, blue_noise};
use measure::link::Link;
use measure::model::PanelModel;
use vsf::colour::convert::apply_matrix_3x3_f32;
use vsf::colour::rec2020::constants::REC2020_2VSF_RGB;
use vsf::colour::spectral::constants::VSF_RGB2LMS;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../measurements");
const N: usize = 466;

/// A decoded image as non-linear-or-linear Rec.2020 values (as stored, normalised so integer full scale = 1).
struct Image {
    w: usize,
    h: usize,
    rgb: Vec<[f32; 3]>,
}

fn decode(path: &Path) -> Image {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "png" { decode_png(path) } else { decode_tiff(path) }
}

fn decode_tiff(path: &Path) -> Image {
    use tiff::decoder::{Decoder, DecodingResult, Limits};
    let mut d = Decoder::new(BufReader::new(File::open(path).expect("open image"))).expect("tiff header").with_limits(Limits::unlimited());
    let (w, h) = d.dimensions().expect("tiff dimensions");
    let ch = match d.colortype().expect("tiff colour type") {
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        other => panic!("unsupported TIFF colour type {other:?}"),
    };
    let samples: Vec<f32> = match d.read_image().expect("tiff data") {
        DecodingResult::U8(v) => v.into_iter().map(|x| x as f32 / 255.0).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x as f32 / 65535.0).collect(),
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
        _ => panic!("unsupported TIFF sample format"),
    };
    let rgb = samples.chunks_exact(ch).map(|p| [p[0], p[1], p[2]]).collect();
    Image { w: w as usize, h: h as usize, rgb }
}

fn decode_png(path: &Path) -> Image {
    let mut r = png::Decoder::new(BufReader::new(File::open(path).expect("open image"))).read_info().expect("png header");
    let mut buf = vec![0u8; r.output_buffer_size().expect("png size")];
    let info = r.next_frame(&mut buf).expect("png data");
    let ch = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        other => panic!("unsupported PNG colour type {other:?}"),
    };
    let deep = info.bit_depth == png::BitDepth::Sixteen;
    let sample = |i: usize| if deep { u16::from_be_bytes([buf[2 * i], buf[2 * i + 1]]) as f32 / 65535.0 } else { buf[i] as f32 / 255.0 };
    let (w, h) = (info.width as usize, info.height as usize);
    let rgb = (0..w * h).map(|p| [sample(p * ch), sample(p * ch + 1), sample(p * ch + 2)]).collect();
    Image { w, h, rgb }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: show <image> <rec2020-linear|rec2020-1886> [white] [crop|stretch] [port]");
    let path = Path::new(&path);
    let space = args.next();
    let linearise: fn(f32) -> f32 = match space.as_deref() {
        Some("rec2020-linear" | "panel-linear") => |v| v,
        Some("rec2020-1886") => |v| v.max(0.0).powf(2.4),
        other => panic!("colour space {other:?}: rec2020-linear | rec2020-1886 | panel-linear"),
    };
    let direct = space.as_deref() == Some("panel-linear");
    let white = args.next().unwrap_or_else(|| "0.5".into());
    let stretch = match args.next().as_deref() {
        None | Some("crop") => false,
        Some("stretch") => true,
        Some(other) => panic!("fit {other}: crop | stretch"),
    };
    let port = args.next().unwrap_or_else(|| "/dev/ttyACM0".into());
    let dither565 = match args.next().as_deref() {
        None | Some("888") => false,
        Some("565") => true,
        Some(other) => panic!("depth {other}: 888 | 565"),
    };

    let img = decode(path);
    println!("{}: {}×{}", path.display(), img.w, img.h);
    let model = PanelModel::load(&Path::new(ROOT).join("i1pro-2026-10-11"), &Path::new(ROOT).join("i1pro-2026-10-11-toe"));

    // Area-average to 466×466 in linear Rec.2020, then to LMS.
    let (sx, sy, sw, sh) = if stretch {
        (0, 0, img.w, img.h)
    } else {
        let s = img.w.min(img.h);
        ((img.w - s) / 2, (img.h - s) / 2, s, s)
    };
    let mut lin = vec![[0f64; 3]; N * N];
    for oy in 0..N {
        let (y0, y1) = (sy + oy * sh / N, (sy + ((oy + 1) * sh).div_ceil(N)).max(sy + oy * sh / N + 1).min(sy + sh));
        for ox in 0..N {
            let (x0, x1) = (sx + ox * sw / N, (sx + ((ox + 1) * sw).div_ceil(N)).max(sx + ox * sw / N + 1).min(sx + sw));
            let mut acc = [0f64; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    let p = img.rgb[y * img.w + x];
                    for c in 0..3 {
                        acc[c] += linearise(p[c]) as f64;
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f64;
            lin[oy * N + ox] = acc.map(|v| v / n);
        }
    }

    let mut codes = vec![0u8; N * N * 3];
    if direct {
        if dither565 {
            let mask = blue_noise();
            let levels = [0, 1, 2].map(|c| Levels::new(&model, c));
            // The same map shifted per channel, so the three channels' dots don't coincide.
            let shift = [(0, 0), (23, 41), (47, 13)];
            for (i, v) in lin.iter().enumerate() {
                let (x, y) = (i % N, i / N);
                for c in 0..3 {
                    let th = mask[(y + shift[c].1) % TILE * TILE + (x + shift[c].0) % TILE];
                    codes[i * 3 + c] = levels[c].pick(v[c].clamp(0.0, 1.0), th);
                }
            }
        } else {
            for (i, v) in lin.iter().enumerate() {
                for c in 0..3 {
                    codes[i * 3 + c] = model.code_for(c, v[c].clamp(0.0, 1.0));
                }
            }
        }
        return finish(path, &codes, &port);
    }
    let lms: Vec<[f64; 3]> = lin
        .iter()
        .map(|v| {
            let l = apply_matrix_3x3_f32(&VSF_RGB2LMS, &apply_matrix_3x3_f32(&REC2020_2VSF_RGB, &v.map(|x| x as f32)));
            l.map(|x| x as f64)
        })
        .collect();

    // Exposure: E white (LMS 1,1,1) at a fixed share of panel max, or the brightest pixel just fitting.
    let s = if white == "auto" {
        let peak = lms.iter().map(|&l| model.rgb_of(l).iter().cloned().fold(0.0, f64::max)).fold(0.0, f64::max);
        assert!(peak > 0.0, "image is black");
        1.0 / peak
    } else {
        model.white_scale(white.parse().expect("white fraction or auto"))
    };

    let mut clipped = 0usize;
    for (i, l) in lms.iter().enumerate() {
        let (c, clip) = model.encode(l.map(|v| v * s));
        codes[i * 3..i * 3 + 3].copy_from_slice(&c);
        clipped += clip as usize;
    }
    println!("E white at {:.3} of panel max; {:.1}% of pixels clipped to the panel gamut", s / model.white_scale(1.0), 100.0 * clipped as f64 / (N * N) as f64);
    finish(path, &codes, &port);
}

/// Save the codes as `<name>.codes.png` in the current directory and put them on the panel (unless the port is `-`).
fn finish(path: &Path, codes: &[u8], port: &str) {
    let name = path.file_stem().unwrap().to_string_lossy();
    let mut enc = png::Encoder::new(BufWriter::new(File::create(format!("{name}.codes.png")).unwrap()), N as u32, N as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(codes).unwrap();
    if port == "-" {
        println!("encoded only");
        return;
    }

    let mut link = Link::open(port);
    link.send("m 24");
    link.send("b 255");
    link.send_image(codes);
    println!("on the panel");
}
