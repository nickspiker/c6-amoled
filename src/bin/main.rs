//! Bring-up for the Waveshare ESP32-C6-Touch-AMOLED-1.32. Currently runs the colorimetry patch server: line commands over the USB serial link draw measurement patches, each answered with `ok` once the screen is updated, so a host script can step an i1Pro through them.
//!
//! Commands (numbers decimal, colour channels 0..=255):
//! - `f r g b` — fill the whole screen
//! - `w s r g b` — an s×s square (s even) centred on black
//! - `b n` — brightness register (51h)
//! - `m 16|24` — RGB565 or RGB888 interface format
//! - `i` — followed by one raw RGB888 frame (466 × 466 × 3 bytes), drawn as it arrives
//! - `ping`
//!
//! Also kept: the bring-up test patterns and brightness sweep. Patterns: colour bars (RGB order + byte order), ramps (16-bit banding), geometry (column offset + rotation), a single-pixel star field and a 1px checkerboard (does the CO5300's frame-memory compression smear fine detail?), then a moving square for partial-update speed.

#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use c6_amoled::co5300::{Co5300, Format, HEIGHT, WIDTH};
use embassy_executor::Spawner;
use embassy_time::{Duration, Timer};
use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    dma_rx_buffer,
    dma_tx_buffer,
    gpio::{Level, Output, OutputConfig},
    spi::{
        Mode,
        master::{Config, Spi},
    },
    time::{Instant, Rate},
    timer::timg::TimerGroup,
    usb::usb_serial_jtag::{UsbSerialJtag, UsbSerialJtagRx},
    Blocking,
};
use esp_println::println;
use static_cell::ConstStaticCell;

esp_bootloader_esp_idf::esp_app_desc!();

/// Rows rendered per band. Sized for RGB888 (466 × 32 × 3 = 44,736 bytes); RGB565 bands use the first two thirds and go out as one DMA transfer.
const BAND_ROWS: u16 = 32;
static BAND: ConstStaticCell<[u8; WIDTH as usize * BAND_ROWS as usize * 3]> = ConstStaticCell::new([0; WIDTH as usize * BAND_ROWS as usize * 3]);

const fn rgb(r: u8, g: u8, b: u8) -> u16 {
    ((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3)
}

const BLACK: u16 = 0;
const WHITE: u16 = rgb(255, 255, 255);

#[allow(clippy::large_stack_frames, reason = "it's not unusual to allocate larger buffers etc. in main")]
#[esp_rtos::main]
async fn main(_spawner: Spawner) -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);

    // GPIO2 latches the board's power on when running from battery; hold it high.
    let _power = Output::new(peripherals.GPIO2, Level::High, OutputConfig::default());

    let spi = Spi::new(peripherals.SPI2, Config::default().with_frequency(Rate::from_mhz(40)).with_mode(Mode::_0))
        .expect("SPI config")
        .with_sck(peripherals.GPIO18)
        .with_sio0(peripherals.GPIO19)
        .with_sio1(peripherals.GPIO20)
        .with_sio2(peripherals.GPIO10)
        .with_sio3(peripherals.GPIO11)
        .with_dma(peripherals.DMA_CH0)
        .with_buffers(dma_rx_buffer!(64).expect("rx buf"), dma_tx_buffer!(c6_amoled::co5300::MAX_TRANSFER).expect("tx buf"));
    let cs = Output::new(peripherals.GPIO22, Level::High, OutputConfig::default());
    let mut rst = Output::new(peripherals.GPIO21, Level::High, OutputConfig::default());
    let delay = Delay::new();
    let mut lcd = Co5300::new(spi, cs, &mut rst, &delay);
    let band = BAND.take();
    let (mut rx, _tx) = UsbSerialJtag::new(peripherals.USB_DEVICE).split();
    println!("CO5300 up: 466x466, QSPI 40 MHz");
    patch_server(&mut lcd, band, &mut rx).await
}

/// Read line commands from the USB serial link and draw measurement patches; see the module docs for the command set. Every command is answered with `ok …` once the panel holds the new image, or `err …`.
async fn patch_server(lcd: &mut Co5300<'_>, band: &mut [u8], rx: &mut UsbSerialJtagRx<'_, Blocking>) -> ! {
    lcd.set_format(Format::Rgb888);
    fill_rgb(lcd, band, 0, (0, 0, 0));
    println!("ready");
    let mut line = [0u8; 64];
    let mut n = 0;
    loop {
        match rx.read_byte() {
            Ok(b'\n' | b'\r') => {
                if n > 0 {
                    if &line[..n] == b"i" {
                        receive_image(lcd, band, rx);
                    } else {
                        command(lcd, band, &line[..n]);
                    }
                    n = 0;
                }
            }
            Ok(b) if n < line.len() => {
                line[n] = b;
                n += 1;
            }
            Ok(_) => {}
            Err(_) => Timer::after(Duration::from_millis(1)).await,
        }
    }
}

/// `i`: read exactly one full frame of raw RGB888 (466 × 466 × 3 bytes, row-major, R G B) from the serial link and stream it to the panel band by band as it arrives. Answers `ok i`, or `err i …` if the panel isn't in 24-bit mode or the data stalls for 5 s (the partial frame stays on screen).
fn receive_image(lcd: &mut Co5300, band: &mut [u8], rx: &mut UsbSerialJtagRx<'_, Blocking>) {
    if lcd.format() != Format::Rgb888 {
        println!("err i needs m 24");
        return;
    }
    let total = WIDTH as usize * HEIGHT as usize * 3;
    let band_bytes = WIDTH as usize * BAND_ROWS as usize * 3;
    lcd.begin(0, 0, WIDTH, HEIGHT);
    let (mut got, mut fill) = (0usize, 0usize);
    let mut last = Instant::now();
    while got < total {
        let want = (band_bytes - fill).min(total - got);
        let n = rx.drain_rx_fifo(&mut band[fill..fill + want]);
        if n == 0 {
            if last.elapsed().as_millis() > 5000 {
                lcd.end();
                println!("err i stalled at {got} of {total} bytes");
                return;
            }
            continue;
        }
        last = Instant::now();
        fill += n;
        got += n;
        if fill == band_bytes || got == total {
            lcd.push(&band[..fill]);
            fill = 0;
        }
    }
    lcd.end();
    println!("ok i");
}

/// Parse and run one patch-server command.
fn command(lcd: &mut Co5300, band: &mut [u8], line: &[u8]) {
    let Ok(text) = core::str::from_utf8(line) else {
        println!("err utf8");
        return;
    };
    let mut words = text.split_ascii_whitespace();
    let op = words.next().unwrap_or("");
    let mut args = [0u16; 4];
    let mut count = 0;
    for w in words {
        match w.parse::<u16>() {
            Ok(v) if count < args.len() => {
                args[count] = v;
                count += 1;
            }
            _ => {
                println!("err arg {w}");
                return;
            }
        }
    }
    let channel = |v: u16| v.min(255) as u8;
    match (op, count) {
        ("ping", 0) => {}
        ("f", 3) => fill_rgb(lcd, band, WIDTH, (channel(args[0]), channel(args[1]), channel(args[2]))),
        ("w", 4) if args[0] % 2 == 0 && args[0] >= 2 && args[0] <= WIDTH - 2 => fill_rgb(lcd, band, args[0], (channel(args[1]), channel(args[2]), channel(args[3]))),
        ("b", 1) => lcd.set_brightness(channel(args[0])),
        ("m", 1) if args[0] == 16 || args[0] == 24 => {
            lcd.set_format(if args[0] == 16 { Format::Rgb565 } else { Format::Rgb888 });
            fill_rgb(lcd, band, 0, (0, 0, 0));
        }
        _ => {
            println!("err {text}");
            return;
        }
    }
    println!("ok {text}");
}

/// Full-screen redraw in the panel's current format: an s×s square of `colour` centred on black (s = 0: all black; s = WIDTH: the whole screen).
fn fill_rgb(lcd: &mut Co5300, band: &mut [u8], s: u16, (r, g, b): (u8, u8, u8)) {
    let lo = (WIDTH - s.min(WIDTH)) / 2;
    let hi = lo + s.min(WIDTH);
    let format = lcd.format();
    let bpp = format.bytes_per_pixel();
    lcd.begin(0, 0, WIDTH, HEIGHT);
    let mut y = 0;
    while y < HEIGHT {
        let rows = BAND_ROWS.min(HEIGHT - y);
        let mut i = 0;
        for yy in y..y + rows {
            for x in 0..WIDTH {
                let (pr, pg, pb) = if x >= lo && x < hi && yy >= lo && yy < hi { (r, g, b) } else { (0, 0, 0) };
                match format {
                    Format::Rgb888 => {
                        band[i] = pr;
                        band[i + 1] = pg;
                        band[i + 2] = pb;
                    }
                    Format::Rgb565 => {
                        let c = rgb(pr, pg, pb);
                        band[i] = (c >> 8) as u8;
                        band[i + 1] = c as u8;
                    }
                }
                i += bpp;
            }
        }
        lcd.push(&band[..i]);
        y += rows;
    }
    lcd.end();
}

#[allow(dead_code)]
/// White screen with a large black readout of the exact value sent to WRDISBV (51h): 63, 127 and 255 (quarter, half and full code), 10 s each, for measuring the panel's brightness curve with a meter.
async fn brightness_sweep(lcd: &mut Co5300<'_>, band: &mut [u8]) {
    const LEVELS: [u8; 3] = [63, 127, 255];
    full_frame(lcd, band, "white", |_, _| WHITE);
    for level in LEVELS {
        lcd.set_brightness(level);
        draw_number(lcd, band, level as u16);
        println!("brightness {level:3} (0x{level:02X})");
        Timer::after(Duration::from_secs(10)).await;
    }
}

const DIGIT_W: u16 = 60;
const DIGIT_H: u16 = 110;
const DIGIT_GAP: u16 = 20;
const STROKE: u16 = 12;
const READOUT_W: u16 = 3 * DIGIT_W + 2 * DIGIT_GAP;
const READOUT_X: u16 = (WIDTH - READOUT_W) / 2 / 2 * 2; // even-aligned
const READOUT_Y: u16 = (HEIGHT - DIGIT_H) / 2 / 2 * 2;

/// Redraw the 3-digit readout box (black seven-segment digits on white, leading zeros blank) as one partial update.
fn draw_number(lcd: &mut Co5300, band: &mut [u8], n: u16) {
    let digits = [n / 100 % 10, n / 10 % 10, n % 10];
    let shown = [n >= 100, n >= 10, true];
    lcd.begin(READOUT_X, READOUT_Y, READOUT_W, DIGIT_H);
    let mut y = 0;
    while y < DIGIT_H {
        let rows = BAND_ROWS.min(DIGIT_H - y);
        let k = render(band, READOUT_W, y, rows, |x, yy| {
            let cell = x / (DIGIT_W + DIGIT_GAP);
            let cx = x % (DIGIT_W + DIGIT_GAP);
            if cell < 3 && cx < DIGIT_W && shown[cell as usize] && segment_lit(digits[cell as usize], cx, yy) { BLACK } else { WHITE }
        });
        lcd.push(&band[..k]);
        y += rows;
    }
    lcd.end();
}

/// Is (x, y) inside a lit segment of seven-segment digit `d` in a DIGIT_W × DIGIT_H cell?
fn segment_lit(d: u16, x: u16, y: u16) -> bool {
    // Segment bits: a top, b top-right, c bottom-right, d bottom, e bottom-left, f top-left, g middle.
    const MAP: [u8; 10] = [0b0111111, 0b0000110, 0b1011011, 0b1001111, 0b1100110, 0b1101101, 0b1111101, 0b0000111, 0b1111111, 0b1101111];
    let (w, h, t) = (DIGIT_W, DIGIT_H, STROKE);
    let segs = MAP[d as usize];
    let on = |bit: u8| segs & (1 << bit) != 0;
    let mid = h / 2;
    (on(0) && y < t)
        || (on(1) && x >= w - t && y < mid)
        || (on(2) && x >= w - t && y >= mid)
        || (on(3) && y >= h - t)
        || (on(4) && x < t && y >= mid)
        || (on(5) && x < t && y < mid)
        || (on(6) && y >= mid - t / 2 && y < mid + t / 2)
}

/// The bring-up test patterns, kept for re-runs; the loop above currently runs the brightness sweep instead.
#[allow(dead_code)]
async fn test_patterns(lcd: &mut Co5300<'_>, band: &mut [u8]) {
    loop {
        full_frame(lcd, band, "colour bars", colour_bars);
        Timer::after(Duration::from_secs(5)).await;
        full_frame(lcd, band, "ramps", ramps);
        Timer::after(Duration::from_secs(5)).await;
        full_frame(lcd, band, "geometry", geometry);
        Timer::after(Duration::from_secs(5)).await;
        full_frame(lcd, band, "star field", stars);
        Timer::after(Duration::from_secs(8)).await;
        full_frame(lcd, band, "fine detail", fine_detail);
        Timer::after(Duration::from_secs(8)).await;
        moving_square(lcd, band);
    }
}

/// Render and send the whole screen band by band from a pure per-pixel function; print render vs transfer time.
fn full_frame(lcd: &mut Co5300, band: &mut [u8], name: &str, px: fn(u16, u16) -> u16) {
    let t0 = Instant::now();
    let (mut render_us, mut send_us) = (0u64, 0u64);
    lcd.begin(0, 0, WIDTH, HEIGHT);
    let mut y = 0;
    while y < HEIGHT {
        let rows = BAND_ROWS.min(HEIGHT - y);
        let t = Instant::now();
        let n = render(band, WIDTH, y, rows, |x, yy| px(x, yy));
        render_us += t.elapsed().as_micros();
        let t = Instant::now();
        lcd.push(&band[..n]);
        send_us += t.elapsed().as_micros();
        y += rows;
    }
    lcd.end();
    let total = t0.elapsed().as_micros();
    let fps10 = 10_000_000 / total.max(1); // tenths of a frame per second, integer only (no FPU on the C6)
    println!("{name:12} frame {total:6} us  (render {render_us:6} us, transfer {send_us:6} us)  -> {}.{} fps max", fps10 / 10, fps10 % 10);
}

/// Fill `band` with `rows` rows of `w` RGB565 pixels starting at row `y0`, high byte first. Returns the byte count.
fn render(band: &mut [u8], w: u16, y0: u16, rows: u16, px: impl Fn(u16, u16) -> u16) -> usize {
    let mut i = 0;
    for y in y0..y0 + rows {
        for x in 0..w {
            let c = px(x, y);
            band[i] = (c >> 8) as u8;
            band[i + 1] = c as u8;
            i += 2;
        }
    }
    i
}

/// Eight vertical bars: white, yellow, cyan, green, magenta, red, blue, black. Swapped red/blue or a byte-order mistake shows immediately.
fn colour_bars(x: u16, _y: u16) -> u16 {
    const BARS: [u16; 8] = [WHITE, rgb(255, 255, 0), rgb(0, 255, 255), rgb(0, 255, 0), rgb(255, 0, 255), rgb(255, 0, 0), rgb(0, 0, 255), BLACK];
    BARS[(x as usize * 8 / WIDTH as usize).min(7)]
}

/// Four horizontal stripes ramping 0→255 left to right: red, green, blue, grey. RGB565 has 32 red/blue and 64 green levels, so the steps are visible on purpose.
fn ramps(x: u16, y: u16) -> u16 {
    let v = (x as u32 * 255 / (WIDTH as u32 - 1)) as u8;
    match y * 4 / HEIGHT {
        0 => rgb(v, 0, 0),
        1 => rgb(0, v, 0),
        2 => rgb(0, 0, v),
        _ => rgb(v, v, v),
    }
}

/// A 1px circle inscribed in the panel, 1px border rectangle (only its midpoints should be visible on a round panel), crosshair, a red dot at the top and a green dot at the right to read the orientation. Off-centre = wrong column offset.
fn geometry(x: u16, y: u16) -> u16 {
    let (cx, cy) = (x as i32 * 2 - 465, y as i32 * 2 - 465); // doubled coords centred on the panel
    let r2 = cx * cx + cy * cy; // (2r)²
    let ring = (r2 - 464 * 464).abs() < 2 * 464; // within ~0.5px of radius 232
    if x == 0 || y == 0 || x == WIDTH - 1 || y == HEIGHT - 1 || ring || x == 232 || x == 233 || y == 232 || y == 233 {
        return WHITE;
    }
    if (x as i32 - 232).abs() < 6 && (y as i32 - 20).abs() < 6 {
        return rgb(255, 0, 0); // top
    }
    if (x as i32 - 446).abs() < 6 && (y as i32 - 232).abs() < 6 {
        return rgb(0, 255, 0); // right
    }
    BLACK
}

/// Black sky with single-pixel stars at four brightnesses, a few 2×2 stars, and a dim one-pixel ring. If compression smears fine detail on black, it shows here.
fn stars(x: u16, y: u16) -> u16 {
    let h = hash(x as u32, y as u32);
    if h % 997 == 0 {
        let v = [255, 160, 90, 40][(h >> 12) as usize % 4];
        return rgb(v, v, v);
    }
    if (x % 64, y % 64) == (31, 31) || (x % 64, y % 64) == (32, 31) || (x % 64, y % 64) == (31, 32) || (x % 64, y % 64) == (32, 32) {
        return WHITE;
    }
    BLACK
}

/// Left half: 1px checkerboard (worst case for block compression). Right half: alternating 1px lines, vertical on top and horizontal below.
fn fine_detail(x: u16, y: u16) -> u16 {
    let on = if x < WIDTH / 2 {
        (x + y) % 2 == 0
    } else if y < HEIGHT / 2 {
        x % 2 == 0
    } else {
        y % 2 == 0
    };
    if on { WHITE } else { BLACK }
}

/// Partial updates: a 32×32 square bouncing around for 4 seconds, erasing its previous position each step. Prints steps per second.
fn moving_square(lcd: &mut Co5300, band: &mut [u8]) {
    const S: u16 = 32;
    lcd.begin(0, 0, WIDTH, HEIGHT);
    for _ in 0..(HEIGHT / BAND_ROWS + 1) {
        let n = render(band, WIDTH, 0, BAND_ROWS, |_, _| BLACK);
        lcd.push(&band[..n]);
    }
    lcd.end();
    let (mut x, mut y, mut dx, mut dy) = (100i32, 60i32, 6i32, 4i32);
    let t0 = Instant::now();
    let mut steps = 0u32;
    while t0.elapsed().as_millis() < 4000 {
        let (ox, oy) = (x, y);
        x += dx;
        y += dy;
        if x < 0 || x > (WIDTH - S) as i32 {
            dx = -dx;
            x += 2 * dx;
        }
        if y < 0 || y > (HEIGHT - S) as i32 {
            dy = -dy;
            y += 2 * dy;
        }
        let sq = |ax: i32, ay: i32, c: u16, lcd: &mut Co5300, band: &mut [u8]| {
            lcd.begin(ax as u16, ay as u16, S, S);
            let n = render(band, S, 0, S, |_, _| c);
            lcd.push(&band[..n]);
            lcd.end();
        };
        sq(ox, oy, BLACK, lcd, band);
        sq(x, y, rgb(255, 160, 0), lcd, band);
        steps += 1;
    }
    println!("moving square: {} steps in 4 s = {} updates/s (two 32x32 rects each)", steps, steps / 4);
}

/// Small integer hash for deterministic star placement.
fn hash(x: u32, y: u32) -> u32 {
    let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0xC2B2_AE3D);
    h ^ (h >> 13)
}
