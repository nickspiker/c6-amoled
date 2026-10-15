//! Firmware for the Waveshare ESP32-C6-Touch-AMOLED-1.32: plays a loop of frames stored in flash, and takes new frames and settings over Wi-Fi (TCP, after a keyed-BLAKE3 challenge) or the USB serial link, which also keeps the colorimetry patch commands.
//!
//! Line commands, the same on both links (numbers decimal, colour channels 0..=255):
//! - `s n [hash]` — followed by one raw RGB565 disk frame (only the pixels whose centres lie within 233 px of the panel's centre, row by row, high byte first: 341,160 bytes), stored in flash slot n (the `frames` partition, see partitions.csv). The flash work waits for the loop's hold on its newest frame, which stretches to cover it, so the animation never stutters; only that slot is erased and written. With `hash` (the frame's BLAKE3, 64 hex digits), the slot is read back from flash and hashed, and anything short of an exact match is answered `err s n hash …`. Storing into the playing loop's oldest slot moves the loop on by one, so the new frame plays last.
//! - `p start count ms [hold]` — play slots start, start+1, … wrapping within 0..count, one every `ms`, holding the last (newest) for `hold` ms before looping (default: `ms`). Kept in flash; the board resumes the loop by itself after a power cut.
//! - `h n` — the BLAKE3 of what slot n holds, to audit the ring
//! - `F len hash` — followed by a glyph blob (tools/measure `fontpack`) of `len` bytes, checked against its BLAKE3 and kept in flash
//! - `C len hash` — followed by the coastline overlay (`goes coast`) of `len` bytes, checked against its BLAKE3 and kept in flash; flickered ±16 codes along the coast, one flip per frame period (see `coast`)
//! - `info` — brightness level, signal strength (dBm, 0 when not connected), font, coastline overlay bytes, touch controller
//!
//! Touch: drag up or down anywhere to change the brightness (144 levels), shown as two dozenal digits while adjusting; the level is kept in flash.
//! - `ping`
//!
//! Over TCP (port 7766, `goes-puck.local`), the board first sends `hello <nonce>`, and nothing but `ping` works until the client answers `auth <BLAKE3 keyed with the shared key, of the nonce>`.
//!
//! USB only:
//! - `wifi <ssid> <password>` — store the network to join and restart; `key <64 hex digits>` — store the shared key; `net` — report the board's address
//! - `t n` — show slot n once, answering with the flash-map and draw times
//! - `f r g b` — fill the whole screen; `w s r g b` — an s×s square (s even) centred on black; `b n` — brightness register (51h); `m 16|24` — RGB565 or RGB888 interface format; `i` — followed by one raw RGB888 frame (466 × 466 × 3 bytes), drawn as it arrives. These stop the loop until the next `p`.
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

extern crate alloc;

mod coast;
mod net;
mod ui;

use core::fmt::Write as _;
use core::sync::atomic::{AtomicBool, Ordering};

use c6_amoled::co5300::{Co5300, Format, HEIGHT, WIDTH};
use c6_amoled::flashmap::Window;
use embassy_executor::Spawner;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_sync::signal::Signal;
use embassy_time::{Duration, Timer, with_timeout};
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
    Async,
};
use esp_println::println;
use esp_storage::FlashStorage;
use static_cell::{ConstStaticCell, StaticCell};

esp_bootloader_esp_idf::esp_app_desc!();

/// Rows rendered per band. Sized for RGB888 (466 × 32 × 3 = 44,736 bytes); RGB565 bands use the first two thirds and go out as one DMA transfer.
const BAND_ROWS: u16 = 32;
const BAND_BYTES: usize = WIDTH as usize * BAND_ROWS as usize * 3;
/// Word-aligned, since the ROM flash routines read and write through `*u32`.
#[repr(C, align(4))]
struct Band([u8; BAND_BYTES]);
static BAND: ConstStaticCell<Band> = ConstStaticCell::new(Band([0; BAND_BYTES]));
static COAST: ConstStaticCell<coast::Blob> = ConstStaticCell::new(coast::Blob([0; coast::SIZE as usize]));

/// First column inside the disk on row y: a pixel is in when its centre lies within 233 px of the panel's centre (the host uses the same test), worked in doubled coordinates so it stays integer. Rows are symmetric, so the span is `start..WIDTH - start`.
const fn disk_start(y: u16) -> u16 {
    let d = 2 * y as i32 + 1 - HEIGHT as i32;
    let r2 = WIDTH as i32 * WIDTH as i32;
    let mut x = 0;
    while x < WIDTH / 2 {
        let e = 2 * x as i32 + 1 - WIDTH as i32;
        if e * e + d * d <= r2 {
            break;
        }
        x += 1;
    }
    x
}

const DISK_START: [u16; HEIGHT as usize] = {
    let mut t = [0; HEIGHT as usize];
    let mut y = 0;
    while y < HEIGHT {
        t[y as usize] = disk_start(y);
        y += 1;
    }
    t
};

/// A stored frame: just the disk's pixels, row by row, RGB565 high byte first.
const FRAME_BYTES: u32 = {
    let mut n = 0;
    let mut y = 0;
    while y < HEIGHT as usize {
        n += (WIDTH - 2 * DISK_START[y]) as u32 * 2;
        y += 1;
    }
    n
};
const _: () = assert!(FRAME_BYTES % 4 == 0, "flash reads and writes go by whole words");

/// The `frames` partition (partitions.csv): frames packed back to back, so one more fits than if each started on a sector (a slot shares its first and last sector with its neighbours; see `write_slot`), then one sector for the play-settings log.
const FRAMES_BASE: u32 = 0xB_4000;
const FRAMES_SIZE: u32 = 0xF4_C000;
const SLOTS: u16 = ((FRAMES_SIZE - FlashStorage::SECTOR_SIZE) / FRAME_BYTES) as u16;
const _: () = assert!(SLOTS == 47, "partitions.csv no longer leaves room for 47 frames");
/// MMU pages to map one slot, which needn't start on a page.
const MAP_PAGES: u32 = FRAME_BYTES.div_ceil(FlashStorage::BLOCK_SIZE) + 1;
/// Rows per rectangle when covering the disk; even, as CO5300 windows must be.
const STRIP_ROWS: u16 = 32;

/// A loop: slots `start`, `start + 1`, … wrapping within `0..count`, one every `ms`, the last held for `hold` ms.
#[derive(Clone, Copy, PartialEq)]
struct PlaySettings {
    start: u16,
    count: u16,
    ms: u16,
    hold: u16,
}

/// The play settings survive power cuts as a log in the frames partition's tail, in the first whole sector past the last slot: 16-byte records appended in order, and once the sector is full it's erased and the log starts over, so it's erased once per 256 changes rather than once per change.
const LOG_BASE: u32 = (FRAMES_BASE + SLOTS as u32 * FRAME_BYTES).div_ceil(FlashStorage::SECTOR_SIZE) * FlashStorage::SECTOR_SIZE;
const LOG_END: u32 = FRAMES_BASE + FRAMES_SIZE;
const RECORD: u32 = 16;
const MAGIC: u32 = 0x504C_4159;
const _: () = assert!(LOG_END - LOG_BASE >= FlashStorage::SECTOR_SIZE, "no room left for the play-settings log");

impl PlaySettings {
    fn words(self) -> [u32; 4] {
        let (a, b) = (self.start as u32 | (self.count as u32) << 16, self.ms as u32 | (self.hold as u32) << 16);
        [MAGIC, a, b, a ^ b ^ 0xA5A5_A5A5]
    }

    fn from_words(w: [u32; 4]) -> Option<PlaySettings> {
        let s = PlaySettings { start: w[1] as u16, count: (w[1] >> 16) as u16, ms: w[2] as u16, hold: (w[2] >> 16) as u16 };
        (w[0] == MAGIC && w[3] == w[1] ^ w[2] ^ 0xA5A5_A5A5 && s.count >= 1 && s.count <= SLOTS && s.start < s.count).then_some(s)
    }
}

/// Scan the log: the last valid record and where the next one goes (the first erased word).
fn load_log(flash: &mut FlashStorage, band: &mut [u8]) -> (Option<PlaySettings>, u32) {
    let mut saved = None;
    let mut at = LOG_BASE;
    while at < LOG_END {
        let chunk = &mut band[..FlashStorage::SECTOR_SIZE as usize];
        if flash.read_nor(at, chunk).is_err() {
            return (saved, LOG_END);
        }
        for r in chunk.chunks_exact(RECORD as usize) {
            let w = [0, 4, 8, 12].map(|i| u32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]));
            if w[0] == u32::MAX {
                return (saved, at);
            }
            saved = PlaySettings::from_words(w).or(saved);
            at += RECORD;
        }
    }
    (saved, at)
}

/// Network, shared key and brightness, kept in the first sector of the `nvs` partition (nothing else here uses NVS).
#[derive(Clone)]
struct Settings {
    ssid: heapless::String<32>,
    password: heapless::String<64>,
    key: Option<[u8; 32]>,
    /// Brightness level (0..=143, see `ui::LEVELS`).
    level: u8,
}

const SETTINGS_ADDR: u32 = 0x9000;
const SETTINGS_MAGIC: u32 = 0x4E45_5431;
/// Magic, lengths, SSID (32 bytes), password (64), key (32).
const SETTINGS_WORDS: usize = 34;

impl Settings {
    fn load(flash: &mut FlashStorage) -> Settings {
        let mut w = [0u32; SETTINGS_WORDS];
        // SAFETY: the u32 array viewed as its bytes, word-aligned as the ROM read needs.
        let bytes = unsafe { core::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, SETTINGS_WORDS * 4) };
        let empty = Settings { ssid: heapless::String::new(), password: heapless::String::new(), key: None, level: ui::TOP_LEVEL };
        if flash.read_nor(SETTINGS_ADDR, bytes).is_err() || w[0] != SETTINGS_MAGIC {
            return empty;
        }
        let (ssid_len, pass_len, has_key) = ((w[1] & 0xFF) as usize, (w[1] >> 8 & 0xFF) as usize, w[1] >> 16 & 1 == 1);
        let ssid = core::str::from_utf8(&bytes[8..8 + ssid_len.min(32)]).ok().and_then(|s| s.try_into().ok());
        let password = core::str::from_utf8(&bytes[40..40 + pass_len.min(64)]).ok().and_then(|s| s.try_into().ok());
        let (Some(ssid), Some(password)) = (ssid, password) else {
            return empty;
        };
        let key = has_key.then(|| bytes[104..136].try_into().unwrap());
        // Stored as level + 1, so a sector written before brightness existed reads as full brightness.
        let level = match (w[1] >> 24) as u8 {
            0 => ui::TOP_LEVEL,
            n => (n - 1).min(ui::TOP_LEVEL),
        };
        Settings { ssid, password, key, level }
    }

    fn save(&self, flash: &mut FlashStorage) -> bool {
        let mut w = [0u32; SETTINGS_WORDS];
        // SAFETY: as in `load`.
        let bytes = unsafe { core::slice::from_raw_parts_mut(w.as_mut_ptr() as *mut u8, SETTINGS_WORDS * 4) };
        bytes[8..8 + self.ssid.len()].copy_from_slice(self.ssid.as_bytes());
        bytes[40..40 + self.password.len()].copy_from_slice(self.password.as_bytes());
        if let Some(k) = self.key {
            bytes[104..136].copy_from_slice(&k);
        }
        w[0] = SETTINGS_MAGIC;
        w[1] = self.ssid.len() as u32 | (self.password.len() as u32) << 8 | (self.key.is_some() as u32) << 16 | (self.level as u32 + 1) << 24;
        let bytes = unsafe { core::slice::from_raw_parts(w.as_ptr() as *const u8, SETTINGS_WORDS * 4) };
        flash.erase(SETTINGS_ADDR, SETTINGS_ADDR + FlashStorage::SECTOR_SIZE).is_ok() && flash.write_nor(SETTINGS_ADDR, bytes).is_ok()
    }
}

/// Everything the tasks share: the panel, the flash store and what's playing.
struct Board {
    lcd: Co5300<'static>,
    band: &'static mut [u8],
    flash: FlashStorage<'static>,
    window: Window,
    /// Next free record in the play-settings log, and the settings it last recorded.
    log: u32,
    saved: Option<PlaySettings>,
    /// The loop on screen; `None` while a patch command has the panel.
    playing: Option<PlaySettings>,
    settings: Settings,
    /// Brightness level now (saved to `settings` once the readout fades).
    level: u8,
    font: Option<ui::Font>,
    overlay: ui::Overlay,
    /// Slots that don't hold a whole frame (bit n for slot n): erased by a store that hasn't verified yet, or found ending in erased flash at boot. The player skips them, so a dropped upload never shows as white.
    incomplete: u64,
    /// The coastline overlay's strips (`coast`), and their length: 0 for none.
    coast: &'static mut [u8; coast::SIZE as usize],
    coast_len: usize,
    /// The coastline's phase on screen: +1 or −1 (× 16 codes). Each new frame flips it, and so does each flicker step in the hold.
    coast_sign: i32,
}

type Shared = Mutex<CriticalSectionRawMutex, Board>;
static BOARD: StaticCell<Shared> = StaticCell::new();
/// The player waits on this while stopped.
static PLAY: Signal<CriticalSectionRawMutex, ()> = Signal::new();
/// A store waiting for the loop's hold; the player answers with `AT_HOLD` and holds until `STORE_DONE`.
static STORE_WANTED: AtomicBool = AtomicBool::new(false);
static AT_HOLD: Signal<CriticalSectionRawMutex, ()> = Signal::new();
static STORE_DONE: Signal<CriticalSectionRawMutex, ()> = Signal::new();
/// The readout changed: the player redraws the frame on screen, even mid-hold.
static OVERLAY: Signal<CriticalSectionRawMutex, ()> = Signal::new();

impl Board {
    fn set_level(&mut self, level: u8) {
        self.level = level;
        self.lcd.set_brightness(ui::LEVELS[level as usize]);
        if let Some(font) = &self.font {
            self.overlay.set(font, level, ui::signal_digit(net::rssi()));
        }
    }

    fn save_level(&mut self) {
        if self.settings.level != self.level {
            self.settings.level = self.level;
            let s = self.settings.clone();
            s.save(&mut self.flash);
        }
    }

    /// Start (or change) the loop, recording it in the log unless it's what the log already holds.
    fn play(&mut self, s: PlaySettings) {
        self.playing = Some(s);
        PLAY.signal(());
        if self.saved == Some(s) {
            return;
        }
        if self.log + RECORD > LOG_END {
            if self.flash.erase(LOG_BASE, LOG_END).is_err() {
                return;
            }
            self.log = LOG_BASE;
        }
        let words = s.words();
        // SAFETY: four u32s viewed as their 16 bytes; the pointer stays word-aligned, as the ROM write needs.
        let bytes = unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, RECORD as usize) };
        if self.flash.write_nor(self.log, bytes).is_ok() {
            self.log += RECORD;
            self.saved = Some(s);
        }
    }

    /// The cache-mapped bytes of slot n.
    fn slot(&mut self, slot: u16) -> &[u8] {
        let base = FRAMES_BASE + slot as u32 * FRAME_BYTES;
        let page = base & !(FlashStorage::BLOCK_SIZE - 1);
        &self.window.map(page)[(base - page) as usize..][..FRAME_BYTES as usize]
    }

    /// Draw slot n on the panel (already in RGB565 mode) from the cache-mapped flash. The CO5300 only takes rectangles, so the disk goes out as strips of STRIP_ROWS rows, each as wide as its widest row (rounded out to even); each row's disk pixels are copied from flash into the band buffer between black margins. Returns the microseconds to map the slot and to draw it.
    fn show_slot(&mut self, slot: u16) -> (u64, u64) {
        let t = Instant::now();
        let base = FRAMES_BASE + slot as u32 * FRAME_BYTES;
        let page = base & !(FlashStorage::BLOCK_SIZE - 1);
        let src = &self.window.map(page)[(base - page) as usize..][..FRAME_BYTES as usize];
        let t2 = Instant::now();
        let (mut y, mut off) = (0, 0);
        while y < HEIGHT {
            let rows = STRIP_ROWS.min(HEIGHT - y);
            let x0 = DISK_START[y as usize..(y + rows) as usize].iter().copied().min().unwrap_or(0) & !1;
            let mut i = 0;
            for yy in y..y + rows {
                let s = DISK_START[yy as usize];
                let margin = (s - x0) as usize * 2;
                let n = (WIDTH - 2 * s) as usize * 2;
                self.band[i..i + margin].fill(0);
                self.band[i + margin..i + margin + n].copy_from_slice(&src[off..off + n]);
                self.band[i + margin + n..i + 2 * margin + n].fill(0);
                i += 2 * margin + n;
                off += n;
            }
            if self.overlay.fade > 0 {
                if let Some(font) = &self.font {
                    self.overlay.apply(font, &mut self.band[..i], x0, y, WIDTH - 2 * x0, rows);
                }
            } else {
                // The coastline goes up with the frame in its current phase. Strips start on even rows, as these bands do, so each lies wholly in one band.
                let w = WIDTH - 2 * x0;
                for (sy, sx, sw, data) in coast::strips(&self.coast[..self.coast_len]).skip_while(|s| s.0 < y).take_while(|s| s.0 < y + rows) {
                    for r in 0..2 {
                        for k in 0..sw {
                            let c = coast::coverage(data, sw, r, k);
                            if c != 0 && sx + k >= x0 && sx + k < WIDTH - x0 {
                                let at = ((sy + r - y) as usize * w as usize + (sx + k - x0) as usize) * 2;
                                coast::nudge(&mut self.band[at..at + 2], c, self.coast_sign);
                            }
                        }
                    }
                }
            }
            self.lcd.begin(x0, y, WIDTH - 2 * x0, rows);
            self.lcd.push(&self.band[..i]);
            self.lcd.end();
            y += rows;
        }
        ((t2 - t).as_micros(), t2.elapsed().as_micros())
    }

    /// Re-push slot n's coastline strips nudged by `sign` (the frame itself stays up): each strip's two rows from flash into the band buffer, black outside the disk, nudged, and sent as its own small window.
    fn coast_strips(&mut self, slot: u16, sign: i32) {
        let base = FRAMES_BASE + slot as u32 * FRAME_BYTES;
        let page = base & !(FlashStorage::BLOCK_SIZE - 1);
        let src = &self.window.map(page)[(base - page) as usize..][..FRAME_BYTES as usize];
        // Where row `row` starts in the slot, walked forward as the strips go down the disk.
        let (mut row, mut off) = (0u16, 0usize);
        for (sy, sx, sw, data) in coast::strips(&self.coast[..self.coast_len]) {
            while row < sy {
                off += (WIDTH - 2 * DISK_START[row as usize]) as usize * 2;
                row += 1;
            }
            let mut o = off;
            for r in 0..2 {
                let s = DISK_START[(sy + r) as usize];
                for k in 0..sw {
                    let (x, at) = (sx + k, (r * sw + k) as usize * 2);
                    let px = &mut self.band[at..at + 2];
                    if x >= s && x < WIDTH - s {
                        let from = o + (x - s) as usize * 2;
                        px.copy_from_slice(&src[from..from + 2]);
                    } else {
                        px.fill(0);
                    }
                    let c = coast::coverage(data, sw, r, k);
                    if c != 0 {
                        coast::nudge(px, c, sign);
                    }
                }
                o += (WIDTH - 2 * s) as usize * 2;
            }
            self.lcd.begin(sx, sy, sw, 2);
            self.lcd.push(&self.band[..sw as usize * 4]);
            self.lcd.end();
        }
    }
}

const fn rgb(r: u8, g: u8, b: u8) -> u16 {
    ((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3)
}

const BLACK: u16 = 0;
const WHITE: u16 = rgb(255, 255, 255);


#[allow(clippy::large_stack_frames, reason = "it's not unusual to allocate larger buffers etc. in main")]
#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    let peripherals = esp_hal::init(esp_hal::Config::default().with_cpu_clock(CpuClock::max()));
    // The Wi-Fi driver allocates; give it the RAM the bootloader used, plus a little more.
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 64 * 1024);
    esp_alloc::heap_allocator!(size: 32 * 1024);
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
    lcd.set_format(Format::Rgb888);
    let band = &mut BAND.take().0;
    fill_rgb(&mut lcd, band, 0, (0, 0, 0));
    let mut flash = FlashStorage::new(peripherals.FLASH);
    let mut window = Window::claim(MAP_PAGES).expect("no free MMU entries for a frame slot");
    // Self-check: the app image at 0x10000 starts with the ESP image magic, 0xE9.
    assert_eq!(window.map(0x1_0000)[0], 0xE9, "flash mapping isn't where expected");
    let (saved, log) = load_log(&mut flash, band);
    let settings = Settings::load(&mut flash);
    let font = load_font(&mut flash, band);
    let coast = &mut COAST.take().0;
    let coast_len = coast::load(&mut flash, coast);
    println!("CO5300 up: 466x466, QSPI 40 MHz; {SLOTS} frame slots of {FRAME_BYTES} bytes, read through the cache");
    if let Some(s) = saved {
        println!("resuming loop: {} frames from slot {}", s.count, s.start);
    }
    let wifi = settings.clone();
    println!("font: {}", if font.is_some() { "loaded" } else { "none (upload with `F`)" });
    let level = settings.level;
    let board = BOARD.init(Mutex::new(Board { lcd, band, flash, window, log, saved, playing: saved, settings, level, font, overlay: ui::Overlay::new(), incomplete: 0, coast, coast_len, coast_sign: 1 }));
    {
        // A store cut short (a power cut, a dropped link) leaves the end of its slot erased.
        let mut b = board.lock().await;
        for slot in 0..SLOTS {
            if b.slot(slot)[FRAME_BYTES as usize - 4..] == [0xFF; 4] {
                b.incomplete |= 1 << slot;
            }
        }
        if b.incomplete != 0 {
            println!("incomplete slots skipped: {:#x}", b.incomplete);
        }
    }
    board.lock().await.set_level(level);
    spawner.spawn(player(board).expect("player task"));
    let i2c = esp_hal::i2c::master::I2c::new(peripherals.I2C0, esp_hal::i2c::master::Config::default().with_frequency(Rate::from_khz(400)))
        .expect("I2C config")
        .with_sda(peripherals.GPIO8)
        .with_scl(peripherals.GPIO7);
    spawner.spawn(ui::touch(i2c, board).expect("touch task"));
    if wifi.ssid.is_empty() {
        println!("no Wi-Fi network set: `wifi <ssid> <password>` over USB");
    } else {
        net::start(&spawner, peripherals.WIFI, &wifi.ssid, &wifi.password, board);
    }

    let (rx, _tx) = UsbSerialJtag::new(peripherals.USB_DEVICE).into_async().split();
    let mut usb = Conn::new(Io::Usb(rx));
    println!("ready");
    loop {
        let _ = serve(&mut usb, board, true).await;
    }
}

/// Plays the loop: one frame every `ms`, then the newest held for `hold` — and, if a store is waiting, held until it's done, so all flash work happens while the picture is still. Touching the screen freezes it until the brightness readout has faded.
#[embassy_executor::task]
async fn player(board: &'static Shared) {
    let mut current: Option<PlaySettings> = None;
    let mut i = 0u16;
    let mut next = embassy_time::Instant::now();
    let mut on_screen = 0u16;
    // Wait for `done` (the hold, or a store during it), redrawing the frame on screen whenever the readout changes meanwhile.
    async fn wait<F: core::future::Future>(board: &'static Shared, slot: u16, done: F) {
        let mut done = core::pin::pin!(done);
        loop {
            match embassy_futures::select::select(&mut done, OVERLAY.wait()).await {
                embassy_futures::select::Either::First(_) => return,
                embassy_futures::select::Either::Second(()) => {
                    let mut b = board.lock().await;
                    if b.playing.is_some() {
                        b.show_slot(slot);
                    }
                }
            }
        }
    }
    // Until `until`, flip the coastline's phase every `step` (strips only), the true image their average. Each frame draws in the opposite phase to the last, so mid-animation the phase flips once a frame (a frame takes ~55 ms to draw of its ~66, leaving no room for a step between); in the hold it keeps that same pace, one flip per frame period. Skipped while the readout is up.
    async fn flicker(board: &'static Shared, slot: u16, until: embassy_time::Instant, step: Duration) {
        let mut at = embassy_time::Instant::now();
        loop {
            at += step;
            if at >= until {
                return Timer::at(until).await;
            }
            Timer::at(at).await;
            let mut b = board.lock().await;
            if b.coast_len > 0 && b.overlay.fade == 0 && b.playing.is_some() {
                let sign = -b.coast_sign;
                b.coast_sign = sign;
                b.coast_strips(slot, sign);
            }
        }
    }
    loop {
        // Frozen while the readout is up (a finger on the glass, then the readout lingering and fading): only the readout changes, redrawn on the frame on screen, and the animation picks up again once it has faded.
        if current.is_some() && board.lock().await.overlay.fade > 0 {
            loop {
                OVERLAY.wait().await;
                let mut b = board.lock().await;
                if b.playing.is_some() {
                    b.show_slot(on_screen);
                }
                if b.overlay.fade == 0 {
                    break;
                }
            }
            next = embassy_time::Instant::now();
        }
        let shown = {
            let mut b = board.lock().await;
            match b.playing {
                None => None,
                Some(s) => {
                    if current != Some(s) {
                        if current.is_none() {
                            b.lcd.set_format(Format::Rgb565);
                            next = embassy_time::Instant::now();
                        }
                        current = Some(s);
                        i = 0;
                    }
                    let slot = (s.start + i) % s.count;
                    // An incomplete slot is skipped: the previous frame simply stays up for its turn.
                    if b.incomplete & 1 << slot == 0 {
                        on_screen = slot;
                        b.coast_sign = -b.coast_sign;
                        b.show_slot(slot);
                    }
                    Some((s, i == s.count - 1))
                }
            }
        };
        let Some((s, last)) = shown else {
            current = None;
            PLAY.wait().await;
            continue;
        };
        if last {
            let held_from = embassy_time::Instant::now();
            if STORE_WANTED.load(Ordering::Acquire) {
                STORE_DONE.reset();
                AT_HOLD.signal(());
                wait(board, on_screen, STORE_DONE.wait()).await;
            }
            next = held_from + Duration::from_millis(s.hold as u64);
            i = 0;
        } else {
            next += Duration::from_millis(s.ms as u64);
            i += 1;
        }
        let now = embassy_time::Instant::now();
        if next > now {
            // Mid-animation the next frame picks up any readout change within one period; only the hold needs redraws of its own. Meanwhile the coastline flickers once a frame period.
            let step = Duration::from_millis(s.ms.max(1) as u64);
            if last {
                wait(board, on_screen, flicker(board, on_screen, next, step)).await;
            } else {
                flicker(board, on_screen, next, step).await;
            }
        } else {
            // Running late (drawing the readout costs time): catch up, but always let the other tasks run once per frame, or touch, USB and the network starve.
            next = now;
            embassy_futures::yield_now().await;
        }
    }
}

/// The packed glyphs from FONT_ADDR, if a valid blob is there (read through the word-aligned band buffer).
fn load_font(flash: &mut FlashStorage, band: &mut [u8]) -> Option<ui::Font> {
    flash.read_nor(ui::FONT_ADDR, &mut band[..8]).ok()?;
    let len = u32::from_le_bytes(band[4..8].try_into().ok()?);
    if &band[..4] != b"FNT1" || len > ui::FONT_SIZE || len % 4 != 0 {
        return None;
    }
    let mut blob = alloc::vec![0u8; len as usize];
    for at in (0..len as usize).step_by(4096) {
        let n = (len as usize - at).min(4096);
        flash.read_nor(ui::FONT_ADDR + at as u32, &mut band[..n]).ok()?;
        blob[at..at + n].copy_from_slice(&band[..n]);
    }
    ui::Font::parse(blob)
}

/// A byte stream the command protocol runs over: the USB serial link or a TCP connection. An enum rather than a trait, so the protocol code is compiled once instead of once per link (code space is tight).
enum Io<'a, 'b> {
    Usb(UsbSerialJtagRx<'static, Async>),
    Tcp(&'a mut embassy_net::tcp::TcpSocket<'b>),
}

impl Io<'_, '_> {
    /// Read at least one byte; `Err` once the link is gone.
    async fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()> {
        match self {
            Io::Usb(rx) => embedded_io_async::Read::read(rx, buf).await.map_err(|_| ()),
            Io::Tcp(socket) => socket.read(buf).await.map_err(|_| ()),
        }
    }

    async fn write(&mut self, mut data: &[u8]) -> Result<(), ()> {
        match self {
            Io::Usb(_) => {
                esp_println::Printer::write_bytes(data);
                Ok(())
            }
            Io::Tcp(socket) => {
                while !data.is_empty() {
                    let n = socket.write(data).await.map_err(|_| ())?;
                    data = &data[n..];
                }
                socket.flush().await.map_err(|_| ())
            }
        }
    }
}

/// Longest command line: `s nn` and a 64-digit hash, with room to spare.
const LINE: usize = 112;

/// One received command line.
struct Line {
    buf: [u8; LINE],
    len: usize,
}

impl Line {
    fn text(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

/// A link with a small read buffer, so a command line and the frame data right behind it can arrive in any chunking.
struct Conn<'a, 'b> {
    io: Io<'a, 'b>,
    buf: [u8; 256],
    at: usize,
    end: usize,
}

impl<'a, 'b> Conn<'a, 'b> {
    fn new(io: Io<'a, 'b>) -> Self {
        Conn { io, buf: [0; 256], at: 0, end: 0 }
    }

    async fn byte(&mut self) -> Result<u8, ()> {
        if self.at == self.end {
            let n = self.io.read(&mut self.buf).await?;
            if n == 0 {
                return Err(());
            }
            (self.at, self.end) = (0, n);
        }
        self.at += 1;
        Ok(self.buf[self.at - 1])
    }

    /// The next non-empty line, without its line ending; overlong lines are cut at LINE bytes.
    async fn line(&mut self) -> Result<Line, ()> {
        let mut line = Line { buf: [0; LINE], len: 0 };
        loop {
            match self.byte().await? {
                b'\n' | b'\r' if line.len > 0 => return Ok(line),
                b'\n' | b'\r' => {}
                b if line.len < LINE => {
                    line.buf[line.len] = b;
                    line.len += 1;
                }
                _ => {}
            }
        }
    }

    /// Fill `out` exactly: whatever's buffered first, then straight from the link.
    async fn read_exact(&mut self, out: &mut [u8]) -> Result<(), ()> {
        let have = (self.end - self.at).min(out.len());
        out[..have].copy_from_slice(&self.buf[self.at..self.at + have]);
        self.at += have;
        let mut got = have;
        while got < out.len() {
            let n = self.io.read(&mut out[got..]).await?;
            if n == 0 {
                return Err(());
            }
            got += n;
        }
        Ok(())
    }

    async fn reply(&mut self, args: core::fmt::Arguments<'_>) -> Result<(), ()> {
        let mut s = heapless::String::<160>::new();
        let _ = s.write_fmt(args);
        let _ = s.push('\n');
        self.io.write(s.as_bytes()).await
    }
}

/// Hex digits, lowercase.
struct Hex<'a>(&'a [u8]);

impl core::fmt::Display for Hex<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

/// Answer commands on one link until it drops. `usb` is the trusted local link: no challenge, and the setup and patch commands are allowed.
async fn serve(conn: &mut Conn<'_, '_>, board: &'static Shared, usb: bool) -> Result<(), ()> {
    let mut nonce = [0u8; 32];
    let mut authed = usb;
    if !usb {
        esp_hal::rng::Rng::new().read(&mut nonce);
        conn.reply(format_args!("hello {}", Hex(&nonce))).await?;
    }
    loop {
        let line = conn.line().await?;
        let Ok(text) = core::str::from_utf8(line.text()) else {
            conn.reply(format_args!("err utf8")).await?;
            continue;
        };
        let mut words = text.split_ascii_whitespace();
        let op = words.next().unwrap_or("");
        match op {
            "ping" => conn.reply(format_args!("ok ping")).await?,
            "auth" if !usb => {
                let key = board.lock().await.settings.key;
                let answer = words.next().and_then(unhex32);
                authed = matches!((key, answer), (Some(k), Some(a)) if blake3::keyed_hash(&k, &nonce) == blake3::Hash::from_bytes(a));
                conn.reply(format_args!("{} auth", if authed { "ok" } else { "err" })).await?;
            }
            _ if !authed => conn.reply(format_args!("err auth first")).await?,
            "s" => match parse_store(text) {
                Some((Some(slot), hash)) if slot < SLOTS => store(conn, board, slot, hash).await?,
                _ => conn.reply(format_args!("err {text}")).await?,
            },
            "p" => match numbers::<4>(words) {
                Some((a, n)) if (n == 3 || n == 4) && a[1] >= 1 && a[1] <= SLOTS && a[0] < a[1] => {
                    board.lock().await.play(PlaySettings { start: a[0], count: a[1], ms: a[2], hold: if n == 4 { a[3] } else { a[2] } });
                    conn.reply(format_args!("ok p")).await?;
                }
                _ => conn.reply(format_args!("err {text}")).await?,
            },
            "F" => {
                let (len, hash) = (words.next().and_then(|w| w.parse::<u32>().ok()), words.next().and_then(unhex32));
                match (len, hash) {
                    (Some(len), Some(hash)) if len % 4 == 0 && len <= ui::FONT_SIZE => store_font(conn, board, len, hash).await?,
                    _ => conn.reply(format_args!("err F <length, multiple of 4, at most {}> <blake3>", ui::FONT_SIZE)).await?,
                }
            }
            "C" => {
                let (len, hash) = (words.next().and_then(|w| w.parse::<u32>().ok()), words.next().and_then(unhex32));
                match (len, hash) {
                    (Some(len), Some(hash)) if len % 4 == 0 && len >= 12 && len <= coast::SIZE => store_coast(conn, board, len as usize, hash).await?,
                    _ => conn.reply(format_args!("err C <length, multiple of 4, at most {}> <blake3>", coast::SIZE)).await?,
                }
            }
            "info" => {
                let b = board.lock().await;
                conn.reply(format_args!("ok info level {} rssi {} font {} coast {} touch {:#04x} y {}", b.level, net::rssi().unwrap_or(0), b.font.is_some(), b.coast_len, ui::TOUCH_ID.load(Ordering::Relaxed), ui::TOUCH_Y.load(Ordering::Relaxed))).await?;
            }
            "h" => match numbers::<1>(words) {
                Some((a, 1)) if a[0] < SLOTS => {
                    let hash = blake3::hash(board.lock().await.slot(a[0]));
                    conn.reply(format_args!("ok h {} {}", a[0], Hex(hash.as_bytes()))).await?;
                }
                _ => conn.reply(format_args!("err {text}")).await?,
            },
            _ if !usb => conn.reply(format_args!("err {text}")).await?,
            "wifi" => {
                let (Some(ssid), Some(pass), None) = (words.next(), words.next(), words.next()) else {
                    conn.reply(format_args!("err wifi <ssid> <password>")).await?;
                    continue;
                };
                let mut b = board.lock().await;
                let (Ok(ssid), Ok(password)) = (ssid.try_into(), pass.try_into()) else {
                    conn.reply(format_args!("err wifi: ssid ≤ 32 bytes, password ≤ 64")).await?;
                    continue;
                };
                b.settings.ssid = ssid;
                b.settings.password = password;
                let ok = { let s = b.settings.clone(); s.save(&mut b.flash) };
                conn.reply(format_args!("{} wifi; restarting", if ok { "ok" } else { "err" })).await?;
                Timer::after(Duration::from_millis(100)).await;
                esp_hal::system::software_reset();
            }
            "key" => match (words.next().and_then(unhex32), words.next()) {
                (Some(k), None) => {
                    let mut b = board.lock().await;
                    b.settings.key = Some(k);
                    let ok = { let s = b.settings.clone(); s.save(&mut b.flash) };
                    conn.reply(format_args!("{} key", if ok { "ok" } else { "err" })).await?;
                }
                _ => conn.reply(format_args!("err key <64 hex digits>")).await?,
            },
            "net" => {
                let ip = net::address();
                conn.reply(format_args!("ok net {}", ip.map_or(heapless::String::<16>::new(), |a| { let mut s = heapless::String::new(); let _ = write!(s, "{a}"); s }))).await?;
            }
            "t" => match numbers::<1>(words) {
                Some((a, 1)) if a[0] < SLOTS => {
                    let mut b = board.lock().await;
                    b.playing = None;
                    b.lcd.set_format(Format::Rgb565);
                    let (map, draw) = b.show_slot(a[0]);
                    drop(b);
                    conn.reply(format_args!("ok t {} map {map} us draw {draw} us", a[0])).await?;
                }
                _ => conn.reply(format_args!("err {text}")).await?,
            },
            "i" => receive_image(conn, board).await?,
            _ => {
                let mut b = board.lock().await;
                b.playing = None;
                let b = &mut *b;
                command(&mut b.lcd, b.band, text);
            }
        }
    }
}

/// Up to N decimal arguments: the values and how many there were, or `None` if one isn't a number or there are too many.
fn numbers<'a, const N: usize>(words: impl Iterator<Item = &'a str>) -> Option<([u16; N], usize)> {
    let mut a = [0u16; N];
    let mut n = 0;
    for w in words {
        if n == N {
            return None;
        }
        a[n] = w.parse().ok()?;
        n += 1;
    }
    Some((a, n))
}

/// `s n [hash]`: `None` if the line isn't an `s` command; otherwise the slot (`None` if malformed) and the optional BLAKE3 hash.
fn parse_store(text: &str) -> Option<(Option<u16>, Option<[u8; 32]>)> {
    let mut words = text.split_ascii_whitespace();
    if words.next() != Some("s") {
        return None;
    }
    let slot = words.next().and_then(|w| w.parse::<u16>().ok());
    let hash = match words.next() {
        None => None,
        Some(hex) => match unhex32(hex) {
            Some(h) => Some(h),
            None => return Some((None, None)),
        },
    };
    if words.next().is_some() {
        return Some((None, None));
    }
    Some((slot, hash))
}

fn unhex32(hex: &str) -> Option<[u8; 32]> {
    let digits = hex.as_bytes();
    if digits.len() != 64 {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; 32];
    for (i, pair) in digits.chunks_exact(2).enumerate() {
        out[i] = nibble(pair[0])? << 4 | nibble(pair[1])?;
    }
    Some(out)
}

/// `F len hash`: receive a glyph blob (tools/measure `fontpack`), check its BLAKE3, write it to FONT_ADDR and use it. Answers `ok F` or `err F …`.
async fn store_font(conn: &mut Conn<'_, '_>, board: &'static Shared, len: u32, hash: [u8; 32]) -> Result<(), ()> {
    let mut blob = alloc::vec![0u8; len as usize];
    match with_timeout(Duration::from_secs(10), conn.read_exact(&mut blob)).await {
        Ok(r) => r?,
        Err(_) => return conn.reply(format_args!("err F stalled")).await,
    }
    if blake3::hash(&blob) != blake3::Hash::from_bytes(hash) {
        return conn.reply(format_args!("err F hash mismatch")).await;
    }
    let Some(font) = ui::Font::parse(blob) else {
        return conn.reply(format_args!("err F not a glyph blob")).await;
    };
    let mut guard = board.lock().await;
    let b = &mut *guard;
    // Drop the old font first: the heap is small.
    b.font = None;
    let mut ok = b.flash.erase(ui::FONT_ADDR, ui::FONT_ADDR + ui::FONT_SIZE).is_ok();
    let bytes = font.bytes();
    for at in (0..bytes.len()).step_by(4096) {
        let n = (bytes.len() - at).min(4096);
        // Through the word-aligned band buffer, as the ROM write needs.
        b.band[..n].copy_from_slice(&bytes[at..at + n]);
        ok &= b.flash.write_nor(ui::FONT_ADDR + at as u32, &b.band[..n]).is_ok();
    }
    b.font = Some(font);
    b.overlay.reset();
    let level = b.level;
    b.set_level(level);
    drop(guard);
    conn.reply(format_args!("{} F", if ok { "ok" } else { "err" })).await
}

/// `s n [hash]`: store one frame into slot n. The flash work waits for the loop to reach its hold on the newest frame (the player holds until it's done), so the animation never stutters; the frame's bytes stream in meanwhile as fast as the flash takes them. Answers `ok s n`, or `err s …`; the link only fails on a dropped connection.
async fn store(conn: &mut Conn<'_, '_>, board: &'static Shared, slot: u16, hash: Option<[u8; 32]>) -> Result<(), ()> {
    let playing = board.lock().await.playing.is_some();
    if playing {
        AT_HOLD.reset();
        STORE_WANTED.store(true, Ordering::Release);
        // A whole pass is at most 47 frames plus the hold; don't wait forever if the player is somehow stuck.
        let _ = with_timeout(Duration::from_secs(15), AT_HOLD.wait()).await;
    }
    let result = write_slot(conn, board, slot, hash).await;
    if playing {
        STORE_WANTED.store(false, Ordering::Release);
        STORE_DONE.signal(());
    }
    match result {
        Ok(Ok(())) => conn.reply(format_args!("ok s {slot}")).await,
        Ok(Err(why)) => conn.reply(format_args!("err s {slot} {why}")).await,
        Err(()) => Err(()),
    }
}

/// Erase slot n, stream FRAME_BYTES into it a 4 KiB page at a time, then (with a hash) read it back through the cache and check its BLAKE3. Storing into the playing loop's first slot (its oldest frame) moves the loop on by one, so the new frame plays last. Outer `Err`: the link failed; inner: why the store did.
async fn write_slot(conn: &mut Conn<'_, '_>, board: &'static Shared, slot: u16, hash: Option<[u8; 32]>) -> Result<Result<(), &'static str>, ()> {
    #[repr(C, align(4))]
    struct Page([u8; 4096]);
    let base = FRAMES_BASE + slot as u32 * FRAME_BYTES;
    {
        let mut b = board.lock().await;
        // Incomplete from the erase until the frame verifies; any failure below leaves it skipped.
        b.incomplete |= 1 << slot;
        if !erase_slot(&mut b, base) {
            return Ok(Err("erase failed"));
        }
    }
    let mut page = Page([0; 4096]);
    let mut got = 0;
    while got < FRAME_BYTES {
        let n = (FRAME_BYTES - got).min(4096) as usize;
        match with_timeout(Duration::from_secs(10), conn.read_exact(&mut page.0[..n])).await {
            Ok(r) => r?,
            Err(_) => return Ok(Err("stalled")),
        }
        if board.lock().await.flash.write_nor(base + got, &page.0[..n]).is_err() {
            return Ok(Err("write failed"));
        }
        got += n as u32;
    }
    let mut b = board.lock().await;
    if let Some(want) = hash {
        if blake3::hash(b.slot(slot)) != blake3::Hash::from_bytes(want) {
            return Ok(Err("hash mismatch"));
        }
    }
    b.incomplete &= !(1 << slot);
    if let Some(s) = b.playing {
        if s.start == slot {
            b.play(PlaySettings { start: (slot + 1) % s.count, ..s });
        }
    }
    Ok(Ok(()))
}

/// Erase the sectors under the frame at `base`. Slots are packed, so the first and last sector also hold the neighbouring frames' edges: those bytes are kept in the band buffer (free during the hold) and written straight back.
fn erase_slot(b: &mut Board, base: u32) -> bool {
    const SECTOR: u32 = FlashStorage::SECTOR_SIZE;
    let (first, end) = (base / SECTOR * SECTOR, (base + FRAME_BYTES).div_ceil(SECTOR) * SECTOR);
    let (head, tail) = ((base - first) as usize, (end - base - FRAME_BYTES) as usize);
    let tail_at = base + FRAME_BYTES;
    let band = &mut *b.band;
    // Both edges are whole words: sectors and frames are multiples of 4 bytes.
    let saved = b.flash.read_nor(first, &mut band[..head]).is_ok() && b.flash.read_nor(tail_at, &mut band[SECTOR as usize..SECTOR as usize + tail]).is_ok();
    saved
        && b.flash.erase(first, end).is_ok()
        && b.flash.write_nor(first, &band[..head]).is_ok()
        && b.flash.write_nor(tail_at, &band[SECTOR as usize..SECTOR as usize + tail]).is_ok()
}

/// `i`: read exactly one full frame of raw RGB888 (466 × 466 × 3 bytes, row-major, R G B) and stream it to the panel band by band as it arrives. Stops the loop. Answers `ok i`, or `err i …` if the panel isn't in 24-bit mode or the data stalls for 5 s (the partial frame stays on screen).
async fn receive_image(conn: &mut Conn<'_, '_>, board: &'static Shared) -> Result<(), ()> {
    let mut guard = board.lock().await;
    let b = &mut *guard;
    b.playing = None;
    if b.lcd.format() != Format::Rgb888 {
        return conn.reply(format_args!("err i needs m 24")).await;
    }
    let total = WIDTH as usize * HEIGHT as usize * 3;
    let band_bytes = WIDTH as usize * BAND_ROWS as usize * 3;
    b.lcd.begin(0, 0, WIDTH, HEIGHT);
    let mut got = 0;
    while got < total {
        let n = band_bytes.min(total - got);
        match with_timeout(Duration::from_secs(5), conn.read_exact(&mut b.band[..n])).await {
            Ok(r) => r?,
            Err(_) => {
                b.lcd.end();
                return conn.reply(format_args!("err i stalled at {got} of {total} bytes")).await;
            }
        }
        b.lcd.push(&b.band[..n]);
        got += n;
    }
    b.lcd.end();
    conn.reply(format_args!("ok i")).await
}

/// Run one of the USB patch commands, answering over USB.
fn command(lcd: &mut Co5300, band: &mut [u8], text: &str) {
    let mut words = text.split_ascii_whitespace();
    let op = words.next().unwrap_or("");
    let Some((args, count)) = numbers::<4>(words) else {
        println!("err {text}");
        return;
    };
    let channel = |v: u16| v.min(255) as u8;
    match (op, count) {
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

/// `C len hash`: receive the coastline overlay (`goes coast`) straight into its buffer, check its BLAKE3, keep it in flash at `coast::ADDR` and use it. Answers `ok C` or `err C …`; a bad upload falls back to the kept one. Holds the board while the bytes arrive, so the loop pauses for that moment.
async fn store_coast(conn: &mut Conn<'_, '_>, board: &'static Shared, len: usize, hash: [u8; 32]) -> Result<(), ()> {
    let mut guard = board.lock().await;
    let b = &mut *guard;
    b.coast_len = 0;
    let problem = match with_timeout(Duration::from_secs(10), conn.read_exact(&mut b.coast[..len])).await {
        Ok(Ok(())) if blake3::hash(&b.coast[..len]) != blake3::Hash::from_bytes(hash) => Some("hash mismatch"),
        Ok(Ok(())) if coast::length(&b.coast[..8]) != Some(len) => Some("not a coastline blob"),
        Ok(Ok(())) => None,
        Ok(Err(())) => {
            b.coast_len = coast::load(&mut b.flash, &mut *b.coast);
            return Err(());
        }
        Err(_) => Some("stalled"),
    };
    if let Some(problem) = problem {
        b.coast_len = coast::load(&mut b.flash, &mut *b.coast);
        drop(guard);
        return conn.reply(format_args!("err C {problem}")).await;
    }
    let ok = b.flash.erase(coast::ADDR, coast::ADDR + coast::SIZE).is_ok() && b.flash.write_nor(coast::ADDR, &b.coast[..len]).is_ok();
    b.coast_len = len;
    drop(guard);
    conn.reply(format_args!("{} C", if ok { "ok" } else { "err" })).await
}
