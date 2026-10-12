//! CO5300 AMOLED driver over quad SPI, as wired on the Waveshare ESP32-C6-Touch-AMOLED-1.32 (466×466).
//!
//! Protocol (CO5300 datasheet §5.2, matching Waveshare's ESP-IDF example): every transaction opens with an 8-bit opcode and a 24-bit address on one wire, the address carrying the register in its middle byte (`0x00_RR_00`). Opcode `0x02` writes a command's parameters on one wire; opcode `0x32` streams pixels on four wires after a memory-write (`0x2C`) address. The panel keeps its own frame memory and refreshes itself, so only changed rectangles are ever sent and the MCU holds no framebuffer.
//!
//! Chip select is a plain GPIO held low for a whole pixel stream, so a rectangle larger than one DMA buffer goes out as several transfers without re-sending the opcode — the same trick Espressif's driver uses.
//!
//! Windows must start on even coordinates and span an even number of pixels in each direction; Waveshare's example enforces this, most likely because the frame memory is compressed in 2×2 blocks.

use esp_hal::{
    Blocking,
    delay::Delay,
    gpio::Output,
    spi::master::{Address, Command, DataMode, SpiDma},
};

pub const WIDTH: u16 = 466;
pub const HEIGHT: u16 = 466;

/// The 466-column panel sits inside the driver's 480-column source range, starting at column 6.
const COL_OFFSET: u16 = 6;

const OP_WRITE: u16 = 0x02;
const OP_PIXELS_QUAD: u16 = 0x32;
const RAMWR: u8 = 0x2C;

/// Largest single DMA transfer: the SPI DMA's copy buffer (esp-hal: anything past 32736 bytes is wasted).
pub const MAX_TRANSFER: usize = 32736;

/// Interface pixel format (COLMOD 3Ah). Pixel bytes go high byte first; RGB888 is R, G, B.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
    Rgb565,
    Rgb888,
}

impl Format {
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Format::Rgb565 => 2,
            Format::Rgb888 => 3,
        }
    }
}

pub struct Co5300<'d> {
    spi: SpiDma<'d, Blocking>,
    cs: Output<'d>,
    streaming: bool,
    format: Format,
}

impl<'d> Co5300<'d> {
    /// Hardware-reset the panel and run Waveshare's init sequence: RGB565, brightness, the 466×466 window, sleep out, display on, rotated 180° (their default orientation, USB at the bottom).
    pub fn new(spi: SpiDma<'d, Blocking>, cs: Output<'d>, rst: &mut Output<'_>, delay: &Delay) -> Self {
        rst.set_high();
        delay.delay_millis(1);
        rst.set_low();
        delay.delay_millis(10);
        rst.set_high();
        delay.delay_millis(150);

        let mut d = Self { spi, cs, streaming: false, format: Format::Rgb565 };
        d.command(0xFE, &[0x00]); // command page 0
        d.command(0xC4, &[0x80]); // QSPI mode (vendor)
        d.command(0x3A, &[0x55]); // COLMOD: 16-bit RGB565
        d.command(0x35, &[0x00]); // tearing-effect output on (the pin isn't wired; harmless)
        d.command(0x53, &[0x20]); // WRCTRLD: brightness control on
        d.command(0x51, &[0xFF]); // brightness (high-brightness mode, 66h/63h, stays off)
        d.command(0x55, &[0x00]); // ACL (auto current limit) off — its power-on default, written so measurements never depend on it
        d.window(0, 0, WIDTH, HEIGHT);
        d.command(0x11, &[]); // sleep out
        delay.delay_millis(100);
        d.command(0x29, &[]); // display on
        d.command(0x36, &[0xC0]); // MADCTL: rotate 180°
        d
    }

    /// One command with its parameters, all on one wire.
    pub fn command(&mut self, reg: u8, params: &[u8]) {
        self.cs.set_low();
        self.spi
            .half_duplex_write(DataMode::Single, Command::_8Bit(OP_WRITE, DataMode::Single), Address::_24Bit((reg as u32) << 8, DataMode::Single), 0, params)
            .expect("CO5300 command write");
        self.cs.set_high();
    }

    /// Switch the interface pixel format; later pixel streams must use [`Format::bytes_per_pixel`] bytes per pixel.
    pub fn set_format(&mut self, format: Format) {
        self.command(0x3A, &[if format == Format::Rgb565 { 0x55 } else { 0x77 }]);
        self.format = format;
    }

    pub fn format(&self) -> Format {
        self.format
    }

    /// Panel brightness (WRDISBV, 51h), 0..=255. Measured with an i1Pro on a white window (2026-10-11): light ≈ ((level + 1) / 256)^2.29 — fitted γ 2.32/2.30/2.28/2.28 at levels 127/63/31/15, so one clean power law; 127 gives 20% of 255's light. The code is near-perceptual; for a target light fraction f, level = 256·f^(1/2.29) − 1. Pixel codes follow the same law, under a toe fixed in absolute light (see measurements/i1pro-2026-10-11/report.txt).
    pub fn set_brightness(&mut self, level: u8) {
        self.command(0x51, &[level]);
    }

    /// Set the write window to the `w`×`h` rectangle at (`x`, `y`). All four must be even.
    fn window(&mut self, x: u16, y: u16, w: u16, h: u16) {
        assert!(x % 2 == 0 && y % 2 == 0 && w % 2 == 0 && h % 2 == 0, "CO5300 windows must be even-aligned: {x},{y} {w}×{h}");
        assert!(w > 0 && h > 0 && x + w <= WIDTH && y + h <= HEIGHT, "window out of range: {x},{y} {w}×{h}");
        let (x0, x1) = (x + COL_OFFSET, x + COL_OFFSET + w - 1);
        let (y0, y1) = (y, y + h - 1);
        self.command(0x2A, &[(x0 >> 8) as u8, x0 as u8, (x1 >> 8) as u8, x1 as u8]);
        self.command(0x2B, &[(y0 >> 8) as u8, y0 as u8, (y1 >> 8) as u8, y1 as u8]);
    }

    /// Start streaming pixels into the `w`×`h` rectangle at (`x`, `y`) (even-aligned). Follow with [`Self::push`] calls totalling exactly w·h·[`Format::bytes_per_pixel`] bytes, then [`Self::end`].
    pub fn begin(&mut self, x: u16, y: u16, w: u16, h: u16) {
        self.window(x, y, w, h);
        self.cs.set_low();
        self.streaming = false;
    }

    /// Stream pixel bytes; the first transfer of a rectangle carries the quad-pixel opcode, later ones are bare data with chip select still held.
    pub fn push(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(MAX_TRANSFER) {
            let (cmd, addr) = if self.streaming {
                (Command::None, Address::None)
            } else {
                (Command::_8Bit(OP_PIXELS_QUAD, DataMode::Single), Address::_24Bit((RAMWR as u32) << 8, DataMode::Single))
            };
            self.spi.half_duplex_write(DataMode::Quad, cmd, addr, 0, chunk).expect("CO5300 pixel write");
            self.streaming = true;
        }
    }

    /// Finish the rectangle (release chip select).
    pub fn end(&mut self) {
        self.cs.set_high();
        self.streaming = false;
    }
}
