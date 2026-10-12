//! Reading flash through the cache instead of the ROM read routine (which manages ~5 MB/s): point a run of free MMU entries at the wanted flash pages, invalidate the cache over them, and read the mapped addresses like memory.
//!
//! The C6 has one 16 MiB linear window for instructions and data, 0x4200_0000.., in 256 entries of 64 KiB; the bootloader maps the app into the low entries, and this claims the highest free run.

use esp32c6::SPI0;

const PAGE: u32 = 0x1_0000;
const ENTRIES: u32 = 256;
const LINEAR_BASE: u32 = 0x4200_0000;

unsafe extern "C" {
    /// ROM: drop cached lines for a virtual address range.
    fn Cache_Invalidate_Addr(addr: u32, size: u32);
}

/// A run of MMU entries this code owns.
pub struct Window {
    first: u32,
    pages: u32,
}

impl Window {
    /// Claim the highest run of `pages` unused MMU entries, or `None` if there isn't one.
    pub fn claim(pages: u32) -> Option<Window> {
        let spi0 = unsafe { &*SPI0::ptr() };
        let mut run = 0;
        for e in (0..ENTRIES).rev() {
            let valid = critical_section::with(|_| {
                spi0.mmu_item_index().write(|w| unsafe { w.mmu_item_index().bits(e) });
                spi0.mmu_item_content().read().valid().bit()
            });
            run = if valid { 0 } else { run + 1 };
            if run == pages {
                return Some(Window { first: e, pages });
            }
        }
        None
    }

    /// Map the window's span of flash starting at page-aligned `paddr` and return it as bytes.
    pub fn map(&mut self, paddr: u32) -> &[u8] {
        debug_assert!(paddr % PAGE == 0);
        let spi0 = unsafe { &*SPI0::ptr() };
        critical_section::with(|_| {
            for i in 0..self.pages {
                spi0.mmu_item_index().write(|w| unsafe { w.mmu_item_index().bits(self.first + i) });
                spi0.mmu_item_content().write(|w| unsafe { w.paddr().bits((paddr / PAGE + i) as u16).valid().set_bit() });
            }
            unsafe { Cache_Invalidate_Addr(self.vaddr(), self.pages * PAGE) };
        });
        unsafe { core::slice::from_raw_parts(self.vaddr() as *const u8, (self.pages * PAGE) as usize) }
    }

    fn vaddr(&self) -> u32 {
        LINEAR_BASE + self.first * PAGE
    }
}
