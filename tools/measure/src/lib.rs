//! Shared pieces of the c6-amoled measurement tools: the serial link to the panel's patch server and the i1Pro ([`link`]), emissive spectra and their cone responses ([`spectra`]), and the panel's colour model ([`model`]).
//!
//! Colorimetry is VSF's throughout: Stockman & Sharpe 2000 10° cone fundamentals (sum-normalised, so equal-energy light is LMS = 1, 1, 1), Illuminant E white, VSF RGB and wavelength-defined Rec.2020, AGB ratios for chromaticity. CIE 1931 appears nowhere; spotread is used only to capture spectra.

pub mod link;
pub mod model;
pub mod spectra;
