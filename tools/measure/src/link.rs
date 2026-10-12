//! The serial link to the panel's patch server, and one-shot i1Pro captures through ArgyllCMS `spotread`.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

pub struct Link {
    port: Box<dyn serialport::SerialPort>,
    reader: BufReader<Box<dyn serialport::SerialPort>>,
}

impl Link {
    /// Open the board's USB serial link and wait until the patch server answers `ping` (opening the port can reset the board, so retry for a while).
    pub fn open(path: &str) -> Link {
        let port = serialport::new(path, 115_200).timeout(Duration::from_millis(500)).open().unwrap_or_else(|e| panic!("open {path}: {e}"));
        let reader = BufReader::new(port.try_clone().expect("clone serial port"));
        let mut link = Link { port, reader };
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(15) {
            if link.try_send("ping", Duration::from_secs(2)) {
                return link;
            }
        }
        panic!("patch server on {path} never answered ping");
    }

    /// Wait for the reply line `ok <cmd>`; false on timeout or `err`.
    fn await_ok(&mut self, cmd: &str, wait: Duration) -> bool {
        let start = Instant::now();
        let mut line = String::new();
        while start.elapsed() < wait {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => {}
                Ok(_) => {
                    let l = line.trim();
                    if l == format!("ok {cmd}") {
                        return true;
                    }
                    if l.starts_with("err") {
                        eprintln!("panel: {l}");
                        return false;
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                Err(e) => panic!("serial read: {e}"),
            }
        }
        false
    }

    /// Send one command and wait for its `ok`; false on timeout or `err`.
    pub fn try_send(&mut self, cmd: &str, wait: Duration) -> bool {
        self.port.write_all(format!("{cmd}\n").as_bytes()).expect("serial write");
        self.await_ok(cmd, wait)
    }

    pub fn send(&mut self, cmd: &str) {
        assert!(self.try_send(cmd, Duration::from_secs(10)), "panel never acknowledged `{cmd}`");
    }

    /// Upload one full RGB888 frame (466 × 466 × 3 bytes) with the `i` command.
    pub fn send_image(&mut self, rgb: &[u8]) {
        assert_eq!(rgb.len(), 466 * 466 * 3, "frame must be 466×466 RGB888");
        self.port.write_all(b"i\n").expect("serial write");
        self.port.write_all(rgb).expect("serial write");
        assert!(self.await_ok("i", Duration::from_secs(20)), "panel never acknowledged the image");
    }
}

/// Capture one i1Pro emissive spectrum into `<dir>/<name>.sp` with the given extra spotread flags; returns the file path. Only the spectrum is kept — spotread's own colorimetry is ignored.
pub fn capture(dir: &Path, name: &str, flags: &[&str]) -> PathBuf {
    let sp = dir.join(format!("{name}.sp"));
    let out = Command::new("spotread").args(["-e", "-N"]).args(flags).arg("-O").arg(&sp).stdin(std::process::Stdio::null()).output().expect("run spotread");
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    if text.contains("needs a calibration") {
        panic!("the i1Pro wants recalibrating: put it on its tile, run `spotread -e -H -O`, put it back and re-run");
    }
    assert!(sp.exists() && text.contains("succeeded"), "spotread didn't save a spectrum for {name}:\n{text}");
    sp
}
