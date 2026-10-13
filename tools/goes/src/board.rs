//! The c6-amoled board over Wi-Fi (`goes-puck.local`, TCP 7766): answer its challenge with the shared key, then line commands answered `ok …` or `err …`.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

const HOST: &str = "goes-puck.local:7766";
const KEY: &str = "/mnt/Harbor/Code/keys/goes-puck.key";

pub struct Board {
    reader: BufReader<TcpStream>,
    stream: TcpStream,
}

fn hex32(s: &str) -> Option<[u8; 32]> {
    let b = s.trim().as_bytes();
    if b.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, pair) in b.chunks_exact(2).enumerate() {
        out[i] = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(out)
}

impl Board {
    /// Connect and authenticate: the board sends `hello <nonce>`; the answer is the nonce's BLAKE3 keyed with the shared key.
    pub fn connect() -> Result<Board, String> {
        let key = hex32(&std::fs::read_to_string(KEY).map_err(|e| format!("key: {e}"))?).ok_or("bad key file")?;
        let stream = TcpStream::connect(HOST).map_err(|e| format!("connect {HOST}: {e}"))?;
        let mut board = Board { reader: BufReader::new(stream.try_clone().map_err(|e| e.to_string())?), stream };
        let hello = board.line(Duration::from_secs(15))?;
        let nonce = hello.strip_prefix("hello ").and_then(hex32).ok_or(format!("no challenge (got {hello:?})"))?;
        board.send(&format!("auth {}", blake3::keyed_hash(&key, &nonce).to_hex()))?;
        Ok(board)
    }

    fn line(&mut self, wait: Duration) -> Result<String, String> {
        self.stream.set_read_timeout(Some(wait)).ok();
        let mut s = String::new();
        match self.reader.read_line(&mut s) {
            Ok(0) => Err("link closed".into()),
            Ok(_) => Ok(s.trim().to_string()),
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn write(&mut self, data: &[u8]) -> Result<(), String> {
        self.stream.write_all(data).map_err(|e| e.to_string())
    }

    /// The next line starting `ok <prefix>`; an `err` line is an error. Other lines are the board's log.
    pub fn reply(&mut self, prefix: &str, wait_s: u64) -> Result<String, String> {
        let deadline = Instant::now() + Duration::from_secs(wait_s);
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            let text = self.line(left.max(Duration::from_millis(10)))?;
            if text.starts_with(&format!("ok {prefix}")) {
                return Ok(text);
            }
            if text.starts_with("err") {
                return Err(text);
            }
            if !text.is_empty() {
                println!("  board: {text}");
            }
        }
        Err(format!("no answer to {prefix}"))
    }

    pub fn send(&mut self, cmd: &str) -> Result<String, String> {
        self.write(format!("{cmd}\n").as_bytes())?;
        self.reply(cmd.split_whitespace().next().unwrap_or(""), 10)
    }

    /// Write one RGB565 disk frame into a flash slot; only that slot is erased. The board waits for the loop's hold to do the flash work, then reads the slot back and checks it against the frame's BLAKE3.
    pub fn store(&mut self, slot: usize, data: &[u8]) -> Result<(), String> {
        self.write(format!("s {slot} {}\n", blake3::hash(data).to_hex()).as_bytes())?;
        self.write(data)?;
        self.reply(&format!("s {slot}"), 60).map(|_| ())
    }
}
