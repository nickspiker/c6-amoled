"""The board's command link, from Python: line commands answered `ok …`, raw frame uploads into flash slots.

Two transports, one interface:
- a serial device path (`/dev/serial/by-id/…`): the USB link, trusted, also for setup (`wifi`, `key`)
- `tcp://host:port` (normally `tcp://goes-puck.local:7766`): Wi-Fi, after answering the board's challenge with a BLAKE3 keyed by the shared key in /mnt/Harbor/Code/keys/goes-puck.key
"""

import os
import secrets
import select
import socket
import termios
import time
import tty
from pathlib import Path

from blake3 import blake3

KEY = Path("/mnt/Harbor/Code/keys/goes-puck.key")


def shared_key():
    """The 32-byte key shared with the board, created on first use (readable only by this user)."""
    if not KEY.exists():
        KEY.parent.mkdir(parents=True, exist_ok=True)
        KEY.touch(mode=0o600)
        KEY.write_text(secrets.token_hex(32) + "\n")
    return bytes.fromhex(KEY.read_text().strip())


class Panel:
    def __init__(self, port):
        self.buf = b""
        self.sock = None
        if port.startswith("tcp://"):
            host, _, tcp_port = port[len("tcp://") :].rpartition(":")
            self.sock = socket.create_connection((host, int(tcp_port)), timeout=15)
            self.sock.settimeout(None)
            self.fd = self.sock.fileno()
            hello = self.line(10)
            if not hello or not hello.startswith("hello "):
                self.close()
                raise RuntimeError(f"no challenge from {port} (got {hello!r})")
            nonce = bytes.fromhex(hello.split()[1])
            self.send(f"auth {blake3(nonce, key=shared_key()).hexdigest()}")
            return
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        # Until it's raw, Linux echoes whatever arrives straight back to the board; drop anything that came in meanwhile.
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        # End whatever partial line the board may hold (a replug, a half-sent command), then wait for a clean answer to ping, riding out any errors the junk provokes.
        self.write(b"\n")
        for _ in range(8):
            try:
                if self.send("ping", wait=2, check=False):
                    return
            except RuntimeError:
                pass
        self.close()
        raise RuntimeError(f"no answer from {port}")

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        elif self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = None

    def fill(self, wait):
        """Read whatever has arrived within `wait` seconds into the line buffer."""
        r, _, _ = select.select([self.fd], [], [], wait)
        if r:
            data = os.read(self.fd, 65536)
            if not data:
                raise RuntimeError("link closed")
            self.buf += data

    def line(self, wait):
        """The next non-empty line, or None after `wait` seconds."""
        end = time.time() + wait
        while time.time() < end:
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                text = line.decode(errors="replace").strip()
                if text:
                    return text
            self.fill(0.2)
        return None

    def reply(self, prefix, wait):
        """The next line starting with `ok <prefix>` (returned) or `err` (raised); None on timeout. Other lines (the board's log) are printed."""
        end = time.time() + wait
        while (text := self.line(max(0.0, end - time.time()))) is not None:
            if text.startswith(f"ok {prefix}"):
                return text
            if text.startswith("err"):
                raise RuntimeError(text)
            print(f"  board: {text}")
        return None

    def write(self, data):
        """Write everything, draining whatever the board sends meanwhile into the line buffer, so neither side can block the other."""
        os.set_blocking(self.fd, False)
        try:
            v = memoryview(data)
            while v:
                r, w, _ = select.select([self.fd], [self.fd], [], 1.0)
                if r:
                    data = os.read(self.fd, 65536)
                    if not data:
                        raise RuntimeError("link closed")
                    self.buf += data
                if w:
                    try:
                        v = v[os.write(self.fd, v[:4096]) :]
                    except BlockingIOError:
                        pass
        finally:
            os.set_blocking(self.fd, True)

    def send(self, cmd, wait=10, check=True):
        self.write(cmd.encode() + b"\n")
        text = self.reply(cmd.split()[0], wait)
        if check and text is None:
            raise RuntimeError(f"no answer to {cmd!r}")
        return text

    def store(self, slot, rgb565):
        """Write one big-endian RGB565 frame into a flash slot; only that slot is erased. The board waits for the loop's hold to do the flash work, then reads the slot back and checks it against the frame's BLAKE3, refusing anything short of an exact copy."""
        self.write(f"s {slot} {blake3(rgb565).hexdigest()}\n".encode())
        self.write(rgb565)
        if self.reply(f"s {slot}", 60) is None:
            raise RuntimeError(f"no answer storing slot {slot}")
