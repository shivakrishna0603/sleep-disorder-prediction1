# ============================================================
# ble_watch.py — Live BLE collector for Fireboltt 046 / Da Fit
# Sleep Disorder Prediction Using Wearable Sensor Data
#
# Implements the Moyoung / Da Fit BLE V1+V2 packet protocol
# (reverse-engineered by Gadgetbridge) so live watch data can be
# captured straight from the laptop's Bluetooth adapter and fed
# through app/smartwatch_import.py -> model features -> prediction.
#
# Data is persisted to a local SQLite DB whose table/column names
# are auto-detected by smartwatch_import.load_records(), so nothing
# in the existing parse/evaluate pipeline needs to change.
#
# Protocol notes (from Gadgetbridge sources):
#   - packets are framed  FE EA <size_hi+32*|16> <size_lo> <cmd> <payload>
#     where full size = 5 + len(payload);  *V2 only, else packet[2]=16
#   - write char:  cmd==1 -> FEE5, cmd==2 -> FEE6, otherwise FEE2
#   - notifications arrive on FEE3 (data in) and FEE1 (realtime steps)
#   - live HR = poll loop cmd 0x6D {0}; response {bpm} (~2-5s per reading)
#   - SpO2 cmd 0x6B {0} -> {percent};   BP cmd 0x69 {0,0,0} -> {unused,sys,dia}
#   - sleep/steps/HR history via commands 0x32/0x33/0x35/0x36/0x37
#   - watch stores time as epoch seconds in GMT+8
# ============================================================

import asyncio
import logging
import os
import re
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

try:
    import bleak
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except Exception:  # pragma: no cover - allow import without hardware
    BLEAK_AVAILABLE = False

sync_session_available = BLEAK_AVAILABLE

log = logging.getLogger("ble_watch")

# ── GATT definitions ──────────────────────────────────────────
UUID_SERVICE_MOYOUNG = "0000feea-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_STEPS = "0000fee1-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_DATA_OUT = "0000fee2-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_DATA_IN = "0000fee3-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_DATA_SPECIAL_1 = "0000fee5-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_DATA_SPECIAL_2 = "0000fee6-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_BATTERY = "00002a19-0000-1000-8000-00805f9b34fb"
UUID_SERVICE_DEVICE_INFO = "0000180a-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_MFR_NAME = "00002a29-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_MODEL = "00002a24-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_FW_VERSION = "00002a26-0000-1000-8000-00805f9b34fb"
UUID_CHARACTERISTIC_HW_VERSION = "00002a27-0000-1000-8000-00805f9b34fb"

# ── Commands (MoyoungConstants) ───────────────────────────────
CMD_SYNC_TIME = 0x31                       # 49  {ts_be, 0x08}
CMD_SYNC_SLEEP = 0x32                      # 50  {} -> {type,h,m}... (today's sleep)
CMD_SYNC_PAST_SLEEP_AND_STEP = 0x33        # 51  {arg} -> mixed data
ARG_SYNC_YESTERDAY_STEPS = 0x01
ARG_SYNC_DAY_BEFORE_YESTERDAY_STEPS = 0x02
ARG_SYNC_YESTERDAY_SLEEP = 0x03
ARG_SYNC_DAY_BEFORE_YESTERDAY_SLEEP = 0x04
CMD_QUERY_TIMING_MEASURE_HEART_RATE = 47   # (*) unused, kept for reference
CMD_QUERY_LAST_DYNAMIC_RATE = 52
CMD_QUERY_PAST_HEART_RATE_1 = 53           # 0x35  {i} -> {i, 72 samples x every 5min}
CMD_QUERY_PAST_HEART_RATE_2 = 54           # 0x36  {i} -> {i, 72 samples x every 1min}
CMD_QUERY_MOVEMENT_HEART_RATE = 55         # 0x37  {} -> 3 workout summaries
CMD_TRIGGER_MEASURE_BLOOD_PRESSURE = 105   # 0x69  {0,0,0} -> {unused, sys, dia}
CMD_TRIGGER_MEASURE_BLOOD_OXYGEN = 107     # 0x6b  {0} -> {percent}
CMD_TRIGGER_MEASURE_HEARTRATE = 109        # 0x6d  {0} -> {bpm}
CMD_QUERY_V2_WORKOUT = 0xB2
CMD_QUERY_V2_WORKOUT_LIST_REQUEST = 0x00
CMD_QUERY_V2_WORKOUT_LIST_RESPONSE = 0x01
CMD_QUERY_V2_WORKOUT_DETAIL_REQUEST = 0x02
CMD_QUERY_V2_WORKOUT_DETAIL_RESPONSE = 0x03
CMD_QUERY_V2_WORKOUT_HR_REQUEST = 0x04
CMD_QUERY_V2_WORKOUT_HR_RESPONSE = 0x05
CMD_ADVANCED_QUERY = 0xB9
CMD_ADVANCED_CMD = 0xBB
ARG_ADVANCED_STRESS_PACKET = 0x11
CMD_RETURN_PRINCIPAL_SCREEN = 83

_WATCH_TZ = timezone(timedelta(hours=8))   # GMT+8, no DST

# Payload matching for name-based scan
_NAME_PATTERNS = [
    r"boltt", r"\b046", r"dafit", r"da\s?fit", r"moyoung", r"danze?",
    r"iwo", r"gooult|goboult", r"c21|colmi", r"smart", r"band", r"watch",
    r"firebolt", r"jodu",
]
_ADDR_PATTERNS = [r"(?i)^[0-9a-f]{12}$"]


def watch_to_local(secs: int) -> datetime:
    """Convert a watch epoch (GMT+8) to naive local time."""
    try:
        return datetime.fromtimestamp(int(secs) - 8 * 3600)
    except (ValueError, OverflowError, OSError):
        return datetime.now()


def local_to_watch(dt: Optional[datetime] = None) -> int:
    """Local (naive) datetime -> watch epoch seconds (GMT+8)."""
    dt = dt or datetime.now()
    return int(dt.timestamp()) + 8 * 3600


# ============================================================
# Framing
# ============================================================

def build_packet(mtu: int, cmd: int, payload: bytes = b"") -> bytes:
    """Encode a full Moyoung packet (V1 fixed or V2 MTU-based header)."""
    packet = bytearray(5 + len(payload))
    packet[0] = 0xFE
    packet[1] = 0xEA
    if mtu == 20:
        packet[2] = 16
    else:
        packet[2] = (32 + ((len(packet)) >> 8)) & 0xFF
    packet[3] = len(packet) & 0xFF
    packet[4] = cmd & 0xFF
    packet[5:] = payload
    return bytes(packet)


def split_fragments(packet: bytes, mtu: int) -> List[bytes]:
    """Split a built packet into GATT write chunks of at most mtu bytes."""
    if mtu <= 0:
        mtu = 20
    return [packet[i:i + mtu] for i in range(0, len(packet), mtu)]


def parse_packet_length(fragment: bytes) -> int:
    """Parse full logical length from the first fragment, or None."""
    if len(fragment) < 2 or fragment[0] != 0xFE or fragment[1] != 0xEA:
        return None
    b2 = fragment[2]
    if b2 == 16:
        if len(fragment) < 4:
            return None
        return fragment[3]
    if (b2 & 0xFF) < 32:
        return None
    if len(fragment) < 4:
        return None
    return (((b2 & 0xFF) - 32) << 8) | (fragment[3] & 0xFF)


class PacketReassembler:
    """Reassemble a fragmented incoming Moyoung packet on FEE3."""

    def __init__(self):
        self._buf = bytearray()
        self._total = None

    def feed(self, fragment: bytes) -> Optional[Tuple[int, bytes]]:
        if not fragment:
            return None
        if self._total is None:
            total = parse_packet_length(fragment)
            if total is None:
                return None
            self._total = total
            self._buf = bytearray()
        self._buf += fragment
        if len(self._buf) < self._total:
            return None
        if len(self._buf) > self._total:
            self._buf = self._buf[: self._total]
        raw = bytes(self._buf)
        self._buf = bytearray()
        self._total = None
        if len(raw) < 5:
            return None
        return raw[4], raw[5:]

    def reset(self):
        self._buf = bytearray()
        self._total = None


# ============================================================
# SQLite persistence — schema-compatible with smartwatch_import
# ============================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS heart_rate (time INTEGER NOT NULL, bpm INTEGER);
CREATE TABLE IF NOT EXISTS steps (time INTEGER NOT NULL, count INTEGER);
CREATE TABLE IF NOT EXISTS spo2 (time INTEGER NOT NULL, value INTEGER);
CREATE TABLE IF NOT EXISTS blood_pressure (time INTEGER NOT NULL, systolic INTEGER, diastolic INTEGER);
CREATE TABLE IF NOT EXISTS stress (time INTEGER NOT NULL, value INTEGER);

-- one row per night: auto-detected by smartwatch_import as 'sleep'
CREATE TABLE IF NOT EXISTS sleep (
    start_time INTEGER NOT NULL,
    end_time  INTEGER,
    quality   INTEGER,
    deep_min  INTEGER,
    light_min INTEGER,
    rem_min   INTEGER
);

-- raw sleep stage segments {type,start}  (table name avoids 'sleep' bucket)
CREATE TABLE IF NOT EXISTS stages (
    start_time INTEGER NOT NULL,
    stage      INTEGER NOT NULL
);

-- workouts -> 'activity' bucket
CREATE TABLE IF NOT EXISTS activity (
    start_time INTEGER NOT NULL,
    end_time   INTEGER,
    kind       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_hr_time ON heart_rate(time);
CREATE INDEX IF NOT EXISTS idx_steps_time ON steps(time);
CREATE INDEX IF NOT EXISTS idx_spo2_time ON spo2(time);
CREATE INDEX IF NOT EXISTS idx_stress_time ON stress(time);
"""


@dataclass
class CaptureDB:
    path: str

    def __post_init__(self):
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(_SCHEMA)
        self._lock = asyncio.Lock()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    async def _execute(self, sql, params=()):
        async with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    async def add_heart_rate(self, ts, bpm):
        if bpm is None or bpm <= 0 or bpm > 255:
            return
        await self._execute(
            "INSERT INTO heart_rate(time, bpm) VALUES (?,?)", (int(ts), int(bpm)))

    async def add_spo2(self, ts, value):
        if value is None or value <= 0 or value >= 255:
            return
        await self._execute(
            "INSERT INTO spo2(time, value) VALUES (?,?)", (int(ts), int(value)))

    async def add_bp(self, ts, systolic, diastolic):
        if systolic is None or diastolic is None or systolic <= 0 or diastolic <= 0:
            return
        await self._execute(
            "INSERT INTO blood_pressure(time, systolic, diastolic) VALUES (?,?,?)",
            (int(ts), int(systolic), int(diastolic)))

    async def add_stress(self, ts, value):
        if value is None or value <= 0 or value >= 255:
            return
        await self._execute(
            "INSERT INTO stress(time, value) VALUES (?,?)", (int(ts), int(value)))

    async def set_daily_steps(self, count):
        """Persist today's cumulative steps (one row per day, upserted)."""
        day_start = int(datetime.now().replace(hour=0, minute=0, second=0,
                                              microsecond=0).timestamp())
        async with self._lock:
            self._conn.execute(
                "DELETE FROM steps WHERE time BETWEEN ? AND ?",
                (day_start, day_start + 86399))
            self._conn.execute(
                "INSERT INTO steps(time, count) VALUES (?,?)", (day_start, int(count)))
            self._conn.commit()

    async def add_past_steps(self, days_ago, count):
        day_start = int((datetime.now() - timedelta(days=days_ago))
                        .replace(hour=0, minute=0, second=0, microsecond=0)
                        .timestamp())
        await self._execute(
            "DELETE FROM steps WHERE time BETWEEN ? AND ?",
            (day_start, day_start + 86399))
        await self._execute(
            "INSERT INTO steps(time, count) VALUES (?,?)", (day_start, int(count)))

    async def add_heart_rate_rows(self, rows):
        """Bulk insert [(ts, bpm), ...]."""
        for ts, bpm in rows:
            await self.add_heart_rate(ts, bpm)

    async def add_sleep_stage(self, start_ts, stage):
        await self._execute(
            "INSERT INTO stages(start_time, stage) VALUES (?,?)",
            (int(start_ts), int(stage)))

    async def add_activity(self, start_ts, end_ts, kind=None):
        await self._execute(
            "INSERT INTO activity(start_time, end_time, kind) VALUES (?,?,?)",
            (int(start_ts), int(end_ts), kind))

    async def finish_night(self, day: str):
        """Aggregate today's segments into one `sleep` row."""
        async with self._lock:
            rows = self._conn.execute(
                "SELECT start_time, stage FROM stages ORDER BY start_time").fetchall()
        if not rows:
            return None
        starts = [r[0] for r in rows]
        night_start = min(starts)
        night_end = _estimate_wake_time(rows)
        dur_min = max(1, (night_end - night_start) / 60.0)
        deep_min = sum(int(r[0] and (r[1] == 2)) for r in rows) * 30  # crude stage length

        deep_ratio = 0.0
        # Precise: a stage lasts until the next stage starts.
        seg = [(starts[i], starts[i + 1] if i + 1 < len(starts) else night_end)
               for i in range(len(starts))]
        deep_min = 0
        rem_min = 0
        light_min = 0
        for i, (a, b) in enumerate(seg):
            mins = max(0, (b - a) / 60.0)
            st = rows[i][1]
            if st == 2:
                deep_min += mins
            elif st == 1:
                light_min += mins
        dur_min = max(1, (night_end - night_start) / 60.0)
        if dur_min > 0:
            deep_ratio = deep_min / dur_min
        quality = int(round(min(10, max(1, 5 + 9 * deep_ratio)))) if deep_ratio else None
        async with self._lock:
            self._conn.execute("DELETE FROM sleep WHERE start_time = ?", (int(night_start),))
            self._conn.execute(
                "INSERT INTO sleep(start_time, end_time, quality, deep_min, light_min, rem_min) "
                "VALUES (?,?,?,?,?,?)",
                (int(night_start), int(night_end), quality,
                 int(round(deep_min)), int(round(light_min)), None))
            self._conn.commit()
            self._conn.execute("DELETE FROM stages")
            self._conn.commit()
        return {"start": night_start, "end": night_end, "duration_hrs": round(dur_min / 60, 2),
                "quality": quality, "deep_min": round(deep_min), "light_min": round(light_min)}

    def summary(self) -> Dict[str, int]:
        cur = self._conn.cursor()
        out = {}
        for tbl in ("heart_rate", "steps", "spo2", "blood_pressure", "stress",
                    "sleep", "stages", "activity"):
            try:
                out[tbl] = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            except Exception:
                out[tbl] = 0
        return out


def _estimate_wake_time(rows):
    """Guess wake-up time from ordered sleep-stage markers.

    Moyoung/Da Fit usually end the sleep stream with a type-0 (sober)
    segment whose start is the actual wake time.  Without one we fall
    back to median-segment-duration extrapolation.
    """
    starts = [r[0] for r in rows]
    last = starts[-1]
    if rows[-1][1] == 0:
        return int(last)
    if len(starts) >= 2:
        seg_len = [starts[i] - starts[i - 1] for i in range(1, len(starts))]
        med = sorted(seg_len)[len(seg_len) // 2]
        return int(last + max(30 * 60, min(med, 150 * 60)))
    return int(last + 6 * 3600)


# ============================================================
# Scanning
# ============================================================

def _matches_watch(name: Optional[str], addr: Optional[str]) -> bool:
    if name:
        n = name.lower()
        if any(re.search(p, n) for p in _NAME_PATTERNS):
            return True
    if addr:
        a = addr.lower()
        if re.match(r"^[0-9a-f]{12}$", a) and (not name or any(k in a for k in
                                               ("46", "68", "85", "60", "0b", "3f"))):
            return True
    return False


@dataclass
class FoundDevice:
    address: str
    name: str
    rssi: int
    service_feea: bool = False


async def scan_watches(timeout: float = 10.0) -> List[FoundDevice]:
    """Scan for nearby Moyoung-family BLE watches."""
    if not BLEAK_AVAILABLE:
        raise RuntimeError("bleak is not installed — run `pip install bleak`")
    found: List[FoundDevice] = []
    seen = set()

    def _got(d, ad):
        try:
            name = d.name or (ad.local_name if ad else None)
        except Exception:
            name = None
        quals = None
        if ad:
            quals = getattr(ad, "service_uuids", None)
        svc = bool(quals and UUID_SERVICE_MOYOUNG.lower() in [q.lower() for q in quals])
        if _matches_watch(name, d.address) or svc:
            key = (d.address, name)
            if key in seen:
                return
            seen.add(key)
            rssi = 0
            try:
                rssi = int(getattr(ad, "rssi", 0) or 0)
            except Exception:
                pass
            found.append(FoundDevice(address=d.address, name=name or "(no name)",
                                     rssi=rssi, service_feea=svc))

    scanner = BleakScanner(detection_callback=_got)
    await scanner.start()
    try:
        await asyncio.sleep(timeout)
    finally:
        await scanner.stop()
    found.sort(key=lambda f: f.rssi, reverse=True)
    return found


# ============================================================
# Client
# ============================================================

def _cmd_write_char(cmd: int) -> str:
    if cmd == 1:
        return UUID_CHARACTERISTIC_DATA_SPECIAL_1
    if cmd == 2:
        return UUID_CHARACTERISTIC_DATA_SPECIAL_2
    return UUID_CHARACTERISTIC_DATA_OUT


class MoyoungClient:
    """Asyncio BLE client speaking the Moyoung protocol."""

    def __init__(self, address: str, db: CaptureDB, mtu_override: Optional[int] = None,
                 **client_kwargs):
        if not BLEAK_AVAILABLE:
            raise RuntimeError("bleak is not installed — run `pip install bleak`")
        self.address = address
        self.db = db
        self.mtu_override = mtu_override
        self.client_kwargs = client_kwargs
        self._client: Optional[BleakClient] = None
        self._reassembler = PacketReassembler()
        self._waiters: Dict[int, asyncio.Future] = {}
        self._stream_handlers: Dict[int, List[Callable[[int, bytes], None]]] = {}
        self.protocol = "V1"
        self.mtu = 20
        self.frag_size = 20
        self.device_info: Dict[str, str] = {}
        self.connected = False
        self.on_step: Optional[Callable[[int], None]] = None
        self._measured: Dict[int, Tuple[float, object]] = {}

    # ---- connection -------------------------------------------------
    async def connect(self, timeout: float = 30.0, max_tries: int = 40):
        """Connect, retrying the whole handshake until it succeeds.

        These watches advertise only in short bursts, so a single connect can
        fail with "device not found" between bursts; service discovery and
        notify setup can also fail right after a burst.  Every attempt re-runs
        the complete connect -> device-info -> notify sequence.
        """
        last_err: Optional[Exception] = None
        for attempt in range(max_tries):
            try:
                await self.close()
            except Exception:
                pass
            self._client = BleakClient(self.address, **self.client_kwargs)
            try:
                await asyncio.wait_for(self._client.connect(), timeout=timeout)
                await asyncio.sleep(0.3)   # let service discovery settle
                await self._read_device_info()
                await self._refresh_mtu()
                await asyncio.wait_for(
                    self._client.start_notify(UUID_CHARACTERISTIC_DATA_IN,
                                              self._handle_data_in),
                    timeout=6.0)
                # some watches expose FEE1 without notify - non-fatal
                try:
                    await asyncio.wait_for(
                        self._client.start_notify(UUID_CHARACTERISTIC_STEPS,
                                                  self._handle_steps),
                        timeout=6.0)
                except Exception as e:
                    log.debug("FEE1 notify unavailable: %s", e)
                self.connected = True
                return self.device_info
            except Exception as e:
                last_err = e
                log.debug("connect attempt %d failed: %s", attempt, e)
                try:
                    await self.close()
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        raise ConnectionError(
            f"Could not connect to {self.address} (watch not advertising / "
            f"sensor service missing). Last error: {last_err}")

    async def _refresh_mtu(self):
        try:
            mtu = getattr(self._client, "mtu_size", None) or 0
        except Exception:
            mtu = 0
        if self.mtu_override:
            self.mtu = self.mtu_override
        elif mtu >= 100:
            self.mtu = mtu
        else:
            self.mtu = 20
        mfr = (self.device_info.get("manufacturer") or "").upper()
        if "V2" in mfr:
            self.protocol = "V2"
            if self.mtu_override is None:
                self.mtu = 508
        else:
            self.protocol = "V1" if self.mtu == 20 else "V2"
        self.frag_size = self.mtu if self.mtu > 20 else 20

    async def _read_device_info(self):
        info = {}
        for label, u in (("manufacturer", UUID_CHARACTERISTIC_MFR_NAME),
                         ("model", UUID_CHARACTERISTIC_MODEL),
                         ("firmware", UUID_CHARACTERISTIC_FW_VERSION),
                         ("hardware", UUID_CHARACTERISTIC_HW_VERSION)):
            try:
                val = await asyncio.wait_for(self._client.read_gatt_char(u), timeout=5.0)
                info[label] = bytes(val).decode(errors="replace").strip()
            except Exception:
                info[label] = None
        self.device_info = info

    async def read_battery(self) -> Optional[int]:
        try:
            val = await asyncio.wait_for(
                self._client.read_gatt_char(UUID_CHARACTERISTIC_BATTERY), timeout=5.0)
            return int(bytes(val)[0])
        except Exception:
            return None

    async def read_today_steps(self) -> Optional[int]:
        try:
            val = await asyncio.wait_for(
                self._client.read_gatt_char(UUID_CHARACTERISTIC_STEPS), timeout=5.0)
            data = bytes(val)
            if len(data) >= 3:
                return int.from_bytes(data[0:3], "little")
        except Exception:
            pass
        return None

    # ---- notification handlers --------------------------------------
    def _note_measurement(self, cmd: int, payload: bytes):
        """Track the latest valid reading from a stream of measure replies."""
        try:
            if cmd == CMD_TRIGGER_MEASURE_HEARTRATE and len(payload) >= 1:
                bpm = payload[0]
                if bpm and bpm != 255 and (len(payload) < 2 or payload[1] != 255):
                    self._measured[cmd] = (time.monotonic(), int(bpm))
            elif cmd == CMD_TRIGGER_MEASURE_BLOOD_OXYGEN and len(payload) >= 1:
                v = payload[0]
                if v and v != 255 and v <= 100:
                    self._measured[cmd] = (time.monotonic(), int(v))
            elif cmd == CMD_TRIGGER_MEASURE_BLOOD_PRESSURE and len(payload) >= 3:
                s, d = payload[1], payload[2]
                if s and d and s != 255 and d != 255:
                    self._measured[cmd] = (time.monotonic(), (int(s), int(d)))
            elif cmd == CMD_ADVANCED_QUERY and len(payload) >= 3 and payload[0] == 0x11 and payload[1] == 0x00:
                v = payload[2]
                if v and v != 255:
                    self._measured[cmd] = (time.monotonic(), int(v))
        except Exception:
            pass

    def _handle_steps(self, _sender, data: bytearray):
        b = bytes(data)
        if len(b) < 3:
            return
        steps = int.from_bytes(b[0:3], "little")
        if self.on_step:
            self.on_step(steps)
        asyncio.create_task(self.db.set_daily_steps(steps))

    def _handle_data_in(self, _sender, data: bytearray):
        parsed = self._reassembler.feed(bytes(data))
        if parsed is None:
            return
        cmd, payload = parsed
        self._note_measurement(cmd, payload)
        fut = self._waiters.get(cmd)
        if fut is not None and not fut.done():
            fut.set_result(payload)
            return
        for handler in self._stream_handlers.get(cmd, []):
            try:
                handler(cmd, payload)
            except Exception as e:
                log.debug("stream handler error for 0x%02x: %s", cmd, e)

    async def _await_packet(self, cmd: int, timeout: float) -> Optional[bytes]:
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._waiters[cmd] = fut
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self._waiters.pop(cmd, None)

    async def send_command(self, cmd: int, payload: bytes = b"", await_response: bool = False,
                           timeout: float = 15.0) -> Optional[bytes]:
        packet = build_packet(self.mtu if self.protocol == "V2" else 20, cmd, payload)
        chunks = split_fragments(packet, self.frag_size if self.protocol == "V2" else 20)
        char = _cmd_write_char(cmd)
        for i, chunk in enumerate(chunks):
            try:
                await asyncio.wait_for(
                    self._client.write_gatt_char(char, bytes(chunk), response=False),
                    timeout=5.0,
                )
            except Exception as e:
                log.debug("write to %s failed: %s", char, e)
                if await_response:
                    return None
                raise
            await asyncio.sleep(0.03)  # pacing between fragments
        if await_response:
            return await self._await_packet(cmd, timeout)
        return None

    # ---- time / info -------------------------------------------------
    async def sync_time(self) -> bool:
        ts = local_to_watch()
        payload = struct.pack(">I", ts & 0xFFFFFFFF) + b"\x08"
        await self.send_command(CMD_SYNC_TIME, payload)
        return True

    # ---- live measurements -------------------------------------------
    async def _await_live(self, cmd: int, start_payload: bytes, timeout: float):
        """Trigger a measurement and watch the push-streamed replies for a
        fresh valid reading (the watch emits 0/0xff while measuring)."""
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        last_trigger = 0.0
        await self.send_command(cmd, start_payload)
        last_trigger = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            entry = self._measured.get(cmd)
            if entry and entry[0] >= started:
                return entry[1]
            if now - last_trigger >= 8.0:
                try:
                    await self.send_command(cmd, start_payload)
                except Exception:
                    pass
                last_trigger = now
            await asyncio.sleep(0.25)
        return None

    async def measure_hr(self, timeout: float = 45.0) -> Optional[int]:
        return await self._await_live(CMD_TRIGGER_MEASURE_HEARTRATE, b"\x00",
                                      timeout)

    async def measure_spo2(self, timeout: float = 45.0) -> Optional[int]:
        return await self._await_live(CMD_TRIGGER_MEASURE_BLOOD_OXYGEN, b"\x00",
                                      timeout)

    async def measure_bp(self, timeout: float = 60.0) -> Optional[Tuple[int, int]]:
        return await self._await_live(CMD_TRIGGER_MEASURE_BLOOD_PRESSURE,
                                      b"\x00\x00\x00", timeout)

    async def measure_stress(self, timeout: float = 40.0) -> Optional[int]:
        return await self._await_live(CMD_ADVANCED_QUERY, b"\x11\x00\x00",
                                      timeout)

    async def live_session(self, seconds: float, on_sample=None,
                           hr_every: float = 8.0, spo2_every: float = 30.0,
                           bp_every: float = 60.0, stop_event=None):
        """Capture a vitals snapshot. The watch runs ONE measurement at a
        time, so HR streams first (repeating), then one SpO2 (~20-40 s),
        then one blood-pressure (~40-60 s). Steps arrive passively via FEE1."""
        start = time.monotonic()
        spo2_due = start + spo2_every
        bp_due = start + bp_every
        spo2_done = bp_done = False

        def submit(kind, v):
            if on_sample:
                try:
                    on_sample(kind, v)
                except Exception:
                    pass

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            now = time.monotonic()
            if now >= start + seconds:
                break
            if not spo2_done and now >= spo2_due:
                spo2_done = True
                val = await self.measure_spo2(timeout=max(15.0, seconds - (now - start)))
                if val is not None:
                    await self.db.add_spo2(time.time(), int(val))
                    submit("spo2", int(val))
            if not bp_done and now >= bp_due:
                bp_done = True
                val = await self.measure_bp(timeout=max(20.0, seconds - (now - start)))
                if val is not None:
                    await self.db.add_bp(time.time(), int(val[0]), int(val[1]))
                    submit("bp", (int(val[0]), int(val[1])))
            hr = await self.measure_hr(timeout=min(40.0, seconds - (now - start)))
            if hr is not None:
                await self.db.add_heart_rate(time.time(), int(hr))
                submit("heart_rate", int(hr))
            await asyncio.sleep(hr_every)

    async def capture_sequential(self, *, hr_seconds: float = 30.0,
                                 hr_every: float = 5.0, do_hr: bool = True,
                                 do_spo2: bool = True, do_bp: bool = True,
                                 do_stress: bool = True, on_sample=None,
                                 on_phase=None, stop_event=None):
        """Capture each watch measurement ONE AFTER ANOTHER to completion:
        HR first, then SpO2, then BP, then stress. The single sensor runs
        each to its full measurement window, so this takes longer but never
        interleaves/repeats a vital."""
        def submit(kind, v):
            if on_sample:
                try:
                    on_sample(kind, v)
                except Exception:
                    pass

        def phase(name):
            if on_phase:
                try:
                    on_phase(name)
                except Exception:
                    pass
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("capture stopped by user")

        # PHASE 1 - heart rate (watch streams continuously)
        if do_hr:
            phase("HR")
            start = time.monotonic()
            while time.monotonic() - start < hr_seconds:
                hr = await self.measure_hr(timeout=30.0)
                if hr is not None:
                    await self.db.add_heart_rate(time.time(), int(hr))
                    submit("heart_rate", int(hr))
                await asyncio.sleep(hr_every)
        # PHASE 2 - one full SpO2 reading (retry once if it times out)
        if do_spo2:
            phase("SpO2")
            val = await self.measure_spo2(timeout=60.0)
            if val is None:
                val = await self.measure_spo2(timeout=60.0)
            if val is not None:
                await self.db.add_spo2(time.time(), int(val))
                submit("spo2", int(val))
        # PHASE 3 - one full blood-pressure reading (retry once)
        if do_bp:
            phase("BP")
            val = await self.measure_bp(timeout=70.0)
            if val is None:
                val = await self.measure_bp(timeout=70.0)
            if val is not None:
                await self.db.add_bp(time.time(), int(val[0]), int(val[1]))
                submit("bp", (int(val[0]), int(val[1])))
        # PHASE 4 - stress reading (many watches ignore on-demand stress,
        # so it is one quick attempt - no long retry when unsupported)
        if do_stress:
            phase("Stress")
            val = await self.measure_stress(timeout=30.0)
            if val is not None:
                await self.db.add_stress(time.time(), int(val))
                submit("stress", int(val))

    # ---- sync ----------------------------------------------------------
    async def sync_history(self, with_workouts: bool = True,
                           on_progress: Optional[Callable[[str], None]] = None):
        """Fetch sleep, steps, HR history from the watch into the DB."""
        done = {"sleep": False, "steps": False, "hr": False, "stress": False}
        self._stream_handlers[CMD_SYNC_SLEEP] = [self._on_today_sleep]
        self._stream_handlers[CMD_SYNC_PAST_SLEEP_AND_STEP] = [self._on_past_sleep_step]
        self._stream_handlers[CMD_QUERY_PAST_HEART_RATE_1] = [self._on_hr_history_1]
        self._stream_handlers[CMD_QUERY_PAST_HEART_RATE_2] = [self._on_hr_history_2]
        self._stream_handlers[CMD_ADVANCED_QUERY] = [self._on_stress_packet]
        self._stream_handlers[CMD_QUERY_V2_WORKOUT] = [self._on_workout]

        try:
            await self.sync_time()
            if on_progress:
                on_progress("Syncing time…")

            for i, arg in enumerate((ARG_SYNC_YESTERDAY_SLEEP,
                                     ARG_SYNC_DAY_BEFORE_YESTERDAY_SLEEP)):
                await self.send_command(CMD_SYNC_PAST_SLEEP_AND_STEP, bytes([arg]))
                await asyncio.sleep(0.2)
            await self.send_command(CMD_SYNC_SLEEP)
            await asyncio.sleep(0.2)
            for arg in (ARG_SYNC_YESTERDAY_STEPS, ARG_SYNC_DAY_BEFORE_YESTERDAY_STEPS):
                await self.send_command(CMD_SYNC_PAST_SLEEP_AND_STEP, bytes([arg]))
                await asyncio.sleep(0.1)

            today_steps = await self.read_today_steps()
            if today_steps:
                await self.db.set_daily_steps(today_steps)
            if on_progress:
                on_progress("Steps synced")

            await self.send_command(CMD_QUERY_MOVEMENT_HEART_RATE)
            await self.send_command(CMD_QUERY_PAST_HEART_RATE_1, b"\x00")
            for i in range(20):
                await self.send_command(CMD_QUERY_PAST_HEART_RATE_2, bytes([i]))
                await asyncio.sleep(0.05)
            if on_progress:
                on_progress("Heart-rate history requested")

            if with_workouts:
                await self.send_command(CMD_QUERY_V2_WORKOUT, b"\x00")
            await self.send_command(CMD_ADVANCED_QUERY, b"\x11\x03\x00")
            await self.send_command(CMD_ADVANCED_QUERY, b"\x11\x03\x01")

            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
            night = await self.db.finish_night(datetime.now().date().isoformat())
            if on_progress:
                on_progress("Aggregating last night's sleep…")
            return night
        finally:
            self._stream_handlers.clear()

    # ---- stream decoders ----------------------------------------------
    def _on_today_sleep(self, cmd: int, payload: bytes):
        self._store_sleep_segments(payload)

    def _on_past_sleep_step(self, cmd: int, payload: bytes):
        if not payload:
            return
        data_type = payload[0]
        data = payload[1:]
        if data_type in (ARG_SYNC_YESTERDAY_STEPS, ARG_SYNC_DAY_BEFORE_YESTERDAY_STEPS):
            if len(data) >= 3:
                count = int.from_bytes(data[0:3], "little")
                days_ago = 1 if data_type == ARG_SYNC_YESTERDAY_STEPS else 2
                asyncio.create_task(
                    self.db.add_past_steps(days_ago, count))
        elif data_type in (ARG_SYNC_YESTERDAY_SLEEP, ARG_SYNC_DAY_BEFORE_YESTERDAY_SLEEP):
            days_ago = 1 if data_type == ARG_SYNC_YESTERDAY_SLEEP else 2
            self._store_sleep_segments(data, days_ago=days_ago)

    def _store_sleep_segments(self, payload: bytes, days_ago: int = 0):
        n = len(payload) // 3
        for i in range(n):
            stage, h, m = payload[3 * i], payload[3 * i + 1], payload[3 * i + 2]
            now = datetime.now()
            day = now.date() - timedelta(days=days_ago)
            if h >= 20:
                day = day - timedelta(days=1)
            ts = datetime(day.year, day.month, day.day, h, m).timestamp()
            asyncio.create_task(self.db.add_sleep_stage(ts, stage))

    def _on_hr_history_1(self, cmd: int, payload: bytes):
        if not payload:
            return
        idx = payload[0]
        # 8 packets total: idx//4 = days ago, (idx%4)*6h = start hour,
        # each packet = 6h of samples every 5 min (72 samples).
        days_ago, start_hour = idx // 4, (idx % 4) * 6
        samples = self._decode_hr_history_5min(payload[1:], days_ago, start_hour)
        asyncio.create_task(self.db.add_heart_rate_rows(samples))
        if idx < 7:
            nxt = bytes([idx + 1])
            asyncio.create_task(self.send_command(CMD_QUERY_PAST_HEART_RATE_1, nxt))

    def _on_hr_history_2(self, cmd: int, payload: bytes):
        if not payload:
            return
        idx = payload[0]
        samples = self._decode_hr_grid(payload[1:], idx * 72, step_min=1)
        asyncio.create_task(self.db.add_heart_rate_rows(samples))

    @staticmethod
    def _decode_hr_history_5min(data: bytes, days_ago: int, start_hour: int):
        """Decode a 0x35 packet: 6h of HR samples every 5 minutes."""
        today = datetime.now().date()
        out = []
        base = datetime(today.year, today.month, today.day) - timedelta(days=days_ago)
        for offset in range(len(data)):
            hr = data[offset] & 0xFF
            if hr == 0 or hr == 255 or hr > 220:
                continue
            minute_of_day = start_hour * 60 + offset * 5
            if minute_of_day >= 1440:
                break
            dt = base + timedelta(minutes=minute_of_day)
            if dt > datetime.now():
                continue
            out.append((dt.timestamp(), hr))
        return out

    @staticmethod
    def _decode_hr_grid(data: bytes, start_min_of_day: int, step_min: int):
        today = datetime.now().date()
        out = []
        base = datetime(today.year, today.month, today.day)
        for offset in range(len(data)):
            hr = data[offset] & 0xFF
            if hr == 0 or hr == 255 or hr > 220:
                continue
            minute_of_day = start_min_of_day + offset * step_min
            if minute_of_day >= 1440:
                break
            dt = base + timedelta(minutes=minute_of_day)
            if dt > datetime.now():
                continue
            out.append((dt.timestamp(), hr))
        return out

    def _on_stress_packet(self, cmd: int, payload: bytes):
        if not payload or payload[0] != ARG_ADVANCED_STRESS_PACKET:
            return
        kind = payload[1]
        if kind == 0x00 and len(payload) >= 3:
            v = payload[2]
            asyncio.create_task(self.db.add_stress(time.time(), v))
        elif kind == 0x03 and len(payload) >= 29:
            days_ago = payload[2]
            for i in range(26):
                v = payload[3 + i]
                if v == 0x00:
                    continue
                hour, minute = i // 2, (i % 2) * 30
                day = datetime.now() - timedelta(days=days_ago)
                ts = datetime(day.year, day.month, day.day, hour, minute).timestamp()
                if ts <= time.time():
                    asyncio.create_task(self.db.add_stress(ts, v))

    def _on_workout(self, cmd: int, payload: bytes):
        if not payload:
            return
        subtype = payload[0]
        if subtype == CMD_QUERY_V2_WORKOUT_LIST_RESPONSE:
            count = len(payload) // 5
            for wid in range(count):
                asyncio.create_task(
                    self.send_command(CMD_QUERY_V2_WORKOUT,
                                      bytes([CMD_QUERY_V2_WORKOUT_DETAIL_REQUEST, wid])))
        elif subtype == CMD_QUERY_V2_WORKOUT_DETAIL_RESPONSE:
            recs = self._parse_workout_details(payload)
            for start, end, kind in recs:
                asyncio.create_task(self.db.add_activity(start, end, kind))

    @staticmethod
    def _parse_workout_details(data: bytes, ver: int = 2):
        """Workouts V2 detail rows (26 B each).

        row = [subtype(0x03), workoutNr, start(4) end(4) valid(2) avgHR(1)
               type(1) steps(4) dist(4) cal(4)]  (watch epochs, GMT+8)
        """
        size = 26
        out = []
        for i in range(0, len(data) - size + 1, size):
            row = data[i:i + size]
            start_w = int.from_bytes(row[2:6], "little")
            end_w = int.from_bytes(row[6:10], "little")
            valid = int.from_bytes(row[10:12], "little")
            if not start_w or not end_w or end_w < start_w:
                continue
            if valid == 0 and all(b == 0 for b in row[14:]):
                continue
            kind = row[13]
            out.append((watch_to_local(start_w).timestamp(),
                        watch_to_local(end_w).timestamp(), kind))
        return out

    async def close(self):
        if self._client and getattr(self._client, "is_connected", False):
            try:
                await self._client.stop_notify(UUID_CHARACTERISTIC_DATA_IN)
                await self._client.stop_notify(UUID_CHARACTERISTIC_STEPS)
            except Exception:
                pass
            try:
                await self._client.disconnect()
            except Exception:
                pass
        self.connected = False


# ============================================================
# Session helpers (run asyncio from sync contexts)
# ============================================================

def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def scan_sync(timeout: float = 10.0) -> List[FoundDevice]:
    """Blocking scan wrapper (usable from Streamlit / CLI)."""
    return _run(scan_watches(timeout))


class SyncSession:
    """Synchronous front-end over the asyncio client, for Streamlit/CLI.

    Usage:
        with SyncSession(addr) as s:
            s.device_info      # dict(manufacturer=..., model=..., firmware=...)
            s.battery(); s.today_steps()
            s.live(seconds=30, on_sample=...)
            night = s.sync(with_workouts=True, on_progress=...)
        s.db.summary()         # row counts
    """

    def __init__(self, address: str, db_path: Optional[str] = None,
                 mtu_override: Optional[int] = None):
        self.address = address
        self.db = CaptureDB(db_path or default_db_path())
        self.mtu_override = mtu_override
        self.loop = asyncio.new_event_loop()
        self.client: Optional[MoyoungClient] = None

    def __enter__(self):
        async def _connect():
            c = MoyoungClient(self.address, self.db, mtu_override=self.mtu_override)
            await c.connect()
            return c

        self.client = self.loop.run_until_complete(_connect())
        return self

    def __exit__(self, *_exc):
        self.close()

    @property
    def device_info(self):
        return self.client.device_info if self.client else {}

    @property
    def protocol(self):
        return self.client.protocol if self.client else ""

    def battery(self) -> Optional[int]:
        if not self.client:
            return None
        return self.loop.run_until_complete(self.client.read_battery())

    def today_steps(self) -> Optional[int]:
        if not self.client:
            return None
        return self.loop.run_until_complete(self.client.read_today_steps())

    def live(self, seconds: float, on_sample=None, hr_every: float = 8.0,
             spo2_every: float = 30.0, bp_every: float = 60.0,
             timeout: Optional[float] = None):
        budget = timeout or (seconds + 30.0)
        self.loop.run_until_complete(
            asyncio.wait_for(
                self.client.live_session(seconds, on_sample=on_sample, hr_every=hr_every,
                                         spo2_every=spo2_every, bp_every=bp_every),
                timeout=budget))

    def capture_all(self, hr_seconds: float = 30.0, hr_every: float = 5.0,
                    do_spo2: bool = True, do_bp: bool = True,
                    do_stress: bool = True, on_sample=None, on_phase=None,
                    stop_event=None, timeout: Optional[float] = None):
        """One measurement after another: HR, SpO2, BP, stress - even if
        that takes longer, nothing gets interleaved on the single sensor."""
        budget = timeout or (hr_seconds + 240.0)
        self.loop.run_until_complete(
            asyncio.wait_for(
                self.client.capture_sequential(
                    hr_seconds=hr_seconds, hr_every=hr_every, do_spo2=do_spo2,
                    do_bp=do_bp, do_stress=do_stress,
                    on_sample=on_sample, on_phase=on_phase, stop_event=stop_event),
                timeout=budget))

    def sync(self, with_workouts: bool = True, on_progress=None,
             timeout: Optional[float] = 120.0):
        return self.loop.run_until_complete(
            asyncio.wait_for(
                self.client.sync_history(with_workouts=with_workouts,
                                         on_progress=on_progress),
                timeout=timeout))

    def close(self):
        try:
            if self.client:
                self.loop.run_until_complete(asyncio.wait_for(
                    self.client.close(), timeout=5.0))
        except Exception:
            pass
        try:
            self.db.close()
        except Exception:
            pass
        try:
            pending = asyncio.all_tasks(self.loop)
            for t in pending:
                t.cancel()
            self.loop.close()
        except Exception:
            pass


def connect_and_sync(address: str, db_path: str, with_workouts: bool = True,
                     on_progress=None):
    """Synchronous convenience wrapper for sync_history."""
    db = CaptureDB(db_path)
    try:
        async def _go():
            client = MoyoungClient(address, db)
            try:
                await client.connect()
                return await client.sync_history(with_workouts=with_workouts,
                                                 on_progress=on_progress)
            finally:
                await client.close()
        night = _run(_go())
        return db, night
    finally:
        pass


def load_records_from_db(db_path: str):
    """Run the existing schema-discovering parser over the capture DB."""
    from smartwatch_import import load_records
    return load_records(db_path)


def default_db_path() -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "capture")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "watch_capture.db")