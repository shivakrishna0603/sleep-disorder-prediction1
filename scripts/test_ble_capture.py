#!/usr/bin/env python3
# ============================================================
# test_ble_capture.py — Mocked-GATT tests for app/ble_watch.py
#
# No physical watch or Bluetooth needed: exercises the Moyoung
# packet framing / reassembly, stream decoders, the capture SQLite
# schema, and the round trip
#     CaptureDB -> smartwatch_import.load_records -> extract_features
#
# Usage:
#   python scripts/test_ble_capture.py
# ============================================================

import os
import struct
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from ble_watch import (
    CaptureDB,
    MoyoungClient,
    PacketReassembler,
    _estimate_wake_time,
    _matches_watch,
    build_packet,
    local_to_watch,
    parse_packet_length,
    split_fragments,
    watch_to_local,
)

_decode_hr_grid = MoyoungClient._decode_hr_grid
_parse_workout_details = MoyoungClient._parse_workout_details

from smartwatch_import import load_records, extract_features, FEATURE_KEYS

_passed = []


def check(name, cond, detail=""):
    assert cond, f"{name} FAILED {detail}"
    _passed.append(name)
    print(f"  PASSED: {name}")


def test_framing_v1():
    print("=" * 60)
    print("  TEST: V1 fixed-size framing (MTU 20)")
    print("=" * 60)
    # HR trigger: cmd 0x6D {0} -> FE EA 10 06 6D 00
    p1 = build_packet(20, 0x6D, b"\x00")
    check("V1 header bytes", list(p1[0:4]) == [0xFE, 0xEA, 16, 6],
          f"got {list(p1)}")
    check("V1 cmd+payload", p1[4] == 0x6D and p1[5] == 0x00)

    # Long payload splits into multiple fragments of <=20 bytes
    long_payload = bytes(range(64))
    p2 = build_packet(20, 0x33, long_payload)
    chunks = split_fragments(p2, 20)
    check("V1 fragmentation count", len(chunks) == 4,
          f"len(packet)={len(p2)} chunks={len(chunks)}")
    check("V1 first fragment full length", parse_packet_length(chunks[0]) == len(p2))
    check("V1 reassembly", b"".join(chunks) == p2)

    # Reassemble through the mock notifier
    ra = PacketReassembler()
    got = None
    for c in chunks:
        r = ra.feed(c)
        if r is not None:
            got = r
    check("V1 reassembled cmd", got is not None and got[0] == 0x33)
    check("V1 reassembled payload", got[1] == long_payload)


def test_framing_v2():
    print("=" * 60)
    print("  TEST: V2 MTU-based framing (MTU 508)")
    print("=" * 60)
    p = build_packet(508, 0x6D, b"\x00")
    check("V2 header", p[0] == 0xFE and p[1] == 0xEA and p[2] == 32 and p[3] == 6)
    check("V2 length parse", parse_packet_length(p) == 6)

    payload = bytes(range(200))
    p2 = build_packet(508, 0xB2, payload)
    check("V2 long len", parse_packet_length(p2) == len(p2) == 205)
    chunks = split_fragments(p2, 508)
    check("V2 single chunk", len(chunks) == 1)
    ra = PacketReassembler()
    got = ra.feed(p2)
    check("V2 reassembled", got == (0xB2, payload))


def test_reassembler_stress_packet():
    print("=" * 60)
    print("  TEST: fragmented day-stress response reassembly")
    print("=" * 60)
    payload = bytes([0x11, 0x03, 0x00]) + bytes([0] * 10 + [42] + [0] * 15)
    p = build_packet(20, 0xB9, payload)  # V1 -> split across 20-byte ATT writes
    chunks = split_fragments(p, 20)
    ra = PacketReassembler()
    got = None
    for c in chunks:
        r = ra.feed(c)
        if r is not None:
            got = r
    check("stress reassembled", got is not None and got[1] == payload)
    # corrupted/new-header fragments are ignored gracefully
    ra2 = PacketReassembler()
    check("garbage ignored", ra2.feed(b"\x00\x00\x00\x00") is None)


def test_watch_time_helpers():
    print("=" * 60)
    print("  TEST: watch time (GMT+8) conversion")
    print("=" * 60)
    local_now = datetime.now()
    wt = local_to_watch(local_now)
    back = watch_to_local(wt)
    check("time round-trip", abs((back - local_now).total_seconds()) < 5)
    check("watch>local drops 8h", wt - int(local_now.timestamp()) == 8 * 3600)


def test_decode_hr_grid():
    print("=" * 60)
    print("  TEST: HR history grid decoder (0x36, 1 sample/minute)")
    print("=" * 60)
    data = bytearray([0] * 72)
    data[10] = 70
    data[11] = 75
    data[12] = 0xFE  # invalid sentinel -> skipped
    data[13] = 80
    rows = _decode_hr_grid(bytes(data), start_min_of_day=0, step_min=1)
    check("HR grid sample count", len(rows) == 3, f"got {rows}")
    check("HR grid values", all(rows[i][1] in (70, 75, 80) for i in range(3)))
    check("HR grid future clip", all(ts <= datetime.now().timestamp() + 5 for ts, _ in rows))


def test_workout_details():
    print("=" * 60)
    print("  TEST: workouts V2 details parse")
    print("=" * 60)
    start_w = local_to_watch(datetime.now() - timedelta(hours=1))
    end_w = local_to_watch(datetime.now() - timedelta(minutes=30))
    row = (b"\x03\x00" +
           struct.pack("<I", start_w) + struct.pack("<I", end_w) +
           struct.pack("<H", 1800) + bytes([71]) + bytes([7]) +
           struct.pack("<I", 4000) + struct.pack("<I", 3500) + struct.pack("<I", 180))
    recs = _parse_workout_details(bytes(row), ver=2)
    check("workout count", len(recs) == 1, f"got {recs}")
    check("workout duration", round((recs[0][1] - recs[0][0]) / 60) == 30,
          f"got {recs[0]}")


def test_name_match():
    print("=" * 60)
    print("  TEST: scan name matching")
    print("=" * 60)
    check("FIRE-BOLTT 046", _matches_watch("FIRE-BOLTT 046", None))
    check("DaFit band", _matches_watch("DaFit", None) or _matches_watch("DA-FIT", None))
    check("MOYOUNG-V2", _matches_watch("MOYOUNG-V2", None))
    check("JODU factory name", _matches_watch("JODU52041437361", None))
    check("random phone", not _matches_watch("iPhone", "AA:BB:CC:DD:EE:FF"))
    check("mac-like addr", _matches_watch(None, "A4C1380B3F46") or
          _matches_watch(None, "A4:C1:38:0B:3F:46"))


def test_streamed_measure_buffer():
    print("=" * 60)
    print("  TEST: ~live~ streamed measure buffer (0x00 frames then real)")
    print("=" * 60)
    import asyncio

    class FakeClient:
        def __init__(self):
            self.sent = []
            self._measured = {}

    async def run():
        db = CaptureDB(":memory:")
        c = MoyoungClient("AA:BB:CC:DD:EE:FF", db, mtu_override=508)
        await db.set_daily_steps(0)
        # watch streams measuring-… status, then a real reading
        c._handle_data_in(None, build_packet(508, 0x6D, b"\x00"))
        c._handle_data_in(None, build_packet(508, 0x6D, b"\x00"))
        c._handle_data_in(None, build_packet(508, 0x6D, b"\x54"))  # 84 bpm
        check("buffer captured 84 bpm", c._measured[0x6D][1] == 84,
              f"got {c._measured.get(0x6D)}")
        c._handle_data_in(None, build_packet(508, 0x6B, b"\x61"))  # 97%
        check("buffer captured spo2 97", c._measured[0x6B][1] == 97,
              f"got {c._measured.get(0x6B)}")
        c._handle_data_in(None, build_packet(508, 0x69, b"\x00\x7d\x4b"))  # 125/75
        check("buffer captured BP 125/75", c._measured[0x69][1] == (125, 75),
              f"got {c._measured.get(0x69)}")
        c._handle_data_in(None, build_packet(508, 0x69, b"\x00\xff\xff"))
        check("ff BP frame ignored, keeps last valid", c._measured[0x69][1] == (125, 75),
              f"got {c._measured.get(0x69)}")
        db.close()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


def test_db_round_trip():
    print("=" * 60)
    print("  TEST: CaptureDB -> smartwatch_import -> features")
    print("=" * 60)
    tmp = tempfile.mktemp(suffix=".db")
    db = CaptureDB(tmp)
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_seed_db(db))
        finally:
            loop.close()

        records = load_records(tmp)
        cats = [k for k in records if not k.startswith("_")]
        check("categories detected", {"heart_rate", "steps", "spo2", "stress", "bp", "sleep",
                                      "activity"} <= set(cats),
              f"got {cats}")

        feat = extract_features(records, window_days=7)
        hf = feat["features"]
        check("Daily Steps", hf["Daily Steps"] == 8000, f"got {hf['Daily Steps']}")
        check("Heart Rate avg", hf["Heart Rate"] is not None,
              f"got {hf['Heart Rate']}")
        check("Sleep Duration ~7.5h", hf["Sleep Duration"] is not None and
              7.0 <= hf["Sleep Duration"] <= 8.2,
              f"got {hf['Sleep Duration']}")
        check("BP present", hf["Systolic_BP"] is not None and hf["Diastolic_BP"] is not None,
              f"got {hf['Systolic_BP']}/{hf['Diastolic_BP']}")
        check("Stress present", hf["Stress Level"] is not None,
              f"got {hf['Stress Level']}")
        filled = {k: v for k, v in hf.items() if v is not None}
        print(f"        features: {filled}")
    finally:
        db.close()
        try:
            os.remove(tmp)
        except OSError:
            pass


async def _seed_db(db):
    now = datetime.now()
    # heart rate: 6 samples this morning
    for i in range(6):
        ts = (now - timedelta(hours=2, minutes=10 * i)).timestamp()
        await db.add_heart_rate(ts, 72 + i)
    await db.set_daily_steps(8000)
    await db.add_spo2((now - timedelta(hours=1)).timestamp(), 97)
    await db.add_bp((now - timedelta(hours=1, minutes=5)).timestamp(), 118, 76)
    await db.add_stress((now - timedelta(minutes=30)).timestamp(), 34)
    # last night's sleep: 22:30 -> 06:00 with light/deep segments + sober tail
    night_start = now.replace(hour=22, minute=30, second=0, microsecond=0) - timedelta(days=1)
    await db.add_sleep_stage(night_start.timestamp(), 1)
    await db.add_sleep_stage((night_start + timedelta(minutes=45)).timestamp(), 2)
    await db.add_sleep_stage((night_start + timedelta(hours=2)).timestamp(), 1)
    await db.add_sleep_stage((night_start + timedelta(hours=3)).timestamp(), 2)
    await db.add_sleep_stage((night_start + timedelta(hours=4, minutes=30)).timestamp(), 1)
    await db.add_sleep_stage((night_start + timedelta(hours=7, minutes=30)).timestamp(), 0)
    await db.finish_night(datetime.now().date().isoformat())
    # a workout yesterday
    start = now - timedelta(days=1, hours=6)
    await db.add_activity(start.timestamp(), (start + timedelta(minutes=40)).timestamp(), 7)


def test():
    test_framing_v1()
    test_framing_v2()
    test_reassembler_stress_packet()
    test_watch_time_helpers()
    test_decode_hr_grid()
    test_workout_details()
    test_name_match()
    test_db_round_trip()

    print("=" * 60)
    print(f"  ALL {len(_passed)} TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test()