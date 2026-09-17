#!/usr/bin/env python3
# ============================================================
# test_watch_pipeline.py — Synthetic test for smartwatch_import
#
# Creates a fake Health Connect-style SQLite DB and a
# Crrepa/GOBOULT-style DB in a zip, runs the parser, and
# asserts the extracted features are sane.
#
# Usage:
#   python scripts/test_watch_pipeline.py
# ============================================================

import io
import os
import sqlite3
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from smartwatch_import import load_records, extract_features, build_model_input, FEATURE_KEYS


def _make_hc_zip_bytes(days=7):
    """Build an in-memory zip containing a Health Connect-style SQLite DB."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE steps_records (
            id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            count INTEGER,
            source TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE heart_rate_records (
            id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            bpm REAL,
            source TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE sleep_session_records (
            id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            score INTEGER,
            deep_min INTEGER,
            source TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE spo2_records (
            id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            value REAL,
            source TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE stress_records (
            id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            value REAL,
            source TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE blood_pressure_records (
            id INTEGER PRIMARY KEY,
            start_time TEXT,
            end_time TEXT,
            systolic REAL,
            diastolic REAL,
            source TEXT
        )
    """)

    now = datetime(2026, 9, 16, 12, 0, 0)
    rows_hr, rows_steps, rows_sleep, rows_spo2, rows_stress, rows_bp = [], [], [], [], [], []

    for day_offset in range(days):
        day = now - timedelta(days=day_offset)
        start = day.replace(hour=7, minute=0)
        end = start.replace(hour=19, minute=0)

        # steps: 8000-12000 per day
        import random
        random.seed(day_offset)
        total_steps = random.randint(6000, 12000)
        rows_steps.append((str(start), str(end), total_steps, "GOBOULT_ZL35"))

        # heart rate: 100 samples through the day
        hr_base = random.uniform(60, 80)
        for i in range(100):
            t = start + timedelta(minutes=i)
            hr_val = hr_base + random.uniform(-8, 12)
            rows_hr.append((str(t), str(t + timedelta(minutes=1)),
                           round(hr_val, 1), "GOBOULT_ZL35"))

        # sleep: one session, 7-8 hours
        sleep_start = (day - timedelta(days=1)).replace(hour=23, minute=0)
        sleep_end = day.replace(hour=random.choice([6, 6, 7]), minute=random.choice([0, 15, 30]))
        deep = random.randint(30, 90)
        score = random.randint(60, 90)
        rows_sleep.append((str(sleep_start), str(sleep_end), score, deep, "GOBOULT_ZL35"))

        # spo2
        rows_spo2.append((str(start.replace(hour=2)), str(start.replace(hour=2, minute=5)),
                          round(random.uniform(94, 99), 1), "GOBOULT_ZL35"))

        # stress
        rows_stress.append((str(start.replace(hour=12)), str(start.replace(hour=12, minute=1)),
                            round(random.uniform(20, 70), 1), "GOBOULT_ZL35"))

        # blood pressure
        sys_v = random.randint(110, 130)
        dia_v = random.randint(70, 90)
        rows_bp.append((str(start), str(end), float(sys_v), float(dia_v), "GOBOULT_ZL35"))

    conn.executemany("INSERT INTO steps_records (start_time, end_time, count, source) VALUES (?,?,?,?)",
                     rows_steps)
    conn.executemany("INSERT INTO heart_rate_records (start_time, end_time, bpm, source) VALUES (?,?,?,?)",
                     rows_hr)
    conn.executemany("INSERT INTO sleep_session_records (start_time, end_time, score, deep_min, source) VALUES (?,?,?,?,?)",
                     rows_sleep)
    conn.executemany("INSERT INTO spo2_records (start_time, end_time, value, source) VALUES (?,?,?,?)",
                     rows_spo2)
    conn.executemany("INSERT INTO stress_records (start_time, end_time, value, source) VALUES (?,?,?,?)",
                     rows_stress)
    conn.executemany("INSERT INTO blood_pressure_records (start_time, end_time, systolic, diastolic, source) VALUES (?,?,?,?,?)",
                     rows_bp)
    conn.commit()

    # Export to bytes (sqlite .dump) then package as zip
    blob = conn.serialize()
    conn.close()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("health_connect_database.db", bytes(blob))
    return buf.getvalue()


def _make_crrepa_zip_bytes():
    """Build a Crrepa/GOBOULT-style SQLite DB in a zip."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE heartrate (
            id INTEGER PRIMARY KEY,
            time INTEGER,
            bpm REAL,
            type INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE steps (
            id INTEGER PRIMARY KEY,
            time INTEGER,
            count INTEGER,
            type INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE sleep (
            id INTEGER PRIMARY KEY,
            bedtime INTEGER,
            wakeup INTEGER,
            score INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE bloodpressure (
            id INTEGER PRIMARY KEY,
            time INTEGER,
            systolic REAL,
            diastolic REAL
        )
    """)

    now = datetime(2026, 9, 16)
    import random
    random.seed(42)
    for day_offset in range(3):
        day = now - timedelta(days=day_offset)
        ts = int(day.timestamp())
        conn.execute("INSERT INTO steps (time, count, type) VALUES (?, ?, ?)",
                     (ts, random.randint(5000, 10000), 1))
        conn.execute("INSERT INTO heartrate (time, bpm, type) VALUES (?, ?, ?)",
                     (ts + 3600, round(random.uniform(62, 85), 1), 1))
        conn.execute("INSERT INTO bloodpressure (time, systolic, diastolic) VALUES (?, ?, ?)",
                     (ts + 7200, float(random.randint(115, 130)), float(random.randint(72, 88))))

    sleep_start = int((now - timedelta(hours=8)).timestamp())
    sleep_end = int(now.timestamp())
    conn.execute("INSERT INTO sleep (bedtime, wakeup, score) VALUES (?, ?, ?)",
                 (sleep_start, sleep_end, 75))
    conn.commit()

    blob = conn.serialize()
    conn.close()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("goboult_fit.db", bytes(blob))
    return buf.getvalue()


def test():
    print("=" * 60)
    print("  TEST 1: Health Connect zip (schema-discovering parser)")
    print("=" * 60)
    hc_zip = _make_hc_zip_bytes(days=7)
    records = load_records(hc_zip)
    assert "steps" in records, "steps not found"
    assert "heart_rate" in records, "heart_rate not found"
    assert "sleep" in records, "sleep not found"
    assert "spo2" in records, "spo2 not found"
    assert "stress" in records, "stress not found"
    assert "bp" in records, "bp not found"
    print(f"  PASSED: categories found = {list(k for k in records if not k.startswith('_'))}")

    feat = extract_features(records, window_days=7)
    hf = feat["features"]
    assert hf["Sleep Duration"] is not None, "Sleep Duration missing"
    assert hf["Heart Rate"] is not None, "Heart Rate missing"
    assert hf["Daily Steps"] is not None, "Daily Steps missing"
    print(f"  PASSED: features extracted: { {k: v for k, v in hf.items() if v is not None} }")

    hf, display, missing = build_model_input(feat, "Male", 30, "Software Engineer", "Normal")
    assert all(k in hf for k in FEATURE_KEYS), "missing model keys"
    print(f"  PASSED: model input has all 12 keys, missing={missing}")
    print()

    print("=" * 60)
    print("  TEST 2: Crrepa / GOBOULT Fit zip (Crrepa table names)")
    print("=" * 60)
    cr_zip = _make_crrepa_zip_bytes()
    records2 = load_records(cr_zip)
    cats_found = [k for k in records2 if not k.startswith('_')]
    print(f"  Categories found: {cats_found}")
    assert "steps" in cats_found, "steps not detected in Crrepa tables"
    assert "heart_rate" in cats_found, "heart_rate not detected"
    print(f"  PASSED: Crrepa tables recognized")

    feat2 = extract_features(records2, window_days=7)
    hf2 = feat2["features"]
    assert hf2["Daily Steps"] is not None, "steps value missing"
    assert hf2["Heart Rate"] is not None, "heart rate missing"
    print(f"  PASSED: features from Crrepa DB: { {k: v for k, v in hf2.items() if v is not None} }")
    print()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test()
