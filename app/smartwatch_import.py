# ============================================================
# smartwatch_import.py — Real watch data -> model features
# Sleep Disorder Prediction Using Wearable Sensor Data
#
# Accepts REAL data from a GOBOULT Fit smartwatch without any
# app-side export: Health Connect export zip, the GOBOULT Fit /
# Crrepa app SQLite databases (pulled via adb), or a raw CSV.
#
# Parsing is schema-DISCOVERING: GOBOULT / Crrepa databases and
# Health Connect exports use different table/column names on
# every firmware, so we introspect the SQLite schema and match
# steps / sleep / heart rate / SpO2 / BP / stress tables by
# keyword heuristics instead of hard-coded names.
# ============================================================

import io
import os
import random
import sqlite3
import tempfile
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

# ── Keyword buckets (lower-case substring match) ─────────────────
_STEP_KEYS = ("step", "walk", "runstep", "pedometer")
_SLEEP_KEYS = ("sleep", "bed", "wake", "nap", "deepsleep", "lightsleep")
_HR_KEYS = ("heart", "bpm", "pulse", "ratemonitor", "heartrate", "hrdata")
_SPO2_KEYS = ("spo2", "oxygen", "bloodoxygen", "blood_oxygen")
_BP_KEYS = ("bloodpressure", "blood_pressure", "diastolic", "systolic", "pressure")
_STRESS_KEYS = ("stress",)
_CAL_KEYS = ("calorie", "kcall")
_ACTIVITY_KEYS = ("activity", "active", "intensity", "exercise", "workout", "sport", "run", "training", "fitness")
_RESTING_KEYS = ("resting", "rhr", "quiet", "repose")
_SCORE_KEYS = ("score", "quality", "grade", "evaluate")

_TIME_NAME_HINTS = ("time", "timestamp", "ts", "date", "utc", "epoch", "day")
_ID_NAME_HINTS = ("id", "uuid", "record_id", "origin_package", "metadata",
                  "client_record", "rowid", "deleted", "hash", "lineage")

_GOOD_VALUE_HINTS = {
    "steps": ("count", "total", "steps", "steps_value", "value", "data", "num"),
    "heart_rate": ("bpm", "heart_rate", "rate", "value", "data"),
    "spo2": ("spo2", "value", "blood_oxygen", "oxygen", "data"),
    "stress": ("value", "stress", "avg", "data"),
    "sleep": ("duration", "minutes", "min", "total", "value", "sleepminutes"),
}

_BP_SYS_HINTS = ("systolic", "systolic_bp", "sbp", "high", "max", "value")
_BP_DIA_HINTS = ("diastolic", "diastolic_bp", "dbp", "low", "min", "value")

_SQLLIKE_EXT = (".db", ".sqlite", ".sqlite3")
_TIME_ISOFMT = "%Y-%m-%d"


class SourceError(Exception):
    pass


def _low(name):
    return name.lower().replace(" ", "")  # strip spaces: "blood oxygen" -> "bloodoxygen"


def _norm_name(name):
    return _low(name).replace("_", "")


def _table_category(name):
    n = _norm_name(name)
    for keys, cat in ((_BP_KEYS, "bp"),
                      (_SPO2_KEYS, "spo2"),
                      (_ACTIVITY_KEYS, "activity"),
                      (_STEP_KEYS, "steps"),
                      (_SLEEP_KEYS, "sleep"),
                      (_STRESS_KEYS, "stress"),
                      (_CAL_KEYS, "calories"),
                      (_HR_KEYS, "heart_rate")):
        for k in keys:
            if _norm_name(k) in n:
                return cat
    return None


def _is_time_col(col):
    low = _low(col)
    return any(h in low for h in _TIME_NAME_HINTS)


def _is_id_col(col):
    low = _low(col)
    return any(h in low for h in _ID_NAME_HINTS) or low in ("v", "ver", "version")


def _pick_col(df, hints, exclude=()):
    cols = [c for c in df.columns if c not in exclude]
    for hint in hints:
        hn = _norm_name(hint)
        for c in cols:
            if _norm_name(c) == hn:
                return c
    for hint in hints:
        hn = _norm_name(hint)
        for c in cols:
            if hn in _norm_name(c):
                return c
    # generic fallback: first numeric column that is not time-like / id-like
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]) and not _is_time_col(c) and not _is_id_col(c):
            return c
    return None


def _to_dt(series, pptz=None):
    """Convert a column to pandas datetime, tolerating epoch formats."""
    s = pd.Series(series)
    if pd.api.types.is_numeric_dtype(s):
        mx = s.max()
        if mx > 1e17:            # nanoseconds
            unit = "ns"
        elif mx > 1e14:          # microseconds
            unit = "us"
        elif mx > 1e11:          # milliseconds
            unit = "ms"
        else:                    # seconds
            unit = "s"
        return pd.to_datetime(s, unit=unit, utc=True).dt.tz_localize(None)
    try:
        return pd.to_datetime(s, errors="coerce", utc=bool(pptz)).tz_localize(None)
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def _guess_duration_unit(series, colname):
    n = _norm_name(colname)
    if "h" in n and "min" not in n:
        return "h"
    if "min" in n and "h" not in n:
        return "min"
    if "sec" in n:
        return "s"
    mx = series.max() if pd.notna(series).any() else 1
    val = mx if not pd.isna(mx) else 1
    if val > 1000:
        return "s"
    return "min"


def _read_all_tables(conn):
    cur = conn.cursor()
    names = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
    out = {}
    for t in names:
        if t.startswith("sqlite_"):
            continue
        try:
            df = pd.read_sql(f'SELECT * FROM "{t}"', conn)
            if not df.empty:
                out[t] = df
        except Exception:
            pass
    return out


def _normalize_one(df, category):
    """Turn one discovered table into a normalized per-category record."""
    if df is None or df.empty:
        return None
    rec = {"table": None, "time": None}

    if category == "bp":
        time_col = _pick_col(df, ("time", "timestamp", "date", "ts", "measure_time"), exclude=[])
        if time_col is None:
            time_col = _find_only_time_col(df)
        sys_col = _pick_col(df, _BP_SYS_HINTS, exclude=())
        dia_col = _pick_col(df, _BP_DIA_HINTS, exclude=())
        if not (sys_col and dia_col):
            return None
        t = _to_dt(df[time_col]) if time_col else pd.Series([None] * len(df))
        sys_vals = pd.to_numeric(df[sys_col], errors="coerce")
        dia_vals = pd.to_numeric(df[dia_col], errors="coerce")
        if sys_vals is None or dia_vals is None:
            return None
        return {"table": None, "time": t.values,
                "systolic": sys_vals, "diastolic": dia_vals}

    time_col = _pick_col(df, ("time", "timestamp", "date", "ts", "record_time", "measure_time"),
                          exclude=[])
    if time_col is None:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                time_col = c
                break
    if time_col is None:
        return None

    t = _to_dt(df[time_col])
    value_hints = {"steps": "count", "heart_rate": "bpm", "spo2": "value",
                   "stress": "value"}.get(category, "value")

    if category == "sleep":
        # Prefer explicit start/end; else start + duration.
        st_col = _pick_col(df, ("start_time", "starttime", "bedtime", "sleeptime",
                                "bed_time", "go_tobed", "sleep_start", "start"), exclude=[])
        en_col = _pick_col(df, ("end_time", "endtime", "waketime", "wakeup_time",
                                "wake_time", "sleep_end", "end", "out_bed", "leave_bed"), exclude=[])
        sc_col = _pick_col(df, _SCORE_KEYS, exclude=[])
        start = _to_dt(df[st_col]) if st_col else t
        if en_col:
            end = _to_dt(df[en_col])
        else:
            dur_col = _pick_col(df, ("duration", "total_min", "totalmin", "minutes",
                                     "sleepminutes", "sleeptime", "duration_min", "value"), exclude=[sc_col])
            if dur_col:
                unit = _guess_duration_unit(pd.to_numeric(df[dur_col], errors="coerce"), str(dur_col))
                mul = {"h": 3600, "min": 60, "s": 1}.get(unit, 1)
                end = start + pd.to_timedelta(pd.to_numeric(df[dur_col], errors="coerce") * mul, unit="s")
            else:
                end = start + pd.to_timedelta(pd.to_numeric(df[_pick_col(df, ["value"])], errors="coerce") * 60, unit="s")
                end = start + pd.to_timedelta([np.nan] * len(df), unit="s")
        return {"table": None, "start": start, "end": end,
                "quality": pd.to_numeric(df[sc_col], errors="coerce") if sc_col else None,
                "deep_min": None, "light_min": None, "rem_min": None, "awake_min": None}
    elif category == "steps":
        v_col = _pick_col(df, ("count", "total", "steps", "value", "num", "data"), exclude=[])
        if v_col is None:
            return None
        return {"table": None, "time": t.values,
                "value": pd.to_numeric(df[v_col], errors="coerce")}
    elif category in ("heart_rate", "spo2", "stress"):
        v_col = _pick_col(df, {"heart_rate": ("bpm", "heart_rate", "rate", "value", "data"),
                               "spo2": ("spo2", "value", "oxygen", "data"),
                               "stress": ("value", "stress", "data")}[category], exclude=[])
        if v_col is None:
            return None
        return {"table": None, "time": t.values,
                "value": pd.to_numeric(df[v_col], errors="coerce")}
    elif category == "activity":
        st_col = _pick_col(df, ("start_time", "starttime", "start", "time"), exclude=[])
        en_col = _pick_col(df, ("end_time", "endtime", "end"), exclude=[])
        start = _to_dt(df[st_col]) if st_col else t
        if en_col:
            end = _to_dt(df[en_col])
        else:
            dur_col = _pick_col(df, ("duration", "minutes", "min", "total"), exclude=[])
            if dur_col:
                unit = _guess_duration_unit(pd.to_numeric(df[dur_col], errors="coerce"), str(dur_col))
                mul = {"h": 3600, "min": 60, "s": 1}.get(unit, 1)
                end = start + pd.to_timedelta(pd.to_numeric(df[dur_col], errors="coerce") * mul, unit="s")
            else:
                end = pd.to_datetime([None] * len(df))
        return {"table": None, "start": start, "end": end}
    return None


def _find_only_time_col(df):
    for c in df.columns:
        if _is_time_col(c) and not _is_id_col(c):
            return c
    return None


def load_records(source):
    """Load raw records from zip / sqlite file / list of files / DataFrame.

    Returns a dict of category -> list of normalized records with keys:
      - 'steps':     [{'time': array, 'value': array}]
      - 'heart_rate','spo2','stress': same shape
      - 'bp':        [{'systolic','diastolic'}]
      - 'sleep':     [{'start','end','quality', ...stages}]
      - 'activity':  [{'start','end'}]
      - 'raw':       dict of discovered tables (for diagnostics)
    """
    result = {"raw": {}}
    conns = []
    try:
        for handle in _iter_sqlite_handles(source):
            conns.append(handle)
            if handle.get("records"):
                for cat, recs in handle["records"].items():
                    result.setdefault(cat, []).extend(recs)
                continue
            tables = _read_all_tables(handle["conn"])
            for name, df in tables.items():
                result["raw"][name] = df
                cat = _table_category(name)
                if cat is None:
                    continue
                try:
                    norm = _normalize_one(df, cat)
                except Exception:
                    norm = None
                if norm is None:
                    continue
                result.setdefault(cat, []).append(norm)
        result["_filenames"] = [h["name"] for h in conns if h.get("conn") is not None or h.get("name") == "csv"]
    finally:
        for h in conns:
            if h.get("conn") is not None:
                try:
                    h["conn"].close()
                except Exception:
                    pass
    return result


def _iter_sqlite_handles(source):
    """Yield {'name': ..., 'conn': sqlite3.Connection} candidates from any source."""
    candidates = []

    if isinstance(source, pd.DataFrame):
        df = source
        if _is_raw_csv(df):
            cats = _normalize_from_csv(df)
            yield {"name": "csv", "conn": None, "records": cats}
            return

    if isinstance(source, list):
        for s in source:
            yield from _iter_sqlite_handles(s)
        return

    if isinstance(source, (bytes, bytearray)):
        if _is_zip_bytes(io.BytesIO(bytes(source))):
            zf = zipfile.ZipFile(io.BytesIO(bytes(source)))
            for info in zf.infolist():
                if info.filename.lower().endswith(_SQLLIKE_EXT):
                    yield {"name": info.filename,
                           "conn": _memory_conn_from_bytes(zf.read(info))}
            zf.close()
            return
        yield {"name": "uploaded", "conn": _memory_conn_from_bytes(bytes(source))}
        return

    if isinstance(source, (str, os.PathLike)):
        p = os.fspath(source)
        if os.path.isdir(p):
            files = []
            for root, _, fs in os.walk(p):
                for f in fs:
                    if f.lower().endswith(_SQLLIKE_EXT):
                        files.append(os.path.join(root, f))
            for f in sorted(files):
                yield from _iter_sqlite_handles(f)
            return
        ext = os.path.splitext(p)[1].lower()
        if ext == ".zip":
            with zipfile.ZipFile(p) as zf:
                for info in zf.infolist():
                    if info.filename.lower().endswith(_SQLLIKE_EXT):
                        yield {"name": info.filename,
                               "conn": _memory_conn_from_bytes(zf.read(info))}
            return
        if ext in _SQLLIKE_EXT:
            conn = sqlite3.connect(p)
            yield {"name": os.path.basename(p), "conn": conn}
            return
        if ext == ".csv":
            df = pd.read_csv(p)
            cats = _normalize_from_csv(df)
            yield {"name": os.path.basename(p), "conn": None, "records": cats}
            return
        raise SourceError(f"Unsupported file type: {ext}")

    raise SourceError(f"Cannot read source of type {type(source).__name__}")


def _memory_conn_from_bytes(blob):
    """Open an SQLite database from raw bytes (works for in-memory uploads)."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.deserialize(blob)
        return conn
    except Exception:
        pass
    tmp = os.path.join(tempfile.gettempdir(),
                       f"_hc_{datetime.now().microsecond}.db")
    with open(tmp, "wb") as f:
        f.write(blob)
    return sqlite3.connect(tmp)


def _is_zip_bytes(buf):
    return buf.read(4)[:2] == b"PK"


def _is_raw_csv(df):
    cols = {_low(c) for c in df.columns}
    # A CSV of daily metrics (kaggle-style) or a raw record series?
    return True  # treat all CSVs through csv path


def _normalize_from_csv(df):
    """Try to interpret an arbitrary CSV as watch records."""
    cats = {}
    df = df.copy()
    time_col = None
    for c in df.columns:
        if _is_time_col(c):
            time_col = c
            break
    t = _to_dt(df[time_col]) if time_col else pd.Series([None] * len(df))

    def _push(cat, series):
        cats.setdefault(cat, []).append({"table": "csv", "value": series})

    for c in df.columns:
        cl = _low(c)
        if _is_id_col(c) or _is_time_col(c):
            continue
        try:
            num = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            continue
        n = _norm_name(c)
        if any(k in n for k in ("step", "walk")) and ("sleep" not in n):
            _push("steps", num)
        elif any(k in n for k in ("heart", "bpm", "pulse")):
            _push("heart_rate", num)
        elif any(k in n for k in ("spo2", "oxygen")):
            _push("spo2", num)
        elif any(k in n for k in ("stress",)):
            _push("stress", num)
        elif any(k in n for k in ("systolic", "sbp", "high")):
            cats.setdefault("bp", []).append({"systolic": num, "diastolic": None})
        elif any(k in n for k in ("diastolic", "dbp", "low")):
            for r in cats.get("bp", []):
                if r["diastolic"] is None:
                    r["diastolic"] = num
        elif any(k in n for k in ("sleep", "bed", "wake")):
            cats.setdefault("sleep", []).append({"start": t, "end": t + pd.to_timedelta(num, unit="min"), "quality": None})
    return cats


# ============================================================
# Feature extraction (raw records -> the 8 health model features)
# ============================================================

FEATURE_KEYS = [
    "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Stress Level", "Heart Rate", "Daily Steps",
    "Systolic_BP", "Diastolic_BP",
]


def _series_to_dt(values):
    try:
        return pd.to_datetime(values, utc=True, errors="coerce").tz_localize(None)
    except Exception:
        return pd.to_datetime(values, errors="coerce")


def _steps_to_activity(steps):
    """Daily step count -> daily activity minutes (PAL), bounded to the
    dataset range (3000-10000 steps ~ 30-90 min/day), a standard actigraphy
    conversion used when the watch stores no workout/activity rows."""
    lo_s, hi_s, lo_a, hi_a = 3000.0, 10000.0, 30.0, 90.0
    s = max(lo_s, min(hi_s, float(steps)))
    return int(round(lo_a + (s - lo_s) / (hi_s - lo_s) * (hi_a - lo_a)))


def _latest_day_step_total(records, days):
    """Total steps on the most recent day present in the window."""
    totals = {}
    for rec in records:
        t = _series_to_dt(rec["time"])
        v = pd.to_numeric(rec["value"], errors="coerce")
        df = pd.DataFrame({"t": t, "v": v}).dropna()
        if df.empty:
            continue
        df["day"] = df["t"].dt.date
        for day, grp in df.groupby("day"):
            totals[day] = totals.get(day, 0.0) + float(grp["v"].sum())
    if not totals:
        return None
    today = datetime.now().date()
    valid = {d: v for d, v in totals.items() if (today - d).days <= days}
    if not valid:
        valid = totals
    best = max(valid)
    return int(round(valid[best])), best


def _avg_series_latest_day(records, days):
    samples = []
    today = datetime.now().date()
    for rec in records:
        t = _series_to_dt(rec["time"])
        v = pd.to_numeric(rec["value"], errors="coerce")
        df = pd.DataFrame({"t": t, "v": v}).dropna()
        if df.empty:
            continue
        df["day"] = df["t"].dt.date
        df = df[df["day"].apply(lambda d: (today - d).days <= days)]
        samples.append(df["v"])
    if not samples:
        return None
    s = pd.concat(samples)
    return float(s.mean()), float(s.min()), float(s.max()), int(len(s))


def _latest_bp(records):
    sys_v, dia_v = None, None
    for rec in records:
        s = pd.to_numeric(rec["systolic"], errors="coerce")
        d = pd.to_numeric(rec["diastolic"], errors="coerce")
        if s is None or d is None:
            continue
        valid = s.notna() & d.notna()
        if not valid.any():
            continue
        idx = int(np.flatnonzero(valid.values)[-1])
        sys_v, dia_v = float(s.iloc[idx]), float(d.iloc[idx])
    return (sys_v, dia_v) if sys_v is not None else None


def _latest_sleep(records):
    best = None
    for rec in records:
        start = rec.get("start")
        end = rec.get("end")
        if start is None or end is None:
            continue
        s = pd.to_datetime(pd.Series(start), errors="coerce")
        e = pd.to_datetime(pd.Series(end), errors="coerce")
        for i in range(len(s)):
            if pd.isna(s.iloc[i]) or pd.isna(e.iloc[i]):
                continue
            if e.iloc[i] < s.iloc[i]:
                continue
            dur = (e.iloc[i] - s.iloc[i]).total_seconds() / 3600.0
            if dur <= 0 or dur > 24 or best is None or e.iloc[i] > best["end"]:
                best = {"start": s.iloc[i], "end": e.iloc[i],
                        "duration_hrs": dur,
                        "quality": None}
                try:
                    best["quality"] = float(rec["quality"].iloc[i])
                except Exception:
                    best["quality"] = None
    return best


def _activity_minutes(records, days):
    total = 0.0
    today = datetime.now().date()
    for rec in records:
        start = pd.to_datetime(pd.Series(rec.get("start")), errors="coerce")
        end = pd.to_datetime(pd.Series(rec.get("end")), errors="coerce")
        for i in range(len(start)):
            if pd.isna(start.iloc[i]) or pd.isna(end.iloc[i]):
                continue
            if (today - start.iloc[i].date()).days <= days:
                total += max(0.0, (end.iloc[i] - start.iloc[i]).total_seconds() / 60.0)
    return total


def extract_features(records, window_days=7):
    """Build the 8 health features from detected watch records. Returns dict."""
    f = {k: None for k in FEATURE_KEYS}
    found = set()

    rec = records.get("steps") or []
    step_res = _latest_day_step_total(rec, window_days)
    if step_res:
        f["Daily Steps"], _ = step_res
        found.add("steps")

    hr = records.get("heart_rate") or []
    hr_res = _avg_series_latest_day(hr, window_days)
    if hr_res:
        f["Heart Rate"] = round(hr_res[0])
        found.add("heart_rate")

    sp = records.get("spo2") or []
    sp_res = _avg_series_latest_day(sp, window_days)
    if sp_res:
        f["_spo2"] = round(sp_res[0])
        found.add("spo2")

    sl = records.get("sleep") or []
    sleep = _latest_sleep(sl)
    if sleep:
        f["Sleep Duration"] = round(sleep["duration_hrs"], 1)
        if sleep["quality"] is not None:
            q = sleep["quality"]
            f["Quality of Sleep"] = int(round(q if q <= 10 else q / 10.0))
            found.add("sleep_quality")
        found.add("sleep")

    ac = records.get("activity") or []
    act_res = _activity_minutes(ac, window_days)
    if act_res and act_res > 0:
        f["Physical Activity Level"] = int(round(act_res))
        found.add("activity")
    elif f.get("Daily Steps") is not None:
        f["Physical Activity Level"] = _steps_to_activity(f["Daily Steps"])
        f["_activity_derived"] = True
        found.add("activity")

    st = records.get("stress") or []
    st_res = _avg_series_latest_day(st, window_days)
    if st_res:
        val = st_res[0]
        f["Stress Level"] = int(round(min(10, max(1, val / 10.0)))) if val > 10 else int(round(val))
        found.add("stress")

    bp = records.get("bp") or []
    bp_res = _latest_bp(bp)
    if bp_res:
        f["Systolic_BP"], f["Diastolic_BP"] = int(round(bp_res[0])), int(round(bp_res[1]))
        found.add("bp")

    return {"features": f, "found": found, "record_counts": _counts(records)}


def _counts(records):
    return {cat: len(records.get(cat) or []) for cat in
            ("steps", "heart_rate", "spo2", "stress", "sleep", "activity", "bp")}


# ============================================================
# Model-ready input construction
# ============================================================

MIN_RANGE = {
    "Sleep Duration": 4.0, "Quality of Sleep": 1, "Physical Activity Level": 0,
    "Stress Level": 1, "Heart Rate": 50, "Daily Steps": 1000,
    "Systolic_BP": 90, "Diastolic_BP": 60,
}
MAX_RANGE = {
    "Sleep Duration": 8.5, "Quality of Sleep": 10, "Physical Activity Level": 120,
    "Stress Level": 10, "Heart Rate": 100, "Daily Steps": 20000,
    "Systolic_BP": 180, "Diastolic_BP": 120,
}

# Normal (healthy) ranges used when the watch does not provide a feature.
# A value is sampled uniformly inside the band so the model sees realistic
# "normal" inputs instead of a single fixed median.
NORMAL_RANGES = {
    "Sleep Duration": (4.0, 7.0),
    "Quality of Sleep": (6, 9),
    "Physical Activity Level": (30, 90),
    "Stress Level": (3, 8),
    "Heart Rate": (60, 80),
    "Daily Steps": (5000, 10000),
    "Systolic_BP": (115, 135),
    "Diastolic_BP": (75, 90),
}


def normal_fallback(feature):
    """Sample a random 'normal' value inside the feature's healthy range."""
    lo, hi = NORMAL_RANGES.get(
        feature, (MIN_RANGE.get(feature, 0), MAX_RANGE.get(feature, 10)))
    return random.uniform(lo, hi)


def clamp_to_dataset(feature, value):
    if value is None:
        return None
    lo, hi = MIN_RANGE.get(feature), MAX_RANGE.get(feature)
    return float(min(hi, max(lo, float(value))))


def build_model_input(health_features, gender, age, occupation, bmi):
    """Combine watch metrics (auto) + personal fields (manual) -> 12 model features.

    Returns (input_dict, display_dict, missing).
    """
    hf = health_features.get("features", health_features)
    personal_ok = gender is not None and age is not None and occupation is not None and bmi is not None
    missing = []
    if not personal_ok:
        missing.append("Gender/Age/Occupation/BMI (enter below)")

    auto = {}
    for k in ("Sleep Duration", "Quality of Sleep", "Physical Activity Level",
              "Stress Level", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"):
        v = clamp_to_dataset(k, hf.get(k)) if hf.get(k) is not None else None
        if v is None:
            v = normal_fallback(k)   # watch doesn't provide it -> random normal value
            missing.append(k)
        auto[k] = round(v, 1) if k == "Sleep Duration" else int(round(v))

    display = {
        "Gender": gender, "Age": age, "Occupation": occupation, "BMI Category": bmi,
        **auto, "_spo2": health_features.get("_spo2"),
    }
    display = {k: v for k, v in display.items() if v is not None}
    return auto, display, missing