# ============================================================
# ble_capture.py — CLI for live capture / sync from a
# Fireboltt 046 / Da Fit (Moyoung protocol) BLE watch.
#
# Usage (from project root):
#   python scripts/ble_capture.py scan
#   python scripts/ble_capture.py info --addr <ADDRESS>
#   python scripts/ble_capture.py live  --addr <ADDRESS> --seconds 60
#   python scripts/ble_capture.py stress --addr <ADDRESS>        # live stress reading (retries)
#   python scripts/ble_capture.py sync  --addr <ADDRESS>
# ============================================================

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

from ble_watch import (  # noqa: E402
    BLEAK_AVAILABLE, CaptureDB, MoyoungClient,
    default_db_path, load_records_from_db, scan_watches,
)


def _p(msg):
    print(msg, flush=True)


def _progress(msg):
    print(f"  - {msg}", flush=True)


async def _run_info(address, db_path):
    db = CaptureDB(db_path)
    client = MoyoungClient(address, db)
    try:
        await client.connect()
        info = client.device_info
        print(f"Connected to {address}  protocol={client.protocol}  mtu={client.mtu}")
        for k, v in info.items():
            if v:
                print(f"  {k}: {v}")
        bat = await client.read_battery()
        if bat is not None:
            print(f"  battery: {bat}%")
        steps = await client.read_today_steps()
        if steps is not None:
            print(f"  today steps: {steps}")
    finally:
        await client.close()
        db.close()


async def _run_live(address, seconds, db_path, hr_every):
    db = CaptureDB(db_path)
    client = MoyoungClient(address, db)
    try:
        await client.connect()
        info = client.device_info
        print(f"Connected to {address}  protocol={client.protocol}  mtu={client.mtu}")
        for k, v in info.items():
            if v:
                print(f"  {k}: {v}")
        bat = await client.read_battery()
        if bat is not None:
            print(f"  battery: {bat}%")
        await client.sync_time()
        print(f"Capturing for {seconds}s (Ctrl+C to stop early)...")
        start = time.monotonic()
        n_hr = 0
        n_spo2 = 0
        n_bp = 0
        n_steps = 0

        def on_step(_s):
            nonlocal n_steps
            n_steps += 1

        client.on_step = on_step
        try:
            await client.capture_sequential(
                hr_seconds=seconds, hr_every=hr_every,
                on_sample=lambda kind, v: print(
                    f"  [{time.strftime('%H:%M:%S')}] {kind}: {v}", flush=True))
        except KeyboardInterrupt:
            pass
        print("\nDone. Sample counts:")
        print(" ", db.summary())
        print(f"\nDB: {db_path}")
    finally:
        await client.close()
        db.close()


async def _run_sync(address, db_path):
    db = CaptureDB(db_path)
    client = MoyoungClient(address, db)
    try:
        await client.connect()
        info = client.device_info
        print(f"Connected to {address}  protocol={client.protocol}  mtu={client.mtu}")
        for k, v in info.items():
            if v:
                print(f"  {k}: {v}")
        night = await client.sync_history(with_workouts=True, on_progress=_progress)
        if night:
            print("\nLast night summary:")
            print(f"  start     : {night['start']} ({time.strftime('%H:%M', time.localtime(night['start']))})")
            print(f"  end       : {time.strftime('%H:%M', time.localtime(night['end']))}")
            print(f"  duration  : {night['duration_hrs']} h")
            print(f"  quality   : {night['quality']}")
        print("\nRow counts:", db.summary())
        print(f"\nDB: {db_path}")
    finally:
        await client.close()
        db.close()


async def _run_stress(address, db_path, attempts, timeout):
    db = CaptureDB(db_path)
    client = MoyoungClient(address, db)
    try:
        await client.connect()
        info = client.device_info
        print(f"Connected to {address}  protocol={client.protocol}  mtu={client.mtu}")
        for k, v in info.items():
            if v:
                print(f"  {k}: {v}")
        await client.sync_time()

        def on_attempt(i, n):
            print(f"  stress attempt {i}/{n} ...", flush=True)

        print(f"Measuring stress (up to {attempts} attempts, ~{timeout}s each)...")
        val = await client.measure_stress_with_retry(
            attempts=attempts, timeout=timeout, on_attempt=on_attempt)
        if val is None:
            last = db.latest_stress()
            if last:
                when = time.strftime('%H:%M', time.localtime(last["ts"]))
                print(f"  watch ignored on-demand stress; last stored value = "
                      f"{last['value']} (at {when})")
            else:
                print("  no stress value captured (watch ignored on-demand stress).")
        else:
            await db.add_stress(time.time(), int(val))
            print(f"  ✅ live stress: {val}")
        print("\nRow counts:", db.summary())
        print(f"\nDB: {db_path}")
    finally:
        await client.close()
        db.close()


def _load_model():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    import pickle
    model = pickle.load(open(os.path.join(model_dir, "best_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    le = pickle.load(open(os.path.join(model_dir, "label_encoder.pkl"), "rb"))
    fnames = pickle.load(open(os.path.join(model_dir, "feature_names.pkl"), "rb"))
    cat_enc = pickle.load(open(os.path.join(model_dir, "cat_encoders.pkl"), "rb"))
    return model, scaler, le, fnames, cat_enc


def _predict_from_db(db_path, window, gender, age, occupation, bmi, with_live=None):
    records = load_records_from_db(db_path)
    counts = records.get("record_counts")
    if counts is None:
        counts = {cat: len(records.get(cat) or []) for cat in
                  ("steps", "heart_rate", "spo2", "stress", "sleep", "activity", "bp")}
    print("  row counts:", {k: v for k, v in counts.items() if v})
    if not any(counts.values()):
        print("  No watch data in the capture DB yet - run 'live'/'sync', or wear it overnight.")
        return None

    from smartwatch_import import build_model_input, extract_features, FEATURE_KEYS
    feat = extract_features(records, window_days=window)
    hf, display, missing = build_model_input(feat, gender, age, occupation, bmi)
    print("  features:", {k: v for k, v in hf.items()})
    print("  missing -> dataset defaults:", missing)

    model, scaler, le, fnames, cat_enc = _load_model()
    import pandas as pd
    d = {
        "Gender": int(cat_enc["Gender"].transform([gender])[0]),
        "Age": int(age),
        "Occupation": int(cat_enc["Occupation"].transform([occupation])[0]),
        "BMI Category": int(cat_enc["BMI Category"].transform([bmi])[0]),
    }
    for k in FEATURE_KEYS:
        if hf.get(k) is not None:
            d[k] = hf[k]
    df = pd.DataFrame([d])
    for col in fnames:
        if col not in df.columns:
            df[col] = 0
    df = df[fnames]
    X = scaler.transform(df)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    return label, proba, list(le.classes_)


async def _run_predict(args, db_path):
    if args.addr:
        db = CaptureDB(db_path)
        client = MoyoungClient(args.addr, db)
        try:
            await client.connect()
            print(f"Connected to {args.addr}  protocol={client.protocol}")
            if args.live and args.live > 0:
                print(f"Capturing vitals one at a time for {args.live}s ...")
                await client.capture_sequential(
                    hr_seconds=args.live, hr_every=2.5,
                    on_sample=lambda kind, v: _progress(f"{kind}: {v}"))
            print("Syncing sleep / steps / HR history ...")
            await client.sync_history(with_workouts=True, on_progress=_progress)
            print(f"Captured rows: {sum(db.summary().values())}\n")
        finally:
            await client.close()
            db.close()
    else:
        print(f"Using existing capture DB: {db_path}\n")

    result = _predict_from_db(db_path, args.window, args.gender, args.age,
                              args.occupation, args.bmi)
    if result is None:
        return
    label, proba, classes = result
    print("=" * 46)
    print(f"  PREDICTION: {label}")
    for c, p in zip(classes, proba):
        bar = "#" * int(round(p * 40))
        print(f"    {c:12s} {p * 100:6.1f}%  {bar}")
    print("=" * 46)
    print(f"  Personal profile (manual): {args.gender}, {args.age}, "
          f"{args.occupation}, {args.bmi}")


def main():
    ap = argparse.ArgumentParser(description="Live BLE capture for Fireboltt 046 / Da Fit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="scan for nearby watches")
    p_scan.add_argument("--timeout", type=float, default=10.0)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--addr", required=True, help="BLE MAC address of the watch")
    common.add_argument("--db", default=None,
                        help="output SQLite DB path (default capture/watch_capture.db)")

    p_info = sub.add_parser("info", parents=[common], help="connect and show device info")
    p_live = sub.add_parser("live", parents=[common], help="live HR/SpO2/BP capture")
    p_live.add_argument("--seconds", type=int, default=60)
    p_live.add_argument("--hr-every", type=float, default=3.0)
    p_sync = sub.add_parser("sync", parents=[common], help="sync sleep/steps/HR history")
    p_stress = sub.add_parser("stress", parents=[common],
                              help="measure a live stress reading (with retries)")
    p_stress.add_argument("--attempts", type=int, default=3,
                          help="on-demand stress probes before giving up")
    p_stress.add_argument("--timeout", type=float, default=20.0,
                          help="seconds to wait per attempt")
    p_feat = sub.add_parser("features",
                            help="extract model features from a capture DB")
    p_feat.add_argument("--db", default=None,
                        help="path to the capture SQLite DB (default capture/watch_capture.db)")
    p_feat.add_argument("--window", type=int, default=7,
                        help="extract_features lookback window (days)")

    p_pred = sub.add_parser("predict",
                            help="pull data from the watch (or reuse --db) and predict a sleep disorder")
    p_pred.add_argument("--addr", default=None,
                        help="BLE MAC address of the watch (omit to predict from the existing --db)")
    p_pred.add_argument("--db", default=None,
                        help="path to the capture SQLite DB (default capture/watch_capture.db)")
    p_pred.add_argument("--window", type=int, default=7,
                        help="extract_features lookback window (days)")
    p_pred.add_argument("--gender", default="Male")
    p_pred.add_argument("--age", type=int, default=30)
    p_pred.add_argument("--occupation", default="Software Engineer")
    p_pred.add_argument("--bmi", default="Normal")
    p_pred.add_argument("--live", type=float, default=0.0,
                        help="run a live HR/SpO2/BP session for N seconds before syncing")

    args = ap.parse_args()
    db_path = getattr(args, "db", None) or default_db_path()

    needs_ble = args.cmd in ("scan", "info", "live", "sync", "stress") or \
        (args.cmd == "predict" and bool(args.addr))
    if not BLEAK_AVAILABLE and needs_ble:
        print("bleak is not installed. Run:  pip install bleak")
        sys.exit(1)

    if args.cmd == "scan":
        print("Scanning for watches ...")
        loop = asyncio.new_event_loop()
        try:
            devs = loop.run_until_complete(scan_watches(args.timeout))
        except Exception as e:
            print(f"Scan failed: {e}")
            sys.exit(1)
        finally:
            loop.close()
        if not devs:
            print("No watches found. Make sure the watch is ON and near the laptop "
                  "(and not connected to your phone's Da Fit app).")
            return
        for d in devs:
            flag = "  [service 0xFEEA]" if d.service_feea else ""
            print(f"  {d.address}  {d.name!r}  rssi={d.rssi}{flag}")
        return

    if args.cmd == "features":
        try:
            records = load_records_from_db(db_path)
        except Exception as e:
            print(f"Could not open DB {db_path}: {e}")
            sys.exit(1)
        from smartwatch_import import extract_features
        feat = extract_features(records, window_days=args.window)
        print("Extracted features:")
        for k, v in feat["features"].items():
            print(f"  {k:26s} {v}")
        print(f"\nSources found: {sorted(feat['found'])}")
        print(f"Record counts: {feat['record_counts']}")
        return

    loop = asyncio.new_event_loop()
    try:
        if args.cmd == "info":
            loop.run_until_complete(_run_info(args.addr, db_path))
        elif args.cmd == "live":
            loop.run_until_complete(_run_live(args.addr, args.seconds, db_path, args.hr_every))
        elif args.cmd == "sync":
            loop.run_until_complete(_run_sync(args.addr, db_path))
        elif args.cmd == "stress":
            loop.run_until_complete(_run_stress(args.addr, db_path,
                                                args.attempts, args.timeout))
        elif args.cmd == "predict":
            loop.run_until_complete(_run_predict(args, db_path))
    finally:
        loop.close()


if __name__ == "__main__":
    main()