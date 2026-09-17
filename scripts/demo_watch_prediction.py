# ============================================================
# demo_watch_prediction.py — Live smartwatch demo: watch -> capture
# -> features -> model -> sleep-disorder prediction.
#
# WITH the watch nearby (Fireboltt 046 / Da Fit):
#   python scripts/demo_watch_prediction.py --addr <MAC>
#   python scripts/demo_watch_prediction.py --addr <MAC> --live 90 --gender Male --age 30
#
# WITHOUT the watch (replays previously captured BLE data):
#   python scripts/demo_watch_prediction.py
#
# Every step is printed AND saved to report/demo_watch_prediction.txt
# so the transcript can be shown / submitted as demo output.
# ============================================================

import argparse
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "app"))

LOG_PATH = os.path.join(BASE, "report", "demo_watch_prediction.txt")

_lines = []
_log = None


def say(msg=""):
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%H:%M:%S")
    print(msg, flush=True)
    _lines.append(f"[{ts}] {msg}")
    if _log is not None:
        _log.write(f"[{ts}] {msg}\n")
        _log.flush()


async def _run_live_session(args, db_path):
    import asyncio
    from ble_watch import CaptureDB, MoyoungClient

    db = CaptureDB(db_path)
    client = MoyoungClient(args.addr, db)
    try:
        def log(*a, **k):
            say(*a, **k)

        await client.connect(max_tries=args.max_tries, timeout=12.0)
        info = client.device_info
        for k, v in info.items():
            if v:
                log(f"         {k:12s}: {v}")
        log("         protocol: " + client.protocol)
        log("")
        log(f" [2/5] Capturing vitals one at a time "
            f"(HR {args.live:.0f}s, then SpO2, then BP, then stress) ...")

        def on_sample(kind, v):
            log(f"         {kind:14s}: {v}")

        def on_phase(name):
            log(f"         -- phase: {name} --")

        await client.capture_sequential(
            hr_seconds=args.live, hr_every=2.5,
            on_sample=on_sample, on_phase=on_phase)
        try:
            steps = await asyncio.wait_for(client.read_today_steps(), timeout=8)
            if steps:
                log(f"         steps today   : {steps}")
        except Exception:
            pass
        try:
            bat = await asyncio.wait_for(client.read_battery(), timeout=8)
            if bat:
                log(f"         battery      : {bat}%")
        except Exception:
            pass
        log("")
        log(" [3/5] Syncing last 2 nights (sleep / step / HR history) ...")
        try:
            night = await asyncio.wait_for(
                client.sync_history(with_workouts=True,
                                    on_progress=lambda m: log("         - " + m)),
                timeout=180)
        except Exception as e:
            log(f"         sync note: {e}")
            night = None
        try:
            await client.close()
        except Exception:
            pass
        return night
    except Exception:
        try:
            await client.close()
        except Exception:
            pass
        raise
    finally:
        db.close()


def main():
    ap = argparse.ArgumentParser(description="Live watch -> prediction demo")
    ap.add_argument("--addr", default=None, help="watch BLE MAC (omit to replay capture DB)")
    ap.add_argument("--live", type=float, default=45.0, help="live vitals capture seconds")
    ap.add_argument("--max-tries", type=int, default=6,
                    help="BLE connect handshake retries (fail fast in demos)")
    ap.add_argument("--gender", default="Male")
    ap.add_argument("--age", type=int, default=30)
    ap.add_argument("--occupation", default="Software Engineer")
    ap.add_argument("--bmi", default="Normal")
    args = ap.parse_args()

    from ble_watch import (BLEAK_AVAILABLE, CaptureDB, MoyoungClient,
                           default_db_path)
    from smartwatch_import import (build_model_input, extract_features,
                                   load_records, FEATURE_KEYS)

    # Model artifacts
    import pickle
    mdir = os.path.join(BASE, "models")
    model = pickle.load(open(os.path.join(mdir, "best_model.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(mdir, "scaler.pkl"), "rb"))
    le = pickle.load(open(os.path.join(mdir, "label_encoder.pkl"), "rb"))
    fnames = pickle.load(open(os.path.join(mdir, "feature_names.pkl"), "rb"))
    cat_enc = pickle.load(open(os.path.join(mdir, "cat_encoders.pkl"), "rb"))
    meta = pickle.load(open(os.path.join(mdir, "model_meta.pkl"), "rb"))
    acc = meta["test_accuracy"]

    say("=" * 64)
    say(" SLEEP DISORDER PREDICTOR - SMARTWATCH DEMO")
    say("=" * 64)
    say(f" [MODEL] {meta['model_name']} loaded  (test accuracy {acc*100:.1f}%)")
    say("")

    db_path = os.path.join(BASE, "capture", "watch_capture.db")

    # ---------- CONNECT / CAPTURE ----------
    if args.addr:
        if not BLEAK_AVAILABLE:
            say(" ERROR: bleak not installed -> pip install bleak")
            return 1
        say(" [1/5] Connecting to smartwatch " + args.addr + " over Bluetooth ...")
        say("         (keep the watch ON and near the laptop)")
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        global _log
        _log = open(LOG_PATH, "w", encoding="utf-8")
        db_path = default_db_path()
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                night = loop.run_until_complete(_run_live_session(args, db_path))
            finally:
                loop.close()
        except Exception as e:
            say(f" ERROR: {e}")
            say("")
            say(" Live session failed. The watch may be connected to the phone's")
            say(" Da Fit app, or its Bluetooth is off. Fix that and re-run:")
            say(f"   python scripts/demo_watch_prediction.py --addr {args.addr}")
            say("")
            say(" Meanwhile, a full demo transcript from the data this watch")
            say(" already captured can be produced with:")
            say("   python scripts/demo_watch_prediction.py")
            return 1
        if night:
            say(f"         Last night    : {night['duration_hrs']} h "
                f"(quality {night['quality']}/10)")
        else:
            say("         (no sleep history on the watch yet - wear it to sleep")
            say("          for a night to capture real sleep duration)")
        db = CaptureDB(db_path)
        say("         captured rows : " + ", ".join(
            f"{k}={v}" for k, v in db.summary().items() if v))
        db.close()
        say("         [connected & captured OK]")
    else:
        say(" [1/5] No --addr given -> using data already captured via BLE")
        say(f"       from capture/watch_capture.db")
        if not os.path.exists(db_path):
            say("       ERROR: capture DB not found. Run:")
            say("       python scripts/ble_capture.py live --addr <MAC>")
            say("       python scripts/ble_capture.py sync --addr <MAC>")
            return 1
    say("")

    # ---------- FEATURES ----------
    say(" [4/5] Extracting model features from the watch data ...")
    records = load_records(db_path)
    feat = extract_features(records, window_days=7)
    hf = feat["features"]
    counts = feat["record_counts"]
    if not any(counts.values()):
        say("       ERROR: no rows inside the capture DB yet.")
        return 1
    health, display, missing = build_model_input(feat, args.gender, args.age,
                                                 args.occupation, args.bmi)
    say("       detected tables: " + ", ".join(
        sorted(k for k in records if not k.startswith("_"))))
    for k in FEATURE_KEYS:
        v = health.get(k)
        mark = "" if k not in missing else "  (dataset-median fallback)"
        if k == "Physical Activity Level" and hf.get("_activity_derived"):
            mark += "  (derived from daily steps)"
        say(f"         {k:22s}: {v}{mark}")
    if hf.get("_spo2") is not None:
        say(f"         {'SpO2 (watch)':22s}: {hf['_spo2']}")
    say("")

    # ---------- PREDICT ----------
    say(f" [5/5] Predicting sleep disorder for profile:")
    say(f"         Gender={args.gender}  Age={args.age}  "
        f"Occupation={args.occupation}  BMI={args.bmi}")
    say("")
    d = {
        "Gender": int(cat_enc["Gender"].transform([args.gender])[0]),
        "Age": int(args.age),
        "Occupation": int(cat_enc["Occupation"].transform([args.occupation])[0]),
        "BMI Category": int(cat_enc["BMI Category"].transform([args.bmi])[0]),
    }
    for k in FEATURE_KEYS:
        if health.get(k) is not None:
            d[k] = health[k]
    import pandas as pd
    df = pd.DataFrame([d])
    for col in fnames:
        if col not in df.columns:
            df[col] = 0
    X = scaler.transform(df[fnames])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]

    say(" RESULT")
    say("-" * 64)
    for c, p in zip(le.classes_, proba):
        bar = "#" * int(round(p * 46))
        say(f"   {c:12s} {p*100:6.1f}%  {bar}")
    say("-" * 64)
    say(f"   >>> PREDICTION: {label}   (confidence {max(proba)*100:.1f}%)")
    say("=" * 64)
    say("")
    if label == "None":
        say(" Your watch signals are consistent with healthy sleep patterns.")
    elif label == "Insomnia":
        say(" Signs consistent with insomnia were detected from the watch data.")
    else:
        say(" Signs consistent with sleep apnea were detected from the watch data.")
    say("Please consult a healthcare professional for a proper diagnosis.")
    say("")

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    say(f" Full transcript saved to: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())