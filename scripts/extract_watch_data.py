#!/usr/bin/env python3
# ============================================================
# extract_watch_data.py — CLI for testing watch data parsing
# without the Streamlit UI.
#
# Usage:
#   python scripts/extract_watch_data.py <path-to-zip-or-db-or-csv> [--window 7] [--json]
#
# Examples:
#   python scripts/extract_watch_data.py export.zip
#   python scripts/extract_watch_data.py goboult.db --window 14
#   python scripts/extract_watch_data.py data.csv --json
# ============================================================

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from smartwatch_import import load_records, extract_features, build_model_input, FEATURE_KEYS


def main():
    parser = argparse.ArgumentParser(description="Extract real features from smartwatch data.")
    parser.add_argument("source", help="Path to ZIP, .db, .sqlite, or .csv file")
    parser.add_argument("--window", type=int, default=7,
                        help="Number of days to look back (default: 7)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of a readable table")
    parser.add_argument("--gender", default="Male", help="Gender (for model input)")
    parser.add_argument("--age", type=int, default=30, help="Age (for model input)")
    parser.add_argument("--occupation", default="Software Engineer",
                        help="Occupation (for model input)")
    parser.add_argument("--bmi", default="Normal", help="BMI Category (for model input)")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: file not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading records from: {args.source}")
    records = load_records(args.source)

    raw_tables = records.get("raw", {})
    print(f"[INFO] Tables discovered: {', '.join(raw_tables.keys()) or 'none'}")
    for name, df in raw_tables.items():
        print(f"  - {name}: {len(df)} rows, columns={list(df.columns)[:6]}...")

    filenames = records.get("_filenames", [])
    print(f"[INFO] Source file(s): {', '.join(filenames) or 'N/A'}")

    print(f"\n[INFO] Extracting features (window={args.window} days)...")
    feat = extract_features(records, window_days=args.window)
    hf, display, missing = build_model_input(
        feat, args.gender, args.age, args.occupation, args.bmi
    )

    if args.json:
        out = {
            "features": hf,
            "display": display,
            "missing": missing,
            "record_counts": feat.get("record_counts", {}),
            "raw_tables": {k: len(v) for k, v in raw_tables.items()},
            "sources": filenames,
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print("  EXTRACTED HEALTH FEATURES")
        print("=" * 60)
        print(f"  {'Feature':<30} {'Value':>10}")
        print("  " + "-" * 44)
        for k in FEATURE_KEYS:
            v = hf.get(k)
            print(f"  {k:<30} {str(v):>10}")
        print("  " + "-" * 44)

        if feat.get("features", {}).get("_spo2"):
            print(f"  {'SpO2 (watch-only, not in model)':<30} {feat['features']['_spo2']:>10}")

        counts = feat.get("record_counts", {})
        print(f"\n  Data sources:   {', '.join(filenames) or 'N/A'}")
        print(f"  Tables found:   {len(raw_tables)}")
        print(f"  Record groups:  steps={counts.get('steps',0)}, hr={counts.get('heart_rate',0)}, "
              f"spo2={counts.get('spo2',0)}, sleep={counts.get('sleep',0)}, "
              f"bp={counts.get('bp',0)}, stress={counts.get('stress',0)}, "
              f"activity={counts.get('activity',0)}")

        if missing:
            print(f"\n  !!  Missing features (manual inputs needed): {', '.join(missing)}")

        print()
        print("=" * 60)
        print("  MODEL-READY INPUT (after encoding)")
        print("=" * 60)
        print(json.dumps(hf, indent=2))


if __name__ == "__main__":
    main()
