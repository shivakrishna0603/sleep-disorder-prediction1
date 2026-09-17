# 🛌 Sleep Disorder Prediction Using Wearable Sensor Data
**B.Tech Final Year Mini Project | Computer Science & IT**

---

## 📌 Problem Statement
Sleep disorders like insomnia and sleep apnea affect millions of people. Early prediction using
wearable sensor data (heart rate, activity, sleep patterns) can help avoid expensive clinical tests.

---

## 🗂️ Project Structure
```
sleep-disorder-prediction/
├── data/
│   └── sleep_health.csv          ← Place your dataset here
├── notebooks/
│   ├── 01_EDA.py                 ← Exploratory Data Analysis
│   └── 02_model_training.py      ← Preprocessing + Model Training
├── models/
│   ├── best_model.pkl            ← Saved best ML model
│   ├── scaler.pkl                ← StandardScaler
│   ├── label_encoder.pkl         ← Target LabelEncoder
│   ├── cat_encoders.pkl          ← Categorical encoders
│   └── feature_names.pkl         ← Feature column order
├── capture/
│   └── watch_capture.db          ← Live BLE data stored here (auto-created)
├── app/
│   ├── app.py                    ← Streamlit Web App
│   ├── smartwatch_import.py      ← Real watch data parser (schema-discovering)
│   └── ble_watch.py              ← Live BLE client (Fireboltt 046 / Da Fit)
├── scripts/
│   ├── ble_capture.py            ← CLI: scan / info / live / sync / features / predict
│   ├── extract_watch_data.py     ← CLI: parse watch data without the UI
│   ├── test_ble_capture.py       ← Mocked-GATT BLE protocol tests
│   └── test_watch_pipeline.py    ← Synthetic end-to-end parser test
├── report/
│   ├── eda_plots.png
│   ├── correlation_heatmap.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset
1. Go to: https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset
2. Download `Sleep_health_and_lifestyle_dataset.csv`
3. Rename it to `sleep_health.csv`
4. Place it in the `data/` folder

### Step 3: Run EDA
```bash
cd notebooks
python 01_EDA.py
```
This generates plots in the `report/` folder.

### Step 4: Train the Model
```bash
python 02_model_training.py
```
This trains and saves the best model to `models/`.

### Step 5: Launch the Web App
```bash
cd app
streamlit run app.py
```
Open your browser at: http://localhost:8501

---

## ⌚ Using Real Data From Your GOBOULT Fit Smartwatch

The **⌚ Smartwatch Data** tab replaces the manual slider inputs with **real data**
from your watch. There are two ways to feed it live data:

1. **Directly over Bluetooth (recommended)** — the laptop's BLE adapter talks to
   a **Fireboltt 046 / Da Fit** watch (Moyoung protocol) in real time: live
   HR / SpO₂ / blood-pressure + last-night sleep and step history, stored in
   `capture/watch_capture.db`. The model then predicts straight from that DB —
   no file transfer or phone needed.
2. **By uploading files** — the GOBOULT Fit app has no export button, so the app
   also reads raw databases directly.

### ⚡ Direct watch → prediction (Bluetooth)

```bash
python scripts/ble_capture.py scan          # find the watch, note its MAC
python scripts/ble_capture.py predict --addr <MAC> \
    --gender Male --age 30 --occupation "Software Engineer" --bmi Normal
```

`predict` connects, runs an optional quick live-vitals session (`--live 60` for
HR/SpO₂/BP), syncs the last 2 nights of sleep + steps + HR history, and prints
the disorder prediction with per-class probabilities. Personal fields the watch
can't read (`Gender`, `Age`, `Occupation`, `BMI Category`) are passed as flags;
any health feature the watch hasn't captured yet falls back to the dataset
median instead of a meaningless 0.

Other CLI commands:

```bash
python scripts/ble_capture.py info  --addr <MAC>      # battery, firmware, steps
python scripts/ble_capture.py live  --addr <MAC> --seconds 90
python scripts/ble_capture.py sync  --addr <MAC>      # sleep + steps + HR history
python scripts/ble_capture.py features                # features from --db
python scripts/ble_capture.py predict --gender ...    # predict from an existing --db
```

The same flow is available in the app under **⌚ Smartwatch Data → 🔴 Live capture**:
scan → connect → Start live capture → **🌙 Sync last 2 days** → click
**🔍 Run Sleep Disorder Prediction from Watch Data**.

### Data sources (pick any)

| Source | How to get it | Notes |
|--------|---------------|-------|
| **Live Bluetooth (laptop)** | `ble_capture.py scan` / **⌚ Smartwatch Data** tab | Fireboltt 046 / Da Fit, Moyoung protocol. No phone needed. |
| **Health Connect zip** | Android 14+: Settings → Health Connect → Manage data → Export (or a daily schedule to Google Drive) | Works if GOBOULT Fit / other apps write to Health Connect. |
| **GOBOULT Fit / Crrepa `.db`** | Copy the app's databases off the phone (`adb` / root — see below) | The app introspects the schema, so `steps`, `heart_rate`, `sleep`, `spo2`, `bloodpressure`, `stress` tables are auto-detected regardless of firmware naming. |
| **CSV** | Export anything with step / sleep / HR columns | Simple fallback for testing. |

### Pulling GOBOULT Fit's database via adb (root / backup)

```bash
adb root                          # on a rooted device
adb shell "cat /data/data/com.crrepa.band.boultfit/databases/*.db" > goboult.db
# OR older Androids (≤11):
adb backup -noapk com.crrepa.band.boultfit -f goboult.ab
# then unpack the .ab (skip 24-byte header, zlib-decompress) into a .db/tar
```
Upload the resulting `.zip` / `.db` files in the **⌚ Smartwatch Data** tab.

### Test parsing without the UI

```bash
python scripts/extract_watch_data.py export.zip --window 7
python scripts/extract_watch_data.py goboult.db --json
python scripts/test_watch_pipeline.py     # synthetic end-to-end check
python scripts/test_ble_capture.py       # mocked-GATT BLE protocol check
```

### How it works

`app/smartwatch_import.py` introspects the SQLite schema (table + column names
vary by firmware) and matches steps / sleep / heart rate / SpO2 / blood pressure /
stress tables by keyword heuristics. `Gender`, `Age`, `Occupation`, `BMI Category`
are still entered manually. Any health feature the watch does not provide falls
back to the dataset median rather than a zero value, so a partial capture still
produces a meaningful prediction.

---

## 🧰 Tech Stack
| Component     | Tool |
|---------------|------|
| Language      | Python 3.10+ |
| ML Models     | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Web UI        | Streamlit |
| Notebook      | Google Colab / Jupyter |

---

## 🤖 Models Trained
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest       | Ensemble, handles noise well |
| Gradient Boosting   | High accuracy, robust |
| XGBoost             | Best performance typically |

---

## 📊 Features Used
- Age, Gender, Occupation
- Sleep Duration, Quality of Sleep
- Physical Activity Level
- Stress Level, BMI Category
- Heart Rate, Daily Steps
- Systolic BP, Diastolic BP

---

## 🎯 Output Classes
- **None** — No sleep disorder detected
- **Insomnia** — Difficulty falling/staying asleep
- **Sleep Apnea** — Breathing interruptions during sleep

---

## ⚠️ Disclaimer
This tool is for educational purposes only. Always consult a healthcare professional for medical diagnosis.
