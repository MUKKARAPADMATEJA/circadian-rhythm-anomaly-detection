# 🧬 Circadian Rhythm Anomaly Detection

> A lightweight deep learning project that detects disruptions in circadian rhythms using Apple Watch wearable sensor data and a compact Autoencoder neural network.

---

## 📌 Abstract

This project explores anomaly detection for circadian rhythm disruptions using wearable sensor data from Apple Health. A lightweight autoencoder is trained on normal behavioral patterns (Heart Rate & Step Count) and uses **reconstruction error analysis** to flag unusual deviations. The approach is designed to be energy-efficient and feasible for real-time, resource-constrained environments.

**SDGs Addressed:**
- 🌱 **Goal 3** – Good Health and Well-Being
- ⚙️ **Goal 9** – Industry, Innovation and Infrastructure

---

## 🏗️ Project Structure

```
.
├── parse_health_data.py     # Step 1: Parses Apple Health XML export → CSV
├── anomaly_detector.py      # Step 2: Trains autoencoder & generates anomaly results
├── dashboard.py             # Step 3: Streamlit visualization dashboard
├── requirements.txt         # Python dependencies
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+
- Apple Health data exported from iPhone (`export.xml`)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Parse Your Apple Health Data

Update the path to your `export.xml` in `parse_health_data.py`, then run:

```bash
python parse_health_data.py
```

This will generate `health_data_parsed.csv`.

### 4. Train the Autoencoder & Detect Anomalies

```bash
python anomaly_detector.py
```

This will:
- Resample data into hourly intervals
- Train a lightweight Autoencoder on 80% of the data (assumed normal)
- Set an anomaly threshold at the **95th percentile** of reconstruction errors
- Save `dashboard_data.csv` and `lightweight_autoencoder.pth`

### 5. Launch the Dashboard

```bash
streamlit run dashboard.py
```

Open your browser at **http://localhost:8501**

---

## 🤖 Model Architecture

A compact **Multilayer Perceptron Autoencoder** built with PyTorch:

```
Input (48-dim: 24hrs × 2 features)
    ↓  Encoder
  Linear(48 → 16) → ReLU
  Linear(16 → 8)  → ReLU   ← Latent Space
    ↓  Decoder
  Linear(8 → 16)  → ReLU
  Linear(16 → 48) → Sigmoid
    ↓
Output (Reconstructed 24-hour cycle)
```

**Anomaly Score** = Mean Squared Error between input and reconstructed output

---

## 📊 Dashboard Features

- **KPI Metrics** — Total hours monitored, anomaly count, anomaly rate, primary anomaly driver
- **👨‍⚕️ Digital Doctor's Analysis** — AI-powered health insights and personalized recommendations
- **Heart Rate Timeline** — With anomaly markers highlighted in red
- **Step Count Activity Chart** — Color-coded by anomaly status
- **Reconstruction Error Plot** — With anomaly threshold line

---

## 📦 Data Used

| Feature | Source |
|---|---|
| Heart Rate (BPM) | `HKQuantityTypeIdentifierHeartRate` |
| Step Count | `HKQuantityTypeIdentifierStepCount` |

Data is sourced from Apple Health XML export and aggregated into **hourly** intervals.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Autoencoder training & inference |
| Pandas / NumPy | Data processing |
| Scikit-learn | MinMax normalization |
| Streamlit | Interactive dashboard UI |
| Plotly | Interactive charts |

---

## 👨‍💻 Author

Built as a research project exploring lightweight AI for wearable health monitoring.
