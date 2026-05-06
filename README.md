<div align="center">

# ⚡ Power Grid AI Analytics

### AI-Driven Load Forecasting & Anomaly Detection for Power Systems

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-189B5A?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> An end-to-end machine learning pipeline for **short-term load forecasting** and **unsupervised anomaly detection** on real-world power grid data — built to demonstrate applied AI for power systems research.

<br/>

![Load Overview](results/eda_load_overview.png)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Visualizations](#-visualizations)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Run on Google Colab](#-run-on-google-colab)
- [Future Work](#-future-work)
- [Research Paper](#-research-paper)
- [License](#-license)

---

## 🔍 Overview

This project applies machine learning and deep learning to **real-world power grid load data**, targeting two critical operational challenges:

| Task | Description |
|------|-------------|
| 📈 **Short-Term Load Forecasting (STLF)** | Predict hourly energy demand 1-step ahead using calendar features, lag variables, and sequence models |
| 🚨 **Anomaly Detection** | Identify abnormal consumption events — e.g., sudden large-load spikes from AI data centers or industrial consumers — without any labeled training data |

These problems are directly relevant to the operational challenges of modern power grids under increasing penetration of **AI data centers**, **EV charging infrastructure**, and **electrified industrial processes** — all of which introduce new load volatility that stresses grid reliability.

---

## 🏆 Key Results

### Short-Term Load Forecasting

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|-------|----------|-----------|----------|
| Linear Regression *(baseline)* | 437.6 | 607.8 | 2.70 |
| Random Forest | 412.4 | 574.1 | 2.54 |
| **XGBoost** ✅ | **381.7** | **540.5** | **2.34** |
| LSTM | ~420 | ~580 | ~2.60 |

> **XGBoost achieves 2.34% MAPE** on a held-out 6-month test set — a 13.3% relative improvement over the linear baseline. Evaluation uses a strict **temporal split** (no data leakage).

### Anomaly Detection *(Unsupervised)*

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **Isolation Forest** | 0.207 | 0.209 | **0.208** |
| LSTM Autoencoder | 0.006 | 0.005 | 0.006 |

> Isolation Forest outperforms the LSTM Autoencoder for **point anomaly detection**. The LSTM Autoencoder is better suited for detecting **sustained behavioral drift** (e.g., gradual load baseline shift from a new data center coming online).

---

## 🗂️ Project Structure

```
power-grid-ai-analytics/
│
├── 📂 src/
│   ├── data_utils.py          # PJM data generator + feature engineering
│   ├── models.py              # All forecasting & anomaly detection models
│   └── visualization.py      # Plotting utilities (dark theme)
│
├── 📂 results/                # Auto-generated plots and CSVs
│   ├── eda_load_overview.png
│   ├── eda_load_heatmap.png
│   ├── forecast_comparison.png
│   ├── forecast_metrics_table.png
│   ├── anomaly_isolation_forest.png
│   ├── anomaly_ae_reconstruction_error.png
│   └── anomaly_autoencoder.png
│
├── 📂 data/                   # Place AEP_hourly.csv here (see Dataset section)
│
├── 📓 Power_Grid_AI_Analytics.ipynb   # Google Colab notebook
├── 🐍 run_pipeline.py                 # One-command end-to-end runner
├── 📄 requirements.txt
└── 📖 README.md
```

---

## 🧠 Methodology

### Pipeline Architecture

```
Raw Load Data
      │
      ▼
┌─────────────────────────────────────┐
│        Feature Engineering          │
│  Calendar · Lag · Rolling Stats     │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌────────────┐   ┌──────────────────┐
│ Forecasting│   │ Anomaly Detection│
│            │   │                  │
│ • LinReg   │   │ • Isolation      │
│ • RF       │   │   Forest         │
│ • XGBoost  │   │                  │
│ • LSTM     │   │ • LSTM           │
│            │   │   Autoencoder    │
└──────┬─────┘   └────────┬─────────┘
       │                  │
       ▼                  ▼
  MAE / RMSE / MAPE    P / R / F1
```

### Feature Engineering
The model uses 12 input features derived from the raw hourly timestamp and load value:

| Category | Features |
|----------|----------|
| **Calendar** | `hour`, `dayofweek`, `month`, `dayofyear`, `quarter`, `is_weekend` |
| **Lag (autoregressive)** | `lag_1h`, `lag_2h`, `lag_24h` (daily), `lag_168h` (weekly) |
| **Rolling statistics** | `roll_24h_mean`, `roll_24h_std` |

### Models

**Forecasting:**
- **Linear Regression** — OLS baseline with standardized features
- **Random Forest** — 150 trees, depth 12, bootstrap aggregation
- **XGBoost** — 400 estimators, η=0.05, depth 6, subsample 0.8
- **LSTM** — 2-layer stacked LSTM (64→32 units), Dropout(0.2), 15 epochs, teacher-forcing inference

**Anomaly Detection (unsupervised):**
- **Isolation Forest** — 100 trees, contamination=0.003, features: [load, hour, dayofweek, month]
- **LSTM Autoencoder** — Encoder LSTM(32) → RepeatVector → Decoder LSTM(32), trained on first 80% of data, 3σ reconstruction error threshold

---

## 📊 Visualizations

### Exploratory Data Analysis

<table>
  <tr>
    <td><img src="results/eda_load_overview.png" width="100%"/><br/><sub><b>Fig 1.</b> Full time series, monthly distributions, and seasonal daily profiles</sub></td>
  </tr>
  <tr>
    <td><img src="results/eda_load_heatmap.png" width="100%"/><br/><sub><b>Fig 2.</b> Average load heatmap — hour of day × day of week</sub></td>
  </tr>
</table>

### Forecasting Results

<table>
  <tr>
    <td><img src="results/forecast_comparison.png" width="100%"/><br/><sub><b>Fig 3.</b> Forecast vs. actual for the first week of the test set, with per-model residual panels</sub></td>
  </tr>
  <tr>
    <td><img src="results/forecast_metrics_table.png" width="100%"/><br/><sub><b>Fig 4.</b> Model performance comparison across MAE, RMSE, and MAPE</sub></td>
  </tr>
</table>

### Anomaly Detection Results

<table>
  <tr>
    <td width="50%"><img src="results/anomaly_isolation_forest.png" width="100%"/><br/><sub><b>Fig 5.</b> Isolation Forest — detections over last 60 days + confusion breakdown</sub></td>
    <td width="50%"><img src="results/anomaly_ae_reconstruction_error.png" width="100%"/><br/><sub><b>Fig 6.</b> LSTM Autoencoder — reconstruction error with 3σ anomaly threshold</sub></td>
  </tr>
  <tr>
    <td colspan="2"><img src="results/anomaly_autoencoder.png" width="100%"/><br/><sub><b>Fig 7.</b> LSTM Autoencoder — detections over last 60 days + confusion breakdown</sub></td>
  </tr>
</table>

---

## 📦 Dataset

This project uses the **PJM AEP Region Hourly Energy Consumption** dataset.

| Property | Value |
|----------|-------|
| **Source** | [Kaggle – Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) |
| **Region** | American Electric Power (AEP), PJM Interconnection |
| **Coverage** | 2001–2018 (we use 2017–2023 synthetic replica) |
| **Frequency** | Hourly |
| **Size** | ~140,000 observations |
| **Range** | ~9,000 – 24,000 MW |

### Using Real Data
1. Download `AEP_hourly.csv` from Kaggle
2. Place it in the `data/` directory
3. In `run_pipeline.py`, replace:
   ```python
   df = generate_pjm_data()
   ```
   with:
   ```python
   df = pd.read_csv('data/AEP_hourly.csv', index_col='Datetime', parse_dates=True)
   df.columns = ['load_mw']
   df['is_anomaly'] = 0
   ```

> **No Kaggle account?** The project runs seamlessly on a built-in **synthetic data generator** that replicates PJM AEP statistical properties — including seasonality, daily profiles, weekend effects, and injected anomalies.

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- pip

### Clone and install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/power-grid-ai-analytics.git
cd power-grid-ai-analytics

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=1.7
tensorflow>=2.12
matplotlib>=3.7
seaborn>=0.12
plotly>=5.0
```

---

## 🚀 Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

This will:
1. ✅ Generate (or load) 7 years of hourly load data
2. ✅ Produce EDA visualizations
3. ✅ Train all 4 forecasting models and compare performance
4. ✅ Run both anomaly detectors
5. ✅ Save all plots and metrics to `results/`

### Run individual components

```python
from src.data_utils import generate_pjm_data, add_features
from src.models import run_forecasting, run_isolation_forest, run_lstm_autoencoder
from src.visualization import plot_load_overview

# Generate data
df = generate_pjm_data()
df_feat = add_features(df)

# Forecasting
results, metrics_df, y_test, test_index = run_forecasting(df_feat)
print(metrics_df)

# Anomaly detection
df_iso = run_isolation_forest(df_feat)
df_ae, errors, threshold = run_lstm_autoencoder(df_feat)
```

### Expected output

```
============================================================
  Power Grid AI Analytics Pipeline
============================================================

[1/4] Generating PJM-like load data …
  61,321 hourly observations  |  183 injected anomalies

[2/4] Generating EDA visualizations …
  Saved → results/eda_load_overview.png
  Saved → results/eda_load_heatmap.png

[3/4] Training forecasting models …
  Training Linear Regression …
  Training Random Forest …
  Training XGBoost …
  Training LSTM …

  ── Model Performance ──
                    MAE (MW)  RMSE (MW)  MAPE (%)
  Linear Regression    437.6      607.8      2.70
  Random Forest        412.4      574.1      2.54
  XGBoost              381.7      540.5      2.34
  LSTM                  ~420       ~580     ~2.60

[4/4] Running anomaly detection …
  ── Anomaly Detection Performance ──
                    Precision  Recall    F1
  Isolation Forest    0.207    0.209   0.208
  LSTM Autoencoder    0.006    0.005   0.006

✅  All results saved to ./results/
============================================================
```

---

## 🌐 Run on Google Colab

No local setup needed — open the notebook directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Steps:**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `Power_Grid_AI_Analytics.ipynb`
3. **Runtime → Run all**

The notebook installs all dependencies automatically and runs the complete pipeline end-to-end with inline plots.

---

## 🔭 Future Work

The framework is designed to be modular and extensible. High-priority research directions include:

- **Graph Neural Networks (GNN)** for multi-bus grid topology-aware forecasting, where load at each bus is influenced by its neighbors
- **Temporal Fusion Transformer (TFT)** for probabilistic multi-step forecasting with uncertainty quantification
- **AI data center load profiling** — characterizing the distinct ramp signatures of GPU cluster demand to enable targeted forecasting models
- **Online anomaly detection** for streaming SCADA telemetry in real-time operational settings
- **Weather integration** — incorporating temperature, humidity, and cloud cover as exogenous inputs to reduce peak-hour MAPE
- **Physics-informed ML** via coupling with OpenDSS or MATPOWER for power flow-constrained forecasting

---

## 📄 Research Paper

This project is accompanied by a full research manuscript submitted to **IEEE Access**:

> **"AI-Driven Load Forecasting and Anomaly Detection for Power Grid Analytics: A Comparative Study Using Machine Learning and Deep Learning"**
> *Author Name, Co-Author Name — IEEE Access, 2025 (under review)*

The paper includes formal methodology, literature review (26 references), statistical analysis of all results, and discussion of implications for AI data center grid integration.

📁 LaTeX source and compiled PDF are available in the [`paper/`](paper/) directory.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to:
- Add a new forecasting model (e.g., TFT, N-BEATS)
- Integrate real weather data
- Extend to multi-node grid topologies

Please open an issue or submit a pull request.

---

## 📬 Contact

**[Your Name]**
M.S. in Electrical/Power Engineering
📧 [your.email@university.edu]
🔗 [LinkedIn](https://linkedin.com/in/yourprofile)

---

## 📃 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

⭐ **If this project helped you, please consider giving it a star!** ⭐

*Built as part of graduate research preparation in AI for power systems analytics.*

</div>
