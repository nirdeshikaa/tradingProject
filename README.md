# 🏆 Mini AI Trade Decision Support Dashboard

> **Single Asset: Gold Futures (GC=F)**  
> A tiny, functional web-based dashboard demonstrating a Personalized AI-Based Trade Decision Support System.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technical Architecture](#technical-architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Technical Indicators](#technical-indicators)
- [Machine Learning Model](#machine-learning-model)
- [Screenshots](#screenshots)
- [Development Timeline](#development-timeline)
- [Technologies Used](#technologies-used)

---

## 📖 Project Overview

This project focuses on a **single asset (Gold Futures — GC=F)** and provides:

| Feature | Description |
|---------|-------------|
| 🔮 **AI Prediction** | Predicts **UP** or **DOWN** for the next trading day |
| 📊 **Interactive Charts** | Candlestick and line charts with SMA/EMA overlays |
| 🧠 **Behavioral Logging** | Track trading emotions (Confident / Neutral / Fearful) |
| 📈 **Metrics Dashboard** | Real-time display of technical indicator values |

---

## ✨ Features

### 1. Single Asset Prediction
- Predicts **Up** or **Down** for the next trading day
- Uses **Random Forest Classifier** (scikit-learn) with 100 estimators
- Computes **SMA(5), SMA(20), EMA(5), EMA(20)** as technical indicators
- Provides **confidence percentage** for each prediction
- Displays **model accuracy** on test data

### 2. Interactive Dashboard / Visualization
- **Candlestick chart** showing OHLC price data (last ~252 trading days)
- **Line chart** alternative with area fill
- **SMA/EMA overlay** lines on both chart types
- **Hover tooltips** with unified crosshair
- **Zoom and pan** capabilities
- **Responsive design** for desktop, tablet, and mobile

### 3. Behavioral Emotion Logging
- Select emotion before trading: **😎 Confident / 😐 Neutral / 😰 Fearful**
- Add optional notes about market sentiment
- Entries saved persistently in **localStorage**
- **Summary table** with count and distribution bars
- Merges CSV seed data with user-entered data

### 4. Metrics Bar
- Last close price
- Price change percentage (color-coded)
- SMA(5) and SMA(20) values
- EMA(5) and EMA(20) values
- Volatility percentage

---

## 📁 Project Structure

```
holographic-comet/
├── index.html              # Main dashboard page (single page)
├── style.css               # Premium dark theme stylesheet
├── script.js               # Dashboard logic (charts, prediction, logging)
├── predict.py              # Python ML prediction engine
├── requirements.txt        # Python dependencies
├── README.md               # This documentation file
│
└── data/
    ├── data.csv            # Historical Gold Futures (GC=F) data (2013-2026)
    ├── emotion_log.csv     # Sample behavioral log entries
    ├── prediction.json     # Generated prediction + chart data (for frontend)
    └── prediction.csv      # Simple UP/DOWN text file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `index.html` | Single-page dashboard with header, metrics bar, chart, prediction card, emotion logger, and summary table |
| `style.css` | High-contrast dark theme with CSS custom properties, animations, and responsive breakpoints |
| `script.js` | Loads prediction data, renders Plotly charts, handles emotion logging, updates metrics |
| `predict.py` | Full ML pipeline: load CSV → compute indicators → train Random Forest → output JSON |
| `data/data.csv` | 3,300+ rows of historical Gold Futures OHLCV data from Yahoo Finance |
| `data/emotion_log.csv` | Seed data for behavioral emotion tracking |
| `data/prediction.json` | ML output consumed by the frontend (prediction + chart data + indicators) |

---

## 🏗 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER (CSV Files)                     │
│                                                               │
│   data/data.csv ──────────── Historical OHLCV Prices         │
│   data/emotion_log.csv ───── Behavioral Log Entries          │
│   data/prediction.json ───── ML Output (generated)           │
└───────────┬───────────────────────────┬──────────────────────┘
            │                           │
            ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────────────┐
│   BACKEND / AI LAYER │   │      FRONTEND / DASHBOARD        │
│                      │   │                                    │
│   predict.py         │   │   index.html + style.css          │
│   ├─ Load CSV data   │──▶│   ├─ Plotly.js Charts             │
│   ├─ Compute SMA/EMA │   │   ├─ AI Prediction Card           │
│   ├─ Train RF Model  │   │   ├─ Emotion Logger               │
│   └─ Output JSON     │   │   ├─ Summary Table                │
│                      │   │   └─ script.js                    │
│   scikit-learn       │   │       ├─ Load prediction.json     │
│   pandas / numpy     │   │       ├─ Render charts            │
│                      │   │       ├─ Handle UI interactions   │
│                      │   │       └─ localStorage persistence │
└──────────────────────┘   └──────────────────────────────────┘
```

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.7+** with pip
- A **modern web browser** (Chrome, Firefox, Safari, Edge)
- A **local HTTP server** (Python's built-in or VS Code Live Server)

### Step 1: Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

This installs:
- `pandas` — Data manipulation & CSV reading
- `numpy` — Numerical computations
- `scikit-learn` — Random Forest classifier

### Step 2: Run the Prediction Engine

```bash
python3 predict.py
```

This will:
1. Load `data/data.csv` (Gold Futures historical data)
2. Compute SMA(5), SMA(20), EMA(5), EMA(20) indicators
3. Train a Random Forest Classifier (80/20 chrono split)
4. Generate `data/prediction.json` with prediction + chart data
5. Generate `data/prediction.csv` with simple UP/DOWN

### Step 3: Launch the Dashboard

Start a local HTTP server (required for fetch API):

```bash
python3 -m http.server 8000
```

Then open in your browser:
```
http://localhost:8000
```

> **Note:** Opening `index.html` directly won't work due to CORS restrictions on local file fetching. You must use an HTTP server.

---

## 📈 Usage

### Viewing the Dashboard
1. The **price chart** shows the last ~252 trading days of Gold Futures
2. Toggle between **Candlestick** and **Line** chart views
3. The **AI Prediction** card shows the model's next-day forecast

### Logging Emotions
1. Click one of the **three emotion buttons**: Confident, Neutral, or Fearful
2. Optionally add **notes** about your market sentiment
3. Click **Save Entry** to record the emotion
4. View the **Emotion Summary** table below for aggregate counts

### Interpreting the Prediction
| Element | Meaning |
|---------|---------|
| **UP ▲** (green) | Model predicts next day's close will be higher |
| **DOWN ▼** (red) | Model predicts next day's close will be lower |
| **Confidence** | Probability % the model assigns to its prediction |
| **Model Accuracy** | Accuracy on the test set (last 20% of data) |

---

## 📐 Technical Indicators

### Simple Moving Average (SMA)
The SMA smooths price data by averaging closing prices over a fixed window.

```
SMA(n) = (Close₁ + Close₂ + ... + Closeₙ) / n
```

- **SMA(5)**: Short-term trend (5-day average)
- **SMA(20)**: Medium-term trend (20-day average)

### Exponential Moving Average (EMA)
The EMA gives more weight to recent prices, reacting faster to changes.

```
EMA_today = Close × k + EMA_yesterday × (1 - k)
where k = 2 / (n + 1)
```

- **EMA(5)**: Short-term weighted trend
- **EMA(20)**: Medium-term weighted trend

### Derived Features
| Feature | Formula | Purpose |
|---------|---------|---------|
| Price Change % | `(Close_today - Close_yesterday) / Close_yesterday × 100` | Daily momentum |
| Volatility | `(High - Low) / Close × 100` | Intraday price range |
| SMA Ratio | `SMA(5) / SMA(20)` | Short vs medium-term trend crossover |
| EMA Ratio | `EMA(5) / EMA(20)` | Weighted trend crossover |
| Price vs SMA20 | `(Close - SMA(20)) / SMA(20) × 100` | Distance from trend |

---

## 🤖 Machine Learning Model

### Model: Random Forest Classifier
- **Algorithm**: Random Forest (ensemble of decision trees)
- **Library**: scikit-learn
- **Estimators**: 100 trees
- **Max Depth**: 5 (to prevent overfitting)
- **Class Weight**: Balanced (handles UP/DOWN imbalance)

### Training Process
1. **Data Split**: 80% training / 20% testing (chronological, no shuffle)
2. **Features**: 9 technical indicators (SMA, EMA, ratios, volatility)
3. **Target**: Binary — 1 (UP) if next day close > today's close, else 0 (DOWN)

### Feature Importance (from trained model)
```
Price_Change_Pct     ████████  (highest importance)
EMA_20               █████
SMA_Ratio            █████
SMA_20               █████
EMA_Ratio            █████
Price_vs_SMA20       █████
SMA_5                ████
Volatility           ████
EMA_5                ████
```

> **Note**: Financial markets are inherently noisy. This model is for **educational demonstration** only and should not be used for real trading decisions.

---

## 🖼 Screenshots

To capture screenshots for your report:
1. Open the dashboard at `http://localhost:8000`
2. Use your browser's screenshot tool (or Cmd+Shift+4 on Mac)
3. Capture: full dashboard, candlestick chart, prediction card, emotion logger

---

## 📅 Development Timeline

| Week | Tasks | Status |
|------|-------|--------|
| **Week 1** | Download CSV data, explore SMA/EMA indicators | ✅ Complete |
| **Week 2** | Build prediction engine (Random Forest), test on historical data | ✅ Complete |
| **Week 3** | Build dashboard: price chart + prediction display + emotion logging | ✅ Complete |
| **Week 4** | Polish UI, test end-to-end, prepare screenshots + mini report | ✅ Complete |

---

## 🛠 Technologies Used

| Component | Technology |
|-----------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+) |
| **Charting** | Plotly.js 2.27 |
| **Typography** | Inter (Google Fonts) |
| **ML Model** | scikit-learn (Random Forest) |
| **Data Processing** | pandas, NumPy |
| **Data Persistence** | localStorage (browser), CSV files |
| **Design** | Custom dark theme (Bloomberg/Binance inspired) |

---

## 📝 License

This project is for academic/educational purposes only.

---

*Built with 🧠 AI-powered prediction and 📊 interactive visualization*
