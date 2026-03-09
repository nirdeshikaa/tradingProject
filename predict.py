"""
╔══════════════════════════════════════════════════════════════════╗
║     Mini AI Trade Decision Support System — Prediction Engine   ║
║     Asset: Gold Futures (GC=F)                                  ║
╚══════════════════════════════════════════════════════════════════╝

This script performs the following pipeline:
  1. Loads historical OHLCV data from CSV (Yahoo Finance format)
  2. Computes technical indicators: SMA(5), SMA(20), EMA(5), EMA(20)
  3. Engineers features for machine learning
  4. Trains a Random Forest Classifier (scikit-learn)
  5. Outputs prediction (UP/DOWN) with confidence and chart data as JSON

Technical Indicators:
  - SMA (Simple Moving Average): windows of 5 and 20 days
  - EMA (Exponential Moving Average): spans of 5 and 20 days
  - Daily price change percentage
  - Volatility (High - Low range as % of Close)

Model: Random Forest Classifier with 100 estimators
Output: data/prediction.json (consumed by the frontend dashboard)

Usage:
  python3 predict.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ──────────────────────────────────────────────────────────────────
# 1. LOAD DATA — handles Yahoo Finance CSV format
# ──────────────────────────────────────────────────────────────────
def load_data(filepath):
    """
    Load historical OHLCV data from a Yahoo Finance CSV.
    The file has:
      Row 1: header  → Price,Close,High,Low,Open,Volume
      Row 2: ticker  → Ticker,GC=F,GC=F,...
      Row 3: label   → Date,,,,,
      Row 4+: data   → 2013-01-02,1687.90,...
    We skip the ticker and label rows, rename 'Price' to 'Date'.
    """
    # Skip rows 1 and 2 (ticker metadata and 'Date' label row)
    df = pd.read_csv(filepath, skiprows=[1, 2])

    # The first column is 'Price' but contains dates — rename it
    df = df.rename(columns={'Price': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Ensure numeric types for price columns
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows with NaN prices
    df = df.dropna(subset=['Close', 'High', 'Low', 'Open'])

    # Sort by date ascending
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"  ✅ Loaded {len(df)} rows of historical data")
    print(f"     Date range: {df['Date'].iloc[0].strftime('%Y-%m-%d')} → {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"     Price range: ${df['Close'].min():,.2f} – ${df['Close'].max():,.2f}")
    return df


# ──────────────────────────────────────────────────────────────────
# 2. COMPUTE TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────────
def compute_indicators(df):
    """
    Compute SMA, EMA, price change %, volatility, and trend ratios.
    These features will be used by the ML model for prediction.
    """
    # ── Simple Moving Averages (SMA) ──
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # ── Exponential Moving Averages (EMA) ──
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # ── Daily percentage change in closing price ──
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100

    # ── Volatility: (High - Low) as a percentage of Close ──
    df['Volatility'] = ((df['High'] - df['Low']) / df['Close']) * 100

    # ── SMA ratio (short-term vs long-term trend) ──
    df['SMA_Ratio'] = df['SMA_5'] / df['SMA_20']

    # ── EMA ratio ──
    df['EMA_Ratio'] = df['EMA_5'] / df['EMA_20']

    # ── Price position relative to SMA_20 (above or below trend) ──
    df['Price_vs_SMA20'] = ((df['Close'] - df['SMA_20']) / df['SMA_20']) * 100

    # ── Target variable ──
    # 1 if next day's close is higher (UP), 0 otherwise (DOWN)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop rows with NaN values (from rolling window calculations)
    df = df.dropna().reset_index(drop=True)

    print(f"  ✅ Computed technical indicators ({len(df)} usable rows)")
    return df


# ──────────────────────────────────────────────────────────────────
# 3. TRAIN PREDICTION MODEL
# ──────────────────────────────────────────────────────────────────
def train_model(df):
    """
    Train a Random Forest Classifier on the computed features.
    Uses 80/20 train-test split (chronological, no shuffle).
    Returns the trained model, feature column names, and accuracy.
    """
    # Feature columns for the model
    feature_cols = [
        'SMA_5', 'SMA_20', 'EMA_5', 'EMA_20',
        'Price_Change_Pct', 'Volatility',
        'SMA_Ratio', 'EMA_Ratio', 'Price_vs_SMA20'
    ]

    X = df[feature_cols]
    y = df['Target']

    # Chronological split: 80% training, 20% testing (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Train Random Forest with balanced class weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n  📊 Model Training Results:")
    print(f"     Training samples: {len(X_train)}")
    print(f"     Testing samples:  {len(X_test)}")
    print(f"     Accuracy:         {accuracy:.2%}")
    print(f"\n     Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['DOWN', 'UP'])
    for line in report.split('\n'):
        print(f"     {line}")

    # Feature importance ranking
    importances = dict(zip(feature_cols, model.feature_importances_))
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print(f"\n     Feature Importance:")
    for feat, imp in sorted_feats:
        bar = '█' * int(imp * 40)
        print(f"       {feat:20s} {imp:.4f} {bar}")

    return model, feature_cols, accuracy


# ──────────────────────────────────────────────────────────────────
# 4. GENERATE PREDICTION FOR NEXT DAY
# ──────────────────────────────────────────────────────────────────
def predict_next_day(model, df, feature_cols):
    """
    Use the trained model to predict the next trading day's direction.
    Returns prediction label ('UP'/'DOWN'), confidence %, and latest row data.
    """
    # Get the last row (most recent trading day)
    latest = df.iloc[-1]

    # Prepare features
    X_latest = latest[feature_cols].values.reshape(1, -1)

    # Predict direction and get probability
    prediction = model.predict(X_latest)[0]
    probabilities = model.predict_proba(X_latest)[0]
    confidence = max(probabilities) * 100

    direction = "UP" if prediction == 1 else "DOWN"

    print(f"\n  🔮 PREDICTION FOR NEXT TRADING DAY:")
    print(f"     Direction:    {direction}")
    print(f"     Confidence:   {confidence:.1f}%")
    print(f"     Last Close:   ${latest['Close']:,.2f}")
    print(f"     SMA(5):       ${latest['SMA_5']:,.2f}")
    print(f"     SMA(20):      ${latest['SMA_20']:,.2f}")
    print(f"     EMA(5):       ${latest['EMA_5']:,.2f}")
    print(f"     EMA(20):      ${latest['EMA_20']:,.2f}")

    return direction, confidence, latest


# ──────────────────────────────────────────────────────────────────
# 5. SAVE RESULTS AS JSON (consumed by dashboard frontend)
# ──────────────────────────────────────────────────────────────────
def save_results(direction, confidence, latest, accuracy, df, output_dir):
    """
    Save prediction + full chart data as JSON for the web dashboard.
    Also writes a simple prediction.csv for backward compatibility.
    """
    # Only send the last 252 trading days (~1 year) for the chart
    chart_df = df.tail(252)

    chart_data = {
        'dates':  chart_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'open':   chart_df['Open'].round(2).tolist(),
        'high':   chart_df['High'].round(2).tolist(),
        'low':    chart_df['Low'].round(2).tolist(),
        'close':  chart_df['Close'].round(2).tolist(),
        'volume': chart_df['Volume'].fillna(0).astype(int).tolist(),
        'sma_5':  chart_df['SMA_5'].round(2).tolist(),
        'sma_20': chart_df['SMA_20'].round(2).tolist(),
        'ema_5':  chart_df['EMA_5'].round(2).tolist(),
        'ema_20': chart_df['EMA_20'].round(2).tolist(),
    }

    result = {
        'prediction': direction,
        'confidence': round(confidence, 1),
        'model_accuracy': round(accuracy * 100, 1),
        'asset': 'Gold Futures (GC=F)',
        'last_close': round(float(latest['Close']), 2),
        'last_date': latest['Date'].strftime('%Y-%m-%d'),
        'indicators': {
            'sma_5':            round(float(latest['SMA_5']), 2),
            'sma_20':           round(float(latest['SMA_20']), 2),
            'ema_5':            round(float(latest['EMA_5']), 2),
            'ema_20':           round(float(latest['EMA_20']), 2),
            'price_change_pct': round(float(latest['Price_Change_Pct']), 2),
            'volatility':       round(float(latest['Volatility']), 2),
        },
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'chart_data': chart_data,
    }

    # Write prediction JSON
    prediction_path = os.path.join(output_dir, 'prediction.json')
    with open(prediction_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  💾 Prediction saved → {prediction_path}")

    # Also write simple prediction.csv (backward compatibility)
    csv_path = os.path.join(output_dir, 'prediction.csv')
    with open(csv_path, 'w') as f:
        f.write(direction)
    print(f"  💾 Simple CSV     → {csv_path}")

    return result


# ──────────────────────────────────────────────────────────────────
# MAIN — run the full pipeline
# ──────────────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 62)
    print("  Mini AI Trade Decision Support System")
    print("  Asset: Gold Futures (GC=F)")
    print("  Model: Random Forest Classifier")
    print("=" * 62)
    print()

    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, 'data', 'data.csv')
    output_dir = os.path.join(script_dir, 'data')

    # Validate data file exists
    if not os.path.exists(data_path):
        print(f"  ❌ Error: Data file not found at {data_path}")
        print(f"     Please ensure data.csv exists in the data/ folder.")
        sys.exit(1)

    # ── Pipeline ──
    df = load_data(data_path)
    df = compute_indicators(df)
    model, features, accuracy = train_model(df)
    direction, confidence, latest = predict_next_day(model, df, features)
    save_results(direction, confidence, latest, accuracy, df, output_dir)

    print()
    print("=" * 62)
    print(f"  ✅ Done — Prediction: {direction} ({confidence:.1f}% confidence)")
    print("=" * 62)
    print()


if __name__ == '__main__':
    main()
