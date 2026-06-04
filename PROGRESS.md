# AI Forex Trading Bot — Progress Documentation

> **Project:** AI-powered Forex trading bot using Smart Money Concepts (SMC) + Machine Learning  
> **Broker:** VantageMarkets Demo (MT5)  
> **Account:** 11210667  
> **Last Update:** June 4, 2026

---

## Table of Contents

1. [Phase 1: Project Setup & Environment](#phase-1-project-setup--environment)
2. [Phase 2: Data Module](#phase-2-data-module)
3. [Phase 3: Strategy Framework & Indicators](#phase-3-strategy-framework--indicators)
4. [Phase 4: MA Crossover Strategy](#phase-4-ma-crossover-strategy)
5. [Phase 5: SMC Strategy — H4 Bias Engine](#phase-5-smc-strategy--h4-bias-engine)
6. [Phase 6: SMC Strategy — M15 Zone Finder](#phase-6-smc-strategy--m15-zone-finder)
7. [Phase 7: SMC Strategy — M5 Entry Trigger (CHoCH)](#phase-7-smc-strategy--m5-entry-trigger-choch)
8. [Phase 8: Risk Management Module](#phase-8-risk-management-module)
9. [Phase 9: Paper Broker / Execution Module](#phase-9-paper-broker--execution-module)
10. [Phase 10: Main Trading Bot Loop](#phase-10-main-trading-bot-loop)
11. [Phase 11: Machine Learning Research](#phase-11-machine-learning-research)
12. [Phase 12: Bug Fixes & Issues Resolved](#phase-12-bug-fixes--issues-resolved)
13. [Phase 13: Multi-Pair ML Filter (June 4, 2026)](#phase-13-multi-pair-ml-filter-june-4-2026)
14. [Phase 14: Per-Symbol Thresholds & Risk Fixes (June 4, 2026)](#phase-14-per-symbol-thresholds--risk-fixes-june-4-2026)
15. [Latest Status (June 4, 2026)](#latest-status-june-4-2026)
16. [Next Steps](#next-steps)

---

## Phase 1: Project Setup & Environment

### Accomplished

- Created Python 3.12 virtual environment (`venv/`)
- Installed all dependencies: `pandas`, `numpy`, `MetaTrader5`, `ta`, `rich`, `python-dotenv`, `schedule`, `python-telegram-bot`
- Created `.env` configuration file with MT5 credentials and bot settings
- Created `.env.example` template for reference
- Set up `.gitignore` to exclude `venv/`, `__pycache__/`, `.env`, `logs/`
- Configured VS Code settings (`interpreter` pointing to venv)

### Key Files

| File                    | Purpose                                           |
| ----------------------- | ------------------------------------------------- |
| `requirements.txt`      | Python package list                               |
| `.env`                  | Live config: MT5 login, risk settings, bot params |
| `.vscode/settings.json` | IDE configuration                                 |

### Current `.env` Values

```
MT5_LOGIN=11210667
MT5_SERVER=VantageMarkets-Demo
USE_MT5=true
RISK_PER_TRADE=0.01 (1%)
PAPER_TRADING=true
SCAN_INTERVAL=5 (minutes)
```

---

## Phase 2: Data Module

### Accomplished

- Built `DataProvider` class (`src/data/provider.py`) with:
  - **MT5 connectivity** — automatic login, symbol selection, rate fetching
  - **Timeframe mapping** — supports M1, M5, M15, M30, H1, H4, D1, W1
  - **Simulated data fallback** — random walk generator when MT5 is offline
  - `PriceData` dataclass with validation (requires open/high/low/close/volume)
- Exported `FOREX_MAJORS` list (7 major pairs)
- Created `src/data/__init__.py` for clean imports

### Files Created

| File                   | Lines                                |
| ---------------------- | ------------------------------------ |
| `src/data/__init__.py` | Exports                              |
| `src/data/provider.py` | 127 lines — core data fetching logic |

---

## Phase 3: Strategy Framework & Indicators

### Accomplished

- Created abstract `Strategy` base class (`src/strategies/base.py`) with:
  - `generate_signal(data, symbol)` — abstract method
  - `add_indicators(df)` — adds SMA 20/50, EMA 12/26, RSI, MACD, Bollinger Bands, ATR
- Defined `Signal` dataclass and `SignalType` enum (BUY, SELL, HOLD, CLOSE_BUY, CLOSE_SELL)

---

## Phase 4: MA Crossover Strategy

### Accomplished

- Built `MACrossoverStrategy` as backup strategy
- Logic: SMA 20 crosses above SMA 50 → BUY (RSI > 30), opposite → SELL (RSI < 70)

---

## Phase 5: SMC Strategy — H4 Bias Engine

### Accomplished (Notebook: `05_smc_bias_engine.ipynb`)

- `find_swings()` — swing highs/lows over 5-bar window
- `get_h4_bias()` — bullish / bearish / neutral based on last 2 swing comparisons

---

## Phase 6: SMC Strategy — M15 Zone Finder

### Accomplished (Notebook: `06_smc_zone_finder.ipynb`)

- `find_zones()` — Fair Value Gaps (FVGs) + Order Blocks (OBs)
- `get_confluence()` — OB + FVG overlap filtered by H4 bias

---

## Phase 7: SMC Strategy — M5 Entry Trigger (CHoCH)

### Accomplished (Notebook: `07_smc_entry_trigger.ipynb`)

- `detect_choch_m5()` — Change of Character detection (higher lows / lower highs break)
- `get_next_liquidity()` — TP targeting via next liquidity level
- Full `SMCStrategy.generate_signal()` pipeline:
  1. H4 bias check (skip if neutral)
  2. M15 confluent zones (skip if none)
  3. Price inside zone check
  4. M5 CHoCH confirmation
  5. SL from swing ±1 pip, TP from liquidity or 2:1 RR fallback
  6. Position size via RiskManager

---

## Phase 8: Risk Management Module

### Accomplished

- `RiskManager` — 1% risk per trade, 6% daily max, 5 max open positions
- Pip value adjustment for JPY pairs
- `TradeSizing` dataclass

---

## Phase 9: Paper Broker / Execution Module

### Accomplished

- `PaperBroker` — simulated orders with spread + slippage (realistic fills)
- Position tracking, P&L calculation, auto SL/TP closing via live MT5 prices
- `LiveBroker` stub for future real MT5 execution

---

## Phase 10: Main Trading Bot Loop

### Accomplished

- `TradingBot` in `main.py` — 5-minute scan cycle over 7 pairs
- Signal execution, trade logging to `logs/trades.csv`
- Rich console table for live positions + P&L
- Graceful shutdown, daily reset, Telegram notifications

---

## Phase 11: Machine Learning Research

### Accomplished (12 Jupyter Notebooks)

| #   | Notebook                       | Purpose                           |
| --- | ------------------------------ | --------------------------------- |
| 01  | `01_data_exploration.ipynb`    | Fetch & visualize MT5 data        |
| 02  | `02_feature_engineering.ipynb` | Create ML features (35+ features) |
| 03  | `03_model_training.ipynb`      | Train Random Forest baseline      |
| 04  | `04_h4_xgboost.ipynb`          | XGBoost on H4 data                |
| 05  | `05_backtest.ipynb`            | Backtest with spread/slippage     |
| 06  | `05_smc_bias_engine.ipynb`     | H4 bias research                  |
| 07  | `06_lstm_model.ipynb`          | LSTM price prediction             |
| 08  | `06_smc_zone_finder.ipynb`     | M15 OB/FVG research               |
| 09  | `07_smc_entry_trigger.ipynb`   | M5 CHoCH research                 |
| 10  | `08_risk_management.ipynb`     | Position sizing research          |
| 11  | `09_smc_backtest.ipynb`        | Walk-forward SMC backtest         |
| 12  | `10_ml_filter.ipynb`           | XGBoost signal filter (EURUSD)    |
| 13  | `11_export_pairs_data.ipynb`   | Export H1 ML data for all 7 pairs |

---

## Phase 12: Bug Fixes & Issues Resolved

### Bug #1: "No data for [symbol], skipping"

**Date:** May 31, 2026  
**Fix:** Added `time.sleep(1)` after MT5 login and `time.sleep(0.5)` before `copy_rates_from_pos()`

### Bug #2: Scan Interval showing 300 minutes

**Date:** June 2, 2026  
**Fix:** Changed `SCAN_INTERVAL=5` in `.env`

### Bug #3: `atr_regime` NaN crash during multi-pair training

**Date:** June 4, 2026  
**Symptom:** `Cannot convert non-finite values (NA or inf) to integer` for 6 of 7 pairs  
**Root Cause:** New CSVs had NaN in `atr_pct` column (ATR warmup rows) — `pd.qcut().astype(int)` crashed  
**Fix:** Added `.fillna(1)` before `.astype(int)` in both `add_features()` and the `else` branch of `train.py`

### Bug #4: `_open_positions` counter never updating

**Date:** June 4, 2026  
**Symptom:** RiskManager always allowed new trades regardless of open positions  
**Root Cause:** `open_trade(sizing)` was never called after a successful order; counter stayed at 0  
**Fix:** Replaced single int counter with `_open_symbols: set[str]`; `open_trade(symbol)` adds to set, `close_trade(symbol)` removes — called correctly in `_execute_signal()` and `_on_trade_closed()`

---

## Phase 13: Multi-Pair ML Filter (June 4, 2026)

### Problem

Original ML filter (`ml_filter.pkl`) was trained on EURUSD only. All 7 pairs were using the same EURUSD model — inaccurate for other pairs.

### Solution — Option A: Per-Symbol Models

Each pair gets its own XGBoost model trained on its own H1 data.

### Changes Made

**`11_export_pairs_data.ipynb`** — exported 4,980–5,000 rows of H1 ML data for all 6 remaining pairs:

- `data/GBPUSD_H1_ML.csv`
- `data/USDJPY_H1_ML.csv`
- `data/USDCHF_H1_ML.csv`
- `data/AUDUSD_H1_ML.csv`
- `data/USDCAD_H1_ML.csv`
- `data/NZDUSD_H1_ML.csv`

**`src/ml/train.py`** — updated to support multi-symbol training:

- `train(data_path, symbol)` — trains one model, saves `models/{SYMBOL}_ml_filter.pkl`
- `train_all()` — loops all 7 pairs automatically
- Usage: `python src/ml/train.py` (all) | `python src/ml/train.py GBPUSD` (one)
- Backward compat: `models/ml_filter.pkl` kept in sync with EURUSD model

**`src/ml/filter.py`** — updated `MLFilter` class:

- `MLFilter(symbol="EURUSD")` — loads `{SYMBOL}_ml_filter.pkl`
- `_model_paths(symbol)` — falls back to generic model if per-symbol pkl missing

**`src/strategies/smc_strategy.py`** — per-symbol filter cache:

- `self._ml_filters: dict[str, MLFilter]` — lazy-loaded per pair on first use
- `_get_threshold(symbol)` — looks up per-symbol threshold

### Training Results

| Symbol | AUC   | Best Threshold | WR @ Threshold                  |
| ------ | ----- | -------------- | ------------------------------- |
| EURUSD | 0.608 | 0.55           | 80.8% (26 signals)              |
| GBPUSD | 0.563 | 0.60           | 76.3% (38 signals)              |
| USDJPY | 0.575 | 0.55           | 78.0% (50 signals)              |
| USDCHF | 0.581 | 0.58           | 74.0% (50 signals)              |
| AUDUSD | 0.547 | 0.60           | 63.8% (47 signals)              |
| USDCAD | 0.551 | 0.60           | 80.4% (46 signals)              |
| NZDUSD | 0.485 | 0.52           | 52.5% (61 signals) — weak model |

> NZDUSD AUC below 0.50 — model is near-random. Kept active at low threshold (0.52) pending more data and retraining.

---

## Phase 14: Per-Symbol Thresholds & Risk Fixes (June 4, 2026)

### Per-Symbol ML Thresholds

**`src/config.py`** — replaced single `ml_threshold` with `ml_thresholds` dict:

```python
ml_thresholds = {
    "EURUSD": 0.55,
    "GBPUSD": 0.60,
    "USDJPY": 0.55,
    "USDCHF": 0.58,
    "AUDUSD": 0.60,
    "USDCAD": 0.60,
    "NZDUSD": 0.52,
}
```

- `get_threshold(symbol)` helper method added
- Fallback default `ml_threshold=0.55` kept for `.env` compatibility

**`src/strategies/smc_strategy.py`** — `_get_threshold(symbol)` uses per-symbol dict with fallback

**`main.py`** — passes `ml_thresholds=config.ml_thresholds` into `SMCStrategy`

### Per-Symbol Position Tracking Fix

**`src/risk/manager.py`** — replaced int counter with symbol-aware set:

- `_open_symbols: set[str]` — tracks which pairs have open positions
- `can_trade(symbol)` — returns `(bool, reason)` for transparent blocking
- `open_trade(symbol)` — adds symbol to set
- `close_trade(symbol)` — removes symbol from set
- Behavior: each pair can hold 1 position independently; max 5 pairs simultaneously

**`main.py`** — fixed `open_trade()` and `close_trade()` call sites

### Trade Flow (Sequential Scan)

Every 5 minutes, the bot scans pairs in order: EURUSD → GBPUSD → USDJPY → USDCHF → AUDUSD → USDCAD → NZDUSD. Each pair goes through:

1. SMC gates (H4 bias → M15 zone → in zone → M5 CHoCH)
2. ML filter gate (per-symbol model + threshold)
3. Position check (symbol not in `_open_symbols`, total < 5)
4. Order placement + `open_trade(symbol)`

On close (SL/TP hit): `close_trade(symbol)` releases the slot.

---

## Latest Status (June 4, 2026)

### What's Working

- ✅ MT5 connection & authentication to VantageMarkets Demo
- ✅ Live price fetching for all 7 major pairs (H4, M15, M5, H1)
- ✅ Paper trading with realistic spread + slippage fills
- ✅ SMC strategy: H4 bias → M15 OB/FVG zones → M5 CHoCH entry
- ✅ Per-symbol XGBoost ML filter (7 separate models)
- ✅ Per-symbol ML thresholds tuned to training results
- ✅ Per-symbol position tracking (each pair independent, max 5 total)
- ✅ 1% risk per trade, 6% daily max
- ✅ SL from swing ±1 pip, TP from liquidity or 2:1 RR
- ✅ Trade logging to `logs/trades.csv`
- ✅ Rich console display of positions + P&L
- ✅ Telegram notifications (trade open/close + daily summary)
- ✅ Graceful shutdown (Ctrl+C)

### Models

| File                          | Description                             |
| ----------------------------- | --------------------------------------- |
| `models/EURUSD_ml_filter.pkl` | EURUSD XGBoost model (AUC 0.608)        |
| `models/GBPUSD_ml_filter.pkl` | GBPUSD XGBoost model (AUC 0.563)        |
| `models/USDJPY_ml_filter.pkl` | USDJPY XGBoost model (AUC 0.575)        |
| `models/USDCHF_ml_filter.pkl` | USDCHF XGBoost model (AUC 0.581)        |
| `models/AUDUSD_ml_filter.pkl` | AUDUSD XGBoost model (AUC 0.547)        |
| `models/USDCAD_ml_filter.pkl` | USDCAD XGBoost model (AUC 0.551)        |
| `models/NZDUSD_ml_filter.pkl` | NZDUSD XGBoost model (AUC 0.485) — weak |
| `models/ml_filter.pkl`        | Backward-compat copy of EURUSD model    |

---

## Next Steps

- [ ] **Session filter** — hard gate per pair (USDJPY → Asian+London only, GBPUSD → London+NY only)
- [ ] **Backtest 7 pairs** — run `09_smc_backtest.ipynb` with new per-symbol ML filter
- [ ] **Re-export EURUSD data** — currently 3,560 rows vs 4,980 for other pairs; normalize to 5,000
- [ ] **NZDUSD retraining** — collect more data, retrain when AUC improves above 0.52
- [ ] **Dashboard / monitoring** — Streamlit or Flask UI reading from `logs/trades.csv`
- [ ] **Walk-forward validation** — rolling train/test for more realistic performance estimates
- [ ] **Model retraining scheduler** — monthly auto-retrain to prevent model drift
- [ ] **Live broker** — activate `LiveBroker` for real MT5 order placement when ready
