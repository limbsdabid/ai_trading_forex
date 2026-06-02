# AI Forex Trading Bot — Progress Documentation

> **Project:** AI-powered Forex trading bot using Smart Money Concepts (SMC) + Machine Learning  
> **Broker:** VantageMarkets Demo (MT5)  
> **Account:** 11210667  
> **Last Update:** June 2, 2026

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
13. [Phase 13: Latest Status (June 2, 2026)](#phase-13-latest-status-june-2-2026)
14. [Next Steps](#next-steps)

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
| File | Purpose |
|------|---------|
| `requirements.txt` | Python package list |
| `.env` | Live config: MT5 login, risk settings, bot params |
| `.vscode/settings.json` | IDE configuration |

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
| File | Lines |
|------|-------|
| `src/data/__init__.py` | Exports |
| `src/data/provider.py` | 127 lines — core data fetching logic |

### How It Works
```
main.py → DataProvider.fetch_rates(symbol, timeframe, bars)
         ├── if MT5 connected → _fetch_mt5() → MetaTrader5 API
         └── if not connected → _fetch_simulated() → random walk
```

---

## Phase 3: Strategy Framework & Indicators

### Accomplished
- Created abstract `Strategy` base class (`src/strategies/base.py`) with:
  - `generate_signal(data, symbol)` — abstract method
  - `add_indicators(df)` — adds SMA 20/50, EMA 12/26, RSI, MACD, Bollinger Bands, ATR
  - Indicator helper methods: `_rsi()`, `_macd()`, `_bollinger()`, `_atr()`
- Defined `Signal` dataclass and `SignalType` enum (BUY, SELL, HOLD, CLOSE_BUY, CLOSE_SELL)
- Created `src/strategies/__init__.py`

### Files Created
| File | Lines |
|------|-------|
| `src/strategies/base.py` | 84 lines |
| `src/strategies/__init__.py` | Exports |

---

## Phase 4: MA Crossover Strategy

### Accomplished
- Built `MACrossoverStrategy` (`src/strategies/ma_crossover.py`) as backup strategy
- Logic: SMA 20 crosses above SMA 50 → BUY (with RSI > 30), opposite → SELL (with RSI < 70)
- Registered in main bot (can be toggled on/off)

### Files Created
| File | Lines |
|------|-------|
| `src/strategies/ma_crossover.py` | 62 lines |

---

## Phase 5: SMC Strategy — H4 Bias Engine

### Accomplished (Notebook: `05_smc_bias_engine.ipynb`)
- Developed H4 bias detection logic:
  - `find_swings()` — identifies swing highs/lows using local extrema over 5-bar window
  - `get_h4_bias()` — compares current close to last 2 swing points
    - Close above previous swing high → **bullish**
    - Close below previous swing low → **bearish**
    - Otherwise → **neutral**
- Handles edge cases: insufficient data (<10 bars), fewer than 2 swings

### Code Location
`src/strategies/smc_strategy.py` → `find_swings()` (line 16), `get_h4_bias()` (line 25)

---

## Phase 6: SMC Strategy — M15 Zone Finder

### Accomplished (Notebook: `06_smc_zone_finder.ipynb`)
- Developed M15 zone detection:
  - `find_zones()` — identifies:
    - **Fair Value Gaps (FVGs)** — 3-candle gap where body of candle 3 does not overlap candle 1
    - **Order Blocks (OBs)** — last bearish/bullish candle before an impulse move
  - `get_confluence()` — matches OBs and FVGs within distance threshold, filtered by H4 bias
- Returns confluent zones (OB + FVG aligned) with top/bot/mid prices

### Code Location
`src/strategies/smc_strategy.py` → `find_zones()` (line 41), `get_confluence()` (line 60)

---

## Phase 7: SMC Strategy — M5 Entry Trigger (CHoCH)

### Accomplished (Notebook: `07_smc_entry_trigger.ipynb`)
- Developed M5 entry trigger:
  - `detect_choch_m5()` — detects **Change of Character (CHoCH)**:
    - **Bullish CHoCH:** Higher lows → price breaks above previous swing high
    - **Bearish CHoCH:** Lower highs → price breaks below previous swing low
  - `get_next_liquidity()` — finds next liquidity level for TP targeting
- Full signal generation in `SMCStrategy.generate_signal()`:
  1. Fetch H4, M15, M5 data
  2. Check H4 bias (skip if neutral)
  3. Find M15 confluent zones (skip if none)
  4. Check if price is inside a recent zone (skip if not)
  5. Confirm M5 CHoCH trigger (skip if no CHoCH)
  6. Calculate SL (below/above swing point), TP (next liquidity)
  7. Calculate position size via risk manager
  8. Return BUY/SELL signal with metadata

### Files Created
| File | Lines |
|------|-------|
| `src/strategies/smc_strategy.py` | 205 lines |

---

## Phase 8: Risk Management Module

### Accomplished (Notebook: `08_risk_management.ipynb`)
- Built `RiskManager` class (`src/risk/manager.py`) with:
  - **Position sizing** — `calculate_size(entry, stop, symbol)`
    - Risk amount = balance × risk_per_trade (1% default)
    - Volume = risk_amount / (sl_pips × 10)
    - Volume normalized to 0.01 step
    - Returns None if max positions reached or daily risk exceeded
  - **Daily risk tracking** — `max_daily_risk` (6% default), auto-resets at midnight
  - **Trade counting** — `open_trade()`, `close_trade()`
  - **Pip value** — adjusts for JPY vs non-JPY pairs
  - `TradeSizing` dataclass with computed volume, SL, TP, risk amount

### Files Created
| File | Lines |
|------|-------|
| `src/risk/__init__.py` | Exports |
| `src/risk/manager.py` | 85 lines |

---

## Phase 9: Paper Broker / Execution Module

### Accomplished
- Built `PaperBroker` class (`src/execution/broker.py`) with:
  - **Order management** — `place_order()` with auto-ID, balance check, execution
  - **Position tracking** — dictionary-based with volume averaging for additions
  - **Position closing** — `close_position()` with P&L calculation
  - **Price updates** — `update_prices()` for unrealized P&L
  - **Order/Position dataclasses** — Order, Position, OrderType, OrderSide enums
  - Mock price generator for simulated fills

### Files Created
| File | Lines |
|------|-------|
| `src/execution/broker.py` | 150 lines |
| `src/execution/__init__.py` | Exports |

---

## Phase 10: Main Trading Bot Loop

### Accomplished
- Built `TradingBot` class (`main.py`) with:
  - **Startup** — MT5 connection, account sync (balance, equity, server info)
  - **Scan cycle** — iterates all symbols, fetches M5 data, runs strategies
  - **Signal execution** — maps `SignalType` to OrderSide, validates risk limits
  - **Trade logging** — CSV log at `logs/trades.csv` with timestamp/symbol/entry/SL/TP/volume
  - **Position display** — Rich table with symbol, side, volume, entry, current price, P&L, SL, TP
  - **Scan interval** — configurable sleep between cycles (default 5 min)
  - **Graceful shutdown** — KeyboardInterrupt handling, MT5 disconnect

### Files Created
| File | Lines |
|------|-------|
| `main.py` | 237 lines |

---

## Phase 11: Machine Learning Research

### Accomplished (12 Jupyter Notebooks)

| # | Notebook | Purpose | Key Finding |
|---|----------|---------|-------------|
| 01 | `01_data_exploration.ipynb` | Fetch & visualize MT5 data | Price patterns, volatility analysis |
| 02 | `02_feature_engineering.ipynb` | Create ML features | Indicators + lags → 35+ features |
| 03 | `03_model_training.ipynb` | Train Random Forest | Baseline model for direction prediction |
| 04 | `04_h4_xgboost.ipynb` | XGBoost on H4 data | Gradient boosting for H4 signals |
| 05 | `05_backtest.ipynb` | Backtest with spread/slippage | Realistic simulation with costs |
| 06 | `05_smc_bias_engine.ipynb` | H4 bias detection | Swing highs/lows + BOS/CHoCH logic |
| 07 | `06_lstm_model.ipynb` | LSTM price prediction | Deep learning sequence model |
| 08 | `06_smc_zone_finder.ipynb` | M15 OB/FVG finder | Order Block + Fair Value Gap detection |
| 09 | `07_smc_entry_trigger.ipynb` | M5 CHoCH trigger | Change of Character entry logic |
| 10 | `08_risk_management.ipynb` | Position sizing | Kelly criterion, fixed %, volatility-based |
| 11 | `09_smc_backtest.ipynb` | Walk-forward SMC backtest | End-to-end SMC system validation |
| 12 | `10_ml_filter.ipynb` | XGBoost signal filter | ML filter on SMC signals to predict win/loss |

### ML Data
- **File:** `data/EURUSD_H1_ML.csv` (3,561 rows, ~2.1 MB)
- **Features:** Price data + SMA 10/20/50, EMA 12/26, RSI, MACD, Bollinger Bands, ATR, returns, body/wick ratios, volume ratios, lags
- **Target:** Binary directional prediction

---

## Phase 12: Bug Fixes & Issues Resolved

### Bug #1: "No data for [symbol], skipping"
**Date:** May 31, 2026  
**Symptom:** Bot connected to MT5 successfully but all 7 symbols returned no data  
**Root Cause:** MT5 symbol selection timing — symbols need time to initialize after login  
**Fix Applied:** Added `time.sleep(1)` after `mt5.login()` and `time.sleep(0.5)` in `_fetch_mt5()` before `copy_rates_from_pos()`  
**Result:** Bot successfully fetches live data for all 7 pairs

### Bug #2: Scan Interval Display (300 minutes)
**Date:** June 2, 2026 (08:33)  
**Symptom:** Log showed "Next scan in 300 minutes"  
**Root Cause:** `SCAN_INTERVAL` was set to 300 in `.env` (likely during testing)  
**Fix Applied:** Changed `SCAN_INTERVAL=5` in `.env`  
**Result:** Scan interval reduced to 5 minutes (last log: "Next scan in 5 minutes")  
**Lesson:** No validation/enforcement of max scan interval — added to future improvements list

### Bug #3: Account Balance Fluctuation
**Date:** May 31 → June 2  
**Observation:** Balance dropped from $4,654.85 → $4,478.57 (loss of ~$176.28)  
**Root Cause:** Paper trades incurred losses over the weekend gap and open positions  
**Status:** Under monitoring — expected in live trading; risk management is working as designed (1% risk per trade)

---

## Phase 13: Latest Status (June 2, 2026)

### What's Working
- ✅ MT5 connection & authentication to VantageMarkets Demo
- ✅ Live price fetching for all 7 major pairs
- ✅ Paper trading (simulated orders with position tracking)
- ✅ SMC strategy pipeline: H4 bias → M15 zones → M5 CHoCH entry
- ✅ Risk management: 1% per trade, 5% daily max, 5 max positions
- ✅ Trade logging to CSV
- ✅ Console display of open positions with P&L
- ✅ Configurable scan interval (currently 5 min)
- ✅ Graceful shutdown (Ctrl+C)

### Latest Bot Log (08:45 AM, June 2)
```
EURUSD: 1.16315
GBPUSD: 1.34562
USDJPY: 159.686
USDCHF: 0.78667
AUDUSD: 0.71601
USDCAD: 1.38432
NZDUSD: 0.59246
Next scan in 5 minutes
```

### Account Status
- **Balance:** $4,478.57
- **Server:** VantageMarkets-Demo
- **Mode:** Paper Trading (PAPER_TRADING=true)
- **Login:** 11210667

### Files Summary
| File | Lines | Status |
|------|-------|--------|
| `main.py` | 237 | ✅ Running |
| `src/config.py` | 47 | ✅ Complete |
| `src/data/provider.py` | 127 | ✅ Complete |
| `src/execution/broker.py` | 150 | ✅ Complete |
| `src/risk/manager.py` | 85 | ✅ Complete |
| `src/strategies/base.py` | 84 | ✅ Complete |
| `src/strategies/smc_strategy.py` | 205 | ✅ Complete |
| `src/strategies/ma_crossover.py` | 62 | ✅ Complete |
| `src/utils/logger.py` | 34 | ✅ Complete |
| **Total** | **~1,031** | |

---

## Next Steps

- [ ] **Unit tests** — create test suite under `tests/`
- [ ] **Live MT5 execution** — `LiveBroker` class for real order placement
- [ ] **Telegram notifications** — integrate `python-telegram-bot` for trade alerts
- [ ] **ML filter integration** — connect XGBoost model to filter SMC signals
- [ ] **Max scan interval validation** — add cap to prevent accidental high values
- [ ] **Post-trade analysis** — store trade outcomes for ML retraining
- [ ] **Docker containerization** — for deployment
- [ ] **Git init** — initialize repository for version control
