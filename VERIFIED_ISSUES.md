# Verified Issues in AI Forex Trading Bot

**Date:** June 5, 2026 | **Status:** Re-analyzed with actual code

I apologize for my initial inaccurate analysis. After careful re-examination of the current code, here are the **VERIFIED** issues:

---

## 🔴 HIGH-SEVERITY ISSUES

### 1. Manual Close Position Not Recorded — CONFIRMED
**File:** main.py:153-157  
**Issue:** When a CLOSE_BUY or CLOSE_SELL signal is executed, the position is closed in the broker but `_on_trade_closed()` is not called, so the trade is not logged to `trades.csv`

```python
if signal.type in (SignalType.CLOSE_BUY, SignalType.CLOSE_SELL):
    side = OrderSide.BUY if signal.type == SignalType.CLOSE_BUY else OrderSide.SELL
    self.broker.close_position(signal.symbol, side)
    log.info(f"Closed {signal.symbol} {side.value} position")
    return  # Missing: _on_trade_closed() call!
```

**Impact:** Manual closes don't appear in trade history; account balance tracking is incomplete  
**Fix:** Add call to `_on_trade_closed()` before returning

---

### 2. Cumulative Daily Risk Flawed Logic — CONFIRMED  
**File:** src/risk/manager.py:39  
**Issue:** Daily risk is tracked as cumulative risk per trade opened, not realized P&L

```python
def can_trade(self, symbol: str) -> tuple[bool, str]:
    if self._daily_risk_used >= self.max_daily_risk:
        return False, f"daily risk limit reached ({self.max_daily_risk*100:.0f}%)"
```

At 1% risk per trade with max 6% daily: Bot stops after 6 trades even if all previous trades closed at +1%  

**Impact:** Trading halts mid-day even if positions are winning  
**Fix:** Track realized losses only, not trade count

---

### 3. Reconcile Positions with Unknown Exit Price — CONFIRMED
**File:** main.py:110  
**Issue:** When a position is missing from `broker.get_positions()`, reconcile assumes it was closed externally and records it with `exit_price=entry_price, pnl=0.0`

```python
if (trade['symbol'], side) not in current:
    self._on_trade_closed(key, trade.get('entry', 0), ..., 0.0, 'closed')
    # exit_price is the entry price, pnl is 0!
```

**Impact:** Incorrect P&L if MT5 closes positions automatically via SL/TP hits  
**Likelihood:** Rare in paper trading, but possible with live MT5  
**Fix:** Get actual exit price and P&L from broker history instead of trade.entry

---

## 🟡 MEDIUM-SEVERITY ISSUES

### 4. Scan Interval Not Validated — CONFIRMED
**File:** src/config.py:49  
**Issue:** Can be set to unreasonably large values (PROGRESS.md bug #2: was set to 300 minutes)

```python
scan_interval_minutes=int(os.getenv("SCAN_INTERVAL", "60"))  # No bounds!
```

**Impact:** Setting `SCAN_INTERVAL=300` causes 5-hour wait between bot cycles  
**Fix:** Add bounds: `max(1, min(1440, ...))`

---

### 5. Unused ML Filter Module — CONFIRMED
**File:** src/ml/filter.py, src/ml/train.py, models/  
**Issue:** Complete ML signal filtering system exists but is never called from the trading bot

- Training script: `src/ml/train.py` ✓
- Filter class: `src/ml/filter.py` ✓
- Integration in bot: ✗ **Missing**

**Impact:** Trained XGBoost models sit unused; no ML-based signal filtering  
**Fix:** Call `MLFilter.should_trade()` in `SMCStrategy.generate_signal()`

---

### 6. MACrossover Strategy Unused — CONFIRMED
**File:** main.py:46-49  
**Issue:** MACrossoverStrategy is defined but never registered in the strategy list

```python
self.strategies: list[Strategy] = [
    SMCStrategy(...)  # Only SMC, MACrossover never added
]
```

**Impact:** CLOSE_BUY/CLOSE_SELL signals can never be generated  
**Fix:** Either remove unused strategy or add it to strategy list

---

## 🟢 LOW-SEVERITY ISSUES

### 7. Duplicate close_trade() Call — CONFIRMED
**File:** main.py:214, 237  
**Issue:** `close_trade(symbol)` called twice in `_on_trade_closed()`

```python
214:    self.risk_manager.close_trade(symbol)   # First call
...
237:    self.risk_manager.close_trade(symbol)   # Second call (redundant)
```

**Impact:** Harmless (set.discard is idempotent) but wastes CPU  
**Fix:** Remove line 237

---

### 8. MT5 Symbol Selection Not Checked — MINOR
**File:** src/data/provider.py:85  
**Issue:** `mt5.symbol_select()` return value not checked

```python
mt5.symbol_select(symbol, True)  # Could return False!
import time
time.sleep(0.5)
rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)  # Will fail if symbol not selected
```

**Impact:** Silent failure if symbol can't be selected; will return None on next call  
**Fix:** Check return value and log error

---

## ✅ WHAT'S WORKING WELL

- **Broker implementation** ✓ Proper margin tracking, spread modeling, correct pip-based PnL
- **Data validation** ✓ PriceData checks for required columns
- **Risk management** ✓ Position tracking per symbol works correctly
- **Telegram notifier** ✓ Has retry logic and rate limit handling
- **Strategy framework** ✓ Clean abstractions with SMC implementation

---

## Summary

| Severity | Count | Issues |
|----------|-------|--------|
| High | 3 | Manual close tracking, daily risk logic, reconcile positions |
| Medium | 3 | Scan interval validation, ML integration, unused strategy |
| Low | 2 | Duplicate call, symbol selection check |
| **Total** | **8** | |

---

## Recommended Fix Priority

1. **Manual close position recording** (HIGH) — Easy fix, important for accuracy
2. **Daily risk logic** (HIGH) — Design change needed
3. **Reconcile positions logic** (HIGH) — Edge case, hard to test
4. **Scan interval validation** (MEDIUM) — One-line fix
5. **ML filter integration** (MEDIUM) — Feature completion
6. **Remove duplicate close_trade** (LOW) — Cleanup
7. **Symbol selection check** (LOW) — Defensive programming
