import logging
import threading
from datetime import datetime
from typing import Optional

import requests

log = logging.getLogger("trading_bot")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)

    # ── core send (with retry + non-blocking) ────────────────────────────

    def _send(self, text: str) -> None:
        """Fire-and-forget: send in background thread so bot scan never blocks."""
        if not self._enabled:
            return
        thread = threading.Thread(target=self._send_with_retry, args=(text,), daemon=True)
        thread.start()

    def _send_with_retry(self, text: str, max_retries: int = 3) -> bool:
        """Attempt delivery up to max_retries times with backoff."""
        import time
        url = TELEGRAM_API.format(token=self.bot_token)
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                    timeout=10,
                )
                if resp.ok:
                    return True
                # Telegram rate-limit: wait and retry
                if resp.status_code == 429:
                    wait = resp.json().get("parameters", {}).get("retry_after", 5)
                    log.warning(f"Telegram rate-limited — retrying in {wait}s")
                    time.sleep(wait)
                    continue
                log.warning(f"Telegram send failed [{attempt}/{max_retries}]: {resp.status_code} {resp.text[:80]}")
            except requests.exceptions.Timeout:
                log.warning(f"Telegram timeout [{attempt}/{max_retries}]")
            except Exception as e:
                log.warning(f"Telegram error [{attempt}/{max_retries}]: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s
        return False

    # ── public notification methods ───────────────────────────────────────

    def send_trade_opened(
        self, symbol: str, side: str, entry: float,
        sl: float, tp: float, volume: float,
        ml_score: Optional[float] = None,
    ):
        emoji = "🟢" if side.lower() == "buy" else "🔴"
        sl_pips = round(abs(entry - sl) * 10000)
        tp_pips = round(abs(entry - tp) * 10000)
        rr = round(tp_pips / sl_pips, 1) if sl_pips else "—"
        ml_line = f"\nML Score: {ml_score:.2f}" if ml_score is not None else ""
        text = (
            f"{emoji} <b>TRADE OPENED</b>\n"
            f"{symbol}  {side.upper()}\n"
            f"Entry : {entry}\n"
            f"SL    : {sl}  ({sl_pips} pips)\n"
            f"TP    : {tp}  ({tp_pips} pips)\n"
            f"R:R   : 1:{rr}\n"
            f"Volume: {volume}{ml_line}"
        )
        self._send(text)

    def send_trade_closed(
        self, symbol: str, side: str, entry: float,
        exit_price: float, pnl: float, result: str,
    ):
        emoji = "✅" if "win" in result.lower() else "❌"
        sign = "+" if pnl >= 0 else ""
        text = (
            f"{emoji} <b>TRADE CLOSED — {result.upper()}</b>\n"
            f"{symbol}  {side.upper()}\n"
            f"Entry : {entry}\n"
            f"Exit  : {exit_price}\n"
            f"PnL   : {sign}${pnl:.2f}"
        )
        self._send(text)

    def send_error(self, message: str):
        self._send(f"⚠️ <b>ERROR</b>\n{message}")

    def send_daily_summary(
        self, trades: int, wins: int, losses: int,
        pnl: float, balance: float,
    ):
        win_rate = (wins / trades * 100) if trades > 0 else 0
        sign = "+" if pnl >= 0 else ""
        emoji = "📈" if pnl >= 0 else "📉"
        text = (
            f"{emoji} <b>DAILY SUMMARY</b>\n"
            f"Trades : {trades}  ({wins}W / {losses}L)\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"PnL    : {sign}${pnl:.2f}\n"
            f"Balance: ${balance:.2f}"
        )
        self._send(text)

    def send_ml_blocked(self, symbol: str, side: str, ml_score: float, threshold: float):
        """Optional: notify when ML filter blocks a signal (useful for monitoring)."""
        text = (
            f"🤖 <b>ML FILTER BLOCKED</b>\n"
            f"{symbol}  {side.upper()}\n"
            f"Score : {ml_score:.3f}  (need ≥ {threshold:.2f})\n"
            f"<i>Signal skipped — low confidence</i>"
        )
        self._send(text)

    @property
    def enabled(self) -> bool:
        return self._enabled