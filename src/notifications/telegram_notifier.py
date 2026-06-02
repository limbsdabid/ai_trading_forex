import logging
from datetime import datetime
from typing import Optional

import requests

log = logging.getLogger("trading_bot")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self._enabled = bool(bot_token and chat_id)

    def _send(self, text: str) -> bool:
        if not self._enabled:
            return False
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if not resp.ok:
                log.warning(f"Telegram send failed: {resp.status_code} {resp.text}")
            return resp.ok
        except Exception as e:
            log.warning(f"Telegram send error: {e}")
            return False

    def send_signal(self, symbol: str, side: str, entry: float, sl: float, tp: float, volume: float, confidence: float = 0.0):
        text = (
            f"\U0001F4E1 <b>SIGNAL</b>\n"
            f"{symbol} {side.upper()}\n"
            f"Entry: {entry}\n"
            f"SL: {sl} | TP: {tp}\n"
            f"Vol: {volume} | Conf: {confidence:.0%}"
        )
        self._send(text)

    def send_trade_opened(self, symbol: str, side: str, entry: float, sl: float, tp: float, volume: float):
        emoji = "\U0001F7E2" if side.lower() == "buy" else "\U0001F534"
        text = (
            f"{emoji} <b>TRADE OPENED</b>\n"
            f"{symbol} {side.upper()}\n"
            f"Entry: {entry}\n"
            f"SL: {sl} | TP: {tp}\n"
            f"Volume: {volume}"
        )
        self._send(text)

    def send_trade_closed(self, symbol: str, side: str, entry: float, exit_price: float, pnl: float, result: str):
        emoji = "\U0001F7E2" if "win" in result.lower() else "\U0001F534"
        sign = "+" if pnl >= 0 else ""
        text = (
            f"{emoji} <b>TRADE CLOSED</b>\n"
            f"{symbol} {side.upper()}\n"
            f"Entry: {entry} | Exit: {exit_price}\n"
            f"PnL: {sign}${pnl:.2f}\n"
            f"Result: {result.upper()}"
        )
        self._send(text)

    def send_error(self, message: str):
        text = f"\u26A0\ufe0f <b>ERROR</b>\n{message}"
        self._send(text)

    def send_daily_summary(self, trades: int, wins: int, losses: int, pnl: float, balance: float):
        win_rate = (wins / trades * 100) if trades > 0 else 0
        sign = "+" if pnl >= 0 else ""
        text = (
            f"\U0001F4CA <b>DAILY SUMMARY</b>\n"
            f"Trades: {trades} | Wins: {wins} | Losses: {losses}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"PnL: {sign}${pnl:.2f}\n"
            f"Balance: ${balance:.2f}"
        )
        self._send(text)
