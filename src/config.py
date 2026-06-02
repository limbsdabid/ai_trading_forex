from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


def _load_secret(service: str, key: str, env_var: str, default: str = "") -> str:
    try:
        import keyring
        val = keyring.get_password(service, key)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(env_var, default)


@dataclass
class Config:
    symbols: list = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD",
    ])
    timeframe: str = "H1"
    bars: int = 500
    scan_interval_minutes: int = 60

    account_balance: float = 10_000
    risk_per_trade: float = 0.02
    max_daily_risk: float = 0.06
    max_positions: int = 5

    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    use_mt5: bool = False

    log_level: str = "INFO"
    paper_trading: bool = True

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            mt5_login=int(os.getenv("MT5_LOGIN", "0")) or None,
            mt5_password=_load_secret("ai_trading_forex", "mt5_password", "MT5_PASSWORD"),
            mt5_server=os.getenv("MT5_SERVER", "").strip() or None,
            use_mt5=os.getenv("USE_MT5", "false").lower() == "true",
            account_balance=float(os.getenv("ACCOUNT_BALANCE", "10000")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.02")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
            timeframe=os.getenv("TIMEFRAME", "H1"),
            scan_interval_minutes=int(os.getenv("SCAN_INTERVAL", "60")),
            telegram_bot_token=_load_secret("ai_trading_forex", "telegram_bot_token", "TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )
