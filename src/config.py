from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()


def _load_secret(service: str, key: str, env_var: str, default: str = "") -> str:
    """Read sensitive config from environment variables (.env file)."""
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

    # ML filter — per-symbol minimum win-probability to allow a trade.
    # Based on training results (test-set confidence gate analysis):
    #   EURUSD: 0.55 → 80.8% WR (26 signals)
    #   GBPUSD: 0.60 → 76.3% WR (38 signals)
    #   USDJPY: 0.55 → 78.0% WR (50 signals)
    #   USDCHF: 0.58 → 74.0% WR (50 signals)
    #   AUDUSD: 0.60 → 63.8% WR (47 signals)
    #   USDCAD: 0.60 → 80.4% WR (46 signals)
    #   NZDUSD: 0.52 → 52.5% WR (61 signals) — weak model, low threshold to keep signals
    # Fallback default used for any symbol not listed here.
    ml_threshold: float = 0.55          # fallback default (kept for .env compat)
    ml_thresholds: dict = field(default_factory=lambda: {
        "EURUSD": 0.55,
        "GBPUSD": 0.60,
        "USDJPY": 0.55,
        "USDCHF": 0.58,
        "AUDUSD": 0.60,
        "USDCAD": 0.60,
        "NZDUSD": 0.52,
    })

    use_mtl: bool = False
    ab_test: bool = True

    def get_threshold(self, symbol: str) -> float:
        """Return the ML threshold for a symbol, falling back to ml_threshold."""
        return self.ml_thresholds.get(symbol.upper(), self.ml_threshold)

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
            scan_interval_minutes=max(1, min(int(os.getenv("SCAN_INTERVAL", "5")), 60)),
            telegram_bot_token=_load_secret("ai_trading_forex", "telegram_bot_token", "TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            ml_threshold=float(os.getenv("ML_THRESHOLD", "0.55")),
            use_mtl=os.getenv("USE_MTL", "false").lower() == "true",
            ab_test=os.getenv("AB_TEST", "true").lower() == "true",
        )
