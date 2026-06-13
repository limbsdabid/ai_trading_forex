from dataclasses import dataclass


SETUP_EXPIRY_M5_CANDLES = 6


@dataclass
class SetupMemory:
    symbol: str
    bias: str
    state: str
    zone_status: str
    zone_top: float
    zone_bot: float
    zone_mid: float
    created_at: object
    last_m5_time: object
    candles_waited: int = 0
    expires_after: int = SETUP_EXPIRY_M5_CANDLES
