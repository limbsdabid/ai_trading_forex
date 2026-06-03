"""
Session Filter — adjusts ML threshold based on current trading session.

All sessions can trade, but Asian session uses a stricter threshold
because of lower volatility and more false signals.

Session times (UTC):
  Asian   : 00:00 – 08:00
  London  : 07:00 – 16:00
  New York: 12:00 – 21:00
  Overlap : 12:00 – 16:00  (London + NY — most liquid)
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class SessionInfo:
    name: str           # 'asian', 'london', 'new_york', 'overlap', 'off_hours'
    threshold_mult: float  # multiplier applied to base ML threshold
    description: str    # human-readable label for logs/Telegram


# Threshold multipliers per session
# Asian  : +25% stricter  (e.g. 0.52 → 0.65)
# London : normal          (0.52 → 0.52)
# NY     : normal          (0.52 → 0.52)
# Overlap: -10% looser    (0.52 → 0.47, more aggressive during best liquidity)
# Off    : +30% stricter  (0.52 → 0.68, weekend gaps / dead hours)
SESSION_CONFIG: dict[str, SessionInfo] = {
    "overlap":   SessionInfo("overlap",   0.90, "🔥 London+NY Overlap"),
    "london":    SessionInfo("london",    1.00, "🇬🇧 London Session"),
    "new_york":  SessionInfo("new_york",  1.00, "🇺🇸 New York Session"),
    "asian":     SessionInfo("asian",     1.25, "🌏 Asian Session (stricter)"),
    "off_hours": SessionInfo("off_hours", 1.30, "🌙 Off Hours (strictest)"),
}


def get_current_session(dt: datetime | None = None) -> SessionInfo:
    """
    Determine the current trading session based on UTC time.
    Priority order: overlap > london > new_york > asian > off_hours
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Normalize to UTC if timezone-aware, otherwise assume UTC
    if dt.tzinfo is not None:
        hour = dt.hour
    else:
        hour = dt.hour  # assume caller passes UTC

    london_open  = 7 <= hour < 16
    ny_open      = 12 <= hour < 21
    asian_open   = hour < 8 or hour >= 22

    if london_open and ny_open:          # 12:00–16:00 UTC
        return SESSION_CONFIG["overlap"]
    elif london_open:                     # 07:00–12:00 UTC
        return SESSION_CONFIG["london"]
    elif ny_open:                         # 16:00–21:00 UTC
        return SESSION_CONFIG["new_york"]
    elif asian_open:                      # 22:00–08:00 UTC
        return SESSION_CONFIG["asian"]
    else:                                 # 21:00–22:00 UTC dead zone
        return SESSION_CONFIG["off_hours"]


def get_session_threshold(base_threshold: float, dt: datetime | None = None) -> tuple[float, SessionInfo]:
    """
    Returns (adjusted_threshold, session_info) for the current session.

    Usage in SMCStrategy:
        threshold, session = get_session_threshold(self.ml_filter.threshold)
        allow = ml_score >= threshold
    """
    session = get_current_session(dt)
    adjusted = round(base_threshold * session.threshold_mult, 3)
    # Hard clamp: never above 0.80 (too restrictive) or below 0.45 (too loose)
    adjusted = max(0.45, min(adjusted, 0.80))
    return adjusted, session
