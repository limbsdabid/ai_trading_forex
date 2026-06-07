import logging
import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("trading_bot")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def send_telegram_message(
    message: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    timeout: int = 5,
) -> bool:
    """Send a lightweight synchronous Telegram message via .env credentials."""
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    target_chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not target_chat_id:
        return False

    try:
        response = requests.post(
            TELEGRAM_API.format(token=token),
            json={"chat_id": target_chat_id, "text": message},
            timeout=timeout,
        )
        if response.ok:
            return True
        log.warning(f"Telegram message failed: {response.status_code} {response.text[:120]}")
    except Exception as exc:
        log.warning(f"Telegram message error: {exc}")
    return False
