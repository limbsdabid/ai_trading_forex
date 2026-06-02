import keyring
from getpass import getpass

SERVICE = "ai_trading_forex"
FIELDS = [
    ("MT5_PASSWORD", "mt5_password", "MT5 account password"),
    ("TELEGRAM_BOT_TOKEN", "telegram_bot_token", "Telegram bot token"),
]


def main():
    print("Saving credentials to Windows Credential Manager (keyring)")
    print("These will be loaded instead of .env values.\n")

    for env_var, keyring_key, label in FIELDS:
        current = keyring.get_password(SERVICE, keyring_key)
        if current:
            masked = current[:4] + "..." + current[-4:] if len(current) > 8 else "***"
            print(f"[{label}] Currently stored: {masked}")
        else:
            print(f"[{label}] Not stored yet")

        val = getpass(f"  Enter {label} (leave empty to skip/keep): ").strip()
        if val:
            keyring.set_password(SERVICE, keyring_key, val)
            print(f"  Saved ✓\n")
        else:
            print(f"  Skipped\n")

    print("Done! Your credentials are now stored securely in Windows Credential Manager.")
    print("You can now remove MT5_PASSWORD and TELEGRAM_BOT_TOKEN from your .env file.")


if __name__ == "__main__":
    main()
