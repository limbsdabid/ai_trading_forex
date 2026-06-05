import time
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from src.config import Config
from src.data import DataProvider, FOREX_MAJORS
from src.strategies import Strategy, Signal, SignalType
from src.strategies.smc_strategy import SMCStrategy
from src.execution import Broker, PaperBroker, LiveBroker, Order, OrderSide, OrderType
from src.risk import RiskManager, TradeSizing
from src.notifications import TelegramNotifier
from src.utils import setup_logger


console = Console()
log = setup_logger()

TRADE_LOG_PATH = Path(__file__).parent / "logs" / "trades.csv"


class TradingBot:
    def __init__(self, config: Config):
        self.config = config
        self.data_provider = DataProvider(
            mt5_login=config.mt5_login,
            mt5_password=config.mt5_password,
            mt5_server=config.mt5_server,
        )
        if config.paper_trading:
            self.broker: Broker = PaperBroker(config.account_balance)
        else:
            self.broker: Broker = LiveBroker()
        # Note: balance sync happens AFTER MT5 connects in start()
        self.risk_manager = RiskManager(
            account_balance=config.account_balance,
            risk_per_trade=config.risk_per_trade,
            max_daily_risk=config.max_daily_risk,
            max_positions=config.max_positions,
        )
        self.strategies: list[Strategy] = [
            SMCStrategy(
                risk_manager=self.risk_manager,
                data_provider=self.data_provider,
                ml_threshold=config.ml_threshold,
                ml_thresholds=config.ml_thresholds,
            ),
        ]
        if config.use_mtl or config.ab_test:
            for strat in self.strategies:
                if isinstance(strat, SMCStrategy):
                    strat.use_mtl = config.use_mtl
                    strat.ab_test = config.ab_test
        self.broker.on_close = self._on_trade_closed
        self.notifier = TelegramNotifier(config.telegram_bot_token, config.telegram_chat_id)
        self.symbols = config.symbols
        self.running = False
        self._open_trades: dict[str, dict] = {}
        self._daily_trades: list[dict] = []
        self._last_summary_date = datetime.now().date()

    def start(self):
        log.info("Bot starting...")
        self.running = True

        if self.config.use_mt5:
            if self.data_provider.connect_mt5():
                log.info("MT5 connected successfully")
                time.sleep(2)
                self._sync_mt5_account()
            else:
                log.warning("MT5 connection failed, using simulated data")
        else:
            log.info("Using simulated data mode")

        try:
            while self.running:
                self._scan_cycle()
                self._sleep_until_next_scan()
        except KeyboardInterrupt:
            log.info("Bot stopped by user")
        finally:
            self.data_provider.disconnect_mt5()

    def stop(self):
        self.running = False

    def _sync_mt5_account(self):
        try:
            import MetaTrader5 as mt5
            acc = mt5.account_info()
            if acc:
                balance = acc.balance
                self.risk_manager.update_balance(balance)
                log.info(f"Account: {acc.login} @ {acc.server}")
                log.info(f"Balance: ${balance:,.2f} | Equity: ${acc.equity:,.2f}")
                console.print(f"[cyan]MT5 Account:[/cyan] {acc.login} ({acc.server})")
                console.print(f"[green]Balance:[/green] ${balance:,.2f}")
            else:
                log.warning("No account info from MT5")
        except Exception as e:
            log.error(f"Failed to sync MT5 account: {e}")

    def _reconcile_positions(self):
        """Reconcile open trades with actual broker positions.
        If a trade is in _open_trades but not in broker, assume it was auto-closed (SL/TP hit)."""
        current = {(p.symbol, p.side) for p in self.broker.get_positions()}
        for key in list(self._open_trades.keys()):
            trade = self._open_trades[key]
            side = OrderSide.BUY if trade['side'] == 'buy' else OrderSide.SELL
            if (trade['symbol'], side) not in current:
                # Position was closed externally. Get current price as best estimate of exit price
                try:
                    data = self.data_provider.fetch_rates(trade['symbol'], "M5", 1)
                    exit_price = data.data.iloc[-1]["close"] if data else trade.get('entry', 0)
                except Exception:
                    exit_price = trade.get('entry', 0)

                # Calculate estimated P&L
                pnl = (exit_price - trade['entry']) * trade['volume'] * 10000 if side == OrderSide.BUY else (trade['entry'] - exit_price) * trade['volume'] * 10000
                log.warning(f"Position {trade['symbol']} {side.value} auto-closed (estimated exit={exit_price}, pnl={pnl:.2f})")
                self._on_trade_closed(key, exit_price, datetime.now().isoformat(), pnl, 'auto_closed')

    def _scan_cycle(self):
        log.info(f"=== Scan cycle: {datetime.now().isoformat()} ===")
        self._reconcile_positions()

        for symbol in self.symbols:
            data = self.data_provider.fetch_rates(
                symbol, "M5", 200
            )
            if not data:
                log.warning(f"No data for {symbol}, skipping")
                continue

            latest_price = data.data.iloc[-1]["close"]
            log.info(f"{symbol}: {latest_price}")

            for strategy in self.strategies:
                signal = strategy.generate_signal(data.data, symbol)
                self._execute_signal(signal, latest_price)

        self._check_daily_reset()
        self._display_positions()

    def _check_daily_reset(self):
        now = datetime.now()
        if now.date() != self._last_summary_date:
            trades = len(self._daily_trades)
            if trades > 0:
                wins = sum(1 for t in self._daily_trades if 'win' in t.get('result', ''))
                losses = sum(1 for t in self._daily_trades if 'loss' in t.get('result', ''))
                pnl = sum(t.get('pnl', 0) for t in self._daily_trades)
                balance = self.broker.get_account_balance()
                self.notifier.send_daily_summary(trades, wins, losses, pnl, balance)
            self._daily_trades.clear()
            self._last_summary_date = now.date()
            self.risk_manager.reset_daily()
            log.info('Daily risk reset')

    def _execute_signal(self, signal: Signal, price: float):
        if signal.type == SignalType.HOLD:
            return

        if signal.type in (SignalType.CLOSE_BUY, SignalType.CLOSE_SELL):
            side = OrderSide.BUY if signal.type == SignalType.CLOSE_BUY else OrderSide.SELL
            position_key = f"{signal.symbol}_{side.value}"
            if self.broker.close_position(signal.symbol, side):
                trade = self._open_trades.pop(position_key, {})
                if trade:
                    pnl = (price - trade['entry']) * trade['volume'] * 10000 if side == OrderSide.BUY else (trade['entry'] - price) * trade['volume'] * 10000
                    self._on_trade_closed(position_key, price, datetime.now().isoformat(), pnl, 'manual_close')
                log.info(f"Closed {signal.symbol} {side.value} position")
            return

        if any(pos.symbol == signal.symbol for pos in self.broker.get_positions()):
            log.info(f"{signal.symbol}: position already exists, skipping")
            return

        side = OrderSide.BUY if signal.type == SignalType.BUY else OrderSide.SELL
        meta = signal.metadata or {}

        sl_price = meta.get('sl', price * (0.995 if side == OrderSide.BUY else 1.005))
        tp_price = meta.get('tp', 0.0)
        volume = meta.get('volume', 0.01)

        sizing = self.risk_manager.calculate_size(price, sl_price, signal.symbol)
        if not sizing:
            log.info(f"{signal.symbol}: risk limits reached, skipping")
            return

        volume = min(sizing.volume, 1.0)

        order = Order(
            symbol=signal.symbol,
            side=side,
            volume=volume,
            order_type=OrderType.MARKET,
            price=price,
            stop_loss=sl_price,
            take_profit=tp_price,
        )

        result = self.broker.place_order(order)
        if result and result.status == "executed":
            self.risk_manager.open_trade(signal.symbol)   # pass symbol, not sizing
            log.info(
                f"{side.value.upper()} {signal.symbol} "
                f"vol={volume} sl={sl_price} "
                f"tp={tp_price} conf={signal.confidence}"
            )
            position_key = f"{signal.symbol}_{side.value}"
            self._open_trades[position_key] = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal.symbol,
                'side': side.value,
                'entry': price,
                'sl': sl_price,
                'tp': tp_price,
                'volume': volume,
            }
            ml_score = meta.get('ml_score', None)
            self.notifier.send_trade_opened(
                signal.symbol, side.value, price, sl_price, tp_price, volume,
                ml_score=ml_score,
            )

    def _on_trade_closed(self, key: str, exit_price: float, exit_time: str, pnl: float, result: str):
        trade = self._open_trades.pop(key, {})
        symbol = trade.get('symbol') or key.split('_')[0]
        self.risk_manager.close_trade(symbol, pnl)   # release symbol slot and track P&L
        file_exists = TRADE_LOG_PATH.exists()
        TRADE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "symbol", "side", "entry", "sl", "tp",
                               "volume", "exit_price", "exit_time", "pnl", "result"])
            writer.writerow([
                trade.get('timestamp', ''),
                trade.get('symbol', key.split('_')[0] if '_' in key else key),
                trade.get('side', ''),
                round(trade['entry'], 5) if 'entry' in trade else '',
                round(trade['sl'], 5) if 'sl' in trade else '',
                round(trade['tp'], 5) if 'tp' in trade else '',
                trade.get('volume', ''),
                round(exit_price, 5),
                exit_time,
                round(pnl, 2),
                result,
            ])

        # Sync actual balance from MT5 after every closed trade (LiveBroker only)
        if self.config.use_mt5 and not self.config.paper_trading:
            self._sync_mt5_account()

        side = trade.get('side', '')
        entry = trade.get('entry', 0.0)
        self.notifier.send_trade_closed(symbol, side, entry, exit_price, pnl, result)
        self._daily_trades.append({'pnl': pnl, 'result': result})

    def _display_positions(self):
        positions = self.broker.get_positions()
        if not positions:
            console.print("[dim]No open positions[/dim]")
            return

        table = Table(title="Open Positions", title_style="bold cyan")
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", style="yellow")
        table.add_column("Vol", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("SL", justify="right")
        table.add_column("TP", justify="right")

        for pos in positions:
            pnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            table.add_row(
                pos.symbol,
                "LONG" if pos.side == OrderSide.BUY else "SHORT",
                str(pos.volume),
                str(pos.entry_price),
                str(pos.current_price),
                f"[{pnl_color}]{pos.unrealized_pnl:.2f}[/{pnl_color}]",
                str(pos.stop_loss),
                str(pos.take_profit),
            )

        console.print(table)

    def _sleep_until_next_scan(self):
        interval = self.config.scan_interval_minutes
        log.info(f"Next scan in {interval} minutes")
        for _ in range(interval * 60):
            if not self.running:
                break
            time.sleep(1)


def main():
    config = Config.from_env()
    bot = TradingBot(config)

    console.print("[bold green]===== SMC Forex Trading Bot =====")
    console.print(f"Symbols: {', '.join(config.symbols)}")
    console.print(f"Strategy: SMC (H4 Bias + M15 Zones + M5 CHoCH)")
    console.print(f"Risk per trade: {config.risk_per_trade*100}%")
    console.print(f"Max lot: 1.0")

    if config.paper_trading:
        console.print("[yellow]Mode: Paper Trading[/yellow]")
    elif config.use_mt5:
        console.print("[cyan]Mode: MT5 Demo[/cyan]")

    bot.start()


if __name__ == "__main__":
    main()