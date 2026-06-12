from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


GATE_PATTERN = re.compile(r"\[(G[2-5])-(PASS|FAIL|NEAR)\]")
SETUP_EVENTS = {
    "SETUP_CREATED",
    "CHOCH_WAITING",
    "SETUP_EXPIRED",
    "TRADE_EXECUTED",
}


def expectancy_from_r(r_values: list[float]) -> dict[str, float]:
    clean = [r for r in r_values if r is not None]
    if not clean:
        return {
            "expectancy": 0.0,
            "win_rate": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "trades": 0,
        }

    wins = [r for r in clean if r > 0]
    losses = [r for r in clean if r <= 0]

    win_rate = len(wins) / len(clean)
    loss_rate = 1.0 - win_rate
    avg_win_r = sum(wins) / len(wins) if wins else 0.0
    avg_loss_r = abs(sum(losses) / len(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

    return {
        "expectancy": round(expectancy, 4),
        "win_rate": round(win_rate, 4),
        "avg_win_R": round(avg_win_r, 4),
        "avg_loss_R": round(avg_loss_r, 4),
        "trades": len(clean),
    }


def _trade_r_multiple(row: dict[str, str]) -> float | None:
    try:
        side = row.get("side", "").lower()
        entry = float(row["entry"])
        sl = float(row["sl"])
        exit_price = float(row["exit_price"])

        risk = abs(entry - sl)
        if risk <= 0:
            return None

        if side == "buy":
            return (exit_price - entry) / risk
        if side == "sell":
            return (entry - exit_price) / risk
    except Exception:
        return None

    return None


def analyze_trade_expectancy(trades_path: str | Path = "logs/trades.csv") -> dict[str, float]:
    path = Path(trades_path)
    if not path.exists():
        return expectancy_from_r([])

    r_values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = _trade_r_multiple(row)
            if r is not None:
                r_values.append(r)

    return expectancy_from_r(r_values)


def analyze_bot_log(
    log_path: str | Path = "logs/bot.log",
    trades_path: str | Path | None = "logs/trades.csv",
) -> dict[str, Any]:
    path = Path(log_path)

    gate_counts = {
        "G2": {"PASS": 0, "FAIL": 0, "NEAR": 0},
        "G3": {"PASS": 0, "FAIL": 0, "NEAR": 0},
        "G4": {"PASS": 0, "FAIL": 0, "NEAR": 0},
        "G5": {"PASS": 0, "FAIL": 0, "NEAR": 0},
    }
    setup_counts = {event: 0 for event in SETUP_EVENTS}

    if path.exists():
        with path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                gate_match = GATE_PATTERN.search(line)
                if gate_match:
                    gate, status = gate_match.groups()
                    gate_counts[gate][status] += 1

                for event in SETUP_EVENTS:
                    if f"[{event}]" in line:
                        setup_counts[event] += 1

    gate_rows = []
    for gate, counts in gate_counts.items():
        total = sum(counts.values())
        fail_rate = counts["FAIL"] / total if total else 0.0
        pass_rate = counts["PASS"] / total if total else 0.0
        near_rate = counts["NEAR"] / total if total else 0.0

        gate_rows.append({
            "gate": gate,
            "pass": counts["PASS"],
            "near": counts["NEAR"],
            "fail": counts["FAIL"],
            "total": total,
            "pass_rate": round(pass_rate, 4),
            "near_rate": round(near_rate, 4),
            "fail_rate": round(fail_rate, 4),
        })

    gate_df = pd.DataFrame(gate_rows)
    bottleneck = None
    if not gate_df.empty and gate_df["total"].sum() > 0:
        bottleneck = gate_df.sort_values("fail_rate", ascending=False).iloc[0].to_dict()

    expectancy = (
        analyze_trade_expectancy(trades_path)
        if trades_path is not None
        else expectancy_from_r([])
    )

    return {
        "gate_stats": gate_df,
        "setup_stats": setup_counts,
        "bottleneck": bottleneck,
        "expectancy": expectancy,
    }


if __name__ == "__main__":
    result = analyze_bot_log()
    print("\nGate Stats")
    print(result["gate_stats"].to_string(index=False))

    print("\nSetup Stats")
    print(result["setup_stats"])

    print("\nBottleneck")
    print(result["bottleneck"])

    print("\nExpectancy")
    print(result["expectancy"])