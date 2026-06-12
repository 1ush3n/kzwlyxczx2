from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def generate_report(database_path: Path, output_directory: Path) -> Path:
    """从数字孪生知识库生成 CSV 与课程答辩对比图。"""

    output_directory.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        samples = pd.read_sql_query(
            "SELECT * FROM twin_samples ORDER BY id",
            connection,
        )
        events = pd.read_sql_query(
            "SELECT * FROM fault_events ORDER BY id",
            connection,
        )
        decisions = pd.read_sql_query(
            "SELECT * FROM decisions ORDER BY id",
            connection,
        )
    samples.to_csv(
        output_directory / "mapek_timeline.csv",
        index=False,
        encoding="utf-8-sig",
    )
    events.to_csv(
        output_directory / "fault_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    decisions.to_csv(
        output_directory / "decisions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report_path = output_directory / "autonomous_cps_report.png"
    if samples.empty:
        raise RuntimeError("知识库中没有数字孪生采样，无法生成报告")

    x = range(len(samples))
    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(x, samples["stress"], label="Stress (N)")
    axes[0].axhline(800.0, color="orange", linestyle="--", label="Warning")
    axes[0].axhline(1500.0, color="red", linestyle="--", label="Critical")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, samples["rtt_sec"] * 1000.0, label="RTT (ms)")
    axes[1].axhline(150.0, color="red", linestyle="--")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, samples["ap1_rssi"], label="AP1 RSSI")
    axes[2].plot(x, samples["ap2_rssi"], label="AP2 RSSI")
    axes[2].step(x, samples["active_ap"] * 10.0 - 100.0, label="Active AP marker")
    axes[2].axhline(-80.0, color="red", linestyle="--")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.25)

    axes[3].plot(x, samples["bd"], label="Bd")
    axes[3].plot(x, samples["kd"], label="Kd")
    axes[3].step(x, samples["control_mode"] * 500.0, label="Control mode")
    axes[3].set_xlabel("Digital twin sample")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.25)
    figure.suptitle("APAL Autonomous CPS: MAPE-K Fault Response")
    figure.tight_layout()
    figure.savefig(report_path, dpi=160)
    plt.close(figure)
    return report_path
