from __future__ import annotations

import csv
import json
import sqlite3
import threading
from pathlib import Path
from typing import Protocol

from core.twin.models import TwinState


class TwinRepository(Protocol):
    def save_state(self, state: TwinState) -> None: ...

    def record_fault(self, fault: str, phase: str, detail: str) -> None: ...

    def record_decision(self, action: str, trigger: str, reason: str) -> int: ...

    def record_execution(
        self,
        decision_id: int,
        success: bool,
        status_code: str,
        detail: str,
    ) -> None: ...


class SQLiteTwinRepository:
    """MAPE-K 的持久化 Knowledge 层。"""

    def __init__(self, database_path: Path):
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path
        self._connection = sqlite3.connect(
            database_path,
            timeout=10.0,
            check_same_thread=False,
        )
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS twin_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    acquired_at TEXT NOT NULL,
                    stress REAL NOT NULL,
                    rtt_sec REAL NOT NULL,
                    ap1_rssi REAL NOT NULL,
                    ap2_rssi REAL NOT NULL,
                    active_ap INTEGER NOT NULL,
                    md REAL NOT NULL,
                    bd REAL NOT NULL,
                    kd REAL NOT NULL,
                    control_mode INTEGER NOT NULL,
                    fault_type TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fault_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    fault TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    decision_id INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    status_code TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
                """
            )

    def save_state(self, state: TwinState) -> None:
        payload = state.to_dict()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO twin_samples (
                    acquired_at, stress, rtt_sec, ap1_rssi, ap2_rssi,
                    active_ap, md, bd, kd, control_mode, fault_type, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["synchronization"]["acquired_at"],
                    state.physical.stress,
                    state.network.rtt_sec,
                    state.network.ap1_rssi,
                    state.network.ap2_rssi,
                    state.network.active_ap,
                    state.control.md,
                    state.control.bd,
                    state.control.kd,
                    state.control.control_mode,
                    state.health.fault_type.value,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def record_fault(self, fault: str, phase: str, detail: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO fault_events (fault, phase, detail) VALUES (?, ?, ?)",
                (fault, phase, detail),
            )

    def record_decision(self, action: str, trigger: str, reason: str) -> int:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO decisions (action, trigger, reason)
                VALUES (?, ?, ?)
                """,
                (action, trigger, reason),
            )
            return int(cursor.lastrowid)

    def record_execution(
        self,
        decision_id: int,
        success: bool,
        status_code: str,
        detail: str,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO executions (
                    decision_id, success, status_code, detail
                ) VALUES (?, ?, ?, ?)
                """,
                (decision_id, int(success), status_code, detail),
            )

    def export_csv(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        self._export_table("twin_samples", output_directory / "mapek_timeline.csv")
        self._export_table("fault_events", output_directory / "fault_events.csv")

    def _export_table(self, table: str, path: Path) -> None:
        with self._lock:
            cursor = self._connection.execute(f"SELECT * FROM {table}")
            columns = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
