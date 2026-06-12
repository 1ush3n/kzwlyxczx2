from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from core.twin.aas import AASExporter
from core.twin.models import (
    ControlTwinState,
    FaultType,
    HealthState,
    IdentityState,
    NetworkTwinState,
    PhysicalTwinState,
    PredictionState,
    Severity,
    SynchronizationState,
    TwinState,
)
from core.twin.repository import SQLiteTwinRepository


def make_state() -> TwinState:
    return TwinState(
        identity=IdentityState(),
        physical=PhysicalTwinState(1.0, 0.5, -1.0, 0.4, 0.01, 500.0),
        network=NetworkTwinState(0.05, 0.01, -60.0, -70.0, 0.2, 1, "route-1"),
        control=ControlTwinState(50.0, 500.0, 3000.0, 0, False),
        health=HealthState(FaultType.NONE, Severity.NORMAL, 0, "good"),
        synchronization=SynchronizationState(
            datetime.now(timezone.utc),
            0.01,
            0,
        ),
        prediction=PredictionState(480.0, 0.004, "low"),
    )


def test_twin_repository_and_csv_export(tmp_path: Path) -> None:
    database = tmp_path / "twin.sqlite3"
    repository = SQLiteTwinRepository(database)
    state = make_state()
    repository.save_state(state)
    repository.record_fault("rtt_spike", "start", "测试")
    decision = repository.record_decision(
        "adapt_impedance",
        "rtt_spike",
        "测试决策",
    )
    repository.record_execution(decision, True, "Good", "完成")
    repository.export_csv(tmp_path)
    repository.close()

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM twin_samples"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM decisions"
        ).fetchone()[0] == 1
    assert (tmp_path / "mapek_timeline.csv").exists()
    assert (tmp_path / "fault_events.csv").exists()


def test_lightweight_aas_contains_required_submodels(tmp_path: Path) -> None:
    path = tmp_path / "AGV1.aas.json"
    exporter = AASExporter(
        {
            "aas_id": "urn:test:aas",
            "global_asset_id": "urn:test:asset",
            "manufacturer": "Test",
            "technical_data": {"frequency": 50},
            "documentation": "test",
        }
    )
    exporter.export(make_state(), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["assetAdministrationShells"][0]["idShort"] == "AGV1"
    assert set(payload["submodels"]) == {
        "Identification",
        "TechnicalData",
        "OperationalData",
        "Communication",
        "ConditionMonitoring",
        "Documentation",
    }
