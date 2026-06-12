from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class FaultType(StrEnum):
    NONE = "none"
    AP_SIGNAL = "ap_signal"
    RTT_SPIKE = "rtt_spike"
    STRESS_WARNING = "stress_warning"
    STRESS_CRITICAL = "stress_critical"
    PLC_DISCONNECTED = "plc_disconnected"
    MODEL_MISMATCH = "model_mismatch"


class Severity(StrEnum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class IdentityState:
    asset_id: str = "AGV1"
    asset_type: str = "主从协同运输 AGV"
    version: str = "1.0"


@dataclass(frozen=True)
class PhysicalTwinState:
    master_position: float
    master_velocity: float
    slave_position: float
    slave_velocity: float
    tracking_error: float
    stress: float


@dataclass(frozen=True)
class NetworkTwinState:
    rtt_sec: float
    jitter_sec: float
    ap1_rssi: float
    ap2_rssi: float
    link_occupancy: float
    active_ap: int
    active_route: str


@dataclass(frozen=True)
class ControlTwinState:
    md: float
    bd: float
    kd: float
    control_mode: int
    safety_stop_latched: bool


@dataclass(frozen=True)
class HealthState:
    fault_type: FaultType
    severity: Severity
    alarm_bits: int
    data_quality: str


@dataclass(frozen=True)
class SynchronizationState:
    acquired_at: datetime
    data_age_sec: float
    consecutive_losses: int


@dataclass(frozen=True)
class PredictionState:
    predicted_stress: float
    normalized_residual: float
    risk_level: str


@dataclass(frozen=True)
class TwinState:
    identity: IdentityState
    physical: PhysicalTwinState
    network: NetworkTwinState
    control: ControlTwinState
    health: HealthState
    synchronization: SynchronizationState
    prediction: PredictionState

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["health"]["fault_type"] = self.health.fault_type.value
        result["health"]["severity"] = self.health.severity.value
        result["synchronization"]["acquired_at"] = (
            self.synchronization.acquired_at.astimezone(timezone.utc).isoformat()
        )
        return result
