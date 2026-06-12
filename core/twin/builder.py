from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from core.comms.models import ControlMode
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


class TwinStateBuilder:
    """将 OPC UA 采样构造成统一数字孪生状态。"""

    def __init__(self, agv_config: Mapping[str, Any]):
        self._config = agv_config
        self._previous_rtt = 0.0
        self._losses = 0

    def build(self, values: Mapping[str, Any]) -> TwinState:
        now = datetime.now(timezone.utc)
        updated = values["LastUpdate"]
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age = max(0.0, (now - updated).total_seconds())
        connected = bool(values["ModbusConnected"])
        self._losses = 0 if connected else self._losses + 1
        rtt = float(values["RTT"])
        jitter = abs(rtt - self._previous_rtt)
        self._previous_rtt = rtt

        master_position = float(values["MasterPosition"])
        slave_position = float(values["SlavePosition"])
        master_velocity = float(values["MasterVelocity"])
        slave_velocity = float(values["SlaveVelocity"])
        physics = self._config["physics"]
        predicted = (
            float(physics["Kw"])
            * (master_position - slave_position - float(physics["L"]))
            + float(physics["Cw"]) * (master_velocity - slave_velocity)
        )
        observed = float(values["ExternalForce"])
        residual = abs(observed - predicted) / float(self._config["rl"]["F_max"])
        mode = int(values["ControlMode"])
        quality = "good" if connected and age <= 0.3 else "bad"
        fault = FaultType.NONE if quality == "good" else FaultType.PLC_DISCONNECTED
        severity = Severity.NORMAL if quality == "good" else Severity.CRITICAL
        return TwinState(
            identity=IdentityState(),
            physical=PhysicalTwinState(
                master_position=master_position,
                master_velocity=master_velocity,
                slave_position=slave_position,
                slave_velocity=slave_velocity,
                tracking_error=float(values["TrackingError"]),
                stress=observed,
            ),
            network=NetworkTwinState(
                rtt_sec=rtt,
                jitter_sec=jitter,
                ap1_rssi=float(values["AP1RSSI"]),
                ap2_rssi=float(values["AP2RSSI"]),
                link_occupancy=float(values["LinkOccupancy"]),
                active_ap=int(values["ActiveAP"]),
                active_route=str(values["ActiveRoute"]),
            ),
            control=ControlTwinState(
                md=float(values["ActualMd"]),
                bd=float(values["ActualBd"]),
                kd=float(values["ActualKd"]),
                control_mode=mode,
                safety_stop_latched=mode == int(ControlMode.SAFETY_STOP),
            ),
            health=HealthState(
                fault_type=fault,
                severity=severity,
                alarm_bits=int(values["AlarmBits"]),
                data_quality=quality,
            ),
            synchronization=SynchronizationState(
                acquired_at=now,
                data_age_sec=age,
                consecutive_losses=self._losses,
            ),
            prediction=PredictionState(
                predicted_stress=predicted,
                normalized_residual=residual,
                risk_level=(
                    "high" if abs(observed) >= 1500.0 else
                    "medium" if abs(observed) >= 800.0 else "low"
                ),
            ),
        )
