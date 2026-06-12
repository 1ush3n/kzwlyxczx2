from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, IntFlag
from typing import Tuple


class PLCStatus(IntEnum):
    """PLC 运行状态码。"""

    OK = 0
    INVALID_IMPEDANCE = 1
    INVALID_COMMAND = 2
    INTERNAL_ERROR = 3
    COMMUNICATION_ERROR = 4


class PLCAlarm(IntFlag):
    """PLC 报警位。"""

    NONE = 0
    IMPEDANCE_OUT_OF_RANGE = 1 << 0
    INVALID_COMMAND = 1 << 1
    SIMULATION_ERROR = 1 << 2
    COMMUNICATION_LOST = 1 << 3
    WATCHDOG_TIMEOUT = 1 << 4
    SAFETY_STOP_LATCHED = 1 << 5


class ControlMode(IntEnum):
    """阻抗控制权模式。"""

    AUTOMATIC = 0
    MANUAL = 1
    SAFETY_STOP = 2


class PLCCommand(IntEnum):
    """Modbus 命令码。"""

    NONE = 0
    RESET = 1
    STEP = 2
    APPLY_MANUAL_IMPEDANCE = 3
    RELEASE_MANUAL_CONTROL = 4
    SAFE_STOP = 5
    RESET_SAFETY = 6


@dataclass(frozen=True)
class PLCSnapshot:
    """一次原子读取获得的 PLC 状态快照。"""

    error: float = 0.0
    error_rate: float = 0.0
    external_force: float = 0.0
    rtt_sec: float = 0.0
    delta_x_cmd: float = 0.0
    master_position: float = 0.0
    master_velocity: float = 0.0
    slave_position: float = 0.0
    slave_velocity: float = 0.0
    simulation_time: float = 0.0
    md: float = 0.0
    bd: float = 0.0
    kd: float = 0.0
    ack_sequence: int = 0
    status: PLCStatus = PLCStatus.OK
    alarm: PLCAlarm = PLCAlarm.NONE
    control_mode: ControlMode = ControlMode.AUTOMATIC
    step_count: int = 0
    connected: bool = True
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def sensor_tuple(self) -> Tuple[float, float, float, float, float]:
        """返回与原环境兼容的五元传感器观测。"""

        return (
            self.error,
            self.error_rate,
            self.external_force,
            self.rtt_sec,
            self.delta_x_cmd,
        )

    @classmethod
    def disconnected(cls, previous: PLCSnapshot | None = None) -> PLCSnapshot:
        """保留最后有效数值并标记通信故障。"""

        base = previous or cls()
        return cls(
            error=base.error,
            error_rate=base.error_rate,
            external_force=base.external_force,
            rtt_sec=base.rtt_sec,
            delta_x_cmd=base.delta_x_cmd,
            master_position=base.master_position,
            master_velocity=base.master_velocity,
            slave_position=base.slave_position,
            slave_velocity=base.slave_velocity,
            simulation_time=base.simulation_time,
            md=base.md,
            bd=base.bd,
            kd=base.kd,
            ack_sequence=base.ack_sequence,
            status=PLCStatus.COMMUNICATION_ERROR,
            alarm=base.alarm | PLCAlarm.COMMUNICATION_LOST,
            control_mode=base.control_mode,
            step_count=base.step_count,
            connected=False,
        )


class PLCCommunicationError(RuntimeError):
    """工业通信链路不可用或响应无效。"""
