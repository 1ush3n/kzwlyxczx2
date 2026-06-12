from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

from core.comms.models import (
    ControlMode,
    PLCAlarm,
    PLCCommand,
    PLCSnapshot,
    PLCStatus,
)


class HoldingRegister(IntEnum):
    MD = 0
    BD = 2
    KD = 4
    MASTER_VELOCITY = 6
    INJECTED_RTT = 8
    CONTROL_MODE = 10
    COMMAND = 11
    COMMAND_SEQUENCE = 12
    SEED = 14
    FLAGS = 15


class InputRegister(IntEnum):
    ERROR = 0
    ERROR_RATE = 2
    EXTERNAL_FORCE = 4
    RTT = 6
    DELTA_X_COMMAND = 8
    MASTER_POSITION = 10
    MASTER_VELOCITY = 12
    SLAVE_POSITION = 14
    SLAVE_VELOCITY = 16
    SIMULATION_TIME = 18
    ACTUAL_MD = 20
    ACTUAL_BD = 22
    ACTUAL_KD = 24
    ACK_SEQUENCE = 26
    STATUS = 28
    ALARM = 29
    CONTROL_MODE = 30
    STEP_COUNT = 31


HOLDING_REGISTER_COUNT = 16
INPUT_REGISTER_COUNT = 32
NO_SEED = 0xFFFF


@dataclass(frozen=True)
class HoldingCommandFrame:
    md: float
    bd: float
    kd: float
    master_velocity: float
    injected_rtt_sec: float
    control_mode: ControlMode
    command: PLCCommand
    sequence: int
    seed: int | None = None
    flags: int = 0


def float32_to_registers(value: float) -> list[int]:
    """按大端字节、高字在前编码 IEEE 754 Float32。"""

    raw = struct.pack(">f", float(value))
    return list(struct.unpack(">HH", raw))


def registers_to_float32(registers: Sequence[int]) -> float:
    """按大端字节、高字在前解码 IEEE 754 Float32。"""

    _require_length(registers, 2, "Float32")
    raw = struct.pack(">HH", _uint16(registers[0]), _uint16(registers[1]))
    return float(struct.unpack(">f", raw)[0])


def uint32_to_registers(value: int) -> list[int]:
    """按大端字节、高字在前编码无符号32位整数。"""

    if not 0 <= int(value) <= 0xFFFFFFFF:
        raise ValueError(f"UInt32 超出范围: {value}")
    raw = struct.pack(">I", int(value))
    return list(struct.unpack(">HH", raw))


def registers_to_uint32(registers: Sequence[int]) -> int:
    """按大端字节、高字在前解码无符号32位整数。"""

    _require_length(registers, 2, "UInt32")
    raw = struct.pack(">HH", _uint16(registers[0]), _uint16(registers[1]))
    return int(struct.unpack(">I", raw)[0])


def encode_holding_frame(frame: HoldingCommandFrame) -> list[int]:
    """编码完整的 Holding Register 命令帧。"""

    registers = [0] * HOLDING_REGISTER_COUNT
    _put_float(registers, HoldingRegister.MD, frame.md)
    _put_float(registers, HoldingRegister.BD, frame.bd)
    _put_float(registers, HoldingRegister.KD, frame.kd)
    _put_float(registers, HoldingRegister.MASTER_VELOCITY, frame.master_velocity)
    _put_float(registers, HoldingRegister.INJECTED_RTT, frame.injected_rtt_sec)
    registers[HoldingRegister.CONTROL_MODE] = int(frame.control_mode)
    registers[HoldingRegister.COMMAND] = int(frame.command)
    registers[
        HoldingRegister.COMMAND_SEQUENCE : HoldingRegister.COMMAND_SEQUENCE + 2
    ] = uint32_to_registers(frame.sequence)
    registers[HoldingRegister.SEED] = NO_SEED if frame.seed is None else _uint16(frame.seed)
    registers[HoldingRegister.FLAGS] = _uint16(frame.flags)
    return registers


def decode_holding_frame(registers: Sequence[int]) -> HoldingCommandFrame:
    """解码完整的 Holding Register 命令帧。"""

    _require_length(registers, HOLDING_REGISTER_COUNT, "Holding Register")
    seed_raw = _uint16(registers[HoldingRegister.SEED])
    return HoldingCommandFrame(
        md=_get_float(registers, HoldingRegister.MD),
        bd=_get_float(registers, HoldingRegister.BD),
        kd=_get_float(registers, HoldingRegister.KD),
        master_velocity=_get_float(registers, HoldingRegister.MASTER_VELOCITY),
        injected_rtt_sec=_get_float(registers, HoldingRegister.INJECTED_RTT),
        control_mode=ControlMode(_uint16(registers[HoldingRegister.CONTROL_MODE])),
        command=PLCCommand(_uint16(registers[HoldingRegister.COMMAND])),
        sequence=registers_to_uint32(
            registers[
                HoldingRegister.COMMAND_SEQUENCE : HoldingRegister.COMMAND_SEQUENCE + 2
            ]
        ),
        seed=None if seed_raw == NO_SEED else seed_raw,
        flags=_uint16(registers[HoldingRegister.FLAGS]),
    )


def encode_input_snapshot(snapshot: PLCSnapshot) -> list[int]:
    """将 PLC 快照编码到 Input Register。"""

    registers = [0] * INPUT_REGISTER_COUNT
    _put_float(registers, InputRegister.ERROR, snapshot.error)
    _put_float(registers, InputRegister.ERROR_RATE, snapshot.error_rate)
    _put_float(registers, InputRegister.EXTERNAL_FORCE, snapshot.external_force)
    _put_float(registers, InputRegister.RTT, snapshot.rtt_sec)
    _put_float(registers, InputRegister.DELTA_X_COMMAND, snapshot.delta_x_cmd)
    _put_float(registers, InputRegister.MASTER_POSITION, snapshot.master_position)
    _put_float(registers, InputRegister.MASTER_VELOCITY, snapshot.master_velocity)
    _put_float(registers, InputRegister.SLAVE_POSITION, snapshot.slave_position)
    _put_float(registers, InputRegister.SLAVE_VELOCITY, snapshot.slave_velocity)
    _put_float(registers, InputRegister.SIMULATION_TIME, snapshot.simulation_time)
    _put_float(registers, InputRegister.ACTUAL_MD, snapshot.md)
    _put_float(registers, InputRegister.ACTUAL_BD, snapshot.bd)
    _put_float(registers, InputRegister.ACTUAL_KD, snapshot.kd)
    registers[
        InputRegister.ACK_SEQUENCE : InputRegister.ACK_SEQUENCE + 2
    ] = uint32_to_registers(snapshot.ack_sequence)
    registers[InputRegister.STATUS] = int(snapshot.status)
    registers[InputRegister.ALARM] = int(snapshot.alarm)
    registers[InputRegister.CONTROL_MODE] = int(snapshot.control_mode)
    registers[InputRegister.STEP_COUNT] = _uint16(snapshot.step_count & 0xFFFF)
    return registers


def decode_input_snapshot(registers: Sequence[int]) -> PLCSnapshot:
    """从 Input Register 解码原子 PLC 快照。"""

    _require_length(registers, INPUT_REGISTER_COUNT, "Input Register")
    return PLCSnapshot(
        error=_get_float(registers, InputRegister.ERROR),
        error_rate=_get_float(registers, InputRegister.ERROR_RATE),
        external_force=_get_float(registers, InputRegister.EXTERNAL_FORCE),
        rtt_sec=_get_float(registers, InputRegister.RTT),
        delta_x_cmd=_get_float(registers, InputRegister.DELTA_X_COMMAND),
        master_position=_get_float(registers, InputRegister.MASTER_POSITION),
        master_velocity=_get_float(registers, InputRegister.MASTER_VELOCITY),
        slave_position=_get_float(registers, InputRegister.SLAVE_POSITION),
        slave_velocity=_get_float(registers, InputRegister.SLAVE_VELOCITY),
        simulation_time=_get_float(registers, InputRegister.SIMULATION_TIME),
        md=_get_float(registers, InputRegister.ACTUAL_MD),
        bd=_get_float(registers, InputRegister.ACTUAL_BD),
        kd=_get_float(registers, InputRegister.ACTUAL_KD),
        ack_sequence=registers_to_uint32(
            registers[InputRegister.ACK_SEQUENCE : InputRegister.ACK_SEQUENCE + 2]
        ),
        status=PLCStatus(_uint16(registers[InputRegister.STATUS])),
        alarm=PLCAlarm(_uint16(registers[InputRegister.ALARM])),
        control_mode=ControlMode(_uint16(registers[InputRegister.CONTROL_MODE])),
        step_count=_uint16(registers[InputRegister.STEP_COUNT]),
        connected=True,
    )


def _put_float(registers: list[int], address: int, value: float) -> None:
    registers[int(address) : int(address) + 2] = float32_to_registers(value)


def _get_float(registers: Sequence[int], address: int) -> float:
    return registers_to_float32(registers[int(address) : int(address) + 2])


def _uint16(value: int) -> int:
    value = int(value)
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"UInt16 超出范围: {value}")
    return value


def _require_length(registers: Sequence[int], expected: int, label: str) -> None:
    if len(registers) != expected:
        raise ValueError(f"{label} 长度必须为 {expected}，实际为 {len(registers)}")
