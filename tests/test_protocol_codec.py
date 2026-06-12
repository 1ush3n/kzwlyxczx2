from __future__ import annotations

import math

import pytest

from core.comms.models import ControlMode, PLCCommand, PLCSnapshot
from core.comms.register_map import (
    HoldingCommandFrame,
    decode_holding_frame,
    decode_input_snapshot,
    encode_holding_frame,
    encode_input_snapshot,
    float32_to_registers,
    registers_to_float32,
    registers_to_uint32,
    uint32_to_registers,
)


@pytest.mark.parametrize("value", [0.0, -1.25, 3.1415926, 5000.0])
def test_float32_big_endian_round_trip(value: float) -> None:
    registers = float32_to_registers(value)
    assert len(registers) == 2
    assert math.isclose(registers_to_float32(registers), value, rel_tol=1e-6)


@pytest.mark.parametrize("value", [0, 1, 0x12345678, 0xFFFFFFFF])
def test_uint32_big_endian_round_trip(value: int) -> None:
    registers = uint32_to_registers(value)
    assert registers_to_uint32(registers) == value


def test_holding_frame_round_trip() -> None:
    frame = HoldingCommandFrame(
        md=50.0,
        bd=500.0,
        kd=3000.0,
        master_velocity=1.5,
        injected_rtt_sec=0.025,
        control_mode=ControlMode.AUTOMATIC,
        command=PLCCommand.STEP,
        sequence=0x12345678,
        seed=42,
        flags=1,
    )
    decoded = decode_holding_frame(encode_holding_frame(frame))
    assert decoded.command is PLCCommand.STEP
    assert decoded.sequence == frame.sequence
    assert decoded.seed == 42
    assert math.isclose(decoded.injected_rtt_sec, 0.025, rel_tol=1e-6)


def test_snapshot_round_trip() -> None:
    snapshot = PLCSnapshot(
        error=0.01,
        error_rate=0.2,
        external_force=123.0,
        rtt_sec=0.02,
        delta_x_cmd=0.03,
        master_position=1.0,
        master_velocity=1.5,
        slave_position=-1.0,
        slave_velocity=1.4,
        simulation_time=2.0,
        md=50.0,
        bd=500.0,
        kd=3000.0,
        ack_sequence=99,
        step_count=7,
    )
    decoded = decode_input_snapshot(encode_input_snapshot(snapshot))
    assert decoded.ack_sequence == 99
    assert decoded.step_count == 7
    assert math.isclose(decoded.external_force, 123.0, rel_tol=1e-6)


def test_invalid_register_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        registers_to_float32([1])
    with pytest.raises(ValueError):
        registers_to_uint32([1])
    with pytest.raises(ValueError):
        decode_input_snapshot([0] * 31)

