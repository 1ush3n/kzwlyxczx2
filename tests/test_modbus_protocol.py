from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pytest

from core.comms.config import PROJECT_ROOT, load_yaml_mapping
from core.comms.models import ControlMode, PLCCommunicationError, PLCStatus
from core.comms.plc_interface import MockPLC, ModbusTCPPLC
from core.physics.agv_kinematics import AGVSystemSim
from env.agv_compliance_env import AGVComplianceEnv
from services.virtual_plc import run_virtual_plc
from tests.protocol_test_utils import (
    build_test_config,
    stop_process,
    wait_for_port,
)


@pytest.fixture
def modbus_service(tmp_path: Path):
    config = build_test_config(tmp_path)
    context = mp.get_context("spawn")
    process = context.Process(target=run_virtual_plc, args=(config,))
    process.start()
    wait_for_port(
        "127.0.0.1",
        int(config["modbus"]["port"]),
        process,
    )
    try:
        yield config
    finally:
        stop_process(process)


def test_modbus_reset_step_and_snapshot(modbus_service: dict) -> None:
    agv_config = load_yaml_mapping(PROJECT_ROOT / "config" / "agv_env_config.yaml")
    plc = ModbusTCPPLC(agv_config, modbus_service)
    try:
        assert plc.connect()
        reset_snapshot = plc.reset(seed=42)
        assert reset_snapshot.step_count == 0
        assert reset_snapshot.slave_position == pytest.approx(-2.0)

        plc.write_impedance(50.0, 500.0, 3000.0)
        plc.inject_tsn_delay(0.025)
        step_snapshot = plc.step_simulation(1.5)
        assert step_snapshot.ack_sequence != reset_snapshot.ack_sequence
        assert step_snapshot.step_count == 1
        assert step_snapshot.rtt_sec == pytest.approx(0.025, rel=1e-6)
        assert step_snapshot.master_position > 0.0
    finally:
        plc.close()


def test_modbus_matches_mock_backend(modbus_service: dict) -> None:
    agv_config = load_yaml_mapping(PROJECT_ROOT / "config" / "agv_env_config.yaml")
    modbus = ModbusTCPPLC(agv_config, modbus_service)
    mock = MockPLC(AGVSystemSim(agv_config), agv_config)
    try:
        modbus.reset(seed=123)
        mock.reset(seed=123)
        for _ in range(5):
            for plc in (modbus, mock):
                plc.write_impedance(45.0, 450.0, 2500.0)
                plc.inject_tsn_delay(0.03)
            network_snapshot = modbus.step_simulation(1.5)
            memory_snapshot = mock.step_simulation(1.5)
            assert network_snapshot.error == pytest.approx(
                memory_snapshot.error,
                rel=1e-5,
                abs=1e-6,
            )
            assert network_snapshot.external_force == pytest.approx(
                memory_snapshot.external_force,
                rel=1e-5,
                abs=1e-5,
            )
    finally:
        modbus.close()


def test_manual_control_has_priority(modbus_service: dict) -> None:
    agv_config = load_yaml_mapping(PROJECT_ROOT / "config" / "agv_env_config.yaml")
    plc = ModbusTCPPLC(agv_config, modbus_service)
    try:
        plc.reset(seed=7)
        manual = plc.apply_manual_impedance(55.0, 600.0, 2500.0)
        assert manual.control_mode is ControlMode.MANUAL

        plc.write_impedance(40.0, 300.0, 3500.0)
        during_manual = plc.step_simulation(1.5)
        assert during_manual.md == pytest.approx(55.0)
        assert during_manual.bd == pytest.approx(600.0)
        assert during_manual.kd == pytest.approx(2500.0)

        released = plc.release_manual_control()
        assert released.control_mode is ControlMode.AUTOMATIC
        automatic = plc.step_simulation(1.5)
        assert automatic.md == pytest.approx(40.0)
        assert automatic.bd == pytest.approx(300.0)
        assert automatic.kd == pytest.approx(3500.0)
    finally:
        plc.close()


def test_invalid_manual_impedance_is_rejected(modbus_service: dict) -> None:
    agv_config = load_yaml_mapping(PROJECT_ROOT / "config" / "agv_env_config.yaml")
    plc = ModbusTCPPLC(agv_config, modbus_service)
    try:
        plc.reset(seed=9)
        with pytest.raises(PLCCommunicationError):
            plc.apply_manual_impedance(1000.0, 500.0, 3000.0)
        snapshot = plc.read_snapshot()
        assert snapshot.status is PLCStatus.INVALID_IMPEDANCE
        assert int(snapshot.alarm) != 0
    finally:
        plc.close()


def test_plc_watchdog_latches_safe_stop(modbus_service: dict) -> None:
    agv_config = load_yaml_mapping(PROJECT_ROOT / "config" / "agv_env_config.yaml")
    plc = ModbusTCPPLC(agv_config, modbus_service)
    try:
        plc.reset(seed=11)
        plc.apply_manual_impedance(50.0, 600.0, 2000.0)
        before = plc.step_simulation(0.5)
        time.sleep(0.4)
        stopped = plc.read_snapshot()
        assert stopped.control_mode is ControlMode.SAFETY_STOP
        assert stopped.master_velocity == pytest.approx(0.0)
        assert stopped.md == pytest.approx(before.md)
        assert stopped.bd == pytest.approx(before.bd)
        assert stopped.kd == pytest.approx(before.kd)

        still_stopped = plc.step_simulation(0.5)
        assert still_stopped.control_mode is ControlMode.SAFETY_STOP
        assert still_stopped.master_velocity == pytest.approx(0.0)

        reset = plc.reset_safety()
        assert reset.control_mode is ControlMode.AUTOMATIC
    finally:
        plc.close()


def test_environment_fails_safe_when_modbus_is_unavailable(tmp_path: Path) -> None:
    config = build_test_config(tmp_path)
    agv_config = load_yaml_mapping(PROJECT_ROOT / "config" / "agv_env_config.yaml")
    plc = ModbusTCPPLC(agv_config, config)
    env = AGVComplianceEnv(plc=plc)
    try:
        _, reset_info = env.reset(seed=1)
        assert not reset_info["plc_connected"]
        _, reward, terminated, truncated, info = env.step(
            np.zeros(3, dtype=np.float32)
        )
        assert reward == -100.0
        assert terminated
        assert not truncated
        assert info["communication_failure"]
    finally:
        env.close()
