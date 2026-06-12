from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, MutableSequence

from pymodbus.constants import ExcCodes
from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import DataType, SimData, SimDevice

from core.comms.config import (
    load_protocol_config,
    load_yaml_mapping,
    to_mapping,
)
from core.comms.models import (
    ControlMode,
    PLCAlarm,
    PLCCommand,
    PLCSnapshot,
    PLCStatus,
)
from core.comms.plc_interface import MockPLC
from core.comms.register_map import (
    HOLDING_REGISTER_COUNT,
    INPUT_REGISTER_COUNT,
    HoldingCommandFrame,
    HoldingRegister,
    decode_holding_frame,
    encode_holding_frame,
    encode_input_snapshot,
)
from core.physics.agv_kinematics import AGVSystemSim


HOLDING_FUNCTION_CODES = {3, 6, 16, 22, 23}
INPUT_FUNCTION_CODE = 4


class VirtualPLCDevice:
    """将 AGV 物理仿真封装为 Modbus 设备。"""

    def __init__(
        self,
        agv_config: Mapping[str, Any],
        device_id: int,
        watchdog_timeout_sec: float = 0.3,
        watchdog_poll_sec: float = 0.02,
    ):
        self._agv_config = agv_config
        self._backend = MockPLC(AGVSystemSim(agv_config), agv_config)
        self._backend.reset()
        self._lock = asyncio.Lock()
        self._ack_sequence = 0
        self._status = PLCStatus.OK
        self._alarm = PLCAlarm.NONE
        self._control_mode = ControlMode.AUTOMATIC
        self._watchdog_timeout_sec = float(watchdog_timeout_sec)
        self._watchdog_poll_sec = float(watchdog_poll_sec)
        self._last_control_time = time.monotonic()
        self._watchdog_armed = False
        self._last_command = HoldingCommandFrame(
            md=float(agv_config["impedance"]["M_base"]),
            bd=float(agv_config["impedance"]["B_base"]),
            kd=float(agv_config["impedance"]["K_base"]),
            master_velocity=0.0,
            injected_rtt_sec=0.0,
            control_mode=ControlMode.AUTOMATIC,
            command=PLCCommand.NONE,
            sequence=0,
        )
        self.device = SimDevice(
            id=device_id,
            simdata=(
                [SimData(address=0, count=1, values=False, datatype=DataType.BITS)],
                [SimData(address=0, count=1, values=False, datatype=DataType.BITS)],
                [
                    SimData(
                        address=0,
                        count=HOLDING_REGISTER_COUNT,
                        values=0,
                        datatype=DataType.REGISTERS,
                    )
                ],
                [
                    SimData(
                        address=0,
                        count=INPUT_REGISTER_COUNT,
                        values=0,
                        datatype=DataType.REGISTERS,
                        readonly=True,
                    )
                ],
            ),
            action=self._on_register_access,
        )

    async def _on_register_access(
        self,
        function_code: int,
        start_address: int,
        address: int,
        count: int,
        current_registers: MutableSequence[int],
        set_values: MutableSequence[int] | MutableSequence[bool] | None,
    ) -> ExcCodes | None:
        async with self._lock:
            if function_code == INPUT_FUNCTION_CODE:
                self._publish_input_registers(current_registers, start_address)
                return None

            if function_code not in HOLDING_FUNCTION_CODES:
                return None

            self._publish_holding_registers(current_registers, start_address)
            if set_values is None:
                return None

            prospective = list(current_registers)
            offset = address - start_address
            prospective[offset : offset + count] = [int(value) for value in set_values]

            command_address = int(HoldingRegister.COMMAND)
            if not address <= command_address < address + count:
                return None

            try:
                frame = decode_holding_frame(prospective[:HOLDING_REGISTER_COUNT])
                await self._execute_command(frame)
            except (ValueError, TypeError):
                self._status = PLCStatus.INVALID_COMMAND
                self._alarm |= PLCAlarm.INVALID_COMMAND

            normalized = self._current_holding_registers()
            for index in range(count):
                absolute_address = address + index
                if absolute_address < HOLDING_REGISTER_COUNT:
                    set_values[index] = normalized[absolute_address]
            return None

    async def _execute_command(self, frame: HoldingCommandFrame) -> None:
        self._last_command = frame
        self._status = PLCStatus.OK

        try:
            if frame.command is PLCCommand.RESET:
                self._backend.reset(frame.seed)
                self._control_mode = ControlMode.AUTOMATIC
                self._alarm = PLCAlarm.NONE
                self._watchdog_armed = True
                self._last_control_time = time.monotonic()
            elif frame.command is PLCCommand.STEP:
                self._watchdog_armed = True
                self._last_control_time = time.monotonic()
                if self._control_mode is ControlMode.SAFETY_STOP:
                    self._backend.step_simulation(0.0)
                elif self._control_mode is ControlMode.AUTOMATIC:
                    if not self._impedance_is_valid(frame.md, frame.bd, frame.kd):
                        self._reject_impedance(frame.sequence)
                        return
                    self._backend.write_impedance(frame.md, frame.bd, frame.kd)
                    if frame.flags & 0x0001:
                        self._backend.inject_tsn_delay(frame.injected_rtt_sec)
                    self._backend.step_simulation(frame.master_velocity)
                else:
                    if frame.flags & 0x0001:
                        self._backend.inject_tsn_delay(frame.injected_rtt_sec)
                    self._backend.step_simulation(frame.master_velocity)
            elif frame.command is PLCCommand.APPLY_MANUAL_IMPEDANCE:
                if self._control_mode is ControlMode.SAFETY_STOP:
                    self._status = PLCStatus.INVALID_COMMAND
                    self._alarm |= PLCAlarm.SAFETY_STOP_LATCHED
                    self._ack_sequence = frame.sequence
                    return
                if not self._impedance_is_valid(frame.md, frame.bd, frame.kd):
                    self._reject_impedance(frame.sequence)
                    return
                self._backend.write_impedance(frame.md, frame.bd, frame.kd)
                self._control_mode = ControlMode.MANUAL
                self._alarm &= ~PLCAlarm.IMPEDANCE_OUT_OF_RANGE
                self._last_control_time = time.monotonic()
            elif frame.command is PLCCommand.RELEASE_MANUAL_CONTROL:
                if self._control_mode is not ControlMode.SAFETY_STOP:
                    self._control_mode = ControlMode.AUTOMATIC
                self._last_control_time = time.monotonic()
            elif frame.command is PLCCommand.SAFE_STOP:
                await self._latch_safety_stop(PLCAlarm.SAFETY_STOP_LATCHED)
            elif frame.command is PLCCommand.RESET_SAFETY:
                self._control_mode = ControlMode.AUTOMATIC
                self._alarm &= ~(
                    PLCAlarm.WATCHDOG_TIMEOUT | PLCAlarm.SAFETY_STOP_LATCHED
                )
                self._watchdog_armed = True
                self._last_control_time = time.monotonic()
            elif frame.command is not PLCCommand.NONE:
                self._status = PLCStatus.INVALID_COMMAND
                self._alarm |= PLCAlarm.INVALID_COMMAND

            self._ack_sequence = frame.sequence
        except Exception:
            self._status = PLCStatus.INTERNAL_ERROR
            self._alarm |= PLCAlarm.SIMULATION_ERROR
            self._ack_sequence = frame.sequence

    async def watchdog_loop(self) -> None:
        """PLC 本地看门狗，不依赖 OPC UA 或 MAPE-K。"""

        while True:
            await asyncio.sleep(self._watchdog_poll_sec)
            async with self._lock:
                if (
                    self._watchdog_armed
                    and self._control_mode is not ControlMode.SAFETY_STOP
                    and time.monotonic() - self._last_control_time
                    >= self._watchdog_timeout_sec
                ):
                    await self._latch_safety_stop(
                        PLCAlarm.WATCHDOG_TIMEOUT
                        | PLCAlarm.SAFETY_STOP_LATCHED
                    )

    async def _latch_safety_stop(self, alarm: PLCAlarm) -> None:
        if self._control_mode is not ControlMode.SAFETY_STOP:
            self._backend.step_simulation(0.0)
        self._control_mode = ControlMode.SAFETY_STOP
        self._alarm |= alarm

    def _reject_impedance(self, sequence: int) -> None:
        self._status = PLCStatus.INVALID_IMPEDANCE
        self._alarm |= PLCAlarm.IMPEDANCE_OUT_OF_RANGE
        self._ack_sequence = sequence

    def _impedance_is_valid(self, md: float, bd: float, kd: float) -> bool:
        impedance = self._agv_config["impedance"]
        limits = (
            (
                md,
                max(float(impedance["M_base"]) - float(impedance["M_delta_max"]), 1e-4),
                float(impedance["M_base"]) + float(impedance["M_delta_max"]),
            ),
            (
                bd,
                max(float(impedance["B_base"]) - float(impedance["B_delta_max"]), 1e-4),
                float(impedance["B_base"]) + float(impedance["B_delta_max"]),
            ),
            (
                kd,
                max(float(impedance["K_base"]) - float(impedance["K_delta_max"]), 1e-4),
                float(impedance["K_base"]) + float(impedance["K_delta_max"]),
            ),
        )
        return all(lower <= value <= upper for value, lower, upper in limits)

    def _snapshot(self) -> PLCSnapshot:
        return replace(
            self._backend.read_snapshot(),
            ack_sequence=self._ack_sequence,
            status=self._status,
            alarm=self._alarm,
            control_mode=self._control_mode,
        )

    def _current_holding_registers(self) -> list[int]:
        snapshot = self._snapshot()
        frame = replace(
            self._last_command,
            md=snapshot.md,
            bd=snapshot.bd,
            kd=snapshot.kd,
            control_mode=self._control_mode,
            command=PLCCommand.NONE,
            sequence=self._ack_sequence,
        )
        return encode_holding_frame(frame)

    def _publish_holding_registers(
        self,
        registers: MutableSequence[int],
        start_address: int,
    ) -> None:
        values = self._current_holding_registers()
        offset = -start_address
        for address, value in enumerate(values):
            index = address + offset
            if 0 <= index < len(registers):
                registers[index] = value

    def _publish_input_registers(
        self,
        registers: MutableSequence[int],
        start_address: int,
    ) -> None:
        values = encode_input_snapshot(self._snapshot())
        offset = -start_address
        for address, value in enumerate(values):
            index = address + offset
            if 0 <= index < len(registers):
                registers[index] = value


async def serve_virtual_plc(config: Mapping[str, Any]) -> None:
    """启动虚拟 Modbus TCP PLC。"""

    project_root = Path(str(config["runtime"]["project_root"]))
    agv_config_path = project_root / str(config["runtime"]["agv_config_path"])
    agv_config = load_yaml_mapping(agv_config_path)
    modbus = config["modbus"]
    virtual_device = VirtualPLCDevice(
        agv_config=agv_config,
        device_id=int(modbus["device_id"]),
        watchdog_timeout_sec=float(config["safety"]["watchdog_timeout_sec"]),
        watchdog_poll_sec=float(config["safety"]["watchdog_poll_sec"]),
    )
    server = ModbusTcpServer(
        context=virtual_device.device,
        address=(str(modbus["bind_host"]), int(modbus["port"])),
    )
    print(
        "虚拟 PLC 已启动: "
        f"modbus.tcp://{modbus['bind_host']}:{modbus['port']} "
        f"device={modbus['device_id']}"
    )
    async with server:
        watchdog_task = asyncio.create_task(virtual_device.watchdog_loop())
        try:
            await server.serve_forever()
        finally:
            watchdog_task.cancel()
            await asyncio.gather(watchdog_task, return_exceptions=True)


def run_virtual_plc(config: Mapping[str, Any] | None = None) -> None:
    """供 multiprocessing 调用的同步入口。"""

    resolved = to_mapping(load_protocol_config()) if config is None else dict(config)
    asyncio.run(serve_virtual_plc(resolved))


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 APAL 虚拟 Modbus TCP PLC")
    parser.add_argument("--config", type=Path, default=None, help="工业协议配置文件")
    args = parser.parse_args()
    config = to_mapping(load_protocol_config(args.config))
    run_virtual_plc(config)


if __name__ == "__main__":
    main()
