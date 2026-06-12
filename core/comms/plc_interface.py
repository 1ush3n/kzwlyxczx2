from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Mapping, Tuple

import numpy as np

from core.comms.models import (
    ControlMode,
    PLCCommand,
    PLCCommunicationError,
    PLCSnapshot,
    PLCStatus,
)
from core.comms.register_map import (
    INPUT_REGISTER_COUNT,
    HoldingCommandFrame,
    decode_input_snapshot,
    encode_holding_frame,
)


class BasePLCInterface(ABC):
    """PLC 通信抽象，环境层不得依赖具体协议实现。"""

    @abstractmethod
    def connect(self) -> bool:
        """建立通信连接。"""

    @abstractmethod
    def close(self) -> None:
        """关闭通信连接。"""

    @abstractmethod
    def reset(self, seed: int | None = None) -> PLCSnapshot:
        """复位 PLC 与物理对象。"""

    @abstractmethod
    def read_snapshot(self) -> PLCSnapshot:
        """原子读取完整状态快照。"""

    @abstractmethod
    def write_impedance(self, md: float, bd: float, kd: float) -> None:
        """缓存自动控制器下发的阻抗参数。"""

    @abstractmethod
    def inject_tsn_delay(self, delay_sec: float) -> None:
        """注入 TSN 端到端延迟。"""

    @abstractmethod
    def step_simulation(self, master_v_cmd: float) -> PLCSnapshot:
        """推进一个仿真控制周期。"""

    @abstractmethod
    def safe_stop(self) -> PLCSnapshot:
        """锁存安全停车状态。"""

    @abstractmethod
    def reset_safety(self) -> PLCSnapshot:
        """显式解除安全停车锁存。"""

    def read_sensors(self) -> Tuple[float, float, float, float, float]:
        """兼容旧版调用方式。"""

        return self.read_snapshot().sensor_tuple()

    @property
    def current_M(self) -> float:
        return self.read_snapshot().md

    @property
    def current_B(self) -> float:
        return self.read_snapshot().bd

    @property
    def current_K(self) -> float:
        return self.read_snapshot().kd


class MockPLC(BasePLCInterface):
    """内存 PLC，用于训练和无网络开销的回归测试。"""

    def __init__(self, sim_engine: Any, config: Mapping[str, Any]):
        self.sim = sim_engine
        self.config = config
        self.base_rtt = float(config["comms"]["base_rtt"])
        self.rtt_noise_std = float(config["comms"]["rtt_noise_std"])
        self.delay_mode = str(config.get("comms", {}).get("delay_mode", "standard"))
        self._base_md = float(config["impedance"]["M_base"])
        self._base_bd = float(config["impedance"]["B_base"])
        self._base_kd = float(config["impedance"]["K_base"])
        self._current_md = self._base_md
        self._current_bd = self._base_bd
        self._current_kd = self._base_kd
        self._current_rtt = self.base_rtt
        self._injected_delay: float | None = None
        self._last_x_cmd = 0.0
        self._step_count = 0
        self._connected = False
        self.np_random = np.random.default_rng()
        self._snapshot = PLCSnapshot(
            md=self._current_md,
            bd=self._current_bd,
            kd=self._current_kd,
            connected=False,
        )
        self._safety_latched = False

    def set_rng(self, rng: np.random.Generator) -> None:
        """兼容既有环境的随机数注入。"""

        self.np_random = rng

    def connect(self) -> bool:
        self._connected = True
        return True

    def close(self) -> None:
        self._connected = False

    def reset(self, seed: int | None = None) -> PLCSnapshot:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.connect()
        self.sim.reset()
        self._current_md = self._base_md
        self._current_bd = self._base_bd
        self._current_kd = self._base_kd
        self._current_rtt = self.base_rtt
        self._injected_delay = None
        self._last_x_cmd = float(self.sim.x_d)
        self._step_count = 0
        self._safety_latched = False
        self._snapshot = self._build_snapshot(delta_x_cmd=0.0)
        return self._snapshot

    def read_snapshot(self) -> PLCSnapshot:
        return self._snapshot

    def write_impedance(self, md: float, bd: float, kd: float) -> None:
        self._current_md = float(md)
        self._current_bd = float(bd)
        self._current_kd = float(kd)
        self._snapshot = replace(
            self._snapshot,
            md=self._current_md,
            bd=self._current_bd,
            kd=self._current_kd,
        )

    def inject_tsn_delay(self, delay_sec: float) -> None:
        self._injected_delay = max(0.0, float(delay_sec))

    def step_simulation(self, master_v_cmd: float) -> PLCSnapshot:
        if self._safety_latched:
            master_v_cmd = 0.0
        self._update_network_delay()
        self.sim.step(
            master_v_cmd=float(master_v_cmd),
            delay_sec=self._current_rtt,
            Md=self._current_md,
            Bd=self._current_bd,
            Kd=self._current_kd,
        )
        self._step_count += 1
        current_x_cmd = float(self.sim.x_d)
        delta_x_cmd = current_x_cmd - self._last_x_cmd
        self._last_x_cmd = current_x_cmd
        self._snapshot = self._build_snapshot(delta_x_cmd=delta_x_cmd)
        return self._snapshot

    def safe_stop(self) -> PLCSnapshot:
        self._safety_latched = True
        self.step_simulation(0.0)
        self._snapshot = replace(
            self._snapshot,
            control_mode=ControlMode.SAFETY_STOP,
        )
        return self._snapshot

    def reset_safety(self) -> PLCSnapshot:
        self._safety_latched = False
        self._snapshot = replace(
            self._snapshot,
            control_mode=ControlMode.AUTOMATIC,
        )
        return self._snapshot

    @property
    def current_M(self) -> float:
        return self._current_md

    @property
    def current_B(self) -> float:
        return self._current_bd

    @property
    def current_K(self) -> float:
        return self._current_kd

    def _update_network_delay(self) -> None:
        if self._injected_delay is not None:
            self._current_rtt = self._injected_delay
            self._injected_delay = None
            return

        if self.delay_mode == "extreme_pareto":
            pareto_sample = self.np_random.pareto(1.5)
            burst = min(0.5, pareto_sample * 0.01)
            self._current_rtt = self.base_rtt + burst
            return

        if self.np_random.random() < 0.05:
            self._current_rtt = self.base_rtt + self.np_random.exponential(scale=0.1)
        else:
            noise = max(0.0, self.np_random.normal(0, self.rtt_noise_std))
            self._current_rtt = self.base_rtt + noise

    def _build_snapshot(self, delta_x_cmd: float) -> PLCSnapshot:
        error, error_rate, external_force = self.sim.get_state()
        return PLCSnapshot(
            error=float(error),
            error_rate=float(error_rate),
            external_force=float(external_force),
            rtt_sec=float(self._current_rtt),
            delta_x_cmd=float(delta_x_cmd),
            master_position=float(self.sim.x_m),
            master_velocity=float(self.sim.v_m),
            slave_position=float(self.sim.x_s),
            slave_velocity=float(self.sim.v_s),
            simulation_time=float(self.sim.t),
            md=self._current_md,
            bd=self._current_bd,
            kd=self._current_kd,
            status=PLCStatus.OK,
            control_mode=ControlMode.AUTOMATIC,
            step_count=self._step_count,
            connected=self._connected,
        )


class ModbusTCPPLC(BasePLCInterface):
    """基于真实 TCP 报文的 Modbus PLC 客户端。"""

    def __init__(
        self,
        agv_config: Mapping[str, Any],
        protocol_config: Mapping[str, Any],
    ):
        from pymodbus.client import ModbusTcpClient

        cfg = protocol_config["modbus"]
        self._device_id = int(cfg["device_id"])
        self._timeout = float(cfg["timeout_sec"])
        self._retries = int(cfg["retries"])
        self._poll_interval = float(cfg["poll_interval_sec"])
        self._client = ModbusTcpClient(
            host=str(cfg["host"]),
            port=int(cfg["port"]),
            timeout=self._timeout,
            retries=self._retries,
        )
        self._lock = threading.RLock()
        self._connected = False
        self._pending_md = float(agv_config["impedance"]["M_base"])
        self._pending_bd = float(agv_config["impedance"]["B_base"])
        self._pending_kd = float(agv_config["impedance"]["K_base"])
        self._pending_delay = 0.0
        self._has_pending_delay = False
        self._last_snapshot = PLCSnapshot(
            md=self._pending_md,
            bd=self._pending_bd,
            kd=self._pending_kd,
            connected=False,
        )
        self._sequence = int(time.time_ns() & 0xFFFFFFFF)

    def connect(self) -> bool:
        with self._lock:
            self._connected = bool(self._client.connect())
            return self._connected

    def close(self) -> None:
        with self._lock:
            self._client.close()
            self._connected = False

    def reset(self, seed: int | None = None) -> PLCSnapshot:
        return self._execute_command(
            command=PLCCommand.RESET,
            master_velocity=0.0,
            seed=seed,
        )

    def read_snapshot(self) -> PLCSnapshot:
        with self._lock:
            try:
                self._ensure_connected()
                response = self._client.read_input_registers(
                    address=0,
                    count=INPUT_REGISTER_COUNT,
                    device_id=self._device_id,
                )
                self._check_response(response, "读取 Input Register")
                snapshot = decode_input_snapshot(response.registers)
                self._last_snapshot = snapshot
                return snapshot
            except Exception as exc:
                self._last_snapshot = PLCSnapshot.disconnected(self._last_snapshot)
                self._connected = False
                raise PLCCommunicationError(f"Modbus 状态读取失败: {exc}") from exc

    def write_impedance(self, md: float, bd: float, kd: float) -> None:
        self._pending_md = float(md)
        self._pending_bd = float(bd)
        self._pending_kd = float(kd)

    def inject_tsn_delay(self, delay_sec: float) -> None:
        self._pending_delay = max(0.0, float(delay_sec))
        self._has_pending_delay = True

    def step_simulation(self, master_v_cmd: float) -> PLCSnapshot:
        snapshot = self._execute_command(
            command=PLCCommand.STEP,
            master_velocity=float(master_v_cmd),
        )
        self._pending_delay = 0.0
        self._has_pending_delay = False
        return snapshot

    def apply_manual_impedance(self, md: float, bd: float, kd: float) -> PLCSnapshot:
        self._pending_md = float(md)
        self._pending_bd = float(bd)
        self._pending_kd = float(kd)
        return self._execute_command(
            command=PLCCommand.APPLY_MANUAL_IMPEDANCE,
            master_velocity=0.0,
            mode=ControlMode.MANUAL,
        )

    def release_manual_control(self) -> PLCSnapshot:
        return self._execute_command(
            command=PLCCommand.RELEASE_MANUAL_CONTROL,
            master_velocity=0.0,
            mode=ControlMode.AUTOMATIC,
        )

    def safe_stop(self) -> PLCSnapshot:
        return self._execute_command(
            command=PLCCommand.SAFE_STOP,
            master_velocity=0.0,
            mode=ControlMode.SAFETY_STOP,
        )

    def reset_safety(self) -> PLCSnapshot:
        return self._execute_command(
            command=PLCCommand.RESET_SAFETY,
            master_velocity=0.0,
            mode=ControlMode.AUTOMATIC,
        )

    @property
    def current_M(self) -> float:
        return self._last_snapshot.md

    @property
    def current_B(self) -> float:
        return self._last_snapshot.bd

    @property
    def current_K(self) -> float:
        return self._last_snapshot.kd

    def _execute_command(
        self,
        command: PLCCommand,
        master_velocity: float,
        mode: ControlMode = ControlMode.AUTOMATIC,
        seed: int | None = None,
    ) -> PLCSnapshot:
        with self._lock:
            last_error: Exception | None = None
            for _ in range(self._retries + 1):
                try:
                    self._ensure_connected()
                    sequence = self._next_sequence()
                    frame = HoldingCommandFrame(
                        md=self._pending_md,
                        bd=self._pending_bd,
                        kd=self._pending_kd,
                        master_velocity=master_velocity,
                        injected_rtt_sec=self._pending_delay,
                        control_mode=mode,
                        command=command,
                        sequence=sequence,
                        seed=seed,
                        flags=1 if self._has_pending_delay else 0,
                    )
                    response = self._client.write_registers(
                        address=0,
                        values=encode_holding_frame(frame),
                        device_id=self._device_id,
                    )
                    self._check_response(response, f"写入命令 {command.name}")
                    snapshot = self._wait_for_ack(sequence)
                    if snapshot.status is not PLCStatus.OK:
                        raise PLCCommunicationError(
                            f"PLC 拒绝命令 {command.name}: "
                            f"status={snapshot.status.name}, alarm={int(snapshot.alarm)}"
                        )
                    self._last_snapshot = snapshot
                    return snapshot
                except Exception as exc:
                    last_error = exc
                    self._client.close()
                    self._connected = False
            self._last_snapshot = PLCSnapshot.disconnected(self._last_snapshot)
            raise PLCCommunicationError(
                f"Modbus 命令 {command.name} 执行失败: {last_error}"
            ) from last_error

    def _wait_for_ack(self, sequence: int) -> PLCSnapshot:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            response = self._client.read_input_registers(
                address=0,
                count=INPUT_REGISTER_COUNT,
                device_id=self._device_id,
            )
            self._check_response(response, "等待命令应答")
            snapshot = decode_input_snapshot(response.registers)
            if snapshot.ack_sequence == sequence:
                return snapshot
            time.sleep(self._poll_interval)
        raise TimeoutError(f"等待 PLC 应答超时，sequence={sequence}")

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        if not self.connect():
            raise ConnectionError("无法连接 Modbus TCP 服务")

    @staticmethod
    def _check_response(response: Any, operation: str) -> None:
        if response is None or response.isError():
            raise PLCCommunicationError(f"{operation} 返回 Modbus 异常: {response}")

    def _next_sequence(self) -> int:
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        if self._sequence == 0:
            self._sequence = 1
        return self._sequence


# 保留旧名称，避免既有配置和外部代码立即失效。
ModbusTCP_PLC = ModbusTCPPLC
