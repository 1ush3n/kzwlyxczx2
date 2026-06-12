from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.comms.config import PROJECT_ROOT, load_protocol_config, load_yaml_mapping
from core.comms.models import PLCCommunicationError, PLCSnapshot
from core.comms.plc_interface import BasePLCInterface, MockPLC, ModbusTCPPLC
from core.physics.agv_kinematics import AGVSystemSim


class AGVComplianceEnv(gym.Env):
    """AGV 柔顺控制环境：强化学习层仅依赖 PLC 通信抽象。"""

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        config_path: str | Path | None = None,
        render_mode: str | None = None,
        proposal_config: dict[str, Any] | None = None,
        plc: BasePLCInterface | None = None,
    ):
        super().__init__()

        resolved_config_path = (
            Path(config_path).resolve()
            if config_path is not None
            else PROJECT_ROOT / "config" / "agv_env_config.yaml"
        )
        self.config = load_yaml_mapping(resolved_config_path)
        self.render_mode = render_mode
        self.proposal_config = proposal_config or {}

        self.plc = plc or self._build_plc()
        # 保留该属性供既有实验脚本注入物理扰动；Modbus 模式下为 None。
        self.sim_engine = getattr(self.plc, "sim", None)
        self.current_snapshot = PLCSnapshot(
            md=float(self.config["impedance"]["M_base"]),
            bd=float(self.config["impedance"]["B_base"]),
            kd=float(self.config["impedance"]["K_base"]),
            connected=False,
        )
        self._communication_failed = False

        self.k = int(self.config["rl"]["frame_stack_k"])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.k, 5),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        self.obs_buffer: deque[np.ndarray] = deque(maxlen=self.k)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 500

        self.alpha = float(self.config["rl"]["alpha"])
        self.beta = float(self.config["rl"]["beta"])
        self.omega_3 = float(self.config["rl"]["omega_3"])

        self.M_base = float(self.config["impedance"]["M_base"])
        self.B_base = float(self.config["impedance"]["B_base"])
        self.K_base = float(self.config["impedance"]["K_base"])
        self.M_delta = float(self.config["impedance"]["M_delta_max"])
        self.B_delta = float(self.config["impedance"]["B_delta_max"])
        self.K_delta = float(self.config["impedance"]["K_delta_max"])

        self.F_max = float(self.config["rl"]["F_max"])
        self.e_max = float(self.config["rl"]["e_max"])

    def _build_plc(self) -> BasePLCInterface:
        protocol = str(self.config.get("comms", {}).get("protocol", "mock"))
        if protocol == "modbus_tcp":
            protocol_path = Path(
                str(
                    self.config["comms"].get(
                        "protocol_config_path",
                        "config/industrial_protocols.yaml",
                    )
                )
            )
            if not protocol_path.is_absolute():
                protocol_path = PROJECT_ROOT / protocol_path
            protocol_config = load_protocol_config(protocol_path)
            return ModbusTCPPLC(self.config, protocol_config)
        if protocol != "mock":
            raise ValueError(f"不支持的 PLC 协议: {protocol}")
        return MockPLC(AGVSystemSim(self.config), self.config)

    def _get_obs(self) -> np.ndarray:
        # 输出形状: [k, 5]
        return np.asarray(self.obs_buffer, dtype=np.float32)

    def _get_info(self, snapshot: PLCSnapshot) -> dict[str, Any]:
        return {
            "error": float(snapshot.error),
            "e_dot": float(snapshot.error_rate),
            "F_ext": float(snapshot.external_force),
            "Md": float(snapshot.md),
            "Bd": float(snapshot.bd),
            "Kd": float(snapshot.kd),
            "plc_connected": bool(snapshot.connected),
            "plc_status": int(snapshot.status),
            "plc_alarm": int(snapshot.alarm),
            "control_mode": int(snapshot.control_mode),
        }

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self._communication_failed = False

        try:
            self.current_snapshot = self.plc.reset(seed)
        except PLCCommunicationError:
            self.current_snapshot = PLCSnapshot.disconnected(self.current_snapshot)
            self._communication_failed = True

        initial_obs_row = self._snapshot_to_observation(self.current_snapshot)
        self.obs_buffer.clear()
        for _ in range(self.k):
            self.obs_buffer.append(initial_obs_row.copy())

        return self._get_obs(), self._get_info(self.current_snapshot)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (3,), f"动作形状必须为 [3]，实际为 {action.shape}"

        if self._communication_failed:
            return self._communication_failure_transition()

        delta_M, delta_B, delta_K = self._smooth_action(action)

        md = max(self.M_base + delta_M * self.M_delta, 1e-4)
        bd = max(self.B_base + delta_B * self.B_delta, 1e-4)
        kd = max(self.K_base + delta_K * self.K_delta, 1e-4)

        try:
            self.plc.write_impedance(md, bd, kd)
            self.current_snapshot = self.plc.step_simulation(master_v_cmd=1.5)
        except PLCCommunicationError:
            self.current_snapshot = PLCSnapshot.disconnected(self.current_snapshot)
            self._communication_failed = True
            return self._communication_failure_transition()

        self.step_count += 1
        self.obs_buffer.append(self._snapshot_to_observation(self.current_snapshot))

        effective_action = np.array(
            [delta_M, delta_B, delta_K],
            dtype=np.float32,
        )
        reward = self._calculate_reward(effective_action)
        self.prev_action = effective_action

        effective_f_max = float(
            self.proposal_config.get("F_max_override", self.F_max)
        )
        termination_force = float(
            self.proposal_config.get("term_F_max", effective_f_max * 3)
        )
        terminated = (
            abs(self.current_snapshot.external_force) > termination_force
            or abs(self.current_snapshot.error) > self.e_max * 10
        )
        truncated = self.step_count >= self.max_steps

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            self._get_info(self.current_snapshot),
        )

    def close(self) -> None:
        self.plc.close()

    def _smooth_action(self, action: np.ndarray) -> tuple[float, float, float]:
        delta_M, delta_B, delta_K = (float(value) for value in action)
        config = self.proposal_config

        if "momentum_max_delta" in config:
            max_delta = float(config["momentum_max_delta"])
            delta_M = float(
                np.clip(
                    delta_M,
                    self.prev_action[0] - max_delta,
                    self.prev_action[0] + max_delta,
                )
            )
            delta_B = float(
                np.clip(
                    delta_B,
                    self.prev_action[1] - max_delta,
                    self.prev_action[1] + max_delta,
                )
            )
            delta_K = float(
                np.clip(
                    delta_K,
                    self.prev_action[2] - max_delta,
                    self.prev_action[2] + max_delta,
                )
            )

        if "action_smooth_weight" in config:
            weight = float(config["action_smooth_weight"])
            delta_M = weight * delta_M + (1 - weight) * float(self.prev_action[0])
            delta_B = weight * delta_B + (1 - weight) * float(self.prev_action[1])
            delta_K = weight * delta_K + (1 - weight) * float(self.prev_action[2])

        return delta_M, delta_B, delta_K

    def _calculate_reward(self, action: np.ndarray) -> float:
        snapshot = self.current_snapshot
        effective_f_max = float(
            self.proposal_config.get("F_max_override", self.F_max)
        )
        reward = (
            -self.alpha * (snapshot.external_force / effective_f_max) ** 2
            - self.beta * (snapshot.error / self.e_max) ** 2
            - self.omega_3 * float(np.sum((action - self.prev_action) ** 2))
        )

        if "risk_threshold" in self.proposal_config:
            threshold = float(self.proposal_config["risk_threshold"])
            boost = float(self.proposal_config.get("risk_boost", 3.0))
            absolute_force = abs(snapshot.external_force)
            if absolute_force > threshold:
                denominator = max(effective_f_max - threshold, 1e-6)
                risk_factor = 1.0 + boost * (absolute_force - threshold) / denominator
                reward *= risk_factor
        return float(reward)

    def _communication_failure_transition(
        self,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.obs_buffer:
            row = self._snapshot_to_observation(self.current_snapshot)
            for _ in range(self.k):
                self.obs_buffer.append(row.copy())
        else:
            self.obs_buffer.append(self.obs_buffer[-1].copy())
        info = self._get_info(self.current_snapshot)
        info["communication_failure"] = True
        return self._get_obs(), -100.0, True, False, info

    @staticmethod
    def _snapshot_to_observation(snapshot: PLCSnapshot) -> np.ndarray:
        # PLC快照 -> 单帧观测形状: [5]
        return np.array(snapshot.sensor_tuple(), dtype=np.float32)

    @property
    def unwrapped(self) -> AGVComplianceEnv:
        return self
