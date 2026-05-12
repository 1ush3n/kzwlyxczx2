import os
import yaml
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics.agv_kinematics import AGVSystemSim
from core.comms.plc_interface import MockPLC


class AGVComplianceEnv(gym.Env):
    """
    AGV 柔顺控制强化学习环境 (Gymnasium 标准接口)
    遵循三层解耦架构: RL环境层 → 通信抽象层 → 物理仿真引擎层
    """
    metadata = {"render_modes": ["human", None]}

    def __init__(self, config_path=None, render_mode=None):
        super().__init__()

        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'agv_env_config.yaml')

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.render_mode = render_mode

        # 物理引擎
        self.sim_engine = AGVSystemSim(self.config)

        # 通信接口
        protocol = self.config.get('comms', {}).get('protocol', 'mock')
        if protocol == 'modbus_tcp':
            from core.comms.plc_interface import ModbusTCP_PLC
            self.plc = ModbusTCP_PLC(self.sim_engine, self.config)
        else:
            self.plc = MockPLC(self.sim_engine, self.config)

        # 观测空间: 帧堆叠 [k=4, 5] (e, e_dot, F_ext, tau, delta_x_cmd)
        self.k = self.config['rl']['frame_stack_k']
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.k, 5),
            dtype=np.float32
        )

        # 动作空间: [delta_M, delta_B, delta_K] ∈ [-1, 1]^3
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # 内部状态
        self.obs_buffer = deque(maxlen=self.k)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 500

        # 奖励参数
        self.alpha = self.config['rl']['alpha']
        self.beta = self.config['rl']['beta']
        self.omega_3 = self.config['rl']['omega_3']

        # 阻抗参数映射基准
        self.M_base = self.config['impedance']['M_base']
        self.B_base = self.config['impedance']['B_base']
        self.K_base = self.config['impedance']['K_base']
        self.M_delta = self.config['impedance']['M_delta_max']
        self.B_delta = self.config['impedance']['B_delta_max']
        self.K_delta = self.config['impedance']['K_delta_max']

        # 归一化参数
        self.F_max = self.config['rl']['F_max']
        self.e_max = self.config['rl']['e_max']

    def _get_obs(self):
        return np.array(self.obs_buffer, dtype=np.float32)

    def _get_info(self):
        e, e_dot, F_ext = self.sim_engine.get_state()
        return {
            'error': float(e),
            'e_dot': float(e_dot),
            'F_ext': float(F_ext),
            'Md': float(self.plc.current_M),
            'Bd': float(self.plc.current_B),
            'Kd': float(self.plc.current_K),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim_engine.reset()
        self.plc.set_rng(self.np_random)
        self.plc.connect()
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.step_count = 0

        e, e_dot, F_ext = self.sim_engine.get_state()
        rtt, delta_x = self.plc.read_sensors()[3], self.plc.read_sensors()[4]
        initial_obs_row = np.array([e, e_dot, F_ext, rtt, delta_x], dtype=np.float32)

        self.obs_buffer.clear()
        for _ in range(self.k):
            self.obs_buffer.append(initial_obs_row.copy())

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        delta_M, delta_B, delta_K = action

        # 动作映射: 归一化[-1,1] → 实际阻抗参数
        Md = max(self.M_base + delta_M * self.M_delta, 1e-4)
        Bd = max(self.B_base + delta_B * self.B_delta, 1e-4)
        Kd = max(self.K_base + delta_K * self.K_delta, 1e-4)

        # 下发阻抗参数至PLC
        self.plc.write_impedance(Md, Bd, Kd)

        # 推进物理仿真一步 (主车匀速1.5 m/s)
        master_v_cmd = 1.5
        self.plc.step_simulation(master_v_cmd)
        self.step_count += 1

        # 读取当前状态
        e, e_dot, F_ext = self.sim_engine.get_state()
        sensors = self.plc.read_sensors()
        rtt = sensors[3]
        delta_x_cmd = sensors[4]

        # 更新观测缓冲区
        new_obs_row = np.array([e, e_dot, F_ext, rtt, delta_x_cmd], dtype=np.float32)
        self.obs_buffer.append(new_obs_row)

        # 计算奖励
        reward = (
            -self.alpha * (F_ext / self.F_max) ** 2
            - self.beta * (e / self.e_max) ** 2
            - self.omega_3 * np.sum((action - self.prev_action) ** 2)
        )

        self.prev_action = action.copy()

        # 终止/截断条件
        terminated = abs(F_ext) > self.F_max * 3 or abs(e) > self.e_max * 10
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    @property
    def unwrapped(self):
        return self
