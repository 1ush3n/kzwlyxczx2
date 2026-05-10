import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple

class BasePLCInterface(ABC):
    """抽象 PLC 通信接口层"""
    
    @abstractmethod
    def connect(self) -> bool:
        pass
        
    @abstractmethod
    def read_sensors(self) -> Tuple[float, float, float, float, float]:
        """
        读取当前所有传感器和通信状态
        :return: (e_t, e_dot_t, F_ext_t, rtt_delay_t, delta_x_cmd_t)
        """
        pass
        
    @abstractmethod
    def write_impedance(self, Md: float, Bd: float, Kd: float):
        """下发最新的虚拟阻抗参数到从车底层控制器"""
        pass
        
    @abstractmethod
    def inject_tsn_delay(self, actual_delay: float):
        """外部注入 TSN 调度算出的真实延迟"""
        pass
        
    @abstractmethod
    def step_simulation(self, master_v_cmd: float):
        """（仅用于仿真）推进环境时间步"""
        pass


class MockPLC(BasePLCInterface):
    """
    纯本地内存模拟的 PLC 通信层。
    包装了 AGVSystemSim 物理引擎，并可注入观测噪声和网络延迟。
    """
    def __init__(self, sim_engine, config: Dict):
        self.sim = sim_engine
        self.base_rtt = config['comms']['base_rtt']
        self.rtt_noise_std = config['comms']['rtt_noise_std']
        self.current_rtt = self.base_rtt
        
        # 记录上一次下发的阻抗，用于仿真步进
        self.current_M = config['impedance']['M_base']
        self.current_B = config['impedance']['B_base']
        self.current_K = config['impedance']['K_base']
        
        # 记录上一周期的 x_cmd
        self.last_x_cmd = 0.0
        
        self.np_random = np.random.default_rng()
        
        # 延迟模式配置 (用于 Domain Randomization)
        self.delay_mode = config.get('comms', {}).get('delay_mode', 'standard')
        
        # TSN 注入的延迟缓冲
        self.injected_delay = None
        
    def set_rng(self, rng):
        self.np_random = rng
        
    def connect(self) -> bool:
        # Mock 始终连接成功
        self.last_x_cmd = self.sim.x_d
        self.current_rtt = self.base_rtt
        return True
        
    def _update_network_delay(self):
        """模拟网络波动 或 使用注入的延迟"""
        if self.injected_delay is not None:
            self.current_rtt = self.injected_delay
            self.injected_delay = None
            return
            
        if self.delay_mode == 'extreme_pareto':
            # Phase 1: 炼蛊皿极限抗压模式 (Domain Randomization)
            # 使用 Pareto 分布制造极端的长尾延迟突刺
            # Pareto shape parameter alpha (usually between 1 and 3 for heavy tails)
            pareto_sample = self.np_random.pareto(1.5)
            # 缩放至合理的极端毫秒级 (如平均几毫秒，偶尔几百毫秒)
            burst = min(0.5, pareto_sample * 0.01) # 最大截断为 500ms
            self.current_rtt = self.base_rtt + burst
        else:
            # 标准模式
            if self.np_random.random() < 0.05:  # 5%的概率发生网络拥塞尖峰
                burst_delay = self.np_random.exponential(scale=0.1)
                self.current_rtt = self.base_rtt + burst_delay
            else:
                self.current_rtt = self.base_rtt + max(0, self.np_random.normal(0, self.rtt_noise_std))
        
    def read_sensors(self) -> Tuple[float, float, float, float, float]:
        e, e_dot, F_ext = self.sim.get_state()
        
        # 模拟 Delta x_cmd 计算 (带延迟的增量)
        current_x_cmd = self.sim.x_d
        delta_x_cmd = current_x_cmd - self.last_x_cmd
        self.last_x_cmd = current_x_cmd
        
        return float(e), float(e_dot), float(F_ext), float(self.current_rtt), float(delta_x_cmd)
        
    def write_impedance(self, Md: float, Bd: float, Kd: float):
        self.current_M = Md
        self.current_B = Bd
        self.current_K = Kd
        
    def inject_tsn_delay(self, actual_delay: float):
        self.injected_delay = actual_delay
        
    def step_simulation(self, master_v_cmd: float):
        self._update_network_delay()
        self.sim.step(
            master_v_cmd=master_v_cmd,
            delay_sec=self.current_rtt,
            Md=self.current_M,
            Bd=self.current_B,
            Kd=self.current_K
        )


class ModbusTCP_PLC(BasePLCInterface):
    """
    基于 ModbusTCP 的真实 PLC 接口占位。
    由于实验环境限制，当前仍将其挂载在 Mock 上，但保留了独立扩展协议的结构。
    """
    def __init__(self, sim_engine, config: Dict):
        self.ip = "192.168.1.100" # 占位
        self.port = 502
        self.mock_backend = MockPLC(sim_engine, config)
        print(f"[ModbusTCP_PLC] Initialized with endpoint {self.ip}:{self.port}")
        
    def connect(self) -> bool:
        print(f"[ModbusTCP_PLC] Attempting connection to {self.ip}:{self.port} ... OK")
        return self.mock_backend.connect()
        
    def read_sensors(self) -> Tuple[float, float, float, float, float]:
        # 未来这里可以替换为 client.read_holding_registers(...)
        return self.mock_backend.read_sensors()
        
    def write_impedance(self, Md: float, Bd: float, Kd: float):
        # 未来这里可以替换为 client.write_registers(...)
        self.mock_backend.write_impedance(Md, Bd, Kd)
        
    def inject_tsn_delay(self, actual_delay: float):
        self.mock_backend.inject_tsn_delay(actual_delay)
        
    def set_rng(self, rng):
        self.mock_backend.set_rng(rng)
        
    def step_simulation(self, master_v_cmd: float):
        self.mock_backend.step_simulation(master_v_cmd)
