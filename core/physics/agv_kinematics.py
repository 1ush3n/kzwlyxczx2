import numpy as np
from typing import Tuple, List, Dict
from collections import deque

class DelayBuffer:
    """
    网络延迟历史缓冲区，用于模拟主车状态通过网络传输给从车时的延迟。
    """
    def __init__(self, max_len: int = 100):
        self.buffer = deque(maxlen=max_len)
    
    def push(self, timestamp: float, data: np.ndarray):
        """压入带时间戳的数据"""
        self.buffer.append((timestamp, data))
        
    def get_delayed_data(self, current_time: float, delay_sec: float) -> np.ndarray:
        """获取 delay_sec 之前的数据。如果没有足够旧的数据，则返回最旧的一条。"""
        if not self.buffer:
            return None
            
        target_time = current_time - delay_sec
        # 从最新往最老找，找到第一个时间 <= target_time 的数据
        for i in range(len(self.buffer)-1, -1, -1):
            if self.buffer[i][0] <= target_time:
                return self.buffer[i][1]
        
        # 如果都在 target_time 之后，说明刚开始运行，还没产生足够延迟的数据，返回最老的数据
        return self.buffer[0][1]

class AGVSystemSim:
    """
    AGV 主从协同系统的物理动力学仿真核心。
    采用 Kelvin-Voigt 模型和阻抗控制状态空间方程，使用 RK4 进行数值积分。
    """
    def __init__(self, config: Dict):
        self.dt = config['physics']['dt']
        self.Kw = config['physics']['Kw']
        self.Cw = config['physics']['Cw']
        self.L = config['physics']['L']
        
        # 内部状态时间
        self.t = 0.0
        
        # 物理状态：从车阻抗控制下的局部偏差状态 [e, \dot{e}]^T
        # e = x_s - x_d
        self.z = np.zeros(2, dtype=np.float32)
        
        # 真实物理位置与速度
        self.x_m = 0.0
        self.v_m = 0.0
        self.x_s = 0.0
        self.v_s = 0.0
        
        # 期望指令状态 (网络传过来的主车状态)
        self.x_d = 0.0
        self.v_d = 0.0
        
        # 受力
        self.F_ext = 0.0
        
        # 主车状态网络延迟缓冲
        self.master_buffer = DelayBuffer(max_len=config['physics']['history_buffer_size'])
        
    def reset(self):
        """重置物理环境到初始状态"""
        self.t = 0.0
        self.z = np.zeros(2, dtype=np.float32)
        self.x_m = 0.0
        self.v_m = 0.0
        self.x_s = -self.L # 从车初始在 L 距离处
        self.v_s = 0.0
        self.x_d = self.x_s
        self.v_d = 0.0
        self.F_ext = 0.0
        
        self.master_buffer.buffer.clear()
        self.master_buffer.push(self.t, np.array([self.x_m, self.v_m]))
        
    def _rk4_step(self, z: np.ndarray, Md: float, Bd: float, Kd: float, F: float) -> np.ndarray:
        """
        四阶龙格-库塔积分 (RK4) 推进单步状态方程
        \dot{z} = A z + B F_{ext}
        z = [e, \dot{e}]^T
        """
        def derivative(z_state):
            # A 矩阵:
            # [0, 1]
            # [-Kd/Md, -Bd/Md]
            e, e_dot = z_state[0], z_state[1]
            e_ddot = (-Kd * e - Bd * e_dot + F) / Md
            return np.array([e_dot, e_ddot], dtype=np.float32)

        k1 = derivative(z)
        k2 = derivative(z + 0.5 * self.dt * k1)
        k3 = derivative(z + 0.5 * self.dt * k2)
        k4 = derivative(z + self.dt * k3)
        
        return z + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step(self, master_v_cmd: float, delay_sec: float, Md: float, Bd: float, Kd: float):
        """
        向前仿真一步 (dt)
        :param master_v_cmd: 主车当前周期的理想运动速度指令 (简化主车为匀速/加速运动的理想被控对象)
        :param delay_sec: 当前周期的网络 RTT (秒)
        :param Md, Bd, Kd: 当前生效的阻抗参数
        """
        # 1. 更新主车状态 (简单积分)
        self.v_m = master_v_cmd
        self.x_m += self.v_m * self.dt
        self.master_buffer.push(self.t, np.array([self.x_m, self.v_m]))
        
        # 2. 从车通过网络获取延迟后的主车状态指令 x_d, v_d
        delayed_master_state = self.master_buffer.get_delayed_data(self.t, delay_sec)
        if delayed_master_state is not None:
            delayed_x_m = delayed_master_state[0]
            delayed_v_m = delayed_master_state[1]
            
            # x_d 是从车“期望到达的理想位置”：应该落后主车 L 的距离
            # 注意这是带延迟的期望，当网络卡顿时，期望位置 x_d 将滞后于真正的主车目标
            self.x_d = delayed_x_m - self.L
            self.v_d = delayed_v_m
            
        # 3. 计算由于从车尚未跟随到位，导致机翼形变产生的被动力 F_ext (Kelvin-Voigt模型)
        delta_x = self.x_m - self.x_s - self.L
        delta_v = self.v_m - self.v_s
        self.F_ext = self.Kw * delta_x + self.Cw * delta_v
        
        # 4. 根据当前的外力和设定的阻抗参数，计算从车下一时刻的误差演化 z = [e, \dot{e}]
        # 防御性编程：钳位质量防止除零或发散
        assert Md > 1e-4, f"Virtual mass Md must be > 0, got {Md}"
        
        self.z = self._rk4_step(self.z, Md, Bd, Kd, self.F_ext)
        
        # 5. 反推真实的从车物理位置和速度 (e = x_s - x_d => x_s = e + x_d)
        self.x_s = self.z[0] + self.x_d
        self.v_s = self.z[1] + self.v_d
        
        # 时间推进
        self.t += self.dt

    def get_state(self) -> Tuple[float, float, float]:
        """获取环境反馈给强化学习的状态"""
        e = self.z[0]
        e_dot = self.z[1]
        return e, e_dot, self.F_ext

