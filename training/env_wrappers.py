import math
import numpy as np
import torch
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from env.agv_compliance_env import AGVComplianceEnv

class NestedGNNEnvWrapper:
    """
    用于 Phase 2 悲观拓扑预训练的环境包装器。
    将底层的刚性 AGV 环境嵌套在 GNN 调度环境中。
    """
    def __init__(self, tsn_env: TSN_GNN_Env, agv_env: AGVComplianceEnv):
        self.tsn_env = tsn_env
        self.agv_env = agv_env
        self.dt = agv_env.config['physics']['dt']
        
        # 定义刚性小车的固定动作 (没有柔顺能力)
        self.rigid_action = np.array([0.0, 0.0, 0.0])
        
    def reset(self):
        self.agv_env.reset()
        # 物理层的初始位置通常为0，我们用这同步拓扑中的 agv_x
        # 但是这里为了简单，我们复用 agv 的 step_count 作为虚拟位置，或者直接不管它
        return self.tsn_env.reset()
        
    def step(self, next_node: int, t_offset: float):
        # 1. 跑 GNN 的一次路由或者一跳
        # 这里为了计算距离，我们需要给 agv_x。
        # 物理层 AGV 的坐标其实是 agv_env.slave_car.x，我们可以获取它。
        agv_x = getattr(self.agv_env, 'slave_car', None)
        agv_pos = agv_x.x if agv_x is not None else 0.0
        
        obs, current_node, mask, reward, terminated, truncated, info = self.tsn_env.step(
            next_node, t_offset, agv_x=agv_pos
        )
        
        # 2. 如果路由完成（无论是成功还是失败，只要产生了总延迟）
        if 'total_delay' in info and info['status'] == 'success':
            rtt_sec = info['total_delay'] / 1e6
            
            # 计算失联盲区的物理步数
            num_blind_steps = math.ceil(rtt_sec / self.dt)
            # 防止步数过长导致无限循环（比如网络极度拥堵）
            num_blind_steps = min(num_blind_steps, 50) 
            
            peak_stress = 0.0
            
            # 3. 物理环境 Rollout 逼近真实应力破坏
            self.agv_env.plc.inject_tsn_delay(rtt_sec)
            
            for _ in range(num_blind_steps):
                agv_obs, agv_reward, agv_term, agv_trunc, agv_info = self.agv_env.step(self.rigid_action)
                
                # agv_obs 是一个堆叠的帧，取最新一帧
                # 状态: [e, e_dot, F_ext, tau, delta_x_cmd]
                current_stress = abs(agv_obs[-1][2]) 
                peak_stress = max(peak_stress, current_stress)
                
                if agv_term or agv_trunc:
                    break
                    
            # 4. 将物理层峰值应力转化为 GNN 的惩罚 (反向传播机制)
            f_max = self.agv_env.config['rewards']['F_max']
            stress_penalty = - (peak_stress / f_max) ** 2 * 10.0 # 放大惩罚因子
            
            # 追加到当前 GNN 步的 reward 中
            reward += stress_penalty
            info['peak_stress'] = peak_stress
            info['stress_penalty'] = stress_penalty
            
        return obs, current_node, mask, reward, terminated, truncated, info
