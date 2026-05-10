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
    def __init__(self, tsn_env: TSN_GNN_Env, agv_env: AGVComplianceEnv, agv_agent=None):
        self.tsn_env = tsn_env
        self.agv_env = agv_env
        self.dt = agv_env.config['physics']['dt']
        self.agv_agent = agv_agent # 可以是 SB3 PPO 对象
        
        # 定义刚性小车的固定动作 (没有柔顺能力)
        self.rigid_action = np.array([0.0, 0.0, 0.0])
        
    def reset(self, seed=None):
        obs, info = self.agv_env.reset(seed=seed)
        return self.tsn_env.reset()
        
    def step(self, next_node: int, t_offset: float):
        # 获取物理层坐标同步
        agv_pos = self.agv_env.sim_engine.x_s
        
        obs, current_node, mask, reward, terminated, truncated, info = self.tsn_env.step(
            next_node, t_offset, agv_x=agv_pos
        )
        
        if 'total_delay' in info and info['status'] == 'success':
            rtt_sec = info['total_delay'] / 1e6
            num_blind_steps = math.ceil(rtt_sec / self.dt)
            num_blind_steps = min(num_blind_steps, 50) 
            
            peak_stress = 0.0
            self.agv_env.plc.inject_tsn_delay(rtt_sec)
            
            # 使用最新一帧作为观测开始 Rollout
            # 注意：这里需要维持 agv_env 的内部状态
            for _ in range(num_blind_steps):
                if self.agv_agent is not None:
                    # 使用当前训练好的 AGV 策略进行动作选择
                    # SB3 expect shape [B, obs_dim], we have history frame
                    agv_obs = self.agv_env.obs_buffer.copy()
                    action, _ = self.agv_agent.predict(agv_obs, deterministic=True)
                else:
                    action = self.rigid_action
                    
                agv_obs, agv_reward, agv_term, agv_trunc, agv_info = self.agv_env.step(action)
                
                current_stress = abs(agv_obs[-1][2]) 
                peak_stress = max(peak_stress, current_stress)
                
                if agv_term or agv_trunc:
                    break
                    
            # 4. 将物理层峰值应力转化为 GNN 的惩罚 (反向传播机制)
            f_max = self.agv_env.config['rl']['F_max']
            stress_penalty = - (peak_stress / f_max) ** 2 * 10.0 # 放大惩罚因子
            
            # 追加到当前 GNN 步的 reward 中
            reward += stress_penalty
            info['peak_stress'] = peak_stress
            info['stress_penalty'] = stress_penalty
            
        return obs, current_node, mask, reward, terminated, truncated, info
