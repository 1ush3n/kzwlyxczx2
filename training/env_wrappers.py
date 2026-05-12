import math
import numpy as np
import torch
import gymnasium as gym
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from env.agv_compliance_env import AGVComplianceEnv


class GNNDelayAGVWrapper(gym.Wrapper):
    """
    Phase 3 Step A 专用环境包装器：
    将 GNN 产生的真实网络延迟注入 AGV 环境，替代随机标准延迟。
    使 AGV 能够针对当前 GNN 策略的延迟分布进行微调。
    """
    def __init__(self, agv_env, tsn_env, gnn_agent, device):
        super().__init__(agv_env)
        self.tsn_env = tsn_env
        self.gnn_agent = gnn_agent
        self.device = device
        self._tsn_obs = None
        self._tsn_curr = None
        self._tsn_mask = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._tsn_obs, self._tsn_curr, self._tsn_mask = self.tsn_env.reset()
        self._inject_gnn_delay()
        return obs, info

    def _inject_gnn_delay(self):
        tsn_obs = self._tsn_obs
        tsn_curr = self._tsn_curr
        tsn_mask = self._tsn_mask
        while True:
            with torch.no_grad():
                h = self.gnn_agent.encode(tsn_obs.to(self.device))
                target_node = self.tsn_env.target_node
                logits = self.gnn_agent.get_routing_logits(
                    h, tsn_curr, target_node, tsn_mask.to(self.device))
                next_node = torch.argmax(logits).item()
                edge_idx = self.tsn_env._get_edge_idx(tsn_curr, next_node)
                edge_attr = self.tsn_env.topo.edge_attr[edge_idx].to(self.device)
                out = self.gnn_agent.scheduling_head(
                    torch.cat([h[tsn_curr], h[next_node], edge_attr]))
                t_offset = torch.sigmoid(out[0]).item()

            agv_pos = self.env.sim_engine.x_s
            next_obs, next_curr, next_mask, _, done, _, info = self.tsn_env.step(
                next_node, t_offset, agv_x=agv_pos)

            if done:
                status = info.get('status', 'success')
                if status == 'success':
                    rtt_sec = info['total_delay'] / 1e6
                    self.env.plc.inject_tsn_delay(rtt_sec)
                self._tsn_obs, self._tsn_curr, self._tsn_mask = self.tsn_env.reset()
                break
            else:
                tsn_obs, tsn_curr, tsn_mask = next_obs, next_curr, next_mask

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated and not truncated:
            self._inject_gnn_delay()
        return obs, reward, terminated, truncated, info


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
        tsn_obs = self.tsn_env.reset()
        # S2: 训练时也注入背景流量，让 GNN 学会在拥塞下路由
        self._inject_bg_traffic()
        return tsn_obs
    
    def _inject_bg_traffic(self):
        bg_prob = np.random.uniform(0.1, 0.4)
        for i in range(self.tsn_env.topo.num_edges):
            if np.random.rand() < bg_prob:
                self.tsn_env.gantt.check_and_add_slot(
                    i, np.random.uniform(0, 300), np.random.uniform(50, 400))
        
    def step(self, next_node: int, t_offset: float):
        # 获取物理层坐标同步
        agv_pos = self.agv_env.sim_engine.x_s
        
        obs, current_node, mask, reward, terminated, truncated, info = self.tsn_env.step(
            next_node, t_offset, agv_x=agv_pos
        )
        
        # 初始化默认值，防止碰撞时 info 中缺少这些键
        info.setdefault('peak_stress', float('nan'))
        info.setdefault('stress_penalty', 0.0)
        info.setdefault('stress_weight', 0)
        
        if 'total_delay' in info and info.get('status', 'success') == 'success':
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
                    
            # 4. 将物理层峰值应力转化为 GNN 的惩罚 (自适应退火权重机制)
            f_max = self.agv_env.config['rl']['F_max']
            normalized_stress = peak_stress / f_max
            
            # 自适应退火：应力越低，penalty权重越小，让GNN更专注于网络指标优化
            if normalized_stress < 0.04:   # peak_stress < 200N，安全区间
                stress_weight = 5.0       # 提升基准权重，让GNN始终感知物理反馈
            elif normalized_stress < 0.10: # 200N <= peak_stress < 500N，警戒区间
                stress_weight = 12.0
            else:                          # peak_stress >= 500N，危险区间
                stress_weight = 50.0       # 极致惩罚，让 GNN 高度厌恶高风险路径
                
            stress_penalty = - (normalized_stress ** 2) * stress_weight
            
            # 追加到当前 GNN 步的 reward 中
            reward += stress_penalty
            info['peak_stress'] = peak_stress
            info['stress_penalty'] = stress_penalty
            info['stress_weight'] = stress_weight
            
        return obs, current_node, mask, reward, terminated, truncated, info
