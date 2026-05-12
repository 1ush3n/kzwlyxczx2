import os
import yaml
import torch
import numpy as np
from typing import Tuple, Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tsn_net.topology import TSNTopology
from tsn_net.gantt_chart import GanttChartManager

class TSN_GNN_Env:
    """
    基于 PyTorch Geometric 的 TSN 调度与路由环境。
    采用了脱离标准 Gym.Box 的 Custom Tensor 接口，以便支持 Autoregressive Action 和 Dynamic Action Mask。
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, 'config', 'tsn_env_config.yaml')
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.topo = TSNTopology(
            num_nodes=self.config['topology']['num_nodes'],
            num_ap=self.config['topology']['num_ap'],
            agv_idx=self.config['topology']['agv_node_idx']
        )
        
        self.gantt = GanttChartManager(
            num_edges=self.topo.num_edges,
            cycle_time=self.config['network']['cycle_time']
        )
        
        # 奖励配置
        self.r_cfg = self.config['rewards']
        
        self.current_node = 0
        self.target_node = self.topo.agv_idx
        self.visited_nodes = set()
        
        self.total_delay = 0.0
        self.step_count = 0
        
        # 当前流的信息 (固定为大小 1500 Bytes 的控制包)
        self.flow_size = 1500 * 8 # Bits
        
    def _update_edge_occupancy(self):
        """同步甘特图状态到图拓扑的边特征中"""
        for i in range(self.topo.num_edges):
            slots = self.gantt.edge_slots[i]
            total_occ = sum([e - s for s, e in slots])
            self.topo.edge_attr[i, 3] = total_occ / self.gantt.cycle_time
            
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重置环境状态
        :return: (PyG_Data, current_node_idx, action_mask)
        """
        self.gantt.reset()
        self.current_node = 0 # 永远从服务器出发
        self.visited_nodes = {self.current_node}
        self.total_delay = 0.0
        self.step_count = 0
        
        # 注入背景流量以训练避障能力 (0.7 概率)
        for i in range(self.topo.num_edges):
            if np.random.rand() < 0.7:
                self.gantt.check_and_add_slot(i, np.random.uniform(0, 300), np.random.uniform(100, 400))
        
        self._update_edge_occupancy() # 同步到特征
        
        # 注入流需求 [Src, Dst, Size, D_max]
        self.topo.u = torch.tensor([[self.current_node, self.target_node, self.flow_size, self.config['network']['max_delay']]], dtype=torch.float)
        
        obs = self.topo.get_pyg_data()
        action_mask = self._get_action_mask(self.current_node)
        
        return obs, self.current_node, action_mask
        
    def _get_action_mask(self, node_idx: int) -> torch.Tensor:
        """
        生成当前节点的合法下一跳掩码。
        1 表示合法，0 表示非法（用于被赋 -inf）。
        """
        mask = torch.zeros(self.topo.num_nodes, dtype=torch.bool)
        neighbors = self.topo.get_neighbors(node_idx)
        for n in neighbors:
            if n not in self.visited_nodes:
                mask[n] = True
        return mask

    def _get_edge_idx(self, u: int, v: int) -> int:
        """根据两端节点找到具体的边索引"""
        for i in range(self.topo.num_edges):
            if self.topo.edge_index[0, i] == u and self.topo.edge_index[1, i] == v:
                return i
        raise ValueError(f"No edge found between {u} and {v}")

    def step(self, next_node: int, t_offset: float, agv_x: float = None) -> Tuple[torch.Tensor, int, torch.Tensor, float, bool, bool, dict]:
        """
        定制化自回归 Step：传入下一跳和这跳的时间偏移
        :param next_node: Actor Routing Head 选择的节点
        :param t_offset: Actor Scheduling Head 预测的时间槽偏移量 (0.0 ~ 1.0)
        :param agv_x: 可选，传入 AGV 当前的物理 x 坐标用于更新 RSSI
        """
        reward = self.r_cfg['step_penalty']
        terminated = False
        truncated = False
        info = {}
        
        # 获取合法掩码
        mask = self._get_action_mask(self.current_node)
        
        # ⚠️ 防坑 3: 检查死胡同 (Dead End)
        if not mask.any():
            terminated = True
            reward += self.r_cfg['dead_end_penalty']
            info['status'] = 'dead_end'
            return self.topo.get_pyg_data(), self.current_node, mask, reward, terminated, truncated, info
            
        # 非法动作检查 (原则上不该发生因为有mask，但作为防线保留)
        if not mask[next_node]:
            terminated = True
            reward += self.r_cfg['dead_end_penalty']
            info['status'] = 'illegal_action'
            return self.topo.get_pyg_data(), self.current_node, mask, reward, terminated, truncated, info
            
        edge_idx = self._get_edge_idx(self.current_node, next_node)
        edge_attr = self.topo.edge_attr[edge_idx]
        bw_mbps = edge_attr[0].item()
        prop_delay = edge_attr[1].item()
        rssi = edge_attr[2].item()
        
        # 如果是极差的 Rssi 链路，可能会带来极大的重传延迟甚至丢包
        retransmission_delay = 0.0
        if rssi < -80.0:
            retransmission_delay = 500.0 # 模拟丢包重传
            
        # 计算该跳物理传输时长 (Size / BW + Prop)
        duration = (self.flow_size / (bw_mbps * 1e6)) * 1e6 + prop_delay + retransmission_delay
        
        # 时间窗分配
        start_time = t_offset * self.config['network']['cycle_time']
        
        # ⚠️ 防坑 2: 甘特图周期卷绕碰撞检测
        success = self.gantt.check_and_add_slot(edge_idx, start_time, duration)
        if not success:
            terminated = True
            reward += self.r_cfg['collision_penalty']
            info['status'] = 'collision'
            return self.topo.get_pyg_data(), self.current_node, mask, reward, terminated, truncated, info
            
        # 无冲突，顺利前进
        self.visited_nodes.add(next_node)
        self.current_node = next_node
        self.total_delay += duration
        self.step_count += 1
        self._update_edge_occupancy() # 关键：更新边负载特征
        info['total_delay'] = self.total_delay
        
        # 更新漫游网络动态状态 (基于传入的 agv_x 或默认步数模拟)
        if agv_x is not None:
            self.topo.update_roaming_rssi(agv_x)
        else:
            # 默认模拟：AGV 以 1.0m/step 的速度从 0 移动到 20
            self.topo.update_roaming_rssi(float(self.step_count))
        
        # 判断到达目标
        if self.current_node == self.target_node:
            terminated = True
            reward += self.r_cfg['success_reward']
            info['status'] = 'success'
            info['total_delay'] = self.total_delay
            
            # 延迟惩罚
            if self.total_delay > self.config['network']['max_delay']:
                excess = self.total_delay - self.config['network']['max_delay']
                reward += self.r_cfg['latency_penalty_factor'] * excess
        
        # 检查是否截断 (跑得太久)
        if self.step_count >= self.config['training']['max_steps_per_flow'] and not terminated:
            truncated = True
            info['status'] = 'timeout'
            
        next_obs = self.topo.get_pyg_data()
        next_mask = self._get_action_mask(self.current_node)
        if not next_mask.any() and not terminated:
            terminated = True
            reward += self.r_cfg['dead_end_penalty']
            info['status'] = 'dead_end'
        
        # 在返回前再次检查下一步是否面临死胡同
        if not terminated and not next_mask.any():
            terminated = True
            reward += self.r_cfg['dead_end_penalty']
            info['status'] = 'dead_end_next'
            
        return next_obs, self.current_node, next_mask, reward, terminated, truncated, info
