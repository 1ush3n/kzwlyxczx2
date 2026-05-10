import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data

class GNNActorCritic(nn.Module):
    """
    基于 PyTorch Geometric 的自回归 Actor-Critic 智能体架构。
    严格遵守路由 -> 调度的自回归依赖。
    """
    def __init__(self, node_dim: int = 3, edge_dim: int = 3, hidden_dim: int = 64):
        super(GNNActorCritic, self).__init__()
        
        # 1. 编码器 (Encoder)
        # GATv2Conv 支持动态关注不同边（如衰减的Rssi），edge_dim 帮助边特征融入注意力计算
        self.conv1 = GATv2Conv(node_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        
        # 2. 状态表达融合层
        self.ctx_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. 路由策略头 (Actor-Routing Head)
        # 输入: H_ctx 拼接所有候选节点 H_v
        self.routing_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出 Logits
        )
        
        # 4. 调度策略头 (Actor-Scheduling Head)
        # ⚠️ 严格自回归：必须传入已选定的目标边特征进行独立预测
        self.scheduling_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 输出 mu, sigma (通过 Softplus 保证 sigma 为正)
        )
        
        # 5. 价值头 (Critic Head)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode(self, data: Data) -> torch.Tensor:
        """特征提取"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        h = self.conv1(x, edge_index, edge_attr)
        h = torch.relu(h)
        h = self.conv2(h, edge_index, edge_attr)
        return torch.relu(h)
        
    def get_routing_logits(self, h: torch.Tensor, current_node: int, target_node: int, action_mask: torch.Tensor) -> torch.Tensor:
        """
        根据上下文提取全图节点的下一跳 Logits。
        自动应用 Action Mask（把非邻居和已访问节点的 Logit 降为 -1e9）。
        """
        # H_ctx = [H_curr || H_target]
        h_ctx = torch.cat([h[current_node], h[target_node]], dim=-1)
        
        # 将 h_ctx 广播到所有节点
        # shape: [N, hidden_dim] -> [N, hidden_dim * 2]
        h_ctx_broadcast = h_ctx.unsqueeze(0).expand(h.size(0), -1)
        
        # 结合候选节点的隐向量
        routing_input = torch.cat([h_ctx_broadcast, h], dim=-1)
        
        logits = self.routing_head(routing_input).squeeze(-1) # shape: [N]
        
        # 施加严苛掩码
        logits[~action_mask] = -1e9
        
        return logits
        
    def get_scheduling_dist(self, h: torch.Tensor, current_node: int, next_node: int, raw_edge_attr: torch.Tensor) -> torch.distributions.Normal:
        """
        自回归第二阶段：基于已经选出的 next_node，预测对应的 T_offset。
        """
        h_curr = h[current_node]
        h_next = h[next_node]
        
        sched_input = torch.cat([h_curr, h_next, raw_edge_attr], dim=-1)
        out = self.scheduling_head(sched_input)
        
        mu = torch.sigmoid(out[0]) # 约束在 [0, 1] 之间
        sigma = torch.nn.functional.softplus(out[1]) + 1e-4
        
        return torch.distributions.Normal(mu, sigma)
        
    def get_value(self, h: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """全局评估图价值"""
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            
        global_h = global_mean_pool(h, batch)
        return self.critic_head(global_h)
