import torch
from torch_geometric.data import Data
import numpy as np

class TSNTopology:
    """
    TSN 网络拓扑结构管理器，负责生成并维护 PyG 的 Data 对象。
    包含节点负载演化以及 AP 漫游衰减（动态边特征）。
    """
    def __init__(self, num_nodes: int, num_ap: int, agv_idx: int):
        self.num_nodes = num_nodes
        self.num_ap = num_ap
        self.agv_idx = agv_idx
        
        # 物理位置定义: {node_idx: (x, y)}
        # 假设车间走廊长 20m, AP1 在 5m 处, AP2 在 15m 处
        self.ap_coords = {
            7: (5.0, 2.0),  # AP1
            8: (15.0, 2.0)  # AP2
        }
        
        # 节点特征: [Type, CpuLoad, QueueLength]
        # Type: 0=Server/Switch, 1=AP, 2=AGV
        self.x = torch.zeros((num_nodes, 3), dtype=torch.float)
        
        # 定义一个环+星型的基础测试拓扑
        # 0 是源服务器, 1..6 是核心/边缘交换机, 7,8是AP, 9是AGV
        edges = [
            (0, 1), (1, 0), (0, 2), (2, 0),
            (1, 3), (3, 1), (2, 4), (4, 2),
            (3, 5), (5, 3), (4, 6), (6, 4),
            (5, 7), (7, 5), (6, 8), (8, 6), # AP 接入交换机
            (7, 9), (9, 7), (8, 9), (9, 8)  # AGV 接入 AP
        ]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.num_edges = self.edge_index.size(1)
        
        # 初始化节点类型
        self.x[0, 0] = 0 # Server
        self.x[1:7, 0] = 0 # Switch
        self.x[7:9, 0] = 1 # AP
        self.x[9, 0] = 2 # AGV
        
        # 边特征: [Bandwidth(Mbps), PropDelay(us), Rssi(dBm)]
        self.edge_attr = torch.zeros((self.num_edges, 3), dtype=torch.float)
        
        for i in range(self.num_edges):
            u, v = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            
            # 默认带宽 1000Mbps，传播延迟 1us
            bw = 1000.0
            prop_delay = 1.0
            rssi = 0.0 # 有线链路不需要 Rssi，置0
            
            # 如果是无线链路 (AP 到 AGV 或者 AGV 到 AP)
            if u >= 7 and v == 9 or u == 9 and v >= 7:
                bw = 100.0 # Wi-Fi 带宽低
                prop_delay = 5.0 # 无线延迟大
                rssi = -50.0 # 初始 Rssi 良好
                
            self.edge_attr[i] = torch.tensor([bw, prop_delay, rssi])
            
        # 全局状态 (待路由的数据流): [Src, Dst, Size(Bytes), D_max(us)]
        # 这个会在环境 reset 或 step 时动态注入
        self.u = torch.zeros((1, 4), dtype=torch.float)
        
    def get_pyg_data(self) -> Data:
        """返回当前的 PyG 图对象"""
        return Data(x=self.x.clone(), edge_index=self.edge_index.clone(), edge_attr=self.edge_attr.clone(), u=self.u.clone())
        
    def get_neighbors(self, node_idx: int) -> list:
        """获取节点的物理相连邻居"""
        mask = self.edge_index[0] == node_idx
        return self.edge_index[1][mask].tolist()
        
    def update_roaming_rssi(self, agv_x: float):
        """基于物理距离的 RSSI 衰减模型 (Log-Distance Path Loss Model)"""
        # RSSI(d) = P_tx - 10 * n * log10(d)
        # 设定基准: 1m 处为 -30dBm, 路径损耗指数 n = 3.0 (工厂复杂环境)
        p_tx = -30.0
        n_loss = 3.0
        agv_y = 0.0 # 假设 AGV 在 y=0 的轨迹上移动
        
        noise = np.random.normal(0, 0.5) # 少量多径效应/阴影衰减噪声
        
        for i in range(self.num_edges):
            u, v = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            
            ap_node = None
            if v == 9 and u in self.ap_coords:
                ap_node = u
            elif u == 9 and v in self.ap_coords:
                ap_node = v
                
            if ap_node is not None:
                ap_x, ap_y = self.ap_coords[ap_node]
                # 计算欧几里得距离
                dist = np.sqrt((agv_x - ap_x)**2 + (agv_y - ap_y)**2)
                
                # 计算 RSSI
                rssi = p_tx - 10 * n_loss * np.log10(dist + 1.0) + noise
                self.edge_attr[i, 2] = max(-95.0, min(-20.0, rssi))
