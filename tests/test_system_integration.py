import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agv_compliance_env import AGVComplianceEnv
from tsn_net.tsn_gnn_env import TSN_GNN_Env

def run_integration_test(steps=200):
    print(f"Starting System Integration Test for {steps} steps...")
    
    # 1. 初始化两个环境
    # 物理环境使用默认配置，TSN环境使用默认拓扑
    agv_env = AGVComplianceEnv()
    tsn_env = TSN_GNN_Env()
    
    agv_env.reset(seed=42)
    tsn_obs, tsn_curr, tsn_mask = tsn_env.reset()
    
    # 数据记录器
    history = {
        'time': [],
        'rtt': [],
        'error': [],
        'stress': [],
        'rssi_ap1': [],
        'rssi_ap2': [],
        'reward_agv': []
    }
    
    dt = agv_env.config['physics']['dt']
    
    for i in range(steps):
        # --- 步骤 A: TSN 网络层决策 ---
        # 在集成测试中，我们模拟一个简单的路由策略：从 0 -> 1 -> 3 -> 5 -> 7 -> 9 (到达AGV)
        # 我们根据当前节点手动选择下一跳，并随机生成一个时间偏移
        path_map = {0: 1, 1: 3, 3: 5, 5: 7, 7: 9}
        
        actual_delay = 0.0
        if tsn_curr in path_map:
            next_hop = path_map[tsn_curr]
            t_offset = np.random.uniform(0.1, 0.9)
            
            # 执行 TSN Step
            obs, tsn_curr, mask, r_tsn, term, trunc, info = tsn_env.step(next_hop, t_offset)
            
            # 如果成功到达或正在中转，累加延迟
            # 在这个简化测试中，我们假设每一跳的延迟都实时反馈给 PLC
            # 真实情况下可能是流完成后反馈，这里为了观察动态效果，我们每步注入
            if 'total_delay' in info:
                actual_delay = info['total_delay'] / 1000.0 # us -> ms (或根据PLC单位调整, PLC默认秒)
                # 转换微秒到秒
                actual_delay_sec = info['total_delay'] / 1e6
                agv_env.plc.inject_tsn_delay(actual_delay_sec)
                
                # 重置 TSN 环境以便下一条流开始 (模拟持续的控制流)
                tsn_env.reset()
                tsn_curr = 0
        
        # --- 步骤 B: AGV 物理层决策 ---
        # 模拟一个固定的阻抗动作 (M, B, K 稍微增加一点)
        agv_action = np.array([0.1, 0.2, -0.1]) # 微调阻抗
        agv_obs, agv_reward, terminated, truncated, agv_info = agv_env.step(agv_action)
        
        # 记录数据
        # 观测状态: [e, e_dot, F_ext, tau, delta_x_cmd]
        # 注意: agv_obs 是堆叠的，取最后一帧
        latest_obs = agv_obs[-1]
        
        history['time'].append(i * dt)
        history['rtt'].append(latest_obs[3] * 1000) # s -> ms
        history['error'].append(latest_obs[0])
        history['stress'].append(latest_obs[2])
        history['reward_agv'].append(agv_reward)
        
        # 记录 RSSI (从拓扑结构中提取)
        # 边 16 是 (7,9), 边 18 是 (8,9)
        rssi1 = tsn_env.topo.edge_attr[16, 2].item()
        rssi2 = tsn_env.topo.edge_attr[18, 2].item()
        history['rssi_ap1'].append(rssi1)
        history['rssi_ap2'].append(rssi2)
        
        if terminated or truncated:
            agv_env.reset()
            
    # --- 步骤 C: 生成可视化报告 ---
    print("Generating Visual Integration Report...")
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3)
    
    # 1. 网络延迟 (TSN 注入结果)
    axs[0, 0].plot(history['time'], history['rtt'], color='blue', label='End-to-End Latency (RTT)')
    axs[0, 0].set_title('TSN Scheduled Network Latency')
    axs[0, 0].set_ylabel('Latency (ms)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 2. 漫游 RSSI 演化
    axs[0, 1].plot(history['time'], history['rssi_ap1'], label='AP1 (Node 7) RSSI', linestyle='--')
    axs[0, 1].plot(history['time'], history['rssi_ap2'], label='AP2 (Node 8) RSSI')
    axs[0, 1].axhline(y=-80, color='red', linestyle=':', label='Retransmission Threshold')
    axs[0, 1].set_title('Dynamic RSSI & Roaming (TSN Topology)')
    axs[0, 1].set_ylabel('RSSI (dBm)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # 3. 跟踪误差
    axs[1, 0].plot(history['time'], history['error'], color='green')
    axs[1, 0].set_title('AGV Tracking Error (Compliance)')
    axs[1, 0].set_ylabel('Error (m)')
    axs[1, 0].grid(True)
    
    # 4. 机翼应力
    axs[1, 1].plot(history['time'], history['stress'], color='red')
    axs[1, 1].set_title('Wing Structural Stress (F_ext)')
    axs[1, 1].set_ylabel('Stress (N)')
    axs[1, 1].grid(True)
    
    # 5. 奖励曲线
    axs[2, 0].plot(history['time'], history['reward_agv'], color='purple')
    axs[2, 0].set_title('AGV Step Reward')
    axs[2, 0].set_ylabel('Reward')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].grid(True)
    
    # 6. 系统状态概览
    axs[2, 1].text(0.1, 0.5, f"Steps: {steps}\nControl Freq: {1/dt:.1f}Hz\nAvg Latency: {np.mean(history['rtt']):.2f}ms\nMax Stress: {np.max(history['stress']):.2f}N", 
                   fontsize=12, bbox=dict(facecolor='wheat', alpha=0.5))
    axs[2, 1].axis('off')
    axs[2, 1].set_title('System Metrics Summary')
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images', 'system_integration_report.png')
    plt.savefig(save_path)
    print(f"Report saved to: {save_path}")
    # plt.show()

if __name__ == "__main__":
    # 确保 images 目录存在
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images'), exist_ok=True)
    run_integration_test(steps=300)
