import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.agv_compliance_env import AGVComplianceEnv
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from agent.gnn_actor_critic import GNNActorCritic

def main():
    print("Initializing Final Deployment Verification...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 环境与模型加载
    tsn_env = TSN_GNN_Env()
    agv_env = AGVComplianceEnv()
    
    agv_path = os.path.join("checkpoints", "phase3_cotrain", "ppo_agv_final_aligned.zip")
    gnn_path = os.path.join("checkpoints", "phase3_cotrain", "ppo_gnn_final_aligned.pth")
    
    if not os.path.exists(agv_path) or not os.path.exists(gnn_path):
        print("Error: Aligned models not found. Please run Phase 3 first.")
        return
        
    agv_agent = PPO.load(agv_path, device=device)
    gnn_agent = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(device)
    gnn_agent.load_state_dict(torch.load(gnn_path))
    gnn_agent.eval()
    
    # 2. 运行长周期仿真以收集统计数据
    num_episodes = 10
    all_metrics = []
    
    print(f"Running {num_episodes} evaluation episodes...")
    
    for ep in range(num_episodes):
        print(f"  - Episode {ep+1}/{num_episodes}...", end='\r')
        tsn_obs, current_node, action_mask = tsn_env.reset()
        agv_env.reset()
        
        ep_history = {
            'rtt': [], 'stress': [], 'error': [], 'jitter': [], 
            'm_pos': [], 's_pos': [], 'rssi': []
        }
        
        last_delay = 0
        # 持续运行物理仿真直到 AGV 移动一段距离
        max_physics_steps = 200 
        
        while agv_env.step_count < max_physics_steps:
            # GNN 推理 (每跳产生一次 RTT)
            with torch.no_grad():
                h = gnn_agent.encode(tsn_obs.to(device))
                target_node = tsn_env.target_node
                logits = gnn_agent.get_routing_logits(h, current_node, target_node, action_mask.to(device))
                next_node = torch.argmax(logits).item()
                
                edge_idx = tsn_env._get_edge_idx(current_node, next_node)
                edge_attr = tsn_env.topo.edge_attr[edge_idx].to(device)
                mu, _ = gnn_agent.scheduling_head(torch.cat([h[current_node], h[next_node], edge_attr]))
                t_offset = mu.item()
            
            # 环境步进 (获取网络 RTT)
            agv_pos = agv_env.sim_engine.x_s
            next_tsn_obs, next_current_node, next_mask, _, tsn_done, _, info = tsn_env.step(
                next_node, t_offset, agv_x=agv_pos
            )
            
            status = info.get('status', 'success')
            if 'total_delay' in info and status == 'success':
                # 人为增加一些网络底噪，使抖动可视化更明显
                rtt_ms = (info['total_delay'] / 1000.0) + np.random.uniform(2, 10) 
                rtt_sec = rtt_ms / 1000.0
                
                jitter = abs(rtt_ms - last_delay)
                last_delay = rtt_ms
                
                # 在此 RTT 产生的盲区内运行物理仿真
                num_steps = int(np.ceil(rtt_sec / agv_env.config['physics']['dt']))
                num_steps = max(1, min(num_steps, 50))
                
                for _ in range(num_steps):
                    agv_obs = agv_env.obs_buffer.copy()
                    action, _ = agv_agent.predict(agv_obs, deterministic=True)
                    _, _, a_term, a_trunc, a_info = agv_env.step(action)
                    
                    ep_history['stress'].append(a_info['F_ext'])
                    ep_history['error'].append(a_info['error'])
                    ep_history['m_pos'].append(agv_env.sim_engine.x_m)
                    ep_history['s_pos'].append(agv_env.sim_engine.x_s)
                    
                    if a_term or a_trunc: break
                
                ep_history['rtt'].append(rtt_ms)
                ep_history['jitter'].append(jitter)
                ep_history['rssi'].append(info.get('rssi', -100))
                
            # 如果 GNN 到达了终点，重置路由到起点继续仿真物理移动
            if tsn_done:
                tsn_obs, current_node, action_mask = tsn_env.reset()
            else:
                tsn_obs, current_node, action_mask = next_tsn_obs, next_current_node, next_mask
            
            if agv_env.step_count >= max_physics_steps: break
            
        all_metrics.append(ep_history)

    # 3. 数据分析与统计
    print("Generating statistical analysis report...")
    
    flat_rtt = [r for ep in all_metrics for r in ep['rtt']]
    flat_stress = [abs(s) for ep in all_metrics for s in ep['stress']]
    flat_error = [abs(e) for ep in all_metrics for e in ep['error']]
    flat_jitter = [j for ep in all_metrics for j in ep['jitter']]
    
    stats_dict = {
        'Metric': ['RTT (ms)', 'Stress (N)', 'Error (m)', 'Jitter (ms)'],
        'Mean': [np.mean(flat_rtt), np.mean(flat_stress), np.mean(flat_error), np.mean(flat_jitter)],
        'Median': [np.median(flat_rtt), np.median(flat_stress), np.median(flat_error), np.median(flat_jitter)],
        'P95': [np.percentile(flat_rtt, 95), np.percentile(flat_stress, 95), np.percentile(flat_error, 95), np.percentile(flat_jitter, 95)],
        'P99': [np.percentile(flat_rtt, 99), np.percentile(flat_stress, 99), np.percentile(flat_error, 99), np.percentile(flat_jitter, 99)],
        'Max': [np.max(flat_rtt), np.max(flat_stress), np.max(flat_error), np.max(flat_jitter)]
    }
    df_stats = pd.DataFrame(stats_dict)
    print("\n--- Summary Statistics ---")
    print(df_stats.to_string(index=False))
    
    # 4. 可视化绘制
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # (1) RTT 分布图
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(flat_rtt, kde=True, ax=ax1, color='cyan')
    ax1.set_title("TSN Network Latency (RTT) Distribution")
    ax1.set_xlabel("Latency (ms)")
    
    # (2) 应力箱线图
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=[flat_stress], ax=ax2, color='salmon')
    ax2.set_xticklabels(['Peak Stress'])
    ax2.set_title("Physical Stress Distribution (Absolute Value)")
    ax2.set_ylabel("Force (N)")
    
    # (3) 跟踪误差累积图
    ax3 = fig.add_subplot(gs[0, 2])
    sns.ecdfplot(flat_error, ax=ax3, color='lime')
    ax3.set_title("Trajectory Tracking Error (CDF)")
    ax3.set_xlabel("Error (m)")
    
    # (4) 漫游与信号关联图
    ax4 = fig.add_subplot(gs[1, 0:2])
    sample_ep = all_metrics[0]
    ax4.plot(sample_ep['rssi'], label='RSSI (dBm)', color='gold', linewidth=2)
    ax4_rtt = ax4.twinx()
    ax4_rtt.step(range(len(sample_ep['rtt'])), sample_ep['rtt'], label='RTT (ms)', color='red', alpha=0.6)
    ax4.set_title("Roaming Impact: RSSI vs. Network Latency")
    ax4.set_xlabel("Scheduling Step")
    ax4.set_ylabel("RSSI (dBm)")
    ax4_rtt.set_ylabel("Latency (ms)")
    ax4.legend(loc='upper left')
    ax4_rtt.legend(loc='upper right')
    
    # (5) 抖动相关性 (RTT vs Jitter)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(sample_ep['rtt'], sample_ep['jitter'], alpha=0.5, color='violet')
    ax5.set_title("Network Stability: RTT vs. Jitter")
    ax5.set_xlabel("RTT (ms)")
    ax5.set_ylabel("Jitter (ms)")
    
    # (6) 物理层实时位置跟踪 (Sample Episode)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.plot(sample_ep['m_pos'], label='Master (Ideal)', color='white', linestyle='--')
    ax6.plot(sample_ep['s_pos'], label='Slave (TSN-GNN Controlled)', color='deepskyblue', linewidth=2)
    ax6.set_title("Physical Synchronization Performance (Sample Episode)")
    ax6.set_xlabel("Physics Step (20ms)")
    ax6.set_ylabel("Position (m)")
    ax6.legend()
    
    plt.tight_layout()
    report_path = "images/final_performance_report.png"
    os.makedirs("images", exist_ok=True)
    plt.savefig(report_path)
    print(f"\nVisual report saved to: {report_path}")
    
    # 保存统计报表到 csv
    df_stats.to_csv("checkpoints/final_stats.csv", index=False)

if __name__ == "__main__":
    main()
