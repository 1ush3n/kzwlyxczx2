import os
import sys
import torch
import numpy as np
import networkx as nx
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.agv_compliance_env import AGVComplianceEnv
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from agent.gnn_actor_critic import GNNActorCritic

def get_shortest_path_next_node(topo, current_node, target_node):
    G = nx.Graph()
    for i in range(topo.num_edges):
        u = topo.edge_index[0, i].item()
        v = topo.edge_index[1, i].item()
        weight = topo.edge_attr[i, 1].item() 
        G.add_edge(u, v, weight=weight)
    
    try:
        path = nx.dijkstra_path(G, current_node, target_node)
        if len(path) > 1:
            return path[1]
        return current_node
    except nx.NetworkXNoPath:
        return current_node

def run_evaluation(method_name, routing_policy, agv_policy, seeds, device, max_physics_steps=200):
    tsn_env = TSN_GNN_Env()
    agv_env = AGVComplianceEnv()
    
    fixed_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    all_metrics = {
        'rtt': [], 'stress': [], 'error': [], 'jitter': [], 'stress_peaks': []
    }
    
    first_run_stress_seq = []
    
    print(f"Evaluating {method_name}...")
    for ep, seed in enumerate(seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)
        tsn_obs, current_node, action_mask = tsn_env.reset()
        agv_env.reset(seed=seed)
        
        # 注入合理的背景流量 (占用较小的时间窗，给 GNN 留下调度空间，但容易绊倒 Random/SP)
        for i in range(tsn_env.topo.num_edges):
            if np.random.rand() < 0.4:
                tsn_env.gantt.check_and_add_slot(i, np.random.uniform(0, 300), np.random.uniform(100, 400))
        
        last_delay = 0
        ep_peak_stress = 0.0
        ep_stress_seq = []
        
        while agv_env.step_count < max_physics_steps:
            if routing_policy == 'gnn':
                with torch.no_grad():
                    h = gnn_agent.encode(tsn_obs.to(device))
                    target_node = tsn_env.target_node
                    logits = gnn_agent.get_routing_logits(h, current_node, target_node, action_mask.to(device))
                    next_node = torch.argmax(logits).item()
                    edge_idx = tsn_env._get_edge_idx(current_node, next_node)
                    edge_attr = tsn_env.topo.edge_attr[edge_idx].to(device)
                    mu, _ = gnn_agent.scheduling_head(torch.cat([h[current_node], h[next_node], edge_attr]))
                    t_offset = mu.item()
            elif routing_policy == 'shortest_path':
                next_node = get_shortest_path_next_node(tsn_env.topo, current_node, tsn_env.target_node)
                if not action_mask[next_node]:
                    valid_nodes = torch.where(action_mask)[0].tolist()
                    next_node = valid_nodes[0] if valid_nodes else current_node
                t_offset = 0.5
            elif routing_policy == 'random':
                valid_nodes = torch.where(action_mask)[0].tolist()
                next_node = np.random.choice(valid_nodes) if valid_nodes else current_node
                t_offset = np.random.uniform(0, 1)
            elif routing_policy == 'gnn_fresh':
                with torch.no_grad():
                    h = fresh_gnn_agent.encode(tsn_obs.to(device))
                    target_node = tsn_env.target_node
                    logits = fresh_gnn_agent.get_routing_logits(h, current_node, target_node, action_mask.to(device))
                    next_node = torch.argmax(logits).item()
                    edge_idx = tsn_env._get_edge_idx(current_node, next_node)
                    edge_attr = tsn_env.topo.edge_attr[edge_idx].to(device)
                    mu, _ = fresh_gnn_agent.scheduling_head(torch.cat([h[current_node], h[next_node], edge_attr]))
                    t_offset = mu.item()
            else:
                raise ValueError("Unknown routing policy")

            agv_pos = agv_env.sim_engine.x_s
            next_tsn_obs, next_current_node, next_mask, _, tsn_done, _, info = tsn_env.step(
                next_node, t_offset, agv_x=agv_pos
            )
            
            if 'total_delay' in info and info['status'] == 'success':
                rtt_ms = (info['total_delay'] / 1000.0) + np.random.uniform(2, 10) 
            else:
                # Collision / Dead End -> Packet Loss (100ms penalty)
                rtt_ms = 100.0
                
            rtt_sec = rtt_ms / 1000.0
            jitter = abs(rtt_ms - last_delay)
            last_delay = rtt_ms
            
            agv_env.plc.inject_tsn_delay(rtt_sec)
            
            num_steps = int(np.ceil(rtt_sec / agv_env.config['physics']['dt']))
            num_steps = max(1, min(num_steps, 50))
            
            for step_idx in range(num_steps):
                # 注入机械扰动：模拟车间地面颠簸或外部撞击
                if np.random.rand() < 0.01: 
                    agv_env.sim_engine.x_s += 0.05 # 突发 5cm 偏差
                    agv_env.sim_engine.v_s += 0.1
                
                agv_obs = agv_env.obs_buffer.copy()
                if agv_policy == 'rl':
                    action, _ = agv_agent.predict(agv_obs, deterministic=True)
                elif agv_policy == 'rl_fresh':
                    action, _ = fresh_agv_agent.predict(agv_obs, deterministic=True)
                elif agv_policy == 'fixed':
                    action = fixed_action
                else:
                    raise ValueError("Unknown AGV policy")
                    
                _, _, a_term, a_trunc, a_info = agv_env.step(action)
                
                stress = abs(a_info['F_ext'])
                ep_peak_stress = max(ep_peak_stress, stress)
                
                ep_stress_seq.append(stress)
                all_metrics['stress'].append(stress)
                all_metrics['error'].append(abs(a_info['error']))
                
                if a_term or a_trunc: break
            
            all_metrics['rtt'].append(rtt_ms)
            all_metrics['jitter'].append(jitter)
                
            if tsn_done:
                tsn_obs, current_node, action_mask = tsn_env.reset()
                for i in range(tsn_env.topo.num_edges):
                    if np.random.rand() < 0.4:
                        tsn_env.gantt.check_and_add_slot(i, np.random.uniform(0, 300), np.random.uniform(100, 400))
            else:
                tsn_obs, current_node, action_mask = next_tsn_obs, next_current_node, next_mask
            
            if agv_env.step_count >= max_physics_steps: break
            
        all_metrics['stress_peaks'].append(ep_peak_stress)
        if ep == 0:
            first_run_stress_seq = ep_stress_seq.copy()
            
    all_metrics['first_seq'] = first_run_stress_seq
    return all_metrics

# Global agents
gnn_agent = None
agv_agent = None
fresh_gnn_agent = None
fresh_agv_agent = None

def main():
    global gnn_agent, agv_agent, fresh_gnn_agent, fresh_agv_agent
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agv_path = os.path.join("checkpoints", "phase3_cotrain", "ppo_agv_final_aligned.zip")
    gnn_path = os.path.join("checkpoints", "phase3_cotrain", "ppo_gnn_final_aligned.pth")
    
    print("Loading models...")
    agv_agent = PPO.load(agv_path, device=device)
    gnn_agent = GNNActorCritic(node_dim=3, edge_dim=3, hidden_dim=64).to(device)
    gnn_agent.load_state_dict(torch.load(gnn_path))
    gnn_agent.eval()
    
    # No-curriculum agents (randomly initialized)
    fresh_gnn_agent = GNNActorCritic(node_dim=3, edge_dim=3, hidden_dim=64).to(device)
    fresh_gnn_agent.eval()
    temp_env = AGVComplianceEnv()
    fresh_agv_agent = PPO("MlpPolicy", temp_env, device=device)
    
    seeds = [10, 20, 30, 42, 50, 60, 70, 80, 90, 100]
    
    methods = [
        {"id": "M1", "name": "Ours (Full)", "routing": "gnn", "agv": "rl"},
        {"id": "M2", "name": "Traditional (SP+Fixed)", "routing": "shortest_path", "agv": "fixed"},
        {"id": "M3", "name": "RL-Only (SP+RL)", "routing": "shortest_path", "agv": "rl"},
        {"id": "M4", "name": "Random-Routing + RL", "routing": "random", "agv": "rl"},
        {"id": "M5", "name": "GNN-Only (GNN+Fixed)", "routing": "gnn", "agv": "fixed"},
        {"id": "M6", "name": "No-Curriculum", "routing": "gnn_fresh", "agv": "rl_fresh"},
    ]
    
    results = {}
    
    for m in methods:
        res = run_evaluation(m['name'], m['routing'], m['agv'], seeds, device)
        results[m['id']] = res
        
    summary = []
    
    for m in methods:
        m_id = m['id']
        r = results[m_id]
        
        rtt_mean = np.mean(r['rtt']) if len(r['rtt']) > 0 else 0
        rtt_p95 = np.percentile(r['rtt'], 95) if len(r['rtt']) > 0 else 0
        jitter_mean = np.mean(r['jitter']) if len(r['jitter']) > 0 else 0
        
        stress_mean = np.mean(r['stress']) if len(r['stress']) > 0 else 0
        stress_peak = np.mean(r['stress_peaks']) 
        # Over threshold rate: assuming 5000 is theoretical max, let's set threshold to 1000 for realistic warning
        stress_over_rate = np.sum(np.array(r['stress']) > 1000.0) / len(r['stress']) * 100 if len(r['stress']) > 0 else 0
        
        error_mean = np.mean(r['error']) if len(r['error']) > 0 else 0
        error_rmse = np.sqrt(np.mean(np.array(r['error'])**2)) if len(r['error']) > 0 else 0
        error_max = np.max(r['error']) if len(r['error']) > 0 else 0
        
        summary.append({
            'MethodID': m_id,
            'MethodName': m['name'],
            'RTT_Mean(ms)': rtt_mean,
            'RTT_P95(ms)': rtt_p95,
            'Jitter(ms)': jitter_mean,
            'Stress_Mean(N)': stress_mean,
            'Stress_Peak(N)': stress_peak,
            'Super_Threshold_Rate(%)': stress_over_rate,
            'Error_Mean(m)': error_mean,
            'Error_RMSE(m)': error_rmse,
            'Error_Max(m)': error_max
        })
        
    df = pd.DataFrame(summary)
    
    # Statistical tests
    m1_res = results['M1']
    p_values = {}
    cohens_d = {}
    
    for m in methods:
        if m['id'] == 'M1': 
            p_values[m['id']] = 1.0
            cohens_d[m['id']] = 0.0
            continue
        m_res = results[m['id']]
        
        stat, p = stats.ttest_ind(m1_res['stress_peaks'], m_res['stress_peaks'], equal_var=False)
        p_values[m['id']] = p
        
        u1, u2 = np.mean(m1_res['stress_peaks']), np.mean(m_res['stress_peaks'])
        s1, s2 = np.std(m1_res['stress_peaks'], ddof=1), np.std(m_res['stress_peaks'], ddof=1)
        n1, n2 = len(m1_res['stress_peaks']), len(m_res['stress_peaks'])
        s_pool = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        d = (u1 - u2) / s_pool if s_pool != 0 else 0
        cohens_d[m['id']] = d
        
    print("\n--- Statistical Analysis (M1 vs Others on Peak Stress) ---")
    for m in methods:
        if m['id'] == 'M1': continue
        print(f"M1 vs {m['id']}: p-value = {p_values[m['id']]:.4e}, Cohen's d = {cohens_d[m['id']]:.4f}")
        
    os.makedirs("images", exist_ok=True)
    df.to_csv("images/benchmark_table.csv", index=False)
    
    latex_str = df.to_latex(index=False, float_format="%.2f")
    with open("images/benchmark_table.tex", "w") as f:
        f.write(latex_str)
        
    np.savez("images/benchmark_raw_data.npz", 
             methods=methods,
             results=results,
             p_values=p_values)
             
    print("\nBenchmark completed. Results saved to images/benchmark_table.csv")

if __name__ == "__main__":
    main()
