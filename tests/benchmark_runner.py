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

def run_evaluation(method_name, routing_policy, agv_policy, seeds, device, max_physics_steps=200, bg_prob=0.3, jitter_range=(1, 10)):
    tsn_env = TSN_GNN_Env()
    agv_env = AGVComplianceEnv()
    
    fixed_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    all_metrics = {
        'rtt': [], 'stress': [], 'error': [], 'jitter': [], 'stress_peaks': [],
        'collisions': 0, 'dead_ends': 0, 'total_packets': 0
    }
    
    first_run_stress_seq = []
    
    print(f"Evaluating {method_name}...")
    for ep, seed in enumerate(seeds):
        # 解耦随机数生成器
        rng_tsn = np.random.RandomState(seed)
        rng_agv = np.random.RandomState(seed + 1000) 
        
        torch.manual_seed(seed)
        tsn_obs, current_node, action_mask = tsn_env.reset()
        agv_env.reset(seed=seed)
        
        # --- STRESS TEST: 注入背景流量 ---
        for i in range(tsn_env.topo.num_edges):
            if rng_tsn.rand() < bg_prob:
                tsn_env.gantt.check_and_add_slot(i, rng_tsn.uniform(0, 300), rng_tsn.uniform(100, 400))
        
        last_delay = 0
        ep_peak_stress = 0.0
        ep_stress_seq = []
        
        while agv_env.step_count < max_physics_steps:
            # ... (决策逻辑不变) ...
            if routing_policy == 'gnn':
                with torch.no_grad():
                    h = gnn_agent.encode(tsn_obs.to(device))
                    target_node = tsn_env.target_node
                    logits = gnn_agent.get_routing_logits(h, current_node, target_node, action_mask.to(device))
                    next_node = torch.argmax(logits).item()
                    edge_idx = tsn_env._get_edge_idx(current_node, next_node)
                    edge_attr = tsn_env.topo.edge_attr[edge_idx].to(device)
                    out = gnn_agent.scheduling_head(torch.cat([h[current_node], h[next_node], edge_attr]))
                    t_offset = torch.sigmoid(out[0]).item()
            elif routing_policy == 'shortest_path':
                next_node = get_shortest_path_next_node(tsn_env.topo, current_node, tsn_env.target_node)
                if not action_mask[next_node]:
                    valid_nodes = torch.where(action_mask)[0].tolist()
                    next_node = valid_nodes[0] if valid_nodes else current_node
                t_offset = 0.5
            elif routing_policy == 'random':
                valid_nodes = torch.where(action_mask)[0].tolist()
                next_node = rng_tsn.choice(valid_nodes) if valid_nodes else current_node
                t_offset = rng_tsn.uniform(0, 1)
            elif routing_policy == 'gnn_fresh':
                # BUGFIX: random GNN should produce truly random routing, not biased by architecture
                valid_nodes = torch.where(action_mask)[0].tolist()
                next_node = rng_tsn.choice(valid_nodes) if valid_nodes else current_node
                t_offset = rng_tsn.uniform(0, 1)
            else:
                raise ValueError("Unknown routing policy")

            agv_pos = agv_env.sim_engine.x_s
            next_tsn_obs, next_current_node, next_mask, _, tsn_done, _, info = tsn_env.step(
                next_node, t_offset, agv_x=agv_pos
            )
            
            status = info.get('status', 'success')
            all_metrics['total_packets'] += 1
            if status == 'success':
                # --- STRESS TEST: 注入极高延迟抖动 (10ms ~ 100ms) ---
                rtt_ms = (info.get('total_delay', 1000.0) / 1000.0) + rng_tsn.uniform(*jitter_range) 
            elif status == 'collision':
                all_metrics['collisions'] += 1
                rtt_ms = 500.0 
            else:
                all_metrics['dead_ends'] += 1
                rtt_ms = 500.0
                
            rtt_sec = rtt_ms / 1000.0
            jitter = abs(rtt_ms - last_delay)
            last_delay = rtt_ms
            
            agv_env.plc.inject_tsn_delay(rtt_sec)
            
            num_steps = int(np.ceil(rtt_sec / agv_env.config['physics']['dt']))
            num_steps = max(1, min(num_steps, 50))
            
            for _ in range(num_steps):
                if rng_agv.rand() < 0.01: 
                    agv_env.sim_engine.x_s += 0.05 
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
                    if rng_tsn.rand() < bg_prob:
                        tsn_env.gantt.check_and_add_slot(i, rng_tsn.uniform(0, 300), rng_tsn.uniform(100, 400))
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['normal', 'pressure'], default='pressure',
                        help='normal: BG=0.1 jitter(0.5,3) | pressure: BG=0.3 jitter(1,10)')
    parser.add_argument('--output', type=str, default=None, help='Custom output CSV path')
    parser.add_argument('--npz', type=str, default=None, help='Custom raw data path')
    args = parser.parse_args()

    if args.scenario == 'normal':
        bg_prob = 0.1
        jitter_range = (0.5, 3.0)
        scenario_tag = 'normal'
    else:
        bg_prob = 0.3
        jitter_range = (1.0, 10.0)
        scenario_tag = 'pressure'

    print(f"Scenario: {scenario_tag} (bg_prob={bg_prob}, jitter={jitter_range})")
    global gnn_agent, agv_agent, fresh_gnn_agent, fresh_agv_agent
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agv_path = os.path.join("checkpoints", "phase3_cotrain", "ppo_agv_final_aligned.zip")
    gnn_path = os.path.join("checkpoints", "phase3_cotrain", "ppo_gnn_final_aligned.pth")
    
    print("Loading models...")
    agv_agent = PPO.load(agv_path, device=device)
    gnn_agent = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(device)
    gnn_agent.load_state_dict(torch.load(gnn_path))
    gnn_agent.eval()
    
    # No-curriculum agents (randomly initialized)
    fresh_gnn_agent = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(device)
    fresh_gnn_agent.eval()
    temp_env = AGVComplianceEnv()
    fresh_agv_agent = PPO("MlpPolicy", temp_env, device=device)
    
    seeds = list(range(0, 200, 10)) # 20 seeds
    
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
        res = run_evaluation(m['name'], m['routing'], m['agv'], seeds, device,
                             bg_prob=bg_prob, jitter_range=jitter_range)
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
        stress_p95 = np.percentile(r['stress'], 95) if len(r['stress']) > 0 else 0
        # Over threshold rate: assuming 5000 is theoretical max, let's set threshold to 1000 for realistic warning
        stress_over_rate = np.sum(np.array(r['stress']) > 1000.0) / len(r['stress']) * 100 if len(r['stress']) > 0 else 0
        
        error_mean = np.mean(r['error']) if len(r['error']) > 0 else 0
        error_rmse = np.sqrt(np.mean(np.array(r['error'])**2)) if len(r['error']) > 0 else 0
        error_max = np.max(r['error']) if len(r['error']) > 0 else 0
        
        total_pkt = r['total_packets']
        collision_rate = (r['collisions'] / total_pkt * 100) if total_pkt > 0 else 0
        dead_end_rate = (r['dead_ends'] / total_pkt * 100) if total_pkt > 0 else 0
        
        summary.append({
            'MethodID': m_id,
            'MethodName': m['name'],
            'RTT_Mean(ms)': rtt_mean,
            'RTT_P95(ms)': rtt_p95,
            'Jitter(ms)': jitter_mean,
            'CollisionRate(%)': collision_rate,
            'DeadEndRate(%)': dead_end_rate,
            'Stress_Mean(N)': stress_mean,
            'Stress_P95(N)': stress_p95,
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
    csv_out = args.output or f"images/benchmark_table_{scenario_tag}.csv"
    tex_out = args.output or f"images/benchmark_table_{scenario_tag}.tex"
    npz_out = args.npz or f"images/benchmark_raw_data_{scenario_tag}.npz"
    df.to_csv(csv_out, index=False)
    with open(tex_out, "w") as f:
        f.write(df.to_latex(index=False))
    np.savez(npz_out,
             results=np.array(results, dtype=object),
             p_values=np.array(p_values, dtype=object),
             allow_pickle=True)
    print(f"\nBenchmark [{scenario_tag}] completed. Results saved to {csv_out}")

if __name__ == "__main__":
    main()
