import os, sys, torch, numpy as np, pandas as pd, argparse, json
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agv_compliance_env import AGVComplianceEnv
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from agent.gnn_actor_critic import GNNActorCritic
import networkx as nx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [10, 20, 30, 42, 50]
BG_PROB = 0.3
JITTER_RANGE = (1, 10)
MAX_PHYSICS_STEPS = 200


def load_models():
    from stable_baselines3 import PPO
    agv = PPO.load("checkpoints/phase3_cotrain/ppo_agv_final_aligned.zip", device=DEVICE)
    gnn = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(DEVICE)
    gnn.load_state_dict(torch.load("checkpoints/phase3_cotrain/ppo_gnn_final_aligned.pth"))
    gnn.eval()
    return agv, gnn


def get_shortest_path_next_node(topo, current_node, target_node):
    G = nx.Graph()
    for i in range(topo.num_edges):
        u = topo.edge_index[0, i].item()
        v = topo.edge_index[1, i].item()
        G.add_edge(u, v, weight=topo.edge_attr[i, 1].item())
    try:
        path = nx.dijkstra_path(G, current_node, target_node)
        return path[1] if len(path) > 1 else current_node
    except nx.NetworkXNoPath:
        return current_node


def evaluate_one(agv_agent, gnn_agent, routing_policy, agv_policy, config, seeds):
    tsn_env = TSN_GNN_Env()
    agv_env = AGVComplianceEnv()
    fixed_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    all_metrics = {
        'rtt': [], 'stress': [], 'error': [], 'jitter': [], 'stress_peaks': [],
        'collisions': 0, 'dead_ends': 0, 'total_packets': 0
    }

    for seed in seeds:
        rng_tsn = np.random.RandomState(seed)
        rng_agv = np.random.RandomState(seed + 1000)
        torch.manual_seed(seed)
        tsn_obs, current_node, action_mask = tsn_env.reset()
        agv_env.reset(seed=seed)

        for i in range(tsn_env.topo.num_edges):
            if rng_tsn.rand() < BG_PROB:
                tsn_env.gantt.check_and_add_slot(i, rng_tsn.uniform(0, 300), rng_tsn.uniform(100, 400))

        last_delay = 0
        ep_peak_stress = 0.0

        while agv_env.step_count < MAX_PHYSICS_STEPS:
            if routing_policy == 'gnn':
                with torch.no_grad():
                    h = gnn_agent.encode(tsn_obs.to(DEVICE))
                    logits = gnn_agent.get_routing_logits(h, current_node, tsn_env.target_node, action_mask.to(DEVICE))
                    next_node = torch.argmax(logits).item()
                    edge_idx = tsn_env._get_edge_idx(current_node, next_node)
                    edge_attr = tsn_env.topo.edge_attr[edge_idx].to(DEVICE)
                    out = gnn_agent.scheduling_head(torch.cat([h[current_node], h[next_node], edge_attr]))
                    t_offset = torch.sigmoid(out[0]).item()
            elif routing_policy == 'shortest_path':
                next_node = get_shortest_path_next_node(tsn_env.topo, current_node, tsn_env.target_node)
                if not action_mask[next_node]:
                    valid_nodes = torch.where(action_mask)[0].tolist()
                    next_node = valid_nodes[0] if valid_nodes else current_node
                t_offset = 0.5
            else:
                raise ValueError(f"Unknown routing policy: {routing_policy}")

            agv_pos = agv_env.sim_engine.x_s
            next_tsn_obs, next_current_node, next_mask, _, tsn_done, _, info = tsn_env.step(
                next_node, t_offset, agv_x=agv_pos)

            status = info.get('status', 'success')
            all_metrics['total_packets'] += 1
            if status == 'success':
                rtt_ms = (info.get('total_delay', 1000.0) / 1000.0) + rng_tsn.uniform(*JITTER_RANGE)
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
                elif agv_policy == 'fixed':
                    action = fixed_action
                else:
                    raise ValueError(f"Unknown AGV policy: {agv_policy}")

                action = _apply_action_config(action, agv_env, config)

                _, _, a_term, a_trunc, a_info = agv_env.step(action)

                stress = abs(a_info['F_ext'])
                ep_peak_stress = max(ep_peak_stress, stress)
                all_metrics['stress'].append(stress)
                all_metrics['error'].append(abs(a_info['error']))

                if a_term or a_trunc:
                    break

            all_metrics['rtt'].append(rtt_ms)
            all_metrics['jitter'].append(jitter)

            if tsn_done:
                tsn_obs, current_node, action_mask = tsn_env.reset()
                for i in range(tsn_env.topo.num_edges):
                    if rng_tsn.rand() < BG_PROB:
                        tsn_env.gantt.check_and_add_slot(i, rng_tsn.uniform(0, 300), rng_tsn.uniform(100, 400))
            else:
                tsn_obs, current_node, action_mask = next_tsn_obs, next_current_node, next_mask

            if agv_env.step_count >= MAX_PHYSICS_STEPS:
                break

        all_metrics['stress_peaks'].append(ep_peak_stress)

    return all_metrics


def _apply_action_config(action, agv_env, config):
    """Apply proposal-specific action modifications."""
    if config is None:
        return action

    # --- Proposal 2: Momentum Action Smoothing ---
    if 'momentum_max_delta' in config:
        delta = config['momentum_max_delta']
        prev = agv_env.prev_action
        action = np.clip(action, prev - delta, prev + delta)

    # --- Proposal 1: Risk-Sensitive Reward ---
    if 'risk_threshold' in config:
        pass  # reward is computed inside env.step, handled by env modification

    return action


def run_proposal(method_id, method_name, routing, agv_policy, config, seeds):
    agv_agent, gnn_agent = load_models()
    r = evaluate_one(agv_agent, gnn_agent, routing, agv_policy, config, seeds)
    return {
        'id': method_id,
        'name': method_name,
        'RTT_Mean': np.mean(r['rtt']) if r['rtt'] else 0,
        'RTT_P95': np.percentile(r['rtt'], 95) if r['rtt'] else 0,
        'Jitter': np.mean(r['jitter']) if r['jitter'] else 0,
        'Collision%': r['collisions'] / r['total_packets'] * 100 if r['total_packets'] else 0,
        'Stress_Mean': np.mean(r['stress']) if r['stress'] else 0,
        'Stress_P95': np.percentile(r['stress'], 95) if r['stress'] else 0,
        'Stress_Peak': np.mean(r['stress_peaks']) if r['stress_peaks'] else 0,
        'Error_RMSE': np.sqrt(np.mean(np.array(r['error'])**2)) if r['error'] else 0,
        'raw_rtt': r['rtt'], 'raw_stress': r['stress'],
        'raw_stress_peaks': r['stress_peaks'], 'raw_error': r['error'],
    }


def head_to_head(baselines, candidate, alpha=0.05):
    """Compare candidate (M7) vs M1 and M5 on stress_peaks."""
    print(f"\n  === {candidate['id']} ({candidate['name']}) vs Baselines ===")
    summary = {}
    for label, b in [('M1', baselines[0]), ('M5', baselines[1])]:
        _, p = stats.ttest_ind(candidate['raw_stress_peaks'], b['raw_stress_peaks'], equal_var=False)
        c_mean = np.mean(candidate['raw_stress_peaks'])
        b_mean = np.mean(b['raw_stress_peaks'])
        d = abs(c_mean - b_mean) / max(np.std(candidate['raw_stress_peaks'], ddof=1), 1)
        better = "BETTER" if c_mean < b_mean else "worse"
        sig = "SIGNIFICANT" if p < alpha else "not sig"
        print(f"    vs {label}: {candidate['id']} peak={c_mean:.0f}N {label}={b_mean:.0f}N | {better} | p={p:.4f} {sig} d={d:.2f}")
        summary[label] = {'p': float(p), 'd': float(d), 'better': c_mean < b_mean}

    print(f"\n  {candidate['id']} full metrics:")
    for key in ['RTT_Mean', 'Jitter', 'Collision%', 'Stress_Mean', 'Stress_P95', 'Stress_Peak', 'Error_RMSE']:
        c_val = candidate[key]
        m1_val = baselines[0][key]
        m5_val = baselines[1][key]
        gap1 = (c_val - m1_val) / abs(m1_val) * 100 if m1_val else 0
        gap5 = (c_val - m5_val) / abs(m5_val) * 100 if m5_val else 0
        tag1 = "M1_better" if gap1 > 0 else "M7_better"
        tag5 = "M5_better" if gap5 > 0 else "M7_better"
        print(f"    {key:15s}: M7={c_val:10.2f} | vs M1 {gap1:+5.1f}% {tag1:10s} | vs M5 {gap5:+5.1f}% {tag5:10s}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='JSON config for M7')
    parser.add_argument('--baseline-only', action='store_true', help='Only run M1+M5 baseline')
    args = parser.parse_args()

    seeds = SEEDS
    print(f"Device: {DEVICE}, Seeds: {len(seeds)}")
    print(f"BG prob: {BG_PROB}, Jitter: {JITTER_RANGE}")

    # Always run M1 + M5 baseline
    print("\n--- M1 (Ours Full) ---")
    m1 = run_proposal('M1', 'Ours (Full)', 'gnn', 'rl', None, seeds)
    print("\n--- M5 (GNN+Fixed) ---")
    m5 = run_proposal('M5', 'GNN+Fixed', 'gnn', 'fixed', None, seeds)
    baselines = [m1, m5]

    if args.baseline_only:
        return

    if args.config:
        cfg = json.loads(args.config)
    else:
        cfg = {}

    print(f"\n--- M7 ({cfg.get('name', 'Proposal')}) ---")
    m7 = run_proposal('M7', cfg.get('name', 'Proposal'), 'gnn', 'rl', cfg, seeds)
    result = head_to_head(baselines, m7)

    print("\n" + "=" * 60)
    print("  DECISION:")
    vs_m1 = result['M1']['better'] and result['M1']['p'] < 0.1
    vs_m5 = result['M5']['better'] and result['M5']['p'] < 0.1
    if vs_m1 and vs_m5:
        print("  ACCEPT: M7 significantly better than both M1 and M5")
    elif vs_m5 and not vs_m1:
        print("  WEAK ACCEPT: M7 better than M5 but not M1 — may still be worth merging")
    elif vs_m1 and not vs_m5:
        print("  PARTIAL: M7 better than M1 but not M5 — investigate further")
    else:
        print("  REJECT: M7 does not improve over baselines")
    print("=" * 60)


if __name__ == "__main__":
    main()
