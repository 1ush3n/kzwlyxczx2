"""论文表 C-2: 逐 Episode 详细统计导出
对 M1 最终模型跑 10 个评估 episode，输出每 ep 的 RTT / Stress / Error / Jitter / Status

用法: python tests/export_episode_stats.py
输出: images/episode_stats_table.csv, images/episode_stats.tex
"""
import os, sys, torch, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.agv_compliance_env import AGVComplianceEnv
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from agent.gnn_actor_critic import GNNActorCritic
from stable_baselines3 import PPO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 10
MAX_STEPS = 200


def main():
    base = os.path.dirname(os.path.dirname(__file__))
    agv_path = os.path.join(base, "checkpoints", "phase3_cotrain", "ppo_agv_final_aligned.zip")
    gnn_path = os.path.join(base, "checkpoints", "phase3_cotrain", "ppo_gnn_final_aligned.pth")

    print("Loading models...")
    agv_agent = PPO.load(agv_path, device=DEVICE)
    gnn_agent = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(DEVICE)
    gnn_agent.load_state_dict(torch.load(gnn_path, map_location=DEVICE))
    gnn_agent.eval()

    rows = []
    for ep in range(NUM_EPISODES):
        seed = ep * 10 + 42
        tsn_env = TSN_GNN_Env()
        agv_env = AGVComplianceEnv()
        rng_tsn = np.random.RandomState(seed)
        torch.manual_seed(seed)
        tsn_obs, curr_node, am = tsn_env.reset()
        agv_env.reset(seed=seed)

        for i in range(tsn_env.topo.num_edges):
            if rng_tsn.rand() < 0.3:
                tsn_env.gantt.check_and_add_slot(i, rng_tsn.uniform(0, 300), rng_tsn.uniform(100, 400))

        ep_rtts = []; ep_stresses = []; ep_errors = []; ep_jitters = []
        last_delay = 0; peak_stress = 0; collisions = 0

        while agv_env.step_count < MAX_STEPS:
            with torch.no_grad():
                h = gnn_agent.encode(tsn_obs.to(DEVICE))
                logits = gnn_agent.get_routing_logits(h, curr_node, tsn_env.target_node, am.to(DEVICE))
                next_node = torch.argmax(logits).item()
                edge_idx = tsn_env._get_edge_idx(curr_node, next_node)
                edge_attr = tsn_env.topo.edge_attr[edge_idx].to(DEVICE)
                out = gnn_agent.scheduling_head(torch.cat([h[curr_node], h[next_node], edge_attr]))
                t_offset = torch.sigmoid(out[0]).item()

            agv_pos = agv_env.sim_engine.x_s
            no, nc, nm, _, td, _, info = tsn_env.step(next_node, t_offset, agv_x=agv_pos)

            status = info.get('status', 'success')
            if status == 'success':
                rtt_ms = (info.get('total_delay', 1000) / 1000) + rng_tsn.uniform(1, 10)
            elif status == 'collision':
                collisions += 1; rtt_ms = 500
            else:
                rtt_ms = 500

            rtt_sec = rtt_ms / 1000
            jitter = abs(rtt_ms - last_delay)
            last_delay = rtt_ms
            agv_env.plc.inject_tsn_delay(rtt_sec)

            num_steps = int(np.ceil(rtt_sec / agv_env.config['physics']['dt']))
            num_steps = max(1, min(num_steps, 50))

            for _ in range(num_steps):
                if np.random.RandomState(seed + agv_env.step_count).rand() < 0.01:
                    agv_env.sim_engine.x_s += 0.05
                obs = np.array(agv_env.obs_buffer, dtype=np.float32)
                act, _ = agv_agent.predict(obs.reshape(1, *obs.shape), deterministic=True)
                act = act.flatten()
                _, _, a_term, a_trunc, ai = agv_env.step(act)
                s = abs(ai['F_ext']); peak_stress = max(peak_stress, s)
                ep_stresses.append(s); ep_errors.append(abs(ai['error']))
                if a_term or a_trunc:
                    break

            ep_rtts.append(rtt_ms); ep_jitters.append(jitter)

            if td:
                tsn_obs, curr_node, am = tsn_env.reset()
                for i in range(tsn_env.topo.num_edges):
                    if rng_tsn.rand() < 0.3:
                        tsn_env.gantt.check_and_add_slot(i, rng_tsn.uniform(0, 300), rng_tsn.uniform(100, 400))
            else:
                tsn_obs, curr_node, am = no, nc, nm
            if agv_env.step_count >= MAX_STEPS:
                break

        rows.append({
            'Ep#': ep + 1,
            'RTT(ms)': round(np.mean(ep_rtts), 2) if ep_rtts else 0,
            'Stress(N)': round(np.mean(ep_stresses), 1) if ep_stresses else 0,
            'PeakStress(N)': round(peak_stress, 1),
            'Error(m)': round(np.mean(ep_errors), 3) if ep_errors else 0,
            'Jitter(ms)': round(np.mean(ep_jitters), 2) if ep_jitters else 0,
            'Collisions': collisions,
            'Status': 'success' if collisions == 0 else 'partial',
        })
        print(f"  - Episode {ep+1}/{NUM_EPISODES}: RTT={rows[-1]['RTT(ms)']}ms, Peak={rows[-1]['PeakStress(N)']}N")

    df = pd.DataFrame(rows)
    out_dir = os.path.join(base, "images")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "episode_stats_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    # Summary row
    print(f"\n--- Summary ---")
    print(f"  RTT(ms)     Mean={df['RTT(ms)'].mean():.2f}  P95={df['RTT(ms)'].quantile(0.95):.2f}")
    print(f"  Stress(N)   Mean={df['Stress(N)'].mean():.1f}  P95={df['Stress(N)'].quantile(0.95):.1f}")
    print(f"  PeakStress  Mean={df['PeakStress(N)'].mean():.1f}  Max={df['PeakStress(N)'].max():.1f}")
    print(f"  Error(m)    Mean={df['Error(m)'].mean():.4f}")
    print(f"  Jitter(ms)  Mean={df['Jitter(ms)'].mean():.2f}  P95={df['Jitter(ms)'].quantile(0.95):.2f}")

    # Also save LaTeX
    tex_path = os.path.join(out_dir, "episode_stats.tex")
    styled = df.copy()
    for c in ['RTT(ms)', 'Jitter(ms)']:
        styled[c] = styled[c].apply(lambda x: f"{x:.2f}")
    for c in ['Stress(N)', 'PeakStress(N)']:
        styled[c] = styled[c].apply(lambda x: f"{x:.1f}")
    styled['Error(m)'] = styled['Error(m)'].apply(lambda x: f"{x:.3f}")
    with open(tex_path, 'w') as f:
        f.write(styled.to_latex(index=False, escape=False))
    print(f"LaTeX saved to: {tex_path}")


if __name__ == "__main__":
    main()
