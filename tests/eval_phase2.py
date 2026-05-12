"""Phase 2 GNN 独立评估脚本
评估训练后的 GNN 智能体在路由任务上的独立表现（不含 AGV 柔顺控制介入）。
输出：路由成功率、碰撞率、死胡同率、平均延迟。

用法: python tests/eval_phase2.py [--model checkpoints/phase2_gnn/ppo_gnn_final.pth] [--episodes 500]
"""
import os, sys, torch, numpy as np, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from agent.gnn_actor_critic import GNNActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_path: str, num_episodes: int = 500):
    tsn_env = TSN_GNN_Env()
    agent = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(DEVICE)
    agent.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.eval()

    successes = 0; collisions = 0; dead_ends = 0
    total_delay = []; total_steps = []
    outcomes = []

    for ep in range(num_episodes):
        obs, current_node, action_mask = tsn_env.reset()
        # 注入背景流量模拟训练条件
        for i in range(tsn_env.topo.num_edges):
            if np.random.rand() < 0.3:
                tsn_env.gantt.check_and_add_slot(
                    i, np.random.uniform(0, 300), np.random.uniform(50, 400))

        done = False; steps = 0
        while not done:
            with torch.no_grad():
                h = agent.encode(obs.to(DEVICE))
                logits = agent.get_routing_logits(
                    h, current_node, tsn_env.target_node, action_mask.to(DEVICE))
                next_node = torch.argmax(logits).item()
                edge_idx = tsn_env._get_edge_idx(current_node, next_node)
                edge_attr = tsn_env.topo.edge_attr[edge_idx].to(DEVICE)
                out = agent.scheduling_head(
                    torch.cat([h[current_node], h[next_node], edge_attr]))
                t_offset = torch.sigmoid(out[0]).item()

            agv_x = np.random.uniform(0, 20)
            obs, current_node, action_mask, _, done, _, info = tsn_env.step(
                next_node, t_offset, agv_x=agv_x)
            steps += 1

        status = info.get('status', 'unknown')
        outcomes.append(status)
        if status == 'success':
            successes += 1
            total_delay.append(info.get('total_delay', 0))
        elif status == 'collision':
            collisions += 1
        else:
            dead_ends += 1
        total_steps.append(steps)

    total = num_episodes
    print(f"\n{'='*60}")
    print(f"  Phase 2 GNN Evaluation — {total} episodes")
    print(f"{'='*60}")
    print(f"  Success:   {successes:5d} / {total}  ({successes/total*100:.1f}%)")
    print(f"  Collision: {collisions:5d} / {total}  ({collisions/total*100:.1f}%)")
    print(f"  Dead End:  {dead_ends:5d} / {total}  ({dead_ends/total*100:.1f}%)")
    if total_delay:
        delays_us = np.array(total_delay)
        print(f"  Avg Delay (success): {np.mean(delays_us)/1000:.2f} ms")
        print(f"  P95 Delay (success): {np.percentile(delays_us,95)/1000:.2f} ms")
    print(f"  Avg Steps:  {np.mean(total_steps):.2f}")
    print(f"{'='*60}")

    # 保存 CSV
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "images", "phase2_eval_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("metric,value\n")
        f.write(f"total_episodes,{total}\n")
        f.write(f"success_rate(%),{successes/total*100:.2f}\n")
        f.write(f"collision_rate(%),{collisions/total*100:.2f}\n")
        f.write(f"dead_end_rate(%),{dead_ends/total*100:.2f}\n")
        if total_delay:
            f.write(f"avg_delay_ms,{np.mean(total_delay)/1000:.2f}\n")
            f.write(f"p95_delay_ms,{np.percentile(total_delay,95)/1000:.2f}\n")
        f.write(f"avg_steps,{np.mean(total_steps):.2f}\n")
    print(f"  Results saved to: {out_path}")
    return successes / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_model = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints", "phase2_gnn", "ppo_gnn_final.pth")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
