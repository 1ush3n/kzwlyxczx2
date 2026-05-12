import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.agv_compliance_env import AGVComplianceEnv
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from agent.gnn_actor_critic import GNNActorCritic
from training.env_wrappers import NestedGNNEnvWrapper, GNNDelayAGVWrapper

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def main():
    print("Starting Phase 3: Ping-Pong Co-Training (Final Alignment)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化环境
    tsn_env = TSN_GNN_Env()
    agv_env = AGVComplianceEnv()
    
    # 2. 加载预训练模型
    agv_model_path = os.path.join("checkpoints", "phase1_agv", "ppo_agv_final.zip")
    if os.path.exists(agv_model_path):
        print(f"Loading Phase 1 AGV Agent from {agv_model_path}")
        agv_agent = PPO.load(agv_model_path, env=agv_env, device=device)
    else:
        print("Warning: Phase 1 model not found, starting AGV from scratch.")
        agv_agent = PPO("MlpPolicy", agv_env, verbose=0, device=device)
        
    gnn_agent = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(device)
    gnn_model_path = os.path.join("checkpoints", "phase2_gnn", "ppo_gnn_final.pth")
    if os.path.exists(gnn_model_path):
        print(f"Loading Phase 2 GNN Agent from {gnn_model_path}")
        gnn_agent.load_state_dict(torch.load(gnn_model_path))
    
    # 初始化包装器
    env = NestedGNNEnvWrapper(tsn_env, agv_env, agv_agent=agv_agent)
    
    num_cycles = 15 # 增加乒乓大循环次数以确保深度收敛
    steps_per_agv_cycle = 10000
    episodes_per_gnn_cycle = 30
    
    gnn_optimizer = optim.Adam(gnn_agent.parameters(), lr=5e-5) # 协同训练使用较小的学习率
    # CosineAnnealing LR decay across all 15 cycles * 30 episodes = 450 steps
    gnn_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        gnn_optimizer, T_max=num_cycles * episodes_per_gnn_cycle, eta_min=1e-6
    )
    
    for cycle in range(num_cycles):
        print(f"\n===== Co-Training Cycle {cycle+1}/{num_cycles} =====")
        
        # --- Step A: 冻结 GNN, 训练 AGV (使用 GNN 产生的真实延迟) ---
        print("--- Step A: Training AGV (GNN is Frozen, using GNN delays) ---")
        # CRITICAL FIX: 使用 GNNDelayAGVWrapper 让 AGV 针对当前 GNN 的延迟分布微调
        # 而非使用随机标准延迟
        agv_env = AGVComplianceEnv()
        agv_wrapped = GNNDelayAGVWrapper(agv_env, tsn_env, gnn_agent, device)
        # 将 agent 重新绑定到 wrapped 环境
        agv_agent.set_env(agv_wrapped)
        agv_agent.learn(total_timesteps=steps_per_agv_cycle)
        # 恢复原始环境引用
        agv_agent.set_env(env.agv_env)
        print(f"AGV fine-tuning complete with GNN delays.")
        
        # --- Step B: 冻结 AGV, 训练 GNN ---
        print("--- Step B: Training GNN (AGV is Frozen) ---")
        # 更新 Wrapper 中的 AGV Agent 以便获取其最新的柔顺反馈
        env.agv_agent = agv_agent
        
        for ep in range(episodes_per_gnn_cycle):
            obs, current_node, action_mask = env.reset()
            log_probs, values, rewards, masks = [], [], [], []
            terminated = truncated = False
            
            while not (terminated or truncated):
                # GNN 决策
                with torch.set_grad_enabled(True):
                    h = gnn_agent.encode(obs.to(device))
                    target_node = env.tsn_env.target_node
                    logits = gnn_agent.get_routing_logits(h, current_node, tsn_env.target_node, action_mask.to(device))
                    
                    if torch.isnan(logits).any():
                        print(f"Warning: NaN logits in Phase 3 GNN training, terminating episode.")
                        terminated = True
                        break
                    
                    routing_dist = torch.distributions.Categorical(logits=logits)
                    next_node = routing_dist.sample()
                    
                    edge_idx = env.tsn_env._get_edge_idx(current_node, next_node.item())
                    edge_attr = env.tsn_env.topo.edge_attr[edge_idx].to(device)
                    sched_dist = gnn_agent.get_scheduling_dist(h, current_node, next_node.item(), edge_attr)
                    t_offset = sched_dist.sample()
                    
                    value = gnn_agent.get_value(h)
                    
                    # 记录概率用于更新
                    total_log_prob = routing_dist.log_prob(next_node) + sched_dist.log_prob(t_offset)
                
                # 环境步进 (这里会调用 agv_agent.predict)
                next_obs, next_current_node, next_mask, reward, terminated, truncated, info = env.step(next_node.item(), t_offset.item())
                
                log_probs.append(total_log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append(1.0 - float(terminated))
                
                obs, current_node, action_mask = next_obs, next_current_node, next_mask
            
            # GNN PPO 更新 (单次梯度下降以保持稳定性)
            if len(rewards) > 0:
                with torch.no_grad():
                    next_val = gnn_agent.get_value(gnn_agent.encode(obs.to(device))).detach()
                    returns = compute_gae(next_val, rewards, masks, [v.detach() for v in values])
                    returns = torch.tensor(returns).to(device)
                    old_values = torch.cat(values).detach().squeeze()
                    old_log_probs = torch.stack(log_probs).detach()
                    advantages = returns - old_values
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 重新计算一次 loss 并反向传播
                ratio = torch.exp(torch.clamp(torch.stack(log_probs) - old_log_probs, -20, 20))
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(returns, torch.cat(values).squeeze())
                
                loss = actor_loss + 0.5 * critic_loss
                
                if torch.isnan(loss):
                    print(f"Warning: NaN loss in Phase 3 GNN training, skipping update.")
                    gnn_optimizer.zero_grad()
                else:
                    gnn_optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(gnn_agent.parameters(), 0.5)
                    gnn_optimizer.step()
                
            if ep % 5 == 0:
                print(f"GNN Episode {ep} - Reward: {sum(rewards):.2f}, Peak Stress: {info.get('peak_stress', 0):.1f}N")
            
            gnn_scheduler.step()  # CosineAnnealing LR decay per GNN episode
            
    # Phase 3 cycle complete, report LR
    print(f"[LR] Cycle {cycle+1}/{num_cycles} complete: gnn_lr = {gnn_scheduler.get_last_lr()[0]:.2e}")
                
    # 最终保存
    save_dir = os.path.join("checkpoints", "phase3_cotrain")
    os.makedirs(save_dir, exist_ok=True)
    agv_agent.save(os.path.join(save_dir, "ppo_agv_final_aligned"))
    torch.save(gnn_agent.state_dict(), os.path.join(save_dir, "ppo_gnn_final_aligned.pth"))
    print("\nPhase 3 Co-Training completed! All agents are now physically and network-aligned.")

if __name__ == "__main__":
    main()
