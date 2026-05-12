import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.gnn_actor_critic import GNNActorCritic
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from env.agv_compliance_env import AGVComplianceEnv
from training.env_wrappers import NestedGNNEnvWrapper

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
    print("Starting Phase 2: GNN Pessimistic Topology Pre-training")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化嵌套环境
    tsn_env = TSN_GNN_Env()
    agv_env = AGVComplianceEnv()
    env = NestedGNNEnvWrapper(tsn_env, agv_env)
    
    # 初始化 GNN Agent
    agent = GNNActorCritic(node_dim=3, edge_dim=4, hidden_dim=64).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=5e-5)
    
    # 学习率调度器：CosineAnnealing 从 lr 到 lr*0.1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=2000, eta_min=5e-6
    )
    
    max_episodes = 2000
    clip_param = 0.2
    ppo_epochs = 4
    
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "phase2_gnn")
    os.makedirs(save_dir, exist_ok=True)
    
    # 简单的 PPO 循环演示
    for episode in range(max_episodes):
        obs, current_node, action_mask = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        masks = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            obs = obs.to(device)
            action_mask = action_mask.to(device)
            
            # 1. 编码图特征
            h = agent.encode(obs)
            
            # 2. 路由预测
            # 我们需要给 get_routing_logits 传一个目标节点，因为环境知道 target_node 是 9
            target_node = env.tsn_env.target_node
            logits = agent.get_routing_logits(h, current_node, target_node, action_mask)
            
            if torch.isnan(logits).any():
                print(f"Warning: NaN logits at episode {episode}, terminating.")
                terminated = True
                break
                
            routing_dist = torch.distributions.Categorical(logits=logits)
            next_node = routing_dist.sample()
            
            # 3. 提取选定边的特征进行时间调度预测
            # 这里简化处理：我们用环境的边属性直接喂给预测头
            edge_idx = env.tsn_env._get_edge_idx(current_node, next_node.item())
            edge_attr = env.tsn_env.topo.edge_attr[edge_idx].to(device)
            
            sched_dist = agent.get_scheduling_dist(h, current_node, next_node.item(), edge_attr)
            t_offset = sched_dist.sample()
            
            # 4. 价值评估
            value = agent.get_value(h)
            
            # 环境步进
            next_obs, next_current_node, next_mask, reward, terminated, truncated, info = env.step(next_node.item(), t_offset.item())
            
            # 收集 Log Prob (包含两部分的联合概率)
            total_log_prob = routing_dist.log_prob(next_node) + sched_dist.log_prob(t_offset)
            
            log_probs.append(total_log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1.0 - float(terminated))
            
            obs = next_obs
            current_node = next_current_node
            action_mask = next_mask
            
        # Episode 结束，准备数据
        if len(rewards) == 0: continue
        
        if episode % 10 == 0:
            print(f"Episode {episode} finished with status: {info.get('status', 'unknown')}, Reward: {sum(rewards):.2f}, Peak Stress Penalty: {info.get('stress_penalty', 0):.2f}")
            
        # 将张量脱离计算图准备计算优势
        with torch.no_grad():
            next_val = agent.get_value(agent.encode(obs.to(device))).detach()
            returns = compute_gae(next_val, rewards, masks, [v.detach() for v in values])
            returns = torch.tensor(returns).to(device)
            old_values = torch.cat(values).detach().squeeze()
            old_log_probs = torch.stack(log_probs).detach()
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update (简单版本：每回合执行一次梯度更新以保证稳定)
        ppo_epochs_to_run = 1 
        
        for _ in range(ppo_epochs_to_run):
            log_probs_tensor = torch.stack(log_probs)
            ratio = torch.exp(torch.clamp(log_probs_tensor - old_log_probs, -20, 20))
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            new_values = torch.cat(values).squeeze()
            if returns.dim() != new_values.dim():
                returns = returns.reshape_as(new_values)
            critic_loss = F.mse_loss(returns, new_values)
            
            loss = actor_loss + 0.5 * critic_loss
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at episode {episode}, skipping update.")
                optimizer.zero_grad()
                break
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
        scheduler.step()  # CosineAnnealing LR decay per epsiode
        
        if episode % 100 == 0:
            print(f"[LR] Episode {episode}: lr = {scheduler.get_last_lr()[0]:.2e}")
            
    torch.save(agent.state_dict(), os.path.join(save_dir, "ppo_gnn_final.pth"))
    print("Phase 2 training script structure completed.")

if __name__ == "__main__":
    main()
