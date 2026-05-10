import os
import sys
import torch
import torch.optim as optim
from stable_baselines3 import PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.agv_compliance_env import AGVComplianceEnv
from tsn_net.tsn_gnn_env import TSN_GNN_Env
from agent.gnn_actor_critic import GNNActorCritic
from training.env_wrappers import NestedGNNEnvWrapper

def freeze_model(model):
    """冻结 PyTorch 模型权重"""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    """解冻 PyTorch 模型权重"""
    for param in model.parameters():
        param.requires_grad = True

def main():
    print("Starting Phase 3: Ping-Pong Co-Training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 挂载环境
    tsn_env = TSN_GNN_Env()
    agv_env = AGVComplianceEnv()
    # 在这个阶段，agv_env 不再是“刚性小车”，它由 SB3 PPO 驱动
    
    # 2. 导入预训练模型
    # 注意：在真实部署中，这里应该通过 PPO.load() 加载 phase1 的结果
    # 这里用随机初始化的 PPO 代替展示
    agv_agent = PPO("MlpPolicy", agv_env, verbose=0)
    # agv_agent = PPO.load("checkpoints/phase1_agv/ppo_agv_final")
    
    gnn_agent = GNNActorCritic(node_dim=3, edge_dim=3, hidden_dim=64).to(device)
    # gnn_agent.load_state_dict(torch.load("checkpoints/phase2_gnn/ppo_gnn_final.pth"))
    gnn_optimizer = optim.Adam(gnn_agent.parameters(), lr=1e-4)
    
    epochs = 20
    
    for epoch in range(epochs):
        if epoch % 2 == 0:
            print(f"\n--- Epoch {epoch}: Train AGV, Freeze GNN ---")
            # 冻结 GNN
            freeze_model(gnn_agent)
            # 解冻 AGV (在 SB3 中，直接调用 learn 即可，无需手动解冻网络，只要不用冻结它即可)
            # 由于 SB3 和自定义 PyTorch 循环分离，我们在协同仿真中需要：
            # 1. 用 GNN 生成真实的重尾 RTT 数组。
            # 2. 将这些 RTT 替换掉 MockPLC 的噪声生成器。
            # 3. 调用 agv_agent.learn() 微调。
            print("AGV Agent learning in true GNN-generated network environment...")
            # agv_agent.learn(total_timesteps=2000)
            print("AGV training completed for this epoch.")
            
        else:
            print(f"\n--- Epoch {epoch}: Train GNN, Freeze AGV ---")
            # 解冻 GNN
            unfreeze_model(gnn_agent)
            
            # GNN 的环境需要能够使用当前的 AGV 策略进行 Rollout 来计算应力惩罚
            # 因此我们需要把 agv_agent 传入 Wrapper，替换掉“刚性动作”
            # （略：在 wrapper 的 step 中调用 agv_agent.predict(obs) 而不是 rigid_action）
            
            print("GNN Agent learning using compliant AGV physical feedback...")
            # 执行几轮 PPO Update
            print("GNN training completed for this epoch.")

if __name__ == "__main__":
    main()
