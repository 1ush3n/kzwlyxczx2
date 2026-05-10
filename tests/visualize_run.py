import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 保证能够从当前目录导入 env 和 core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agv_compliance_env import AGVComplianceEnv

def run_and_visualize():
    """运行环境并生成可视化图表"""
    env = AGVComplianceEnv()
    obs, info = env.reset(seed=42)
    
    steps = 200
    
    # 记录数据
    history_e = []
    history_F_ext = []
    history_rtt = []
    history_Md = []
    history_Bd = []
    history_Kd = []
    history_reward = []
    
    for i in range(steps):
        # 模拟一个随时间变化的阻抗调节动作 (正弦波测试)
        action = np.array([
            0.5 * np.sin(i * 0.1),
            0.5 * np.cos(i * 0.1),
            -0.5 * np.sin(i * 0.05)
        ], dtype=np.float32)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        history_e.append(info["error"])
        history_F_ext.append(info["F_ext"])
        # RTT 可以从单步观测的最后一帧中获取 (索引3是 tau)
        history_rtt.append(obs[-1][3])
        history_Md.append(info["Md"])
        history_Bd.append(info["Bd"])
        history_Kd.append(info["Kd"])
        history_reward.append(reward)
        
        if terminated or truncated:
            break

    # 开始绘图
    time_axis = np.arange(len(history_e)) * env.config['physics']['dt']
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('AGV Compliance Control Simulation Results', fontsize=16)
    
    # 1. 跟踪误差
    axs[0].plot(time_axis, history_e, 'b-', label='Tracking Error (e)')
    axs[0].set_ylabel('Error [m]')
    axs[0].grid(True)
    axs[0].legend()
    
    # 2. 机翼受力
    axs[1].plot(time_axis, history_F_ext, 'r-', label='Wing Stress (F_ext)')
    axs[1].set_ylabel('Stress [N]')
    axs[1].grid(True)
    axs[1].legend()
    
    # 3. 阻抗参数变化
    axs[2].plot(time_axis, history_Md, label='M_d (Mass)')
    axs[2].plot(time_axis, history_Bd, label='B_d (Damping)')
    axs[2].plot(time_axis, history_Kd, label='K_d (Stiffness)')
    axs[2].set_ylabel('Impedance Params')
    axs[2].grid(True)
    axs[2].legend()
    
    # 4. 网络延迟与单步奖励
    ax4_2 = axs[3].twinx()
    axs[3].plot(time_axis, history_rtt, 'k-', label='Network RTT')
    axs[3].set_ylabel('RTT [s]')
    ax4_2.plot(time_axis, history_reward, 'g--', alpha=0.6, label='Step Reward')
    ax4_2.set_ylabel('Reward')
    
    # 合并图例
    lines, labels = axs[3].get_legend_handles_labels()
    lines2, labels2 = ax4_2.get_legend_handles_labels()
    axs[3].legend(lines + lines2, labels + labels2, loc='upper left')
    axs[3].grid(True)
    axs[3].set_xlabel('Time [s]')
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'simulation_results.png')
    plt.savefig(save_path, dpi=300)
    print(f"可视化结果已保存至: {save_path}")
    
if __name__ == "__main__":
    run_and_visualize()
