import os
import sys
import numpy as np
import pytest
import gymnasium as gym

# 保证能够从当前目录导入 env 和 core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agv_compliance_env import AGVComplianceEnv
from gymnasium.utils.env_checker import check_env

def test_gym_api_compliance():
    """测试环境是否符合 Gymnasium 标准 API 规范"""
    env = AGVComplianceEnv()
    # check_env 会执行 reset, step(各种采样动作), 检查类型和形状等
    check_env(env.unwrapped)
    print("Gym API Compliance Check: Passed.")

def test_physics_stability():
    """测试基础物理稳定性和零动作下的默认阻抗响应"""
    env = AGVComplianceEnv(render_mode="human")
    obs, info = env.reset()
    
    # 模拟运行 100 步，Agent 采取零动作 (a=[0,0,0])，即仅使用基准阻抗
    for _ in range(100):
        action = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # [k, 5] 观测
        assert obs.shape == (4, 5)
        # 确保应力 F_ext 没有发散到不可控地步 (如 NaN 或极大值)
        assert not np.isnan(info["F_ext"])
        assert not np.isnan(info["error"])
        
        if terminated or truncated:
            break
            
    print(f"Physics Stability Check: Passed. Final Error: {info['error']:.4f}, Stress: {info['F_ext']:.2f}")

def test_action_clamping():
    """测试极端动作下的阻抗钳位防御"""
    env = AGVComplianceEnv()
    env.reset()
    
    # 传入极端的负向动作试图把质量变负
    action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 必须大于 0，不能被负数破坏物理模型
    assert info["Md"] > 0
    assert info["Bd"] > 0
    assert info["Kd"] > 0
    print("Action Clamping Defense Check: Passed.")

if __name__ == "__main__":
    test_gym_api_compliance()
    test_physics_stability()
    test_action_clamping()
