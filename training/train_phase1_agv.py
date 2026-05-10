import os
import sys
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.agv_compliance_env import AGVComplianceEnv

def main():
    print("Starting Phase 1: AGV Domain Randomization (Extreme Network Delays)")
    
    # 强制修改环境配置以启用 extreme_pareto 模式
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'agv_env_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 临时覆盖配置，仅供训练时使用
    if 'comms' not in config:
        config['comms'] = {}
    config['comms']['delay_mode'] = 'extreme_pareto'
    
    # 定义环境构造函数
    def make_env():
        # 传入覆盖后的配置 (如果有接口的话)
        # 当前 AGVComplianceEnv 在 __init__ 中读取 yaml，我们可以通过直接修改类实例的 config 或通过参数传入
        # 为了兼容性，我们在环境里稍微绕一下，创建一个子类或包装器，由于是在 Python 中，可以直接传入 config 字典 (如果环境支持)
        # 如果不支持，我们可以通过全局或者重写 config
        env = AGVComplianceEnv()
        env.config = config
        env.plc.delay_mode = 'extreme_pareto' # 强制注入模式
        return Monitor(env)

    # 创建矢量化环境
    vec_env = make_vec_env(make_env, n_envs=4)
    
    eval_env = make_env()
    
    # 设置模型保存路径
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "phase1_agv")
    os.makedirs(save_dir, exist_ok=True)
    
    # EvalCallback 在训练过程中定期评估模型并保存最佳权重
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_dir,
                                 log_path=save_dir, eval_freq=2000,
                                 deterministic=True, render=False)
                                 
    # 配置 PPO
    model = PPO("MlpPolicy", vec_env, verbose=1, 
                learning_rate=3e-4, 
                n_steps=2048, 
                batch_size=64, 
                n_epochs=10, 
                gamma=0.99, 
                gae_lambda=0.95, 
                clip_range=0.2, 
                tensorboard_log="./runs/phase1_agv_tensorboard/")

    # 开始训练
    print("Training started... Check tensorboard using `tensorboard --logdir ./runs/`")
    model.learn(total_timesteps=100000, callback=eval_callback)
    
    # 保存最终模型
    model.save(os.path.join(save_dir, "ppo_agv_final"))
    print("Phase 1 training completed successfully.")

if __name__ == "__main__":
    main()
