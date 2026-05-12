# AGV 主从协同控制与 TSN 网络调度强化学习系统

> **README Part 1/3** — 项目概述 · 技术栈 · 目录结构 · 代码文件详解

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 技术栈](#2-技术栈)
- [3. 完整目录结构](#3-完整目录结构)
- [4. 各代码文件功能描述与实现逻辑](#4-各代码文件功能描述与实现逻辑)
  - [4.1 物理仿真引擎 — `core/physics/agv_kinematics.py`](#41-物理仿真引擎--corephysicsagv_kinematicspy)
  - [4.2 通信接口层 — `core/comms/plc_interface.py`](#42-通信接口层--corecommsplc_interfacepy)
  - [4.3 TSN 拓扑管理器 — `tsn_net/topology.py`](#43-tsn-拓扑管理器--tsn_nettopologypy)
  - [4.4 TSN 甘特图调度器 — `tsn_net/gantt_chart.py`](#44-tsn-甘特图调度器--tsn_netgantt_chartpy)
  - [4.5 TSN-GNN 强化学习环境 — `tsn_net/tsn_gnn_env.py`](#45-tsn-gnn-强化学习环境--tsn_nettsn_gnn_envpy)
  - [4.6 GNN Actor-Critic 智能体 — `agent/gnn_actor_critic.py`](#46-gnn-actor-critic-智能体--agentgnn_actor_criticpy)
  - [4.7 环境包装器 — `training/env_wrappers.py`](#47-环境包装器--trainingenv_wrapperspy)
  - [4.8 第一阶段训练 — `training/train_phase1_agv.py`](#48-第一阶段训练--trainingtrain_phase1_agvpy)
  - [4.9 第二阶段训练 — `training/train_phase2_gnn.py`](#49-第二阶段训练--trainingtrain_phase2_gnnpy)
  - [4.10 第三阶段训练 — `training/train_phase3_cotrain.py`](#410-第三阶段训练--trainingtrain_phase3_cotrainpy)
  - [4.11 配置文件 — `config/`](#411-配置文件--config)
  - [4.12 测试与可视化 — `tests/`](#412-测试与可视化--tests)
  - [4.13 检查点与日志 — `checkpoints/` & `runs/`](#413-检查点与日志--checkpoints--runs)

---

## 1. 项目概述

本项目构建了一套**多层架构的强化学习控制系统**，用于解决工业 AGV（Automated Guided Vehicle）主从协同搬运场景中的两大核心问题：

1. **物理层的柔顺控制（Compliance Control）**：在多 AGV 协同搬运大型结构件（如飞机机翼）时，由于网络延迟、主从车位置失步，会在连接结构上产生巨大的内部应力。本系统采用基于 **Kelvin-Voigt 模型**的阻抗控制状态空间方程，通过 **PPO（Proximal Policy Optimization）** 强化学习智能体动态调节虚拟阻抗参数 (M, B, K)，实现在不确定网络条件下的柔顺减振。

2. **网络层的 TSN 调度与路由**：在 IEEE 802.1 TSN（Time-Sensitive Networking）车间网络中，数据流的端到端延迟和抖动直接影响物理层控制品质。本系统使用 **图神经网络（GATv2 + GNN）** 驱动的自回归 Actor-Critic 架构，同时学习路由选择和时隙调度，并与物理层形成闭环反馈——延迟导致的物理应力峰值会反向惩罚网络决策。

**三阶段课程训练策略**：
- **Phase 1（领域随机化）**：AGV 在极端延迟分布下独立训练柔顺策略
- **Phase 2（悲观拓扑预训）**：GNN 在刚性 AGV 物理反馈下预训练路由策略
- **Phase 3（乒乓协同训练）**：GNN 与 AGV 交替冻结/训练，实现最终联合对齐

---

## 2. 技术栈

| 层级 | 技术/框架 | 用途 |
|------|-----------|------|
| **深度学习框架** | PyTorch 2.x | GNN 模型构建与训练 |
| **图神经网络** | PyTorch Geometric (torch_geometric) | GATv2Conv 图卷积编码 |
| **强化学习框架** | Stable-Baselines3 (SB3) | PPO 算法实现与 AGV 策略训练 |
| **环境接口** | Gymnasium (gym) | 标准化 RL 环境 API |
| **数值计算** | NumPy, SciPy | 物理仿真（RK4 积分）、统计分析 |
| **可视化** | Matplotlib, Seaborn | 仿真结果绘图、基准测试可视化 |
| **图算法** | NetworkX | 最短路径基准测试 |
| **数据处理** | Pandas | 统计数据汇总与导出 |
| **配置管理** | PyYAML | YAML 配置文件解析 |
| **仿真核心** | 自研物理引擎 | Kelvin-Voigt 模型 + RK4 数值积分 |
| **通信协议** | Modbus TCP（预留接口） | 真实 PLC 通信占位 |

---

## 3. 完整目录结构

```
2/
├── agent/
│   └── gnn_actor_critic.py          # GNN Actor-Critic 智能体模型定义
├── checkpoints/
│   ├── final_stats.csv              # 最终部署验证统计结果
│   ├── phase1_agv/
│   │   ├── best_model.zip           # Phase 1 AGV 最佳模型
│   │   ├── evaluations.npz          # Phase 1 评估日志
│   │   └── ppo_agv_final.zip        # Phase 1 AGV 最终模型
│   ├── phase2_gnn/
│   │   └── ppo_gnn_final.pth        # Phase 2 GNN 最终模型
│   └── phase3_cotrain/
│       ├── ppo_agv_final_aligned.zip # Phase 3 AGV 对齐后模型
│       └── ppo_gnn_final_aligned.pth # Phase 3 GNN 对齐后模型
├── config/
│   ├── agv_env_config.yaml          # AGV 物理层环境配置
│   └── tsn_env_config.yaml          # TSN 网络层环境配置
├── core/
│   ├── comms/
│   │   └── plc_interface.py         # PLC 通信抽象层
│   └── physics/
│       └── agv_kinematics.py        # AGV 动力学物理仿真引擎
├── docs/
│   ├── 方案设计.docx                 # 方案设计文档
│   └── 研究背景现状.docx             # 研究背景与现状文档
├── runs/
│   └── phase1_agv_tensorboard/      # Phase 1 TensorBoard 训练日志
├── tests/
│   ├── benchmark_runner.py          # 基准对比测试运行器
│   ├── benchmark_visualizer.py      # 基准测试可视化
│   ├── final_deployment_verification.py # 最终部署验证脚本
│   ├── test_system.py               # AGV 物理环境单元测试
│   ├── test_system_integration.py   # TSN-AGV 系统集成测试
│   ├── test_tsn.py                  # TSN 环境单元测试
│   └── visualize_run.py             # AGV 物理仿真可视化
├── training/
│   ├── env_wrappers.py              # TSN-AGV 嵌套环境包装器
│   ├── train_phase1_agv.py          # Phase 1 AGV 训练入口
│   ├── train_phase2_gnn.py          # Phase 2 GNN 训练入口
│   └── train_phase3_cotrain.py      # Phase 3 协同训练入口
├── tsn_net/
│   ├── gantt_chart.py               # TSN 甘特图时隙调度器
│   ├── topology.py                  # TSN 网络拓扑管理器
│   └── tsn_gnn_env.py              # TSN-GNN 强化学习环境
├── .gitignore
├── 初步构建.md                       # 初始构建说明
├── README-1.md                      # 本文档 (Part 1/3)
├── README-2.md                      # 依赖·安装·运行 (Part 2/3)
└── README-3.md                      # 配置·测试·排查·贡献 (Part 3/3)
```

---

## 4. 各代码文件功能描述与实现逻辑

### 4.1 物理仿真引擎 — `core/physics/agv_kinematics.py`

**核心职责**：AGV 主从协同系统的物理动力学仿真核心，是整个系统的 "数字孪生" 基础。

**类与关键方法**：

| 类/方法 | 功能 |
|---------|------|
| `DelayBuffer` | 带时间戳的网络延迟历史缓冲区，模拟主车状态经网络传输给从车时的时序错位 |
| `AGVSystemSim` | 主从车动力学仿真器，采用 Kelvin-Voigt 模型 + RK4 数值积分 |

**`DelayBuffer` 实现逻辑**：
1. 使用 `collections.deque` 维护一个最大长度为 `history_buffer_size` 的环形缓冲
2. 每次主车状态更新时，将 `(timestamp, [x_m, v_m])` 压入缓冲区
3. 从车获取主车状态时，根据 `current_time - delay_sec` 查找对应的历史数据
4. 若所有数据都比目标时间新（仿真刚启动），则返回最旧的数据

**`AGVSystemSim` 物理模型**：

状态空间方程（阻抗控制）：

```
ż = A·z + B·F_ext

其中:
  z = [e, e_dot]^T                  — 跟踪误差及其导数
  A = [[0,    1    ],               — 二阶阻抗系统
       [-Kd/Md, -Bd/Md]]
  B = [0, 1/Md]^T                  — 外力输入矩阵
  e = x_s - x_d                     — 从车位置相对于期望位置的偏差
  F_ext = Kw·δx + Cw·δv            — Kelvin-Voigt 机翼结构应力
  δx = x_m - x_s - L                — 主从车相对位移超出理想跨距的部分
```

**`_rk4_step` 方法**：使用四阶龙格-库塔法（RK4）进行数值积分，推进一个仿真步长 (dt=0.02s，即50Hz)，计算精度远高于欧拉积分，确保长时间仿真不发散。

**`step` 方法执行流程**：
1. 更新主车状态（由速度指令 `master_v_cmd` 驱动）
2. 将主车状态压入延迟缓冲区
3. 从缓冲区获取延迟后的主车状态，计算期望指令 `x_d = x_m_delayed - L`
4. 计算 Kelvin-Voigt 被动力：`F_ext = Kw * (x_m - x_s - L) + Cw * (v_m - v_s)`
5. 使用 RK4 推进阻抗状态方程，更新 `z = [e, e_dot]`
6. 反推从车真实位置：`x_s = e + x_d`，`v_s = e_dot + v_d`

---

### 4.2 通信接口层 — `core/comms/plc_interface.py`

**核心职责**：抽象 PLC 通信协议，支持 Mock 仿真与 Modbus TCP 真实协议的双模式切换。

**类层次结构**：

```
BasePLCInterface (ABC 抽象基类)
├── MockPLC          (纯内存仿真，包含网络噪声模拟)
└── ModbusTCP_PLC    (Modbus TCP 真实协议占位，当前委托给 MockPLC)
```

**`BasePLCInterface` 抽象接口**：

| 抽象方法 | 功能 |
|----------|------|
| `connect()` | 建立 PLC 通信连接 |
| `read_sensors()` | 读取传感器状态 (e, e_dot, F_ext, rtt, delta_x_cmd) |
| `write_impedance(Md, Bd, Kd)` | 下发虚拟阻抗参数到从车控制器 |
| `inject_tsn_delay(actual_delay)` | 外部注入 TSN 调度产生的真实延迟 |
| `step_simulation(master_v_cmd)` | 推进仿真时间步 |

**`MockPLC` 关键特性**：

1. **双延迟模式**：
   - `standard`：以 5% 概率触发指数分布的网络拥塞尖峰，其余时间使用高斯噪声
   - `extreme_pareto`（Phase 1 炼蛊皿模式）：使用 Pareto 分布产生极端长尾延迟刺突，最大截断 500ms

2. **延迟注入优先级**：若外部通过 `inject_tsn_delay()` 注入延迟，则本次步进使用注入值并立即清空标记，确保 TSN 反馈闭环

3. **`read_sensors()` 输出元组**：
   ```python
   (e_t, e_dot_t, F_ext_t, rtt_delay_t, delta_x_cmd_t)
   ```

---

### 4.3 TSN 拓扑管理器 — `tsn_net/topology.py`

**核心职责**：生成并维护 TSN 车间网络的图拓扑结构，向 GNN 智能体提供 PyG 标准图数据对象。

**网络拓扑结构**（10 节点环+星型混合拓扑）：

```
Node 0 (Server) ──┬──→ Node 1 ──→ Node 3 ──→ Node 5 ──→ Node 7 (AP1) ──→ Node 9 (AGV)
                   │
                   └──→ Node 2 ──→ Node 4 ──→ Node 6 ──→ Node 8 (AP2) ──→ Node 9 (AGV)
```

**PyG Data 对象属性**：

| 属性 | 维度 | 说明 |
|------|------|------|
| `x` | `[10, 3]` | 节点特征：`[Type, CpuLoad, QueueLength]` |
| `edge_index` | `[2, 20]` | 有向边索引 (双向边) |
| `edge_attr` | `[20, 3]` | 边特征：`[Bandwidth(Mbps), PropDelay(us), Rssi(dBm)]` |
| `u` | `[1, 4]` | 全局图特征：`[Src, Dst, Size, D_max]` 待路由流信息 |

**节点类型编码**：
- `0` = Server/Switch（核心交换机）
- `1` = AP（无线接入点）
- `2` = AGV（自动导引车）

**`update_roaming_rssi(agv_x)` 方法**：基于对数距离路径损耗模型（Log-Distance Path Loss Model）动态更新无线链路的 RSSI 值：

```
RSSI(d) = P_tx - 10 * n * log10(d + 1) + noise

其中:
  P_tx = -30 dBm (1m 参考信号强度)
  n = 3.0 (工厂复杂环境路径损耗指数)
  noise ~ N(0, 0.5²)
  RSSI 钳位范围: [-95, -20] dBm
```

当 RSSI 低于 -80 dBm 时，TSN 环境中会模拟 500μs 的丢包重传延迟。

**AP 坐标**：
- AP1 (Node 7)：`(5.0m, 2.0m)`
- AP2 (Node 8)：`(15.0m, 2.0m)`

---

### 4.4 TSN 甘特图调度器 — `tsn_net/gantt_chart.py`

**核心职责**：管理 TSN 网络中每条边的时间槽分配，支持 GCL 周期卷绕碰撞检测。

**`GanttChartManager` 类**：

**关键参数**：
- `num_edges`: 需要管理的边总数
- `cycle_time`: GCL 周期时间，默认为 1000μs
- `edge_slots`: 每条边维护一个已占用时间槽列表 `[(start, end), ...]`

**`check_and_add_slot(edge_idx, start_time, duration)` 方法**：

1. **周期归一化**：将 `start_time` 映射到 `[0, cycle_time)` 周期内
   ```python
   mod_start = start_time % cycle_time
   mod_end = mod_start + duration
   ```
2. **卷绕处理**（Wrap-around）：如果 `mod_end > cycle_time`，则将时间段拆分为两段：
   - 段1：`[mod_start, cycle_time]`
   - 段2：`[0, mod_end - cycle_time]`
3. **碰撞检测**：逐一检查新段与已有段的区间重叠 (`max(s1, s2) < min(e1, e2)`)
4. **原子分配**：若无冲突，将所有新段持久化加入；若有冲突，直接拒绝并返回 `False`

**防坑设计**：
- 每条边的传输时长不能超过一个完整周期 (`duration <= cycle_time`)
- 跨越周期边界的卷绕处理确保不会遗漏碰撞

---

### 4.5 TSN-GNN 强化学习环境 — `tsn_net/tsn_gnn_env.py`

**核心职责**：基于 PyTorch Geometric 的自定义 RL 环境，支持自回归动作和动态动作掩码。

> ⚠️ 本环境未继承 `gymnasium.Env`，而是使用了自定义张量接口，以支持图结构观测和自回归联合动作空间。

**`TSN_GNN_Env` 类**：

**自回归动作空间**（两阶段决策）：
```
Step 1 → Actor-Routing Head:  选择下一跳节点 (Categorical over masked neighbors)
Step 2 → Actor-Scheduling Head:  预测时间槽偏移量 (Normal distribution, μ ∈ [0,1])
```

**观测空间**：
- `obs`: PyTorch Geometric `Data` 对象（包含节点/边/全局特征）
- `current_node`: 当前所在节点索引
- `action_mask`: 合法下一跳的布尔掩码（排除已访问节点）

**奖励函数**：

| 事件 | 奖励值 | 配置键 |
|------|--------|--------|
| 每路由一步 | -0.01 | `step_penalty` |
| 时间槽碰撞 | -100.0 | `collision_penalty` |
| 走进死胡同 | -50.0 | `dead_end_penalty` |
| 超时惩罚 | -0.1 × excess_ratio | `latency_penalty_factor` |
| 成功到达目标 | +10.0 | `success_reward` |

**`step` 方法执行流程**：

1. 获取当前节点的合法动作掩码
2. **防坑3 — 死胡同检查**：若掩码全为 `False`，立即终止并施加惩罚
3. **防坑 — 非法动作检查**：即使掩码外动作被传入，也立即惩罚
4. 获取选定边的物理特征 (带宽、传播延迟、RSSI)
5. **RSSI 惩罚**：若 RSSI < -80dBm，叠加 500μs 重传延迟
6. 计算物理传输时长：`duration = Size/BW + PropDelay + RetransmissionDelay`
7. 调用甘特图管理器进行碰撞检测与时隙分配
8. 若无冲突：推进当前节点、更新访问集、更新漫游 RSSI
9. 判断是否到达目标节点（施加成功奖励和潜在的延迟超限惩罚）
10. **防坑 — 下一步死胡同预检**：提前检查下一步是否有合法动作

**`_get_action_mask(node_idx)`**：生成当前节点的合法下一跳掩码
- 遍历拓扑中当前节点的所有物理邻居
- 排除已访问过的节点（避免环路）
- 返回 `[num_nodes]` 的布尔张量

---

### 4.6 GNN Actor-Critic 智能体 — `agent/gnn_actor_critic.py`

**核心职责**：定义 GNN 智能体的完整神经网络架构，实现自回归路由-调度联合策略。

**网络架构**：

```
输入: PyG Data (x, edge_index, edge_attr)
  │
  ├──→ GATv2Conv (node_dim→hidden_dim, edge_dim=3)
  │    └── ReLU
  ├──→ GATv2Conv (hidden_dim→hidden_dim)        ← 编码器
  │    └── ReLU → H [N, hidden_dim]
  │
  ├──→ ctx_mlp([H_curr || H_target] → H_ctx)    ← 上下文融合
  │
  ├──→ routing_head([H_ctx || H_candidate])      ← Actor-Routing Head
  │    └── Logits [N] (masked with -1e9 for invalid nodes)
  │
  ├──→ scheduling_head([H_curr || H_next || edge_attr]) ← Actor-Scheduling Head
  │    └── (μ, σ) → Normal(μ, σ)
  │
  └──→ critic_head(global_mean_pool(H))          ← Critic Head
       └── V(s) scalar
```

**关键设计**：

1. **GATv2Conv**：带边特征的图注意力卷积，自动关注动态衰减的 RSSI 信号，比 GATv1 具有更强的表达能力（注意力权重对查询键的排序不再单调）

2. **自回归依赖**：
   - `get_routing_logits` 先选择节点
   - `get_scheduling_dist` 随后基于已选节点的边特征预测时间偏移
   - 两个头的输出共同构成联合概率：`log π = log π_routing + log π_scheduling`

3. **动作掩码**：在路由 Logits 中，将不合法节点的值设为 `-1e9`，经过 softmax 后概率趋近于 0

4. **调度分布约束**：
   - `μ = sigmoid(output[0])` → 约束在 `[0, 1]`
   - `σ = softplus(output[1]) + 1e-4` → 保证正值

---

### 4.7 环境包装器 — `training/env_wrappers.py`

**核心职责**：将 TSN 网络环境与 AGV 物理环境嵌套，形成统一的训练接口，并为 Phase 2 提供"物理反馈 → 网络惩罚"的闭环机制。

**`NestedGNNEnvWrapper` 类**：

**`step(next_node, t_offset)` 执行流程**：

1. 获取 AGV 当前物理位置 `agv_pos = agv_env.sim_engine.x_s`
2. 调用 `tsn_env.step(next_node, t_offset, agv_x=agv_pos)` 获得路由结果
3. **仅在"成功到达"状态时**触发物理层反馈：
   - 将 `total_delay` 从微秒转换为秒，注入 AGV PLC 延迟
   - 计算盲步数 `num_blind_steps = ceil(rtt_sec / dt)`，最大钳位 50 步
   - 在盲步期间使用 AGV 策略（训练好的 RL 模型或刚性固定动作）执行 Rollout
   - 记录峰值应力 `peak_stress`
4. 将峰值应力转化为 GNN 惩罚：
   ```python
   stress_penalty = -(peak_stress / F_max)² × 10.0
   reward += stress_penalty
   ```

**`reset(seed=None)`**：同时重置 AGV 物理环境和 TSN 网络环境。

---

### 4.8 第一阶段训练 — `training/train_phase1_agv.py`

**目标**：AGV 柔顺控制在极端网络延迟下的领域随机化（Domain Randomization）训练。

**训练设置**：
- **算法**：Stable-Baselines3 PPO
- **延迟模式**：`extreme_pareto`（Pareto 分布产生长尾延迟刺突，模拟工业无线网络最恶劣工况）
- **并行环境数**：`n_envs=4`（4 个并行仿真环境）
- **总步数**：200,000 steps
- **神经网络**：`MlpPolicy`（多层感知机策略）

**PPO 超参数**：
```python
model = PPO("MlpPolicy", vec_env, verbose=1,
            learning_rate=3e-4,   # Adam 学习率
            n_steps=2048,         # 每次更新收集的步数
            batch_size=64,        # 小批量大小
            n_epochs=10,          # 每次更新的优化轮数
            gamma=0.99,           # 折扣因子
            gae_lambda=0.95,      # GAE (广义优势估计) 参数
            clip_range=0.2,       # PPO 裁剪范围
            tensorboard_log="./runs/phase1_agv_tensorboard/")
```

**评估回调**：`EvalCallback` 每 2000 步进行一次确定性评估，保存最优模型到 `checkpoints/phase1_agv/best_model.zip`。

**模型保存**：
- 最佳模型：`checkpoints/phase1_agv/best_model.zip`
- 最终模型：`checkpoints/phase1_agv/ppo_agv_final.zip`
- 评估日志：`checkpoints/phase1_agv/evaluations.npz`

---

### 4.9 第二阶段训练 — `training/train_phase2_gnn.py`

**目标**：GNN 在刚性 AGV（无柔顺能力的固定阻抗参数）物理反馈下进行悲观拓扑预训练。

**关键设计**：
- **刚性 AGV 动作**：`rigid_action = np.array([0.0, 0.0, 0.0])`，即使用默认基准阻抗，不进行任何柔顺调节
- **GNN Agent**：`GNNActorCritic(node_dim=3, edge_dim=3, hidden_dim=64)`
- **PPO 简化版**：手动实现 PPO 更新循环，每回合执行一次梯度更新以保持稳定性
- **训练规模**：1000 episodes

**PPO 更新逻辑**（手动实现）：
```python
# 1. 使用 compute_gae 计算广义优势估计 (GAE)
returns = compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95)

# 2. 计算优势
advantages = returns - old_values
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# 3. PPO 裁剪损失
ratio = exp(log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-0.2, 1+0.2) * advantages
actor_loss = -min(surr1, surr2).mean()

# 4. 总损失
loss = actor_loss + 0.5 * MSE(returns, new_values)
```

**模型保存**：`checkpoints/phase2_gnn/ppo_gnn_final.pth`

---

### 4.10 第三阶段训练 — `training/train_phase3_cotrain.py`

**目标**：GNN 与 AGV 通过乒乓协同训练实现最终的物理-网络联合对齐。

**乒乓协同训练 (Ping-Pong Co-Training)**：

```
for cycle in 1..15:
    ┌──────────────────────────────────────────────┐
    │ Step A: 冻结 GNN，训练 AGV                    │
    │   - AGV 在 GNN 产生的真实网络延迟下微调       │
    │   - learn(total_timesteps=10000)              │
    ├──────────────────────────────────────────────┤
    │ Step B: 冻结 AGV，训练 GNN                    │
    │   - GNN 在 AGV 柔顺反馈下优化路由/调度策略    │
    │   - 每回合更新一次 GNN                        │
    │   - 30 episodes per cycle                    │
    └──────────────────────────────────────────────┘
```

**模型加载与初始化**：
1. 加载 Phase 1 AGV 模型 (`ppo_agv_final.zip`)
2. 加载 Phase 2 GNN 模型 (`ppo_gnn_final.pth`)
3. GNN 使用更小的学习率 (`lr=5e-5`) 进行精调

**最终模型保存**：
- AGV：`checkpoints/phase3_cotrain/ppo_agv_final_aligned.zip`
- GNN：`checkpoints/phase3_cotrain/ppo_gnn_final_aligned.pth`

---

### 4.11 配置文件 — `config/`

**`agv_env_config.yaml`** — AGV 物理层环境配置：

| 配置键 | 默认值 | 说明 |
|--------|--------|------|
| `physics.dt` | `0.02` | 仿真控制步长 (s)，对应 50Hz |
| `physics.Kw` | `10000.0` | 机翼等效刚度 (N/m) |
| `physics.Cw` | `500.0` | 机翼等效结构阻尼 (N·s/m) |
| `physics.L` | `2.0` | 主从车理想几何跨距 (m) |
| `physics.history_buffer_size` | `500` | 延迟缓冲区最大容量 |
| `impedance.M_base` | `50.0` | 虚拟质量基准值 (kg) |
| `impedance.B_base` | `500.0` | 虚拟阻尼基准值 (N·s/m) |
| `impedance.K_base` | `3000.0` | 虚拟刚度基准值 |
| `impedance.M_delta_max` | `30.0` | 动作调整 M 的最大幅度 |
| `impedance.B_delta_max` | `300.0` | 动作调整 B 的最大幅度 |
| `impedance.K_delta_max` | `2500.0` | 动作调整 K 的最大幅度 |
| `rl.frame_stack_k` | `4` | 堆叠历史状态帧数 (O_t) |
| `rl.F_max` | `5000.0` | 归一化最大受力 (N) |
| `rl.e_max` | `0.5` | 归一化最大误差 (m) |
| `rl.alpha` | `3.0` | 应力平方惩罚权重 |
| `rl.beta` | `1.0` | 误差平方惩罚权重 |
| `rl.omega_3` | `0.8` | 动作平滑惩罚权重 |
| `comms.protocol` | `"mock"` | 通信协议 (mock/modbus_tcp) |
| `comms.base_rtt` | `0.05` | 基础网络延迟 (s) |
| `comms.rtt_noise_std` | `0.01` | 延迟抖动标准差 (s) |

**`tsn_env_config.yaml`** — TSN 网络层环境配置：

| 配置键 | 默认值 | 说明 |
|--------|--------|------|
| `network.cycle_time` | `1000.0` | GCL 周期时间 (μs) |
| `network.max_delay` | `2000.0` | 全局流 D_max 阈值 (μs) |
| `topology.num_nodes` | `10` | 节点总数 |
| `topology.num_ap` | `3` | 无线接入点数量 |
| `topology.agv_node_idx` | `9` | AGV 在图中的节点索引 |
| `rewards.step_penalty` | `-0.01` | 每路由一步的惩罚 |
| `rewards.collision_penalty` | `-100.0` | 时间槽重叠惩罚 |
| `rewards.dead_end_penalty` | `-50.0` | 死胡同惩罚 |
| `rewards.latency_penalty_factor` | `-0.1` | 超时惩罚系数 |
| `rewards.success_reward` | `10.0` | 成功路由基础奖励 |
| `training.max_steps_per_flow` | `15` | 单条流最大路由跳数 |

---

### 4.12 测试与可视化 — `tests/`

| 文件 | 类型 | 说明 |
|------|------|------|
| `test_system.py` | 单元测试 | AGV 物理环境 API 合规性、物理稳定性、动作钳位防御测试 |
| `test_tsn.py` | 单元测试 | TSN 甘特图卷绕碰撞检测、路由死胡同惩罚测试 |
| `test_system_integration.py` | 集成测试 | TSN-AGV 联合仿真（500步），生成 6 面板可视化报告 |
| `benchmark_runner.py` | 基准测试 | 6 种方法的对比评估 (M1~M6)，含统计分析（Welch's t-test, Cohen's d） |
| `benchmark_visualizer.py` | 可视化 | 基准测试结果的多视角可视化（柱状图、序列图、雷达图、小提琴图、p值热力图） |
| `final_deployment_verification.py` | 部署验证 | 加载 Phase 3 对齐模型，运行长周期仿真并生成综合性能报告 |
| `visualize_run.py` | 可视化 | 单个 AGV 仿真的 4 面板实时过程可视化 |

**基准测试 6 种方法对照**：

| ID | 方法 | 路由策略 | AGV 策略 |
|----|------|----------|----------|
| M1 | Ours (Full) | GNN | RL (Phase 3) |
| M2 | Traditional | 最短路径 (Dijkstra) | 固定阻抗 |
| M3 | RL-Only | 最短路径 (Dijkstra) | RL (Phase 3) |
| M4 | Random + RL | 随机路由 | RL (Phase 3) |
| M5 | GNN-Only | GNN | 固定阻抗 |
| M6 | No-Curriculum | 随机初始化 GNN | 随机初始化 RL |

---

### 4.13 检查点与日志 — `checkpoints/` & `runs/`

**checkpoints/final_stats.csv** — 最终部署验证的统计摘要：

```
Metric      | Mean      | Median     | P95       | P99       | Max
RTT (ms)    | 6.18      | 6.18       | 9.78      | 10.10     | 10.18
Stress (N)  | 48.40     | 22.98      | 137.90    | 509.36    | 2786.18
Error (m)   | 0.025     | 0.027      | 0.042     | 0.051     | 0.079
Jitter (ms) | 2.66      | 2.33       | 6.19      | 7.19      | 10.12
```

**runs/phase1_agv_tensorboard/** — Phase 1 训练 TensorBoard 事件文件，可用于可视化训练过程中的奖励曲线、损失曲线、策略熵等指标。

---

# AGV 主从协同控制与 TSN 网络调度强化学习系统

> **README Part 2/3** — 依赖库版本要求 · 环境配置与安装 · 编译/运行/部署完整流程

---

## 目录

- [5. 依赖库及其版本要求](#5-依赖库及其版本要求)
  - [5.1 核心依赖](#51-核心依赖)
  - [5.2 可选依赖](#52-可选依赖)
  - [5.3 一键安装](#53-一键安装)
- [6. 环境配置与安装步骤](#6-环境配置与安装步骤)
  - [6.1 系统要求](#61-系统要求)
  - [6.2 创建虚拟环境](#62-创建虚拟环境)
  - [6.3 依赖安装](#63-依赖安装)
  - [6.4 PyTorch Geometric 特殊安装](#64-pytorch-geometric-特殊安装)
  - [6.5 验证安装](#65-验证安装)
  - [6.6 环境变量配置](#66-环境变量配置)
- [7. 编译/运行/部署的完整流程](#7-编译运行部署的完整流程)
  - [7.1 快速开始（仅运行测试）](#71-快速开始仅运行测试)
  - [7.2 三阶段训练流程](#72-三阶段训练流程)
    - [Phase 1: AGV 领域随机化训练](#phase-1-agv-领域随机化训练)
    - [Phase 2: GNN 悲观拓扑预训练](#phase-2-gnn-悲观拓扑预训练)
    - [Phase 3: 乒乓协同训练](#phase-3-乒乓协同训练)
  - [7.3 基准测试与对比评估](#73-基准测试与对比评估)
  - [7.4 最终部署验证](#74-最终部署验证)
  - [7.5 使用 TensorBoard 监控训练](#75-使用-tensorboard-监控训练)

---

## 5. 依赖库及其版本要求

### 5.1 核心依赖

| 库名 | 最低版本 | 推荐版本 | 用途 |
|------|---------|---------|------|
| **Python** | 3.9 | 3.10 / 3.11 | 运行环境 |
| **PyTorch** | 2.0.0 | 2.1.x / 2.2.x | 深度学习框架，GNN 模型训练 |
| **torch_geometric** | 2.3.0 | 2.4.x / 2.5.x | 图神经网络库（GATv2Conv 等） |
| **stable-baselines3** | 2.1.0 | 2.3.x | PPO 强化学习算法 |
| **gymnasium** | 0.28.0 | 0.29.x / 1.0.x | 标准化 RL 环境接口 |
| **numpy** | 1.23.0 | 1.26.x | 数值计算（物理仿真） |
| **scipy** | 1.10.0 | 1.12.x | 统计检验与科学计算 |
| **matplotlib** | 3.6.0 | 3.8.x | 仿真结果可视化 |
| **seaborn** | 0.12.0 | 0.13.x | 高级统计可视化 |
| **pandas** | 1.5.0 | 2.1.x | 数据处理与 CSV 导出 |
| **networkx** | 2.8.0 | 3.2.x | 图算法（最短路径基准测试） |
| **pyyaml** | 6.0 | 6.0.x | YAML 配置文件解析 |
| **pytest** | 7.0.0 | 7.4.x | 单元测试框架 |

### 5.2 可选依赖

| 库名 | 用途 | 安装条件 |
|------|------|----------|
| **tensorboard** | 训练过程可视化监控 | 可选，Phase 1 训练需配合 TensorBoard |

### 5.3 一键安装

创建 `requirements.txt` 并使用 pip 安装所有核心依赖：

```txt
# requirements.txt
torch>=2.0.0
torch_geometric>=2.3.0
stable-baselines3>=2.1.0
gymnasium>=0.28.0
numpy>=1.23.0
scipy>=1.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
pandas>=1.5.0
networkx>=2.8.0
pyyaml>=6.0
pytest>=7.0.0
tensorboard>=2.12.0
```

```bash
pip install -r requirements.txt
```

---

## 6. 环境配置与安装步骤

### 6.1 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 10/11, Ubuntu 20.04+, macOS 12+ |
| **Python** | 3.9 – 3.11 (PyTorch Geometric 兼容性考虑) |
| **GPU（推荐）** | NVIDIA GPU + CUDA 11.8 / 12.1 (PyTorch 2.x 对应版本) |
| **内存** | 至少 8GB RAM |
| **磁盘** | 约 2GB 可用空间（含模型检查点） |

### 6.2 创建虚拟环境

**Windows (PowerShell)**：

```powershell
# 创建虚拟环境
python -m venv agv_tsn_env

# 激活虚拟环境
.\agv_tsn_env\Scripts\Activate.ps1

# 若遇到执行策略限制，先执行：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux / macOS**：

```bash
# 创建虚拟环境
python3 -m venv agv_tsn_env

# 激活虚拟环境
source agv_tsn_env/bin/activate
```

### 6.3 依赖安装

```bash
# 1. 确保 pip 为最新版本
pip install --upgrade pip setuptools wheel

# 2. 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (不推荐，训练极慢)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 安装核心依赖
pip install stable-baselines3 gymnasium numpy scipy matplotlib seaborn pandas networkx pyyaml pytest tensorboard
```

### 6.4 PyTorch Geometric 特殊安装

> ⚠️ PyTorch Geometric 需要与 PyTorch 和 CUDA 版本精确匹配，不能直接 `pip install torch_geometric`。

```bash
# 方式一：使用 PyG 官方安装脚本（推荐）
# 替换 ${CUDA} 为 cu118 或 cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+${CUDA}.html

# 方式二：如果方式一失败，尝试 conda 安装
conda install pyg -c pyg
```

**验证 PyG 安装**：

```python
import torch
import torch_geometric
print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

### 6.5 验证安装

进入项目根目录，运行以下命令验证所有依赖是否正确安装：

```bash
cd "d:\OneDrive\研究生\控制网络与现场总线\2"

# 1. 验证核心依赖导入
python -c "import torch; import torch_geometric; import gymnasium; import stable_baselines3; import numpy; import yaml; import networkx; print('All dependencies imported successfully.')"

# 2. 运行 AGV 物理层单元测试
python tests/test_system.py

# 3. 运行 TSN 网络层单元测试
python tests/test_tsn.py

# 4. 快速 AGV 可视化运行（生成 images/simulation_results.png）
python tests/visualize_run.py
```

如果以上 4 步全部通过，则环境安装成功。

### 6.6 环境变量配置

```bash
# Windows PowerShell
$env:PYTHONPATH = "d:\OneDrive\研究生\控制网络与现场总线\2"

# Linux / macOS
export PYTHONPATH="/path/to/控制网络与现场总线/2"
```

> ℹ️ 项目代码中已通过 `sys.path.append()` 处理了模块导入路径，通常情况下无需额外设置 `PYTHONPATH`。

---

## 7. 编译/运行/部署的完整流程

本项目为纯 Python 项目，无需编译步骤。以下为完整的运行流程。

### 7.1 快速开始（仅运行测试）

如果您只想验证系统功能而不进行完整训练，可以直接运行测试套件：

```bash
cd "d:\OneDrive\研究生\控制网络与现场总线\2"

# ===== 单元测试 =====

# AGV 物理环境测试（Gym API 合规性、物理稳定性、动作钳位）
python tests/test_system.py

# TSN 网络环境测试（甘特图碰撞检测、路由死胡同惩罚）
python tests/test_tsn.py

# ===== 集成测试 =====

# 系统联合仿真并生成可视化报告
python tests/test_system_integration.py

# ===== 可视化 =====

# 单个 AGV 仿真过程可视化
python tests/visualize_run.py
```

### 7.2 三阶段训练流程

> ⚠️ **重要**：三个阶段必须按顺序执行，因为每个后续阶段依赖前一阶段的模型输出。

**训练执行顺序一览**：

```
┌─────────────────────────────────────────────────────────────┐
│  1. python training/train_phase1_agv.py                     │
│     ↓ 生成: checkpoints/phase1_agv/ppo_agv_final.zip       │
│                                                             │
│  2. python training/train_phase2_gnn.py                     │
│     ↓ 生成: checkpoints/phase2_gnn/ppo_gnn_final.pth        │
│                                                             │
│  3. python training/train_phase3_cotrain.py                 │
│     ↓ 生成: checkpoints/phase3_cotrain/ppo_agv_final_aligned.zip │
│             checkpoints/phase3_cotrain/ppo_gnn_final_aligned.pth  │
└─────────────────────────────────────────────────────────────┘
```

---

#### Phase 1: AGV 领域随机化训练

**目标**：训练 AGV 智能体在极端网络延迟（Pareto 分布）下的柔顺控制策略。

**运行命令**：

```bash
python training/train_phase1_agv.py
```

**预计耗时**：200,000 步 × 4 并行环境 ≈ 15–30 分钟（取决于硬件）

**输出文件**：

```
checkpoints/phase1_agv/
├── best_model.zip         # 最佳模型（EvalCallback 自动选择）
├── evaluations.npz        # 评估日志
└── ppo_agv_final.zip      # 最终模型
```

**TensorBoard 监控**（在另一个终端中运行）：

```bash
tensorboard --logdir runs/
# 浏览器打开 http://localhost:6006
```

**训练过程中的关键指标**：
- `rollout/ep_rew_mean`：回合平均奖励（应逐渐上升）
- `train/approx_kl`：策略 KL 散度（应保持在小范围）
- `train/clip_fraction`：裁剪比例
- `train/entropy_loss`：策略熵（防止过早收敛）

---

#### Phase 2: GNN 悲观拓扑预训练

**前提条件**：Phase 1 训练完成，存在 `checkpoints/phase1_agv/ppo_agv_final.zip`（虽然不是必须，但完整流程需要）。

**运行命令**：

```bash
python training/train_phase2_gnn.py
```

**预计耗时**：1000 episodes ≈ 5–10 分钟

**输出文件**：

```
checkpoints/phase2_gnn/
└── ppo_gnn_final.pth      # GNN 模型权重 (state_dict)
```

**训练日志示例**：

```
Starting Phase 2: GNN Pessimistic Topology Pre-training
Using device: cuda
Episode 0 finished with status: success, Reward: 5.32, Peak Stress Penalty: -2.10
Episode 10 finished with status: success, Reward: 6.77, Peak Stress Penalty: -0.64
...
Phase 2 training script structure completed.
```

**关键说明**：
- GNN 在此阶段面对的是**刚性 AGV**（`rigid_action = [0, 0, 0]`），即从车以默认基准阻抗运行
- 训练目的是让 GNN 学会在"最坏情况"（无柔顺能力）下选择低风险的网络路径
- 每 10 个 episode 打印一次训练状态

---

#### Phase 3: 乒乓协同训练

**前提条件**：
- Phase 1 模型：`checkpoints/phase1_agv/ppo_agv_final.zip`
- Phase 2 模型：`checkpoints/phase2_gnn/ppo_gnn_final.pth`

**运行命令**：

```bash
python training/train_phase3_cotrain.py
```

**预计耗时**：15 cycles × (10,000 AGV steps + 30 GNN episodes) ≈ 30–60 分钟

**输出文件**：

```
checkpoints/phase3_cotrain/
├── ppo_agv_final_aligned.zip    # AGV 最终对齐模型
└── ppo_gnn_final_aligned.pth    # GNN 最终对齐模型
```

**训练日志示例**：

```
Starting Phase 3: Ping-Pong Co-Training (Final Alignment)
Loading Phase 1 AGV Agent from checkpoints/phase1_agv/ppo_agv_final.zip
Loading Phase 2 GNN Agent from checkpoints/phase2_gnn/ppo_gnn_final.pth

===== Co-Training Cycle 1/15 =====
--- Step A: Training AGV (GNN is Frozen) ---
AGV fine-tuning complete.
--- Step B: Training GNN (AGV is Frozen) ---
GNN Episode 0 - Reward: 8.45, Peak Stress: 32.1N
GNN Episode 5 - Reward: 9.12, Peak Stress: 18.3N
...

Phase 3 Co-Training completed! All agents are now physically and network-aligned.
```

**乒乓协同训练的核心机制**：

```
循环 15 次:
  ├── Step A（冻结 GNN）：
  │   AGV 学习适应 GNN 当前产生的网络延迟分布
  │   AGV 在真实延迟环境下学习更优的柔顺策略
  │
  └── Step B（冻结 AGV）：
      GNN 学习在 AGV 当前柔顺策略下的最优路由
      通过 stress_penalty 反向传播物理反馈
```

---

### 7.3 基准测试与对比评估

**前提条件**：Phase 3 训练完成，存在对齐后的模型文件。

```bash
# 1. 运行基准测试（6种方法对比）
python tests/benchmark_runner.py

# 基准测试会输出：
#   - images/benchmark_table.csv         (对比数据表格)
#   - images/benchmark_table.tex         (LaTeX 格式表格)
#   - images/benchmark_raw_data.npz      (原始数据)
#
# 控制台输出统计分析：
#   M1 vs M2: p-value = 0.0012, Cohen's d = -1.4523
#   M1 vs M3: p-value = 0.0345, Cohen's d = -0.8921
#   ...
```

```bash
# 2. 生成基准测试可视化图表
python tests/benchmark_visualizer.py

# 输出到 images/benchmark_plots/：
#   - 1_grouped_bar_chart.png/pdf       (分组柱状图)
#   - 2_stress_sequence.png/pdf         (应力时序对比图)
#   - 3_radar_chart.png                 (雷达图)
#   - 4_violin_plot.png                 (小提琴图)
#   - 5_pvalue_heatmap.png              (p值热力图)
```

---

### 7.4 最终部署验证

**前提条件**：Phase 3 训练完成。

```bash
python tests/final_deployment_verification.py
```

**输出**：

1. **控制台**：输出与 `checkpoints/final_stats.csv` 一致的统计摘要表
2. **图表**：`images/final_performance_report.png`（6 面板综合性能报告）：
   - 面板1：TSN 网络延迟 (RTT) 分布直方图
   - 面板2：物理应力分布箱线图
   - 面板3：跟踪误差累积分布函数 (CDF)
   - 面板4：漫游 RSSI 与网络延迟的关联分析
   - 面板5：RTT vs 抖动散点图
   - 面板6：物理层主从车位置实时跟踪曲线
3. **数据**：`checkpoints/final_stats.csv`

---

### 7.5 使用 TensorBoard 监控训练

**启动 TensorBoard**：

```bash
# 在项目根目录运行
tensorboard --logdir runs/ --bind_all

# 输出类似：
# TensorBoard 2.x.x at http://localhost:6006/
```

**关键监控面板**：

| 指标 | 含义 | 期望趋势 |
|------|------|----------|
| `rollout/ep_rew_mean` | 回合平均奖励 | 上升 |
| `rollout/ep_len_mean` | 回合平均步数 | 稳定或缓慢上升 |
| `train/loss` | 总损失 | 下降并收敛 |
| `train/approx_kl` | KL 散度 | < 0.02 |
| `train/clip_fraction` | 裁剪比例 | 10%–20% |
| `train/entropy_loss` | 策略熵 | 缓慢下降，不低于 -2 |
| `train/value_loss` | 价值函数损失 | 下降 |

---

# AGV 主从协同控制与 TSN 网络调度强化学习系统

> **README Part 3/3** — API 接口文档 · 配置参数详解 · 测试用例说明 · 常见问题排查指南 · 贡献规范 · 更新日志模板

---

## 目录

- [8. API 接口文档](#8-api-接口文档)
  - [8.1 AGVComplianceEnv — AGV 物理环境](#81-agvcomplianceenv--agv-物理环境)
  - [8.2 TSN_GNN_Env — TSN 网络环境](#82-tsn_gnn_env--tsn-网络环境)
  - [8.3 NestedGNNEnvWrapper — 嵌套环境包装器](#83-nestedgnnenvwrapper--嵌套环境包装器)
  - [8.4 GNNActorCritic — GNN 智能体](#84-gnnactorcritic--gnn-智能体)
  - [8.5 GanttChartManager — 甘特图管理器](#85-ganttchartmanager--甘特图管理器)
  - [8.6 TSNTopology — 拓扑管理器](#86-tsntopology--拓扑管理器)
  - [8.7 AGVSystemSim — 物理引擎](#87-agvsystemsim--物理引擎)
  - [8.8 BasePLCInterface / MockPLC / ModbusTCP_PLC — 通信接口](#88-baseplcinterface--mockplc--modbustcp_plc--通信接口)
- [9. 配置参数详解](#9-配置参数详解)
- [10. 测试用例说明](#10-测试用例说明)
- [11. 常见问题排查指南](#11-常见问题排查指南)
- [12. 贡献规范](#12-贡献规范)
- [13. 更新日志模板](#13-更新日志模板)

---

## 8. API 接口文档

### 8.1 AGVComplianceEnv — AGV 物理环境

AGV 柔顺控制的 RL 环境，遵循 `gymnasium.Env` 标准接口。

**初始化**：

```python
from env.agv_compliance_env import AGVComplianceEnv

env = AGVComplianceEnv(
    config_path=None,   # str | None — 配置文件路径，None 使用默认 `config/agv_env_config.yaml`
    render_mode=None    # str | None — 渲染模式（当前未实现图形渲染）
)
```

**观测空间 (Observation Space)**：

```python
# Box shape: (k, 5) = (4, 5)
# 每行 [e, e_dot, F_ext, tau, delta_x_cmd]
# dtype: float32
```

| 维度索引 | 名称 | 含义 | 单位 |
|----------|------|------|------|
| 0 | `e` | 从车跟踪误差 (x_s - x_d) | m |
| 1 | `e_dot` | 跟踪误差变化率 | m/s |
| 2 | `F_ext` | 机翼结构应力 | N |
| 3 | `tau` | 当前网络 RTT 延迟 | s |
| 4 | `delta_x_cmd` | 期望指令增量 | m |

**动作空间 (Action Space)**：

```python
# Box shape: (3,), range: [-1, 1]
# [delta_M, delta_B, delta_K]  — 阻抗参数的相对调整量
# dtype: float32
```

| 维度索引 | 名称 | 含义 | 实际值映射 |
|----------|------|------|-----------|
| 0 | `delta_M` | 虚拟质量调整 | `M = M_base + delta_M * M_delta_max` |
| 1 | `delta_B` | 虚拟阻尼调整 | `B = B_base + delta_B * B_delta_max` |
| 2 | `delta_K` | 虚拟刚度调整 | `K = K_base + delta_K * K_delta_max` |

**核心方法**：

```python
# 重置环境
obs, info = env.reset(seed=None)
#   obs: np.ndarray   — shape (4, 5)
#   info: dict         — {'error': float, 'F_ext': float, 'Md': float, 'Bd': float, 'Kd': float}

# 执行一步
obs, reward, terminated, truncated, info = env.step(action)
#   action: np.ndarray  — shape (3,)
#   obs: np.ndarray     — shape (4, 5)
#   reward: float       — 奖励值
#   terminated: bool    — 是否因终端条件终止
#   truncated: bool     — 是否因步数上限截断
#   info: dict          — 包含 'error', 'F_ext', 'Md', 'Bd', 'Kd'
```

**奖励函数**：

```
R = -alpha * (F_ext / F_max)^2 - beta * (e / e_max)^2 - omega_3 * sum((delta_a_i)^2)
```

**终止/截断条件**：
- `F_ext > F_max * 3` → `terminated = True`（应力超限）
- `abs(e) > e_max * 10` → `terminated = True`（误差超限）
- `step_count >= max_steps_per_episode` → `truncated = True`（步数截断）

---

### 8.2 TSN_GNN_Env — TSN 网络环境

> ⚠️ 本环境使用自定义接口，未继承 `gymnasium.Env`。观测和动作均为 PyTorch 张量。

**初始化**：

```python
from tsn_net.tsn_gnn_env import TSN_GNN_Env

env = TSN_GNN_Env(
    config_path=None  # str | None — 配置文件路径
)
```

**核心方法**：

```python
# 重置环境
obs, current_node, action_mask = env.reset()
#   obs: torch_geometric.data.Data  — 图数据对象
#   current_node: int               — 当前节点索引
#   action_mask: torch.Tensor       — shape (num_nodes,), dtype bool

# 执行一步（自回归接口）
next_obs, current_node, action_mask, reward, terminated, truncated, info = env.step(
    next_node,   # int — Actor-Routing Head 选择的下一跳节点
    t_offset,    # float — Actor-Scheduling Head 预测的时间偏移 (0~1)
    agv_x=None   # float | None — AGV 当前物理 x 坐标（用于 RSSI 更新）
)
#   next_obs: Data — 更新后的图数据
#   current_node: int — 新位置
#   action_mask: Tensor — 新动作掩码
#   reward: float — 该步奖励
#   terminated: bool — 是否终止（成功/碰撞/死胡同）
#   truncated: bool — 是否截断（超步数）
#   info: dict — {'status': str, 'total_delay': float, ...}

# 获取当前节点的合法动作掩码
mask = env._get_action_mask(node_idx)
#   mask: torch.Tensor — shape (num_nodes,), dtype bool

# 根据两端节点查找边索引
edge_idx = env._get_edge_idx(u, v)
#   edge_idx: int
```

**Info 字典状态码**：

| `info['status']` | 含义 |
|------------------|------|
| `'success'` | 成功到达目标节点 (Node 9) |
| `'collision'` | TSN 时间槽发生重叠碰撞 |
| `'dead_end'` | 当前节点无合法下一跳 |
| `'dead_end_next'` | 下一步将面临死胡同 |
| `'illegal_action'` | 执行了非法动作（被掩码拦截） |
| `'timeout'` | 超过最大路由步数 |

---

### 8.3 NestedGNNEnvWrapper — 嵌套环境包装器

将 TSN + AGV 环境嵌套，实现物理反馈闭环。

**初始化**：

```python
from training.env_wrappers import NestedGNNEnvWrapper

wrapper = NestedGNNEnvWrapper(
    tsn_env,      # TSN_GNN_Env — TSN 网络环境
    agv_env,      # AGVComplianceEnv — AGV 物理环境
    agv_agent=None  # PPO | None — AGV 强化学习智能体（None 则使用刚性动作）
)
```

**核心方法**：

```python
# 重置
obs, current_node, action_mask = wrapper.reset(seed=None)

# 执行一步
obs, current_node, mask, reward, terminated, truncated, info = wrapper.step(
    next_node,   # int — 下一跳节点
    t_offset     # float — 时间偏移 (0~1)
)
#   当 info['status'] == 'success' 时，info 额外包含：
#     'peak_stress': float    — AGV 盲步期间的峰值应力 (N)
#     'stress_penalty': float — 反向传播给 GNN 的应力惩罚
```

---

### 8.4 GNNActorCritic — GNN 智能体

基于 GATv2 的自回归 Actor-Critic 网络。

**初始化**：

```python
from agent.gnn_actor_critic import GNNActorCritic

agent = GNNActorCritic(
    node_dim=3,     # 节点特征维度
    edge_dim=3,     # 边特征维度
    hidden_dim=64   # 隐藏层维度
)
```

**核心方法**：

```python
# 图特征编码
h = agent.encode(data)
#   data: torch_geometric.data.Data
#   h: torch.Tensor — shape (num_nodes, hidden_dim)

# 获取路由策略Logits（含动作掩码）
logits = agent.get_routing_logits(h, current_node, target_node, action_mask)
#   current_node: int
#   target_node: int
#   action_mask: torch.Tensor — shape (num_nodes,), dtype bool
#   logits: torch.Tensor — shape (num_nodes,)

# 获取调度策略分布
sched_dist = agent.get_scheduling_dist(h, current_node, next_node, raw_edge_attr)
#   current_node: int — 当前节点
#   next_node: int — 已选定的下一跳节点
#   raw_edge_attr: torch.Tensor — shape (edge_dim,) 边特征
#   sched_dist: torch.distributions.Normal — 正态分布

# 获取状态价值评估
value = agent.get_value(h, batch=None)
#   h: torch.Tensor
#   batch: torch.Tensor | None — 图批次索引
#   value: torch.Tensor — scalar
```

---

### 8.5 GanttChartManager — 甘特图管理器

**初始化**：

```python
from tsn_net.gantt_chart import GanttChartManager

gantt = GanttChartManager(
    num_edges=20,       # 管理边数
    cycle_time=1000.0   # GCL 周期时间 (μs)
)
```

**核心方法**：

```python
# 重置所有时间槽
gantt.reset()

# 尝试分配时间槽
success = gantt.check_and_add_slot(
    edge_idx,    # int — 边索引
    start_time,  # float — 绝对开始时间 (μs)
    duration     # float — 传输时长 (μs)
)
#   success: bool — True = 分配成功, False = 发生碰撞
```

---

### 8.6 TSNTopology — 拓扑管理器

**初始化**：

```python
from tsn_net.topology import TSNTopology

topo = TSNTopology(
    num_nodes=10,    # 总节点数
    num_ap=3,        # 无线 AP 数
    agv_idx=9        # AGV 节点索引
)
```

**核心方法**：

```python
# 获取 PyG 图数据对象
data = topo.get_pyg_data()

# 获取节点邻居列表
neighbors = topo.get_neighbors(node_idx)

# 更新无线链路 RSSI
topo.update_roaming_rssi(agv_x)
```

---

### 8.7 AGVSystemSim — 物理引擎

**初始化**：

```python
from core.physics.agv_kinematics import AGVSystemSim

sim = AGVSystemSim(config)
#   config: dict — 包含 'physics' 键的配置字典
```

**核心方法**：

```python
# 重置物理环境
sim.reset()

# 推进一个仿真步
sim.step(
    master_v_cmd,  # float — 主车速度指令 (m/s)
    delay_sec,     # float — 当前网络 RTT (s)
    Md,            # float — 虚拟质量 (kg)
    Bd,            # float — 虚拟阻尼 (N·s/m)
    Kd             # float — 虚拟刚度 (N/m)
)

# 获取当前状态
e, e_dot, F_ext = sim.get_state()
```

---

### 8.8 BasePLCInterface / MockPLC / ModbusTCP_PLC — 通信接口

**抽象接口** `BasePLCInterface`：

```python
class BasePLCInterface(ABC):
    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def read_sensors(self) -> Tuple[float, float, float, float, float]: ...

    @abstractmethod
    def write_impedance(self, Md: float, Bd: float, Kd: float): ...

    @abstractmethod
    def inject_tsn_delay(self, actual_delay: float): ...

    @abstractmethod
    def step_simulation(self, master_v_cmd: float): ...
```

**MockPLC** 额外方法：

```python
# 设置随机数生成器（保证多回合随机数一致性）
mock_plc.set_rng(rng)

# 注入 TSN 延迟（优先级高于自动生成的噪声延迟）
mock_plc.inject_tsn_delay(actual_delay)
```

---

## 9. 配置参数详解

### 9.1 `agv_env_config.yaml`

详见 [README-1.md §4.11](#411-配置文件--config) 中的完整参数表。

**调参建议**：

| 场景 | 推荐调整 |
|------|----------|
| 超大型结构件（更大应力） | 增大 `alpha`（应力惩罚权重）至 5.0–8.0 |
| 高精度跟踪需求 | 增大 `beta`（误差惩罚权重）至 3.0–5.0 |
| 网络极不稳定 | 增大 `base_rtt` 至 0.1–0.2s，增大 `rtt_noise_std` |
| 更快的控制响应 | 减小 `physics.dt` 至 0.01（100Hz），注意同步调整 RL 训练参数 |
| 探索更大的阻抗范围 | 增大 `M_delta_max`, `B_delta_max`, `K_delta_max` |

### 9.2 `tsn_env_config.yaml`

详见 [README-1.md §4.11](#411-配置文件--config) 中的完整参数表。

**调参建议**：

| 场景 | 推荐调整 |
|------|----------|
| 更大的网络拓扑 | 增大 `topology.num_nodes` 并适配 `topology.py` 中的拓扑定义 |
| 更严格的延迟要求 | 减小 `network.max_delay` 至 1000μs |
| 更多无线干扰 | 在 `topology.py` 中增大路径损耗指数 `n_loss` |
| 加快路由探索 | 增大 `training.max_steps_per_flow` 至 20–25 |

---

## 10. 测试用例说明

### 10.1 单元测试

#### `test_system.py` — AGV 物理环境测试

```bash
python tests/test_system.py
```

| 测试函数 | 验证内容 | 通过标准 |
|----------|----------|----------|
| `test_gym_api_compliance` | Gymnasium API 规范合规性 | `check_env` 无异常 |
| `test_physics_stability` | 100 步零动作仿真不发散 | F_ext 无 NaN，收敛于合理值 |
| `test_action_clamping` | 极端负动作下阻抗钳位 | Md, Bd, Kd 均 > 0 |

#### `test_tsn.py` — TSN 环境测试

```bash
python tests/test_tsn.py
```

| 测试函数 | 验证内容 | 通过标准 |
|----------|----------|----------|
| `test_gantt_chart_circular_overlap` | 周期卷绕碰撞检测 | 跨越周期的分配拒绝发生碰撞的请求 |
| `test_tsn_env_routing_and_dead_end` | 路由通路与死胡同惩罚 | 正确到达目标，死胡同触发终止 |

### 10.2 集成测试

#### `test_system_integration.py` — TSN-AGV 联合仿真

```bash
python tests/test_system_integration.py
```

- 运行 500 步（10 秒）联合仿真
- 生成 6 面板可视化报告：`images/system_integration_report.png`
  - TSN 网络延迟曲线
  - 漫游 RSSI 演化
  - AGV 跟踪误差
  - 机翼结构应力
  - 步奖励曲线
  - 系统指标汇总

### 10.3 基准测试

#### `benchmark_runner.py` — 6 种方法对比

```bash
python tests/benchmark_runner.py
```

- 6 种方法（M1–M6）在相同随机种子下运行
- 使用 Welch's t-test 和 Cohen's d 进行统计显著性分析
- 注入重度背景流量（70% 概率）和极高延迟抖动（10–100ms）
- 输出对比表格和原始数据

#### `benchmark_visualizer.py` — 基准测试可视化

```bash
python tests/benchmark_visualizer.py
```

- 分组柱状图：峰值应力、碰撞率、抖动对比
- 应力序列图：三种方法的逐帧应力对比
- 雷达图：多维度鲁棒性评估
- 小提琴图：峰值应力分布分析
- p 值热力图：M1 与其他方法的统计显著性

### 10.4 部署验证

#### `final_deployment_verification.py` — 最终部署验证

```bash
python tests/final_deployment_verification.py
```

- 加载 Phase 3 对齐模型
- 运行 10 个评估 episode
- 生成综合统计报告和 6 面板可视化

### 10.5 可视化

#### `visualize_run.py` — 运行过程可视化

```bash
python tests/visualize_run.py
```

- 200 步 AGV 仿真过程
- 4 面板实时图表：跟踪误差、机翼应力、阻抗参数变化、网络延迟与奖励

---

## 11. 常见问题排查指南

### 11.1 安装问题

**Q：`torch_geometric` 安装后无法导入**

```bash
# 症状
ImportError: DLL load failed: The specified module could not be found.

# 解决方案：确认 PyTorch 和 PyG 的 CUDA 版本匹配
python -c "import torch; print(torch.version.cuda)"
# 根据输出的 CUDA 版本重新安装 PyG
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

**Q：`ModuleNotFoundError: No module named 'env'`**

```bash
# 解决方案：项目代码已用 sys.path.append 处理了路径问题
# 确保从项目根目录运行脚本
cd "d:\OneDrive\研究生\控制网络与现场总线\2"
python tests/test_system.py
```

### 11.2 训练问题

**Q：Phase 1 训练时 CUDA Out of Memory**

```bash
# 解决方案 1：减少并行环境数
# 编辑 train_phase1_agv.py，将 n_envs 从 4 改为 1 或 2

# 解决方案 2：切换到 CPU 模式
# 代码会自动检测 CUDA 可用性，若无 GPU 则回退到 CPU
# 但训练速度会显著降低
```

**Q：Phase 2/3 训练中 GNN 报错 `IndexError: index out of range`**

```bash
# 原因：action_mask 中所有条目都为 False（死胡同）但未正确处理
# 解决方案：检查 tsn_gnn_env.py 中的 _get_action_mask 是否正确排除了 visited_nodes

# 紧急应对：在 GNN 策略采样前手动检查掩码
if not action_mask.any():
    # 处理死胡同：随机选择一个邻居或重置
    break
```

**Q：训练奖励不收敛或震荡剧烈**

| 可能原因 | 排查方法 | 解决方案 |
|----------|----------|----------|
| 学习率过高 | 检查 TensorBoard `train/loss` 是否有剧烈跳变 | 降低 `learning_rate` 至 1e-4 或 5e-5 |
| `clip_range` 太小 | 检查 `train/clip_fraction` 是否接近 0 | 增大 `clip_range` 至 0.3 |
| 奖励规模失衡 | 检查各奖励分量的量级 | 调整 `alpha`/`beta` 使各分量在同一数量级 |
| GAE 参数不当 | 检查 `train/approx_kl` | 调整 `gae_lambda` 至 0.9 |

### 11.3 运行时问题

**Q：物理仿真出现 NaN**

```bash
# 原因：Md（虚拟质量）变为 0 或负值
# 系统已有防御性编程 (assert Md > 1e-4)，但极端情况下可能被绕过

# 解决方案：检查 config 中 impedance 参数的 delta_max 是否合理
# 确保: M_base - M_delta_max > 1e-4
```

**Q：甘特图碰撞误报或漏报**

```bash
# 可能原因：cycle_time 设置不合理
# 确保每条边的 duration <= cycle_time

# 排查方法：在 check_and_add_slot 中添加调试打印
print(f"Edge {edge_idx}: mod_start={mod_start}, mod_end={mod_end}, segments={segments_to_check}")
```

### 11.4 模型加载问题

**Q：加载 `.zip` 模型报 `KeyError` 或不兼容**

```bash
# 原因：Stable-Baselines3 版本不匹配
# 解决方案：
pip install stable-baselines3==2.3.2  # 使用训练时的版本

# 或重新训练
python training/train_phase1_agv.py
```

**Q：加载 `.pth` 模型报 `Missing key(s) in state_dict`**

```bash
# 原因：GNN 模型架构与保存时不一致
# 解决方案：确认 GNNActorCritic 参数与训练时一致
agent = GNNActorCritic(node_dim=3, edge_dim=3, hidden_dim=64)
agent.load_state_dict(torch.load("checkpoints/phase2_gnn/ppo_gnn_final.pth"))
```

---

## 12. 贡献规范

### 12.1 代码风格

- **语言**：Python 3.9+，类型注解推荐使用
- **命名规范**：
  - 类名：`PascalCase`（如 `AGVSystemSim`, `GanttChartManager`）
  - 函数/方法：`snake_case`（如 `_rk4_step`, `get_routing_logits`）
  - 常量：`UPPER_SNAKE_CASE`（如 `F_max`）
  - 私有方法/属性：前缀 `_`（如 `_get_action_mask`）
- **文档字符串**：使用 Google-style docstring 格式
- **防御性编程**：对关键物理参数添加 `assert` 校验，防止 NaN 传播
- **禁止**：提交包含 API 密钥、密码、内部 IP 地址的代码

### 12.2 分支策略

```
main — 稳定发布版本
  └── develop — 开发分支
       ├── feature/xxx — 新功能分支
       ├── bugfix/xxx — 缺陷修复分支
       └── experiment/xxx — 实验性分支
```

### 12.3 提交信息规范

```
<type>(<scope>): <subject>

# 示例
feat(agent): add multi-head attention support for routing prediction
fix(tsn_env): correct dead-end detection in _get_action_mask
refactor(physics): extract RK4 integrator to standalone function
docs(readme): update installation guide for PyG 2.5
test(benchmark): add M7 baseline (round-robin routing)
```

**Type 类型**：`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`

### 12.4 提交前检查清单

- [ ] 所有单元测试通过：`python tests/test_system.py && python tests/test_tsn.py`
- [ ] 集成测试通过（可选但推荐）：`python tests/test_system_integration.py`
- [ ] 无硬编码路径，所有路径使用 `os.path.join` 和相对路径
- [ ] 新代码包含必要的文档字符串
- [ ] 配置文件中的参数有注释说明
- [ ] 无 `print` 调试语句残留（使用 `logging` 模块替代）
- [ ] 随机数种子已正确设置以保证可复现性

---

## 13. 更新日志模板

### 格式说明

版本号遵循 [Semantic Versioning 2.0.0](https://semver.org/lang/zh-CN/) 规范：
- **主版本号**（MAJOR）：不兼容的 API 修改
- **次版本号**（MINOR）：向下兼容的功能性新增
- **修订号**（PATCH）：向下兼容的问题修正

---

### [Unreleased]

#### Added
- （新增功能）

#### Changed
- （功能变更）

#### Deprecated
- （即将移除的功能）

#### Removed
- （已移除的功能）

#### Fixed
- （缺陷修复）

#### Security
- （安全相关修复）

---

### [1.0.0] — 2026-05-11

#### Added
- 初始版本发布
- **核心物理引擎**：Kelvin-Voigt 模型 + RK4 数值积分的 AGV 主从协同动力学仿真
- **通信接口层**：MockPLC（含极端 Pareto 延迟模式）+ ModbusTCP_PLC 占位接口
- **TSN 网络层**：环+星型拓扑的 GCL 甘特图调度器，支持周期卷绕碰撞检测
- **AGV RL 环境**：遵循 Gymnasium 标准接口的柔顺控制环境
- **TSN-GNN 环境**：自回归路由-调度联合决策的自定义环境
- **GNN 智能体**：基于 GATv2Conv + 自回归 Actor-Critic 架构
- **三阶段训练**：AGV 领域随机化 → GNN 悲观预训练 → 乒乓协同训练
- **基准测试**：6 种方法的对比评估与统计分析

#### Dependencies
- Python 3.9+
- PyTorch 2.0+
- torch_geometric 2.3+
- stable-baselines3 2.1+
- gymnasium 0.28+

---

> 📖 **回到开头**: [README-1.md](./README-1.md) — 项目概述 · 技术栈 · 目录结构 · 代码文件详解


