# Roboyi-Flat 策略接口与控制参数

整理日期：2026-11-09。

本文针对本仓库的 **10 自由度 `Roboyi-Flat` 任务**，记录观测排列、历史堆叠顺序、策略关节顺序、动作缩放、默认关节角和控制周期。依据为当前源代码、已保存的运行验证记录，以及 `logs/rsl_rl/roboyi_flat/2026-11-08_14-59-49_roboyi_walk/params/` 中的训练配置。

不同任务（例如 `RPO-*`）的关节数量和接口可能不同。使用已有 checkpoint 时，应以该 checkpoint 配套的 `params/env.yaml`、`params/agent.yaml` 和模型归一化统计为准。

## 1. 参数速查

| 项目 | Roboyi-Flat 的配置 |
|---|---|
| 动作维数 | 10 |
| actor 单帧观测维数 | 39 |
| actor 历史帧数 | 10 |
| actor 最终输入维数 | 390，形状为 `[batch_size, 390]` |
| critic 单帧观测维数 | 74 |
| critic 历史帧数 | 10 |
| critic 最终输入维数 | 740 |
| 历史排列 | 按完整帧排列，从旧到新 |
| `action_scale` | 0.15 |
| 默认关节角 | 全部为 0 rad |
| 默认关节速度 | 全部为 0 rad/s |
| 物理步长 `sim.dt` | 0.005 s，即 5 ms，200 Hz |
| `decimation` | 4 |
| 策略控制周期 | 0.02 s，即 20 ms，50 Hz |
| 策略包装器动作裁剪 | ±3.0 |
| 环境动作裁剪 | ±100.0，在包装器裁剪之后执行 |
| 观测裁剪 | ±100.0，在历史堆叠之后执行 |
| 关节位置目标软限位裁剪 | 开启，软限位系数为 0.9 |
| 执行器目标延迟 | 每次重置随机采样 0、1 或 2 个训练物理步，即 0、5 或 10 ms |

## 2. actor 单帧观测排列

下列索引采用 Python 切片规则：左闭右开。

| 索引 | 内容 | 维数 | 单位或含义 |
|---|---|---:|---|
| `0:3` | 机身角速度 `[ωx, ωy, ωz]` | 3 | 机身坐标系，rad/s |
| `3:6` | 重力投影方向 `[gx, gy, gz]` | 3 | 机身坐标系中的单位重力方向；直立时为 `[0, 0, -1]` |
| `6:9` | 速度指令 `[vx, vy, ωz]` | 3 | 前进、横移和转向指令；单位依次为 m/s、m/s、rad/s |
| `9:19` | 相对默认关节位置 `q - q_default` | 10 | rad，按第 4 节的策略关节顺序 |
| `19:29` | 相对默认关节速度 `dq - dq_default` | 10 | rad/s，按第 4 节的策略关节顺序 |
| `29:39` | 上一策略周期动作 `a_previous` | 10 | 已经过策略包装器裁剪、尚未进行环境动作裁剪和动作缩放 |

单帧形式为：

```python
frame = concat(
    root_ang_vel_b,              # 3
    projected_gravity_b,         # 3
    command,                    # 3: vx, vy, wz
    q - q_default,              # 10
    dq - dq_default,            # 10
    a_previous,                # 10
)
```

以上各项的 `obs_scales` 全部为 `1.0`。当前任务关闭地形高度扫描，actor 也不包含机身线速度。

机器人机身坐标约定为：`+X` 向前、`+Y` 向左、`+Z` 向上。

代码依据：

- [base_env.py](robolab/robolab/tasks/direct/base/base_env.py)：`compute_current_observations()`，约第 143–163 行。
- [base_config.py](robolab/robolab/tasks/direct/base/base_config.py)：`ObsScalesCfg`、`NormalizationCfg`，约第 100–117 行。
- [roboyi_env_cfg.py](robolab/robolab/tasks/direct/base/roboyi_env_cfg.py)：关闭高度扫描，约第 135–138 行。
- [standing_pose.json](model/standing_pose.json)：`root_axes`。

## 3. 历史堆叠顺序

actor 保存 **10 帧**完整观测，按时间从旧到新展开：

```text
policy_obs = [frame[t-9], frame[t-8], ..., frame[t-1], frame[t]]
```

每一帧内部均使用第 2 节的 39 维排列。堆叠方式是按帧连续拼接，不是把某一个观测项的 10 帧历史先集中拼在一起。

| 最终输入切片 | 对应帧 |
|---|---|
| `0:39` | 最旧帧 `frame[t-9]` |
| `39:78` | `frame[t-8]` |
| `...` | `...` |
| `312:351` | `frame[t-1]` |
| `351:390` | 当前帧 `frame[t]` |

展平前的缓冲区形状为 `[batch_size, 10, 39]`，展平后为 `[batch_size, 390]`。

重置后首次加入观测时，用首次观测重复填满历史。sim2sim 中上一动作初始化为 10 个零；此后每次推理将历史向前移动一帧，并把当前帧写到末尾。

`action_history_length=3` 是独立动作缓冲区的长度。单帧观测只读取其中最新的 10 维动作，并不直接追加三组动作。

代码依据：

- [base_config.py](robolab/robolab/tasks/direct/base/base_config.py)：`RobotCfg`，约第 86–91 行。
- [base_env.py](robolab/robolab/tasks/direct/base/base_env.py)：观测缓冲区初始化、`action_buffer.buffer[:, -1, :]`、`_get_observations()`。
- [sim2sim_roboyi.py](robolab/scripts/mujoco/sim2sim_roboyi.py)：`reset()`、`infer()`，约第 229–286 行；明确实现首次重复填充及从旧到新的移动。
- [verify_roboyi_sim2sim.py](scripts/verify_roboyi_sim2sim.py)：约第 55–64 行，包含历史移动及上一动作位置的检查。

## 4. 策略关节顺序

观测中的关节位置、关节速度，以及策略输出动作，统一使用以下顺序：

| 索引 | 关节名称 | 默认角度（rad） |
|---:|---|---:|
| 0 | `Left_pitch` | 0.0 |
| 1 | `Right_pitch` | 0.0 |
| 2 | `Left_roll` | 0.0 |
| 3 | `Right_roll` | 0.0 |
| 4 | `Left_yaw` | 0.0 |
| 5 | `Right_yaw` | 0.0 |
| 6 | `Left_knee` | 0.0 |
| 7 | `Right_knee` | 0.0 |
| 8 | `Left_ankle` | 0.0 |
| 9 | `Right_ankle` | 0.0 |

可直接使用的列表：

```python
POLICY_JOINTS = [
    "Left_pitch", "Right_pitch",
    "Left_roll", "Right_roll",
    "Left_yaw", "Right_yaw",
    "Left_knee", "Right_knee",
    "Left_ankle", "Right_ankle",
]
```

这是运行时 articulation 的关节顺序。`standing_pose.json` 中的字典则是先列整条左腿、再列整条右腿；不能把字典插入顺序直接当成策略顺序。对接其他仿真器或硬件时，应按关节名称映射位置、速度和动作。

依据：

- [runtime.json](outputs/roboyi_ppo_verification/runtime.json)：已保存的运行验证记录，`joints` 字段。
- [sim2sim_roboyi.py](robolab/scripts/mujoco/sim2sim_roboyi.py)：`POLICY_JOINTS`，约第 28–31 行；关节与执行器映射，约第 157–162 行。

## 5. action_scale 和动作处理

`RoboyiFlatEnvCfg` 将基类的 `action_scale=0.25` 覆盖为 **`0.15`**。

使用当前训练配置时，动作处理顺序为：

```python
# raw_action: 网络原始输出，形状为 [batch_size, 10]
a = clip(raw_action, -3.0, 3.0)          # RSL-RL 包装器
a_previous_for_next_frame = a           # 环境先记录动作
a_env = clip(a, -100.0, 100.0)           # 环境内部裁剪
q_target = q_default + 0.15 * a_env      # 位置目标，单位 rad
q_target = clip(q_target, soft_lower, soft_upper)
```

含义如下：

- 动作为相对默认角的关节位置偏移指令，经过缩放后形成位置目标。
- 单位动作对应 `0.15 rad`，约 `8.59°` 的目标角偏移。
- 包装器裁剪到 ±3 后，软限位裁剪前的最大偏移为 ±0.45 rad，约 ±25.8°。
- 当前开启 `clip_joint_targets=True`；最终目标还要裁剪到各关节软限位。
- 软限位系数为 `0.9`，表示围绕关节范围中心缩小到原范围的 90%。
- 下一帧观测保存的是 `a`，不是缩放后的偏移、最终位置目标或经过执行器延迟后的目标。

代码依据：

- [roboyi_env_cfg.py](robolab/robolab/tasks/direct/base/roboyi_env_cfg.py)：约第 150–151 行。
- [roboyi_agent_cfg.py](robolab/robolab/tasks/direct/base/agents/roboyi_agent_cfg.py)：`clip_actions=3.0`，约第 17 行。
- [base_env.py](robolab/robolab/tasks/direct/base/base_env.py)：`_pre_physics_step()`、`_apply_action()`，约第 242–252 行。
- [roboyi.py](robolab/robolab/assets/robots/roboyi.py)：`soft_joint_pos_limit_factor=0.9`，约第 42 行。
- [sim2sim_roboyi.py](robolab/scripts/mujoco/sim2sim_roboyi.py)：约第 279–285 行，对应部署端动作处理。

## 6. 默认关节角和零位含义

按策略顺序：

```python
q_default = [0.0] * 10   # rad
dq_default = [0.0] * 10  # rad/s
```

默认关节角来自 `standing_pose.json` 的 `new_zero_joint_positions_rad`，资产配置将其赋给 `init_state.joint_pos`。

该模型的零位定义为：躯干直立、膝关节伸直、双脚测得的鞋底平面水平。JSON 中的 `old_urdf_q_equals_new_q_plus_rad` 记录旧 URDF 与新模型坐标的偏移，**不是当前策略的默认关节角**。文件还注明实体硬件零位未改变；硬件侧需要按实际标定转换到模型坐标。

默认角与环境重置时加入的随机关节扰动也是两个不同概念：观测中的位置偏差始终减去默认角。

代码依据：

- [standing_pose.json](model/standing_pose.json)：`zero_definition`、`new_zero_joint_positions_rad`、`physical_hardware_zero_not_changed`。
- [roboyi.py](robolab/robolab/assets/robots/roboyi.py)：约第 37–40 行。
- [roboyi_env_cfg.py](robolab/robolab/tasks/direct/base/roboyi_env_cfg.py)：重置关节位置扰动配置，约第 164 行。

## 7. 控制周期与执行器延迟

```python
physics_dt = 0.005
decimation = 4
policy_dt = physics_dt * decimation  # 0.02 s
```

策略以 **50 Hz** 更新动作。一个策略周期包含 4 个物理步，关节位置目标在这 4 步内保持；PD 执行器以 **200 Hz** 根据最新关节状态计算力矩。

训练执行器为 `DelayedPDActuator`。整组腿部关节每次重置随机采样 0、1 或 2 个训练物理步的目标延迟，即 0、5 或 10 ms；这不会改变 20 ms 的策略更新周期。

sim2sim 默认使用训练物理步长，也允许通过 `--physics-dt` 细分物理步长，但策略周期仍取配套训练配置中的 `sim.dt * decimation`。启用执行器延迟时，新的物理步长还必须能整除训练物理步长，以保持相同延迟时间。

代码依据：

- [base_config.py](robolab/robolab/tasks/direct/base/base_config.py)：约第 259–262 行。
- [base_env.py](robolab/robolab/tasks/direct/base/base_env.py)：`step()`，约第 189–203 行。
- [roboyi.py](robolab/robolab/assets/robots/roboyi.py)：执行器类型及 `min_delay=0`、`max_delay=2`，约第 44–56 行。
- [sim2sim_roboyi.py](robolab/scripts/mujoco/sim2sim_roboyi.py)：`RoboyiSim.__init__()`、`physics_step()`。

## 8. critic 观测补充

critic 单帧在 actor 的 39 维后依次追加以下信息：

| 索引 | 内容 | 维数 |
|---|---|---:|
| `0:39` | actor 单帧基础观测 | 39 |
| `39:42` | 机身坐标系线速度 | 3 |
| `42:44` | 双脚接触标志 | 2 |
| `44:50` | 双脚世界坐标系接触力，每脚三个分量 | 6 |
| `50:52` | 双脚当前腾空时间 | 2 |
| `52:54` | 双脚相对地面的高度 | 2 |
| `54:64` | 关节加速度 | 10 |
| `64:74` | 实际施加的关节力矩 | 10 |

critic 同样按完整帧从旧到新堆叠 10 帧，总计 740 维。常规 actor 部署推理仅使用 390 维策略观测。

依据：[base_env.py](robolab/robolab/tasks/direct/base/base_env.py) 的 `compute_current_observations()`，约第 165–183 行。

## 9. 部署时的观测归一化

`obs_scales=1.0` 不代表网络没有观测归一化。已核对的训练配置开启了 `actor_obs_normalization` 和 `critic_obs_normalization`；推理时需要保留 checkpoint 中的归一化统计。

当前 sim2sim 导出的推理封装为：

```python
action = actor(actor_obs_normalizer(policy_obs))
```

其中 `policy_obs` 是前述经过历史堆叠和 ±100 裁剪的 390 维输入。如果使用已包含归一化的导出模型，应直接传入该输入，避免重复归一化。

依据：

- [训练 agent.yaml](logs/rsl_rl/roboyi_flat/2026-11-08_14-59-49_roboyi_walk/params/agent.yaml)：`actor_obs_normalization`、`critic_obs_normalization`。
- [训练 env.yaml](logs/rsl_rl/roboyi_flat/2026-11-08_14-59-49_roboyi_walk/params/env.yaml)：历史长度、动作缩放、观测维数及控制周期。
- [sim2sim_roboyi.py](robolab/scripts/mujoco/sim2sim_roboyi.py)：`InferencePolicy`，约第 93–100 行。
