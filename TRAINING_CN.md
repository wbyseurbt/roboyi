# 当前机器的 PPO 训练指南

本指南面向 `/root/roboyi_train`，使用机器已有的 `/isaac-sim` 和
`/workspace/isaaclab`。项目 `.venv` 复用 Isaac Sim 自带的 Python 3.11 和
PyTorch 2.7.0+cu128，另外安装 RoboLab 和仓库固定的 RSL-RL 3.3.0。

## 当前机器人：Roboyi-Flat

主目录 `model/` 中的机器人已接入独立任务 **`Roboyi-Flat`**，直接加载
`model/isaacsim/roboyi.usda`，保留其中的自碰撞过滤。`RPO-*` 任务保留给
原 RoboParty 机器人；训练自己的机器人请使用 `Roboyi-Flat`。

2026-11-11 质量调整：当前模型全部连杆的质量和惯量统一乘以 **1.4**，
标称总质量为 **7.071565749 kg**（原为 5.051118392 kg）。URDF、USD 和 MuJoCo
模型同步修改；质心和几何保持原值。相关记录见 `model/mass_properties.json`。
足底力惩罚门槛/上限为 140 N，权重为 `-0.003 / 1.4`；基座附加质量扰动为
±0.28 kg。PD、执行器力矩/速度上限和电机 armature 沿用原值。
新建训练或重新启动训练后生效，当前已运行进程不会热更新。旧 checkpoint 尚未
针对新质量训练，旧配置引用相同模型路径，加载旧 checkpoint 也会使用当前的新质量。

该任务使用 10 个动作，actor/critic 单帧分别为 39/74 维，10 帧历史堆叠后
实际输入为 390/740 维。默认姿态沿用模型记录的零位；位置目标限制在关节软限位内。
显式 PD 使用 kp=100、kd=2、力矩限幅 20 N·m、PhysX 速度上限 3 rad/s、
armature=0.01，并保留独立被动阻尼 0.01。USD 内置驱动增益由 Isaac Lab 清零，
避免同时施加两套 PD。

电机目标指令现已启用与 RPO 相同的 `DelayedPDActuator` 延迟随机化：
`min_delay=0, max_delay=2`，单位是物理步。当前物理步长 0.005 s，对应
**0、5、10 ms**。每次环境重置时为该环境的整组腿部关节抽取一个延迟，
该回合内保持不变。延迟作用于目标指令进入 PD 之前，PD 仍读取最新关节反馈；
没有加入 IMU 或编码器回传延迟、丢包和逐包抖动。该范围沿用 RPO 仿真设定，
不代表仅凭电机型号相同就已确认实物通信时延相同。

旧策略需要在新延迟设置下重新训练或续训以学习适应。当前运行的进程不会热更新。
MuJoCo 的 roboyi 入口也已支持读取 checkpoint 对应参数中的延迟设置，旧零延迟
checkpoint 保持零延迟；含延迟 checkpoint 的物理步长换算见 `SIM2SIM_CN.md`。
该设置已通过 64 环境、10 轮 PPO 短测、带延迟 checkpoint 的 GPU 推理/重置检查，
以及 Isaac Lab/MuJoCo 延迟时序对照。记录位于 `outputs/roboyi_delay_verification/`。

当前指令为前后速度 −0.5～1.0 m/s、横向速度固定为 0、转向角速度
−0.5～0.5 rad/s；保留每 10 秒重采样、20% 站立环境和目标朝向控制。
平面速度跟踪中的 Y 分量用于抑制侧向漂移。

缺少躯干和手臂自由度的奖励已在 `roboyi_env_cfg.py` 中注释；腿部关节偏移
奖励去除了不存在的踝 roll 匹配。脚掌水平、接触、抬脚和滑动奖励仍然有效并保留。
足部传感器路径、足底高度 0.025 m、膝部选择和域随机化均已适配。

### 当前抬腿、着地与反弓奖励（更新至 2026-11-11）

当前奖励配置为：

- `feet_height` 使用摆动脚离地门槛奖励，权重 **0.3**。离地高度 **超过 3 cm**
  才给高度奖励，等于或低于 3 cm 不给分，超过后不再随高度增加；结合脚的平面速度，0.1 m/s
  达到速度因子满分。要求有运动指令、另一只脚支撑且摆动脚不接触地面，
  静止悬脚、双脚腾空、无效地面射线和倒置姿态不给分。
- `swing_knee_flexion` 权重 **0.1**，现已简化为只检查悬空脚对应的膝关节角度。
  单步贡献为 `0.02 × 0.1 × sum(悬空标记 × clamp(-q_knee / radians(20), 0, 1))`。
  正常屈膝 10°为半分，20°及以上封顶；直膝、反弓和着地腿不给此项奖励。
  接触判定仍采用最近接触力历史中的最大力是否超过 1 N。左右脚与膝按同侧配对。
  不检查离地高度、水平速度、运动指令、机身姿态或另一只脚是否支撑；因此静止悬腿
  也能得分，双脚同时悬空且正常屈膝时两侧奖励相加。每条悬空腿最多贡献 **0.002/步**，
  两腿合计最多 **0.004/步**。此项与 `feet_height` 分开计算，不修改后者的条件。
  原复合奖励的验证保留在 `outputs/roboyi_swing_knee_verification/rewards.json`；
  简化版验证使用 `outputs/roboyi_airborne_knee_verification/rewards.json`。
  简化版已通过 14 组接触/膝角用例（含左右顺序互换）和 16 环境、64 控制步 GPU
  运行检查；原抬脚、反弓和着地脚方向用例也通过。本次未进行 PPO 训练。
  不增加左右交替约束或悬空超时规则，不修改膝关节物理限位或默认姿态。
- `feet_air_time` 权重为 **0.25**，时间奖励在 **0.6 秒**封顶（2026-11-10 更新）。
  使用摆动脚腾空时间与支撑脚接触时间的较小值；超过 0.6 秒后保持封顶奖励，不归零。
- `feet_contact_orientation_l2` 当前权重 **−0.2**，只惩罚当前着地脚的倾斜。
  脚掌水平且局部 +Z 朝上时该项代价为零，倾斜越大代价越大；只绕竖直轴转向不受罚，
  悬空脚不受这一新增项惩罚。原有全程脚掌方向项 −0.1 仍保留。
- 新增 `knee_hyperextension`，权重 **−5.0**。本模型两侧膝关节的正常屈膝
  方向均为负角度，正角度会反弓，此方向已用模型运动学核对。代价为两膝
  `max(q, 0)² / (10°)²` 之和：单膝反弓 10° 对应 −5，20° 对应 −20，
  两膝代价相加。正常屈膝和直膝不受这一项惩罚，站立和跌倒时仍计算。
- 平面速度跟踪的 `std` 恢复为 **0.5**。

上述数值是乘控制周期之前的奖励率；环境会乘以 0.02 秒。例如单膝反弓 10°
在一个控制步内扣 0.1。两个速度跟踪项每步合计最多加 0.04，因此反弓惩罚较强。
这次只调整奖励，不修改模型、膝关节物理限位、默认零位或策略输入输出维数。
奖励不能提供机械硬限位保证。

新设置需要在新启动的训练进程中学习。仅运行 Play 加载旧 checkpoint 不会改变
已有策略；当前正在运行的训练进程也不会热更新。建议使用新的 run name，例如
`--run_name roboyi_gait_v2`，便于与旧奖励结果对比。

奖励行为和实际环境验证：

```bash
./scripts/python.sh scripts/verify_roboyi_gait_rewards.py --headless
```

验证结果及短程 PPO 日志位于 `outputs/roboyi_gait_reward_verification/`。
本次已通过膝关节方向检查、奖励边界用例、16 环境 GPU 运行检查及 64 环境
10 轮 PPO 短测（15,360 个环境步），权重已更新，损失、checkpoint 和优化器数值有限。
短测单独使用 `agent.experiment_name=roboyi_gait_reward_check`，不会成为正式
`roboyi_flat` 实验目录里默认选取的最新模型；结果不代表步态已经收敛。
上述 10 轮 PPO 短测对应此前连续高度奖励版本。后续按要求改回超过 2 cm 的门槛，
当前门槛奖励已通过边界用例，结果见 `threshold_reward_cases.json`；反弓惩罚未改变。

当前四项调整（抬脚 0.3、腾空时间 0.25、速度跟踪 std=0.5、新增着地脚水平约束）
另已通过 16 环境 GPU 检查和 64 环境、10 轮 PPO 短测。落地水平项覆盖水平、倾斜、
悬空、仅绕竖直轴转向、翻转及左右脚选择等用例。记录位于
`outputs/roboyi_stance_reward_verification/`，短测使用独立实验目录
`roboyi_stance_reward_check`。按用户要求，本次不加入悬空超时或左右落步交替约束；
短测不代表站立稳定性和跛行问题已收敛解决。

重新运行 PPO 短程测试：

```bash
cd /root/roboyi_train
./scripts/python.sh robolab/scripts/rsl_rl/train.py \
  --task Roboyi-Flat --headless --logger tensorboard \
  --num_envs 64 --max_iterations 10 --seed 42 --run_name roboyi_smoke \
  agent.save_interval=5
```

继续较长时间的训练（256 环境已通过本机续训验证）：

```bash
./scripts/python.sh robolab/scripts/rsl_rl/train.py \
  --task Roboyi-Flat --headless --logger tensorboard \
  --num_envs 4096 --max_iterations 5001 --seed 42 --run_name roboyi_walk

./scripts/python.sh robolab/scripts/rsl_rl/play.py \
  --task Roboyi-Flat --num_envs 1 --plane \
  --checkpoint '/root/roboyi_train/logs/rsl_rl/roboyi_flat/2026-11-08_04-29-17_roboyi_walk/model_5000.pt'

./scripts/python.sh robolab/scripts/rsl_rl/play.py \
  --task Roboyi-Flat --num_envs 1 --plane \
  --checkpoint '/root/roboyi_train/model_1300.pt'

./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --checkpoint /root/roboyi_train/logs/rsl_rl/roboyi_flat/2026-11-08_04-29-17_roboyi_walk/model_5000.pt \
 --duration 60 --keyboard

export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY=localhost,127.0.0.1,::1

./scripts/python.sh -m tensorboard.main \
  --logdir logs/rsl_rl/roboyi_flat \
  --host 127.0.0.1 --port 6600
在服务器的浏览器打开：http://127.0.0.1:6006
```


从本次验证的模型继续训练：

```bash
./scripts/python.sh robolab/scripts/rsl_rl/train.py \
  --task Roboyi-Flat --headless --logger tensorboard --num_envs 256 \
  --resume --load_run 2026-11-07_17-56-23_roboyi_resume_check \
  --checkpoint model_10.pt --max_iterations 9000 --run_name roboyi_walk_resume
```

2026-11-07 验证结果：64 环境训练 10 轮，共 15,360 个环境步，保存
`logs/rsl_rl/roboyi_flat/2026-11-07_17-55-02_roboyi_smoke/model_9.pt`；
256 环境从该权重及优化器恢复并训练 2 轮，共 12,288 个环境步，保存
`logs/rsl_rl/roboyi_flat/2026-11-07_17-56-23_roboyi_resume_check/model_10.pt`。
两次训练的策略权重均发生更新，checkpoint、优化器和全部 TensorBoard 标量均为有限数值。

加载 checkpoint 做 GPU 环境、策略推理和重置检查：

```bash
./scripts/python.sh scripts/verify_roboyi_runtime.py --headless \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_17-56-23_roboyi_resume_check/model_10.pt
```

上述脚本已通过 16 环境、128 步验证，覆盖观测形状、10 个执行器、关节目标限位、
被动阻尼、没有双重 PD、横移指令为零，以及强制超时后的正常重置。
结果和日志保存在 `outputs/roboyi_ppo_verification/`。

回放时使用相同任务，并默认显示机器人上方的速度箭头：**绿色为目标平面速度，
蓝色为实际平面速度**，箭头朝向表示方向、长度随速度变化。启动时指令为零，
绿色箭头长度为零；点击仿真视窗后按两次 `W` 可给出 0.1 m/s 前进指令。
这些箭头不表示转向角速度。训练和回放均默认开启箭头显示；无窗口训练时
也会创建标记，但需要带窗口运行才能直接看到。

```bash
./scripts/python.sh robolab/scripts/rsl_rl/play.py \
  --task Roboyi-Flat --num_envs 1 --plane \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_17-56-23_roboyi_resume_check/model_10.pt
```

这确认的是 PPO 训练、保存、恢复和推理链路，尚未验证步态收敛。现有 MuJoCo
`sim2sim_rpo.py` 仍对应原机器人；自己的机器人使用 `sim2sim_roboyi.py`，见
`SIM2SIM_CN.md`。该入口已适配关节顺序、PD、指令延迟和限速近似；直接使用原始
XML 不会自动获得这些控制逻辑。

以下 `RPO-*` 章节是原 RoboParty 机器人的训练记录和命令参考。

## 运行环境

```bash
cd /root/roboyi_train
# 首次配置或重建项目环境时执行
./scripts/setup_env.sh
# 查看已注册任务
./scripts/python.sh robolab/scripts/tools/list_envs.py
```

后续命令统一使用 `./scripts/python.sh`，它会同时配置 Isaac Sim 动态库路径和
项目 Python 环境，无需激活 Conda。仅执行 `source .venv/bin/activate` 并不足以
配置 Isaac Sim。`constraints-isaacsim51.txt` 固定基础依赖，避免 OpenCV 自动
升级到需要 NumPy 2 的版本。

启动脚本还将项目源码和 `.venv` 放在 Python 搜索路径前面，避免 Isaac Sim
预装的 RSL-RL 3.1.2 覆盖本仓库的 3.3.0。可确认实际加载的版本：

```bash
./scripts/python.sh -c 'import rsl_rl, importlib.metadata as m; print(m.version("rsl-rl-lib"), rsl_rl.__file__)'
```

输出应为 `3.3.0`，源码路径应位于 `/root/roboparty_train/rsl_rl/rsl_rl/`。

安装方式参考 [Isaac Lab 二进制安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)。

## 快速训练验证

`RPO-Flat` 使用 PPO 学习速度跟踪，不需要 AMP/BeyondMimic 动作数据集。
默认训练地形包含平地和轻微起伏。

```bash
./scripts/python.sh robolab/scripts/rsl_rl/train.py \
  --task RPO-Flat --headless --logger tensorboard \
  --num_envs 64 --max_iterations 10 --seed 42 --run_name ppo_smoke
```

10 轮训练用于检查仿真、采样、PPO 更新和模型保存；步态质量需要完整训练后评估。
每轮每个环境采样 24 步，64 个环境运行 10 轮共采样 15,360 个环境步。
日志在 `logs/rsl_rl/rpo_flat/<时间戳>_ppo_smoke/`，最终模型为 `model_9.pt`
（轮次从 0 开始），配置位于该目录的 `params/`。

## 正式训练

```bash
./scripts/python.sh robolab/scripts/rsl_rl/train.py \
  --task RPO-Flat --headless --logger tensorboard \
  --num_envs 2048 --max_iterations 9001 --seed 42 --run_name flat_ppo \
  agent.save_interval=100
```

当前 RTX 4090 已通过 2048 个环境、10 轮训练的验证，可从此并行数开始，
根据实际显存占用和每秒采样量调整。
内存不足时降低到 1024/512；资源充足时可尝试 4096。README 的 8192 是示例值，
当前任务和硬件的最大并行数需要实际测量。

上述命令每 100 轮保存一次模型，并在正常完成时保存最终模型 `model_9000.pt`。
原始配置的保存间隔为 1000 轮。按 `Ctrl+C` 中断不会自动额外保存一次，续训时
请选择已落盘的 `model_*.pt`。

`--task RPO-Rough` 用于复杂地形任务，实验目录为 `logs/rsl_rl/rpo_rough/`。
本次验证范围为 `RPO-Flat`。基础任务配置在
`robolab/robolab/tasks/direct/base/rpo_env_cfg.py`，PPO 参数在
`robolab/robolab/tasks/direct/base/agents/rpo_agent_cfg.py`。
常用 Hydra 覆盖示例：`agent.save_interval=100`、
`agent.algorithm.learning_rate=0.0003`。

## 监控与续训

```bash
# 另开终端；远程访问时可通过 SSH 转发 6006 端口
./scripts/python.sh -m tensorboard.main \
  --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
```

浏览器打开 `http://127.0.0.1:6006`，查看奖励、回合长度、value/surrogate loss
及吞吐量。短跑期间奖励波动不能用来判断是否已收敛。

```bash
# 替换为真实的运行目录名与 checkpoint 文件名
./scripts/python.sh robolab/scripts/rsl_rl/train.py \
  --task RPO-Flat --headless --logger tensorboard --num_envs 2048 \
  --resume --load_run '<运行目录名>' --checkpoint 'model_1000.pt' \
  --max_iterations 8000 --run_name flat_resume
```

续训会恢复权重和优化器，并创建新的日志目录。
`--max_iterations` 表示此次额外运行的轮数。

## 加载测试、录像与导出

```bash
# 替换为实际的完整模型路径；有桌面时显示仿真窗口
./scripts/python.sh robolab/scripts/rsl_rl/play.py \
  --task RPO-Flat --num_envs 1 --plane \
  --checkpoint '/root/roboparty_train/logs/rsl_rl/rpo_flat/<运行目录>/model_9000.pt'

./scripts/python.sh robolab/scripts/rsl_rl/play.py \
  --task Roboyi-Flat --num_envs 1 --plane \
  --checkpoint '/root/roboyi_train/logs/rsl_rl/roboyi_flat/2026-11-08_04-29-17_roboyi_walk/model_9000.pt'

# 无窗口测试：录制 200 步（4 秒仿真时间）后自动退出
./scripts/python.sh robolab/scripts/rsl_rl/play.py \
  --task RPO-Flat --num_envs 1 --plane --headless \
  --checkpoint '/root/roboparty_train/logs/rsl_rl/rpo_flat/<运行目录>/model_9.pt' \
  --video --video_length 200
```

`play.py` 自动导出模型到同一运行目录的 `exported/policy.pt` 和
`exported/policy.onnx`，录像位于 `videos/play/`。
`model_*.pt` 是可续训的 checkpoint，`exported/policy.pt` 是推理用 TorchScript。
`--plane` 只将测试地形改为平面。测试启动时速度指令为零；图形界面中可用键盘
调整指令：`W/S` 前进/后退，`A/D` 左右平移，`Q/E` 左右转向，`X` 停止，
`R` 重置。具体键位以 `robolab/robolab/utils/keyboard.py` 中的实现为准。

## 已完成的验证

2026-10-16 已完成以下验证，产物保留在当前工作区：

| 验证 | 配置 | 结果 |
| --- | --- | --- |
| 短跑 | `RPO-Flat`，64 环境，10 轮 | 15,360 个环境步，保存 `model_9.pt` |
| 并行训练 | `RPO-Flat`，2048 环境，10 轮 | 491,520 个环境步，策略权重更新，权重和损失数值有限 |
| 断点续训 | 加载上述 `model_9.pt`，2048 环境，额外 2 轮 | 恢复权重及优化器，保存 `model_10.pt` |
| 回放与导出 | 短跑模型，1 环境，200 步 | 导出 TorchScript、ONNX 和 4 秒视频 |

2048 环境训练模型：
`logs/rsl_rl/rpo_flat/2026-10-16_11-56-05_ppo_verify_2048/model_9.pt`。
续训模型：
`logs/rsl_rl/rpo_flat/2026-10-16_11-57-48_ppo_resume_verify/model_10.pt`。
训练日志为 `logs/ppo_verify_2048.log`，续训日志为 `logs/ppo_resume_verify.log`，
检查结果为 `logs/ppo_verification.json`。

可直接从已验证的模型继续训练：

```bash
./scripts/python.sh robolab/scripts/rsl_rl/train.py \
  --task RPO-Flat --headless --logger tensorboard --num_envs 2048 \
  --resume --load_run 2026-10-16_11-56-05_ppo_verify_2048 \
  --checkpoint model_9.pt --max_iterations 9000 --seed 42 \
  --run_name flat_resume agent.save_interval=100
```

本仓库的 RSL-RL 从 checkpoint 记录的轮次开始计数，因此从第 9 轮额外运行
2 轮时，日志编号为 9、10，最终文件为 `model_10.pt`。
这些短跑只确认训练链路可用，尚未验证步态收敛。

## 本地兼容处理

预装 Isaac Lab 2.3.2 快照没有新版入口引用的
`handle_deprecated_rsl_rl_cfg`。训练和普通 PPO 测试入口通过同目录 `compat.py`
使用兼容处理；较新的 Isaac Lab 仍调用其原生函数。
请保留这些本地修改以及项目的依赖约束。


cd /root/roboyi_train
./scripts/python.sh -m tensorboard.main \
  --logdir logs/rsl_rl/roboyi_flat \
  --host 127.0.0.1 --port 6006
在服务器的浏览器打开：http://127.0.0.1:6006



cd /root/roboyi_train

./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-08_00-06-38_roboyi_walk/model_9000.pt \
 --duration 60 --keyboard
