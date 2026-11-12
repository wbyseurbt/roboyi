# Roboyi：Isaac Lab → MuJoCo Sim2Sim

入口为 `robolab/scripts/mujoco/sim2sim_roboyi.py`，专用于自己的 **10 自由度 Roboyi-Flat** 策略。它加载训练权重，在 `model/mujoco/standing_scene.xml` 中运行机器人，不启动 Isaac Sim，也不重新训练。

当前模型已于 2026-11-11 将全部连杆质量及惯量统一调整为原来的 **1.4 倍**，
标称总质量为 **7.071565749 kg**，详见 `model/mass_properties.json`。旧 checkpoint
及其 env.yaml 不会恢复旧模型质量：XML 路径指向当前资产，复现加重前的结果需使用
质量调整记录中的备份模型。本文历史行走结果来自加重前模型，不代表新质量下的表现。

## 直接运行训练权重

在项目根目录运行。当前机器通过 `scripts/python.sh` 获得已安装的 PyTorch 和 MuJoCo 环境：

```bash
cd /root/roboyi_train
./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/model_4700.pt \
  --duration 60 --keyboard
```

窗口启动时速度指令为零。点击窗口后：

| 按键 | 作用 |
| --- | --- |
| W / S | 前进速度增加 / 减少，每次 0.05 m/s |
| Q / E | 转向角速度增加 / 减少，每次 0.1 rad/s；Q 左转 |
| 空格 | 三个速度指令全部归零 |
| R | 恢复初始站姿，清空动作和观测历史；保持当前速度指令 |
| Esc | 关闭窗口 |

A/D 可调整横移指令，但本次训练的横移范围是 `[0, 0]`，所以这两个按键不会产生横移指令。键盘指令会限制在 sim2sim 指令范围内，默认前进上限为 **1.0 m/s**。按住 W/S 可以连续调整，窗口标题会显示当前指令。

键盘模式使用独立的窗口按键处理，W/S/A/D/Q/E、空格和 R 不会再触发查看器的线框、颜色、透明度或暂停快捷键。鼠标旋转、平移和滚轮缩放仍可使用；未启用 `--keyboard` 时保留原生 MuJoCo 查看器。

机器人发生跌倒或非脚部的指定终止接触时会停止，需重新运行；这不会自动把失败轨迹重置后继续统计。

也可以直接给定速度，无需键盘：

```bash
./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/model_4700.pt \
  --vx 0.1 --wz 0.0 --duration 30
```

当前 sim2sim 默认范围为 `vx ∈ [-0.2, 1.0] m/s`、`vy = 0`、`wz ∈ [-0.5, 0.5] rad/s`。增加 `--max-vx` 可以调整前进速度指令上限，命令行和 W/S 键使用同一范围。例如设置上限 1.2 m/s，以 0.6 m/s 指令启动：

```bash
./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/model_4700.pt \
  --vx 0.6 --max-vx 1.2 --keyboard --duration 30
```

本次训练范围仍是 `vx ∈ [-0.2, 0.4] m/s`，提高仿真指令上限不会自动提高策略能稳定跟踪的速度；超过 0.4 m/s 属于训练范围外的测试。此参数控制机身前进指令，单位 m/s；它不控制关节的 rad/s 限制。训练配置和机器人 XML 不会被改写。

命令行给出超出 sim2sim 指令范围的值会报错。更换权重时替换 `--checkpoint` 的文件路径；脚本从该文件相邻的 `params/env.yaml` 和 `params/agent.yaml` 读取当次训练配置，不自动选择最新权重。

## 无界面运行和录像

服务器没有桌面时，使用 `--headless`，无需 OpenGL 或显示器：

```bash
./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/model_4700.pt \
  --headless --vx 0.1 --duration 30 \
  --output-dir outputs/sim2sim_roboyi/forward
```

无界面默认尽快运行；`--duration 30` 指 30 秒仿真时间。加入 `--real-time` 可以按实际时间运行。窗口模式默认按实际时间运行。

增加 `--video` 保存视频：

```bash
MUJOCO_GL=egl ./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/model_4700.pt \
  --headless --vx 0.1 --duration 10 \
  --video outputs/sim2sim_roboyi/forward.mp4
```

视频需要可用的 EGL/OpenGL 驱动以及 `imageio`、`imageio-ffmpeg`；当前机器已具备这些依赖。`--headless --video` 未设置 `MUJOCO_GL` 时，脚本默认选择 EGL。如果视频渲染不可用，可以去掉 `--video`，物理仿真和结果记录仍可运行。

## 导出并使用 policy.pt

支持直接读取 checkpoint，无需先打开 Isaac Lab 的 play 窗口。也可以在运行时同时导出 actor 和观测归一化参数：

```bash
./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/model_4700.pt \
  --headless --duration 1 \
  --export-policy outputs/roboyi_4700/policy.pt
```

随后运行此 TorchScript 策略：

```bash
./scripts/python.sh robolab/scripts/mujoco/sim2sim_roboyi.py \
  --load-model outputs/roboyi_4700/policy.pt \
  --config-dir logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/params \
  --vx 0.1 --duration 30
```

对于训练目录下标准的 `exported/policy.pt`，脚本会自动查找上一层的 `params/`。导出文件移到其他位置时应显式传入 `--config-dir`。必须选择与权重配套的训练配置。现有 `play.py` 导出的 Roboyi TorchScript 策略也可通过 `--load-model` 读取。此入口暂不支持 ONNX、RPO 23 关节策略、循环策略或带地形扫描的策略。

## 如何看结果

每次运行默认写到 `outputs/sim2sim_roboyi/<北京时间时间戳>/`，也可通过 `--output-dir` 指定。文件包括：

- `summary.json`：实际运行时长、停止原因、是否跌倒、最大倾角、各关节峰值速度/力矩、速度超限比例和 MuJoCo 数值警告；同时记录 sim2sim 与训练的速度指令范围。
- `trajectory.npz`：每个策略周期记录的观测、动作、目标角度、实际角度、关节速度、机身位置/四元数、机身速度和力矩。关节数组均按策略关节顺序保存，文件内有 `joint_names`。

关节限速默认开启，阈值读取训练配置，当前每个关节为 **±3 rad/s**。报告中的 `velocity_limit_enforced=true` 表示开启；`velocity_limit_active_steps` 为限速发生的物理步数，`peak_pre_limit_joint_speed_rad_s` 是每步施加限速前的速度，`peak_joint_speed_rad_s` 是实际接受的限速后速度。`peak_velocity_limit_impulse_Nm_s` 为各关节峰值限速约束冲量。轨迹中的 `velocity_limit_impulse` 记录每个控制周期最后一个物理步的冲量，单位 N·m·s；整个过程的峰值和介入次数以报告为准。

`stop_reason=duration_reached` 表示运行到了指定时长，不表示速度跟踪一定良好；还需要观察步态、倾角和实际速度。`base_height`、`base_orientation`、`forbidden_body_contact` 表示触发对应的跌倒/接触终止条件。发生跌倒或数值警告时 Python 脚本返回码为 2，正常完成或主动关闭窗口为 0。本机 `scripts/python.sh` 调用的 Isaac Sim 启动器会把非零返回码统一转成 1，并打印 `There was an error running python`；此时先查看已保存的 `summary.json` 区分策略跌倒和程序异常。

轨迹中状态、观测、动作和目标对应每个控制周期开始时刻；`torque` 是该周期最后一个物理步的执行器力矩。`base_velocity` 顺序为机身坐标系的 `[wx, wy, wz, vx, vy, vz]`，`base_quat` 为 `wxyz`。`time` 是整段记录的累计仿真时间，`episode` 标记键盘重置后的片段。

例如读取运行结果：

```bash
./scripts/python.sh -c 'import numpy as np; d=np.load("outputs/sim2sim_roboyi/forward/trajectory.npz"); print("关节:", d["joint_names"]); print("平均前进速度:", d["base_velocity"][len(d["time"])//5:, 3].mean())'
```

## 已对齐的接口

- 单帧为 `角速度(3) + 重力方向(3) + 指令(3) + 相对默认关节位置(10) + 相对默认关节速度(10) + 上一拍动作(10)`，共 39 维。10 帧历史从旧到新排列，首次观测填满历史，与 Isaac Lab `CircularBuffer` 相同。
- 使用机身坐标系角速度和重力方向；MuJoCo 通过 `mjOBJ_XBODY` 获取机身速度，避免误用惯性主轴坐标系。参见 [MuJoCo 的坐标帧实现](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_core_util.c)。
- 策略顺序为 `Left_pitch, Right_pitch, Left_roll, Right_roll, Left_yaw, Right_yaw, Left_knee, Right_knee, Left_ankle, Right_ankle`。按名称找到 MuJoCo 的 qpos、qvel 和执行器地址，不依赖 XML 数组顺序。
- 加载 RSL-RL 的原始网络和归一化统计，使用确定性动作均值，不添加训练时的动作采样噪声或观测噪声。
- 当前配置中策略动作先裁剪至 ±3，再乘 0.15，加默认站姿，并应用 0.9 倍关节软限位。上一拍动作使用裁剪后、缩放前的动作，按训练环境的处理顺序存入观测。
- 默认物理步长 0.005 s，策略周期 0.02 s；每个物理步重算 `tau = kp*(q_target-q) + kd*(dq_default-dq)`，并做力矩限幅。XML 中的位置执行器在内存中改为单位增益力矩执行器，避免重复 PD；源 XML 不修改。
- kp、kd、armature、被动阻尼、力矩限制和初始站姿从保存的训练配置读取。当前为 kp=100、kd=2、力矩 ±20 N·m、armature=0.01、被动阻尼=0.01。
- 支持保存配置中的 `min_delay/max_delay` 电机目标指令延迟。重置时为整组关节
  抽取延迟，回合内固定；当前新训练为 0/1/2 个训练物理步，即 0/5/10 ms。
  延迟缓冲作用于位置目标，PD 仍使用最新关节位置和速度；旧零延迟 checkpoint
  保持原行为。随机数使用保存的环境 seed；不保证与 Isaac Lab 抽到完全相同的序列。
  启动输出和 `summary.json` 会记录所用的延迟范围和当前回合的抽样值。
- 可细化 MuJoCo 物理步长，策略仍为 50 Hz。零延迟配置可用 `--physics-dt 0.002`，
  步长必须整除 0.02 s；含延迟配置还必须精确整除保存的训练物理步长，例如
  `--physics-dt 0.001` 可精确表示 5 ms 延迟，而 0.002 s 无法精确表示，会明确报错。
  改变物理步长属于与默认步长不同的实验条件。

## 当前物理差异

训练中的 PhysX 使用 3 rad/s 的关节速度约束。MuJoCo 3.3.3 没有相同的原生关节参数，本脚本默认通过**质量矩阵加权的速度约束冲量**补齐每步的速度上限：

1. MuJoCo 完成当前步的电机、重力和接触动力学预测。
2. 如果预测的关节速度超过训练限制，求解约束冲量，使修正后的所有关节速度处于限制内。目标是最小化质量矩阵度量下的速度变化；冲量通过完整质量矩阵作用于耦合关节和浮动基座，避免独立裁剪每个关节速度。
3. 使用修正后的广义速度，从该步原始位置重新积分，再刷新机身状态。不会保留按超速计算的关节位置。

质量矩阵求解和位置积分分别使用 MuJoCo 的 [`mj_solveM`](https://mujoco.readthedocs.io/en/3.3.3/APIreference/APIfunctions.html#mj-solvem) 和 [`mj_integratePos`](https://mujoco.readthedocs.io/en/3.3.3/APIreference/APIfunctions.html#mj-integratepos)。约束求解容差为 1e-10 rad/s，报告按 1e-6 rad/s 容差统计超限；它约束实际仿真速度，而不只是限制目标角度变化率。

这是**分步求解的限速近似**，与 PhysX 将速度约束和接触约束一起处理的结果不完全相同。额外的约束冲量与执行器力矩分开记录；电机本身仍维持 ±20 N·m 限幅，不能把“约束冲量 / 步长”当成真实电机输出。它用于对齐训练中的理想速度约束，并不等于已经标定真实电机的转速—力矩特性。

默认运行命令无需增加参数即可启用限速。需要与旧行为对比时，额外传入 `--disable-joint-velocity-limit`。该选项仅用于对照；源 XML 和训练配置均不修改，直接在其他程序加载 XML 不会自动获得这里的限速逻辑。

摩擦、接触求解和积分器也有引擎差异。MuJoCo 使用模型内的标称质量、惯量、摩擦和碰撞过滤；首轮迁移不复制训练中的随机质量、随机摩擦、推力等域随机化。非脚部接触终止按 MuJoCo 的瞬时接触力近似训练条件，无法完全复现 Isaac Lab 接触传感器的历史聚合。

## 验证接口一致性

```bash
./scripts/python.sh scripts/verify_roboyi_sim2sim.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/model_4700.pt
```

该检查将 checkpoint 和 TorchScript 输出与训练用 `ActorCritic.act_inference` 对比，并验证关节映射、历史观测顺序、上一拍动作、机身角速度坐标系、重力方向、PD 力矩、软限位、重置以及两个加载入口的轨迹一致性。限速检查包括浮动基座动量耦合、能量不增、位置积分一致性、未触发时不改变状态，以及 5 ms / 2 ms 步长下正反向强驱动和初始超速恢复。它验证接口和限速，不替代长期行走测试。

真实窗口的按键回归检查可以在安装了 `xvfb`、`xdotool` 的 Linux 环境运行：

```bash
xvfb-run -a ./scripts/python.sh scripts/verify_roboyi_sim2sim.py \
  --checkpoint logs/rsl_rl/roboyi_flat/2026-11-07_18-09-15_roboyi_walk/model_4700.pt \
  --viewer
```

该检查通过 X11 发送真实按键，确认 W 可超过原来的 0.4 m/s 指令上限，且所有机器人控制按键均不改变显示标志、几何体颜色、透明度或暂停状态。

## 当前权重的实测结果

以下为**补充关节限速之前**的对照结果。2026-11-07 使用 `model_4700.pt`，默认 0.005 s 物理步长、标称模型、无外部推力，结果保存于 `outputs/sim2sim_roboyi_validation/`：

| 指令 vx / wz | 运行结果 | 去掉前 20% 样本后的平均机身前进速度 |
| --- | --- | --- |
| 0 / 0 | 约 12.455 s 触发机身高度终止 | — |
| 0.1 m/s / 0 | 运行满 30 s，没有触发跌倒条件 | 约 0.0027 m/s |
| 0.1 m/s / 0.2 rad/s | 运行满 30 s，没有触发跌倒条件 | 约 0.0243 m/s |
| 0.3 m/s / 0 | 运行满 30 s，没有触发跌倒条件 | 约 0.1623 m/s |

接口一致性检查通过；窗口模式、TorchScript 入口和 EGL 录像也已实际运行。上述物理仿真均未出现 MuJoCo 数值警告。0.3 m/s 指令测试约 16.97% 的物理步有至少一个关节超过训练中的 3 rad/s 限制；转向测试约 0.15%，其余两项未超限。

因此，这个 checkpoint 已能接入 MuJoCo 并在部分指令下行走，但零指令稳定性、速度跟踪和航向保持仍需评估，不能仅凭脚本跑通就认定迁移质量合格。更换后续权重后，应重新运行相同指令测试。

新增限速后，使用同一权重、`vx=0.3 m/s` 重新运行 30 秒：未触发跌倒条件，各关节速度不超过 3 rad/s（约 1e-10 数值误差），超限步比例为 0，无 MuJoCo 数值警告。去掉前 20% 样本后的平均机身前进速度约 0.152 m/s。结果位于 `outputs/sim2sim_roboyi_velocity_limit/forward_03/`。这确认了限速生效，尚不能据此认定真机步频或速度跟踪已完全一致。

## 其他机器的依赖

已安装项目训练环境时直接使用该环境。单独准备 MuJoCo 环境时，使用 Python 3.11，并安装 `torch`、`numpy`、`PyYAML`、`mujoco==3.3.3`；读取 checkpoint 还需要本仓库的 RSL-RL 及其依赖，可执行 `pip install -e ./rsl_rl`。仅运行导出的 TorchScript 无需 RSL-RL 或 Isaac Lab。键盘窗口模式使用项目已安装的 `mujoco-python-viewer` 包，单独安装时运行 `pip install mujoco-python-viewer`。录像额外安装 `imageio`、`imageio-ffmpeg`。窗口模式需要桌面显示，SSH 无显示时使用 `--headless`。
