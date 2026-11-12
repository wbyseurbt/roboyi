# roboyi

三份模型使用相同的机器人名称、关节名称和杆件名称，保留左右侧的 `Left` / `Right` 大小写。

## 模型入口

- URDF：`mujoco/meshes/roboyi.urdf`
- MuJoCo：`mujoco/roboyi.xml`
- MuJoCo 带地面站立场景：`mujoco/standing_scene.xml`
- USD：`isaacsim/roboyi.usda`，默认根节点 `/roboyi`
- USD 带地面站立场景：`isaacsim/standing_scene.usda`

USD 必须连同 `isaacsim/Payload` 目录使用；URDF 和 XML 的网格位于 `mujoco/meshes`。网格文件名已与杆件名同步，网格内容未改变。

## 统一命名

| 左关节 | 右关节 | 左杆件 | 右杆件 |
| --- | --- | --- | --- |
| Left_pitch | Right_pitch | Left_hip_pitch_link | Right_hip_pitch_link |
| Left_roll | Right_roll | Left_hip_roll_link | Right_hip_roll_link |
| Left_yaw | Right_yaw | Left_thigh_link | Right_thigh_link |
| Left_knee | Right_knee | Left_shank_link | Right_shank_link |
| Left_ankle | Right_ankle | Left_foot_link | Right_foot_link |

躯干保持 `base_link`。关节名描述转动自由度，杆件名描述对应实体。

## 坐标约定

三份模型均采用与 RPO 相同的右手坐标约定：**+X 向前、+Y 向左、+Z 向上**。标准零位时，11 个杆件的参考坐标轴与本体轴对齐，左右脚掌局部 +Z 也朝上。运动时各杆件坐标随杆件转动。

旧 CAD 本体坐标到新坐标的映射为 `(x新, y新, z新) = (y旧, -x旧, z旧)`。本次重新表达了杆件、网格、质心、惯量和关节框架，保留机械结构、关节角零位及正方向。CAD 导出的真实关节轴倾斜仍保留，不把所有转轴强制改为理想轴。原始 STL 字节未变，网格由各模型的局部变换定位。

`coordinate_convention.json` 保存转换矩阵、每个杆件的局部坐标映射及修改前备份路径。零位、限位和执行器参数保持不变；训练代码未改，10 自由度任务配置与控制接入仍需单独完成。

## 姿态和参数

- 新零位为躯干直立、膝伸直、双脚底面水平；10 个驱动关节均为 0。
- Roll ±15°，其他关节 ±45°；目标速度上限为 3 rad/s。USD PhysX 使用原生速度限制；URDF 声明上限，需导入器执行；当前 MuJoCo XML 的位置执行器没有实现该速度硬约束。
- kp=100、kv=2，执行器力矩限幅 ±20 N·m。
- RoboParty DM4340 系列仿真参考：被动阻尼 0.01 N·m·s/rad、关节侧附加惯量 0.01 kg·m²。
- `actuator_parameters.json` 保存参数来源；`standing_pose.json` 保存软件零位映射；`naming_map.json` 保存此次更名前后的对应关系。
- 标准 URDF 没有通用的 armature 字段，后续接入配置应读取配套参数表。URDF 网格使用同目录下的 `./*.STL` 相对路径，不再依赖 ROS 包映射。
- USD 的 11 个刚体为 `/roboyi/Geometry` 下的同级节点，已移除固定基座关节，关节局部框架已随新坐标同步换算。

## 质量调整（2026-11-11）

当前 URDF、MuJoCo XML 和 USD Physics 层的 11 个连杆质量均为原来的 **1.4 倍**，
标称总质量由 **5.051118392 kg** 增至 **7.071565749 kg**。各连杆惯性张量同时乘以
1.4；按几何和质量分布比例不变处理，质心、主惯性轴、网格、关节框架及默认站姿不变。
USD 同时更新 PhysX 的主惯量及 Newton 的完整惯性张量。

`mass_properties.json` 记录各连杆原始/当前质量、缩放假设及原文件备份位置。
缩放已写入模型文件，加载时不得再次乘以 1.4。电机额外转子惯量 armature、PD 增益、
被动阻尼、力矩和速度上限沿用原电机参数；增加连杆质量不代表增加电机能力。

Roboyi 训练配置同步调整基座附加质量扰动至 ±0.28 kg，保持其原相对幅度；
足底力惩罚的门槛和超量上限从 100 N 增至 140 N，权重除以 1.4，以维持同等
相对体重下的惩罚。腿部质量 ±10% 随机化继续围绕当前的新质量执行。

旧 checkpoint、导出策略及其 `params/env.yaml` / `agent.yaml` 没有重写或重新训练。
这些配置引用的模型路径仍相同，因此用旧策略启动当前模型时也会加载新质量；
复现加重前仿真需要恢复备份模型，不能只使用旧 env.yaml。
已运行的 Isaac/MuJoCo 进程不会热加载修改，需重启后生效。

## 碰撞过滤

保留自碰撞，排除 `base_link` 与左右髋 pitch、髋 roll、大腿共 6 对装配碰撞。MJCF 与现成 USD 已内置相同过滤。

`collision_filters.json` 和 `mujoco/meshes/roboyi.srdf` 保存这 6 对配置。标准 URDF 无法独立表达这些过滤；重新导入 URDF 时必须显式应用，不能假定 Isaac Sim 自动读取 SRDF。需要直接使用已包含过滤的资产时，加载现成 USD。

## 验证范围

2026-11-07 坐标转换后的正式模型检查：9 组关节姿态下，URDF/MJCF 的网格、质心和关节轴均与原模型做相同坐标转换的结果一致；USD 几何及关节框架同步通过检查。MuJoCo 站立 8 秒，双脚接地、无数值警告，最大倾角约 0.810°。

独立 ovphysx CPU 引擎：11 个杆件、10 个关节；2 ms 与 5 ms 步长均站立 2 秒，最大躯干倾角分别约 0.821°、0.888°；无地面场景自由落体验证浮动基座。运行时读回的 kp、kv、力矩、速度限制、armature 和被动阻尼均与参数表一致。

本次没有修改训练代码，也未运行 Isaac Lab/PPO、Newton 或真机。最新坐标转换说明和结果位于 `../model_checks/20261107/axis_alignment/`；较早的模型修复记录保留在 `../model_checks/20261107/report_fixes/`，其中“CAD 坐标保留”的描述属于本次转换前版本。
