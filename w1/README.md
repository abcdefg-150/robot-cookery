# 双 W1/G2 做饭仿真环境

当前入口为 `w1_finals_real_v7`，包版本为 `w1-cookery-dual-v3`。此目录是 RoboTwin 2.0 的覆盖包，包含运行所需的 W1 模型、处理后的锅与食材资产、任务脚本、相机标定参考和采集接口。原始实机视频、ZIP、训练 HDF5 不进入本仓库。

## 场景与任务

- 两条黑色 W1 六轴机械臂，各带 G2；底座中心左右相距 0.60 m，桌面高度 0.74 m。
- 锅位于中央；锅铲位于锅右侧，柄朝前，**初始背面朝上、工作面朝下**，由真实碰撞接触自然静置。
- 食材为两颗青椒、土豆、茄子。锅、锅铲、青椒、茄子使用用户提供的模型，土豆使用 RoboTwin 资产。
- 流程：左右臂同时抓取大/小青椒、同步搬运、交错入锅 → 左右臂同时抓取茄子/土豆、同步搬运、交错入锅 → 左臂避让 → 右臂抓锅铲 → 小幅往返推动 → 锅铲放回桌面 → 双臂归位、夹爪闭合。

翻炒按用户确认的标准判断：锅内真实接触任一种食材、可见食材位移、小幅往返推动，累计有效运动至少 2 秒。实现中要求被接触食材至少移动 5 mm、推动方向至少反转一次（有效行程至少 8 mm）。计时仅包括锅内往返阶段、工具速度大于 3 mm/s 且最近 0.5 秒内发生过食材接触的运动；静止等待不计时。记录实际接触时长，**不要求 45° 转角或两次翻面**。最终还检查所有食材完整留在锅内、受支撑且连续静止至少 1 秒、锅铲释放归位及无严重碰撞。

锅的 26 cm 是用户提供的待核实尺寸，内径/外径含义未确定。本版按实机画面拟合，锅体外宽约 30.0 cm；资产清单保留估计来源。食材尺寸及腕相机安装也是图像拟合值。前相机采用实机内参及外参；当前锅固定，实机中锅会移动。质量、摩擦与驱动参数为仿真参数，尚未进行实机动力学辨识。

## 搬运姿态与轨迹检查

两轮抓取和搬运由双臂同步执行；进入共享锅区时右手先放、左手在锅外等待，随后右手退出与左手进入协同执行。松爪后采用向上并向外的退出路径，给已放食材留出空间。两轮之间直接前往下一颗食材，全部食材入锅后再归位。

翻炒按当前锅内分布搜索低转角推动走廊，双臂放置模式将推动起点向食材靠近 15 mm。

双臂段采用 C2 关节曲线及五次时间参数化，段端速度、加速度为零，按较慢机械臂延长另一条轨迹，不通过加速追赶同步。关节 jerk 上限为 8 rad/s³。每个 250 Hz 控制点检查自碰、桌面、锅体和双臂球体包络（含所持食材），双臂预测净间隙至少 12 mm；下降与张爪路径在入锅前预检。物理执行中任何双臂有效接触均拒绝采集。实际反馈另外检查两臂重叠运动时间及高频残差，失败候选不会加入 accepted。

原生真值记录 `arm_fsm_state`，HDF5 同时保存 `additional_info/left_fsm_state` 和 `right_fsm_state`，以表示两臂并行执行的阶段。

仿真 HDF5 在 `additional_info/fsm_instruction` 保存与每帧观测对齐的英文阶段指令。双臂模式还保存 `left_fsm_instruction` 和 `right_fsm_instruction`。这些标签由专家状态机生成，`instructions` 继续保存整条轨迹的任务指令。

食材入锅没有指定 yaw。状态机优先保持抬起后的实际姿态；不可达时才按小角度递增倾斜，通过连续路径检查后执行。锅铲保持背面朝上的初始摆放，抓起后优先选择与当前姿态接近的推动角度，并预检抬起、入锅和推动走廊。接触后按观察到的夹持姿态退出和放回，放回位置仍为原桌面区域。

检查覆盖整个运动过程。除了限位、避碰与物理接触，还累计每个控制点之间的 SO(3) 转角，拒绝起终点接近但途中绕圈的 IK 分支。空爪归位也按完整轨迹选取腕部解。采集后在原生约 250 Hz 末端反馈上再次独立检查：四次食材搬运各不超过 30°，锅铲入锅不超过 75°，下降不超过 15°；各阶段累计转角超出起终点最短转角不超过 25°。这些是采集质量上限，实际轨迹优先保持姿态。

```bash
python scripts/analyze_w1_motion.py data/w1_cookery_001/candidates/seed_0000 \
  --output data/w1_cookery_001/motion_quality.json --require-pass
```

采集器自动执行该检查，并将结果写入候选的 `motion_quality.json` 和 HDF5 的场景元数据。不合格候选不会加入 accepted 列表。工作区已清理历史示范、旧预览和失败测试输出，仅保留当前 v7 的双臂验收示范及原始实机输入。

## 安装

需要已安装的 RoboTwin、XPolicyLab、cuRobo、SAPIEN、可工作的 NVIDIA GPU 和 `ffmpeg`。已测试的 Python 包版本见 [tested_dependencies.json](tested_dependencies.json)，基线提交见 [manifest.json](manifest.json)。请复用原 RoboTwin 的 Python/CUDA 环境。

在 `robot-cookery` 仓库根目录执行，将路径换成自己的 RoboTwin 安装位置：

```bash
python w1/install.py --target ~/RoboTwin --dry-run
python w1/install.py --target ~/RoboTwin
```

安装器先核对包内文件哈希、核心补丁和注册表冲突，通过后再写入。原有核心文件备份到目标目录的 `.w1_install_backups/`，安装记录为 `.w1_cookery_install.json`。再次安装相同版本会跳过相同内容。发生冲突时请先处理提示的文件，不要直接覆盖整个 RoboTwin 目录。现有面包任务及其他机器人的注册表条目会保留。

安装器会根据目标目录从 `curobo_tmp.yml` 生成 `curobo.yml`，因此包内不保存开发机器的绝对路径。需要更改规划器参数时修改模板，再重新安装。RoboTwin 通用资源索引只在缺失时补入，不覆盖已有索引。

`overlay/` 中的相对路径均以 RoboTwin 根目录为基准；`patches/` 包含必要的框架和 π0.5 数据读取补丁。运行仿真不需要 `episode_0001.zip`，相机参考和原生记录格式模板已经独立保存。

## 预览与接口检查

以下命令都在安装后的 RoboTwin 根目录执行，使用 RoboTwin 环境的 Python。GPU 程序串行运行。输出目录需要使用新名称，避免覆盖旧记录。

```bash
# 录制正面及左右腕部三视角预览，不导出训练示范
python scripts/preview_w1_current.py --config w1_finals_real_v7 --output data/w1_preview_001

# 30 Hz reset/step 接口的保持动作检查；此输出不属于成功示范
python scripts/example_w1_policy_collection.py --config w1_finals_real_v7 \
  --steps 30 --record-dir data/w1_api_smoke_001 --report data/w1_api_smoke_001_report.json

# 判定器和接触过滤测试
python -m unittest discover -s scripts/tests -v
python -m unittest discover -s XPolicyLab/policy/Pi_05/tests -v
```

预览默认输出 `current_scene_1x.mp4` 和 `preview_report.json`。检查报告中的任务结果；有画面并不表示整条任务通过。预览为 15 FPS 合成画面，训练记录使用下文的三路独立 30 Hz RGB-D。

## 采集一条完整示范

```bash
python scripts/collect_w1_real_style.py --config w1_finals_real_v7_diverse \
  --output data/w1_cookery_001 --episodes 1 --max-attempts 5
```

输出结构：

```text
data/w1_cookery_001/
  collection_summary.json                  # 实际成功数量、失败尝试和独立验证结果
  provenance.json                          # 本批关键输入的 SHA256
  task_config.yml / asset_manifest.yml
  candidates/seed_XXXX/
    meta.json / scene_info.json
    actions/{left_arm,right_arm,car}.jsonl + .idx
    sensors/cam_{front,left,right}_{rgb.mp4,depth.mkv} + .tsf
    extra/ground_truth.jsonl + .idx
  native_episode_0000.zip
  w1_finals_real/w1_g2/data/
    episode_0000000.hdf5
    episode_0000000.alignment.json
    episode_0000000.validation.json
```

采集器默认使用 `w1_finals_real_v7_diverse`。每个 seed 独立采样食材摆位、双臂分工、锅和锅铲位置、三路相机与照明。正面相机以实采两套标定为中心做扰动，并在需要时加宽视野让食材保持在画面内。照明在整条 episode 内保持固定。原生元数据和 HDF5 场景记录保存实际采样参数、静置后位姿，以及正面画面的可见性检查结果。使用 `--config w1_finals_real_v7` 可复现固定参考场景。

分阶段标签与实采 `w1_lerobot_subtask_v1` 对齐。HDF5 的 `additional_info/subtask_index` 和 `subtask_instruction` 按帧记录两颗青椒入锅、土豆和茄子入锅、锅铲翻炒三个阶段，`metadata/subtasks` 保存 ID 对照表。细粒度状态机指令同时保留。仿真完整任务包含土豆，因此使用三个任务阶段的 ID 0–2。

只使用 `collection_summary.json` 中 accepted 列表对应的成功示范。失败候选保留用于排查，转换器会拒绝 `success=false` 的仿真记录。不要把预览、保持动作检查、失败候选或旧版本数据直接混入本批训练集。

原生机器人轨迹约 250 Hz；三路 RGB-D 为 640×480、30 Hz。RGB 使用 H.264，深度是 FFV1 保存的 uint16 毫米。统一 HDF5 在共同的真实时间轴上重采样到 **30 Hz、14 维**，保留源相机帧号和时间戳。

## VLA 接口与数据约定

```python
from scripts.w1_sim_interface import W1Simulation

sim = W1Simulation("w1_finals_real_v7", control_hz=30)
try:
    obs = sim.reset(seed=0)
    action = obs["state"].copy()  # 在这里替换为 VLA 输出
    obs, reward, terminated, truncated, info = sim.step(action)
finally:
    sim.close()
```

在线接口的动作与状态顺序为 `[左 q1…q6, 左开度, 右 q1…q6, 右开度]`。关节目标用 rad 表示绝对位置，每个关节相对当前反馈位置每步最多变化 0.05 rad。夹爪总开度用米，范围为 `[0, 0.072]`，例如 `0.036` 表示 36 mm。观测包含 `state[14]`、三路 RGB `images`、三路 uint16 毫米深度 `depths` 和时间戳。物理频率 250 Hz，接口每次推进 8 或 9 步，实现 30 Hz 控制。

π0.5 若使用本包离线 HDF5 训练，模型输入前需将在线 state 的第 6/13 维除以 0.072，模型输出的第 6/13 维乘以 0.072 后再传给 step。离线 HDF5 转换仍使用归一化开度。其 state 为从手宽度 / 0.072，action 为主手宽度 / 0.0479。原生 EE 四元数为机械臂底座坐标系的 xyzw，仿真物体真值为世界坐标的 wxyz。时间戳保留 float64。仿真 effort 为零占位。实机 `master_qpos` 记录遥操主手编码器参考，原始记录未提供实际发送给从手的网络命令。

若需要转换另一个同格式实机记录：

```bash
python scripts/convert_w1_native.py /path/to/unpacked_episode \
  data/my_real_aligned/episode_0000000.hdf5
```

## π0.5 数据准备

本包提供 14 维 W1 注册和现有 π0.5 读取器的适配。RGB 编解码统一使用 XPolicyLab 的共享函数。HDF5 保留深度，现有 π0.5 输入默认使用 RGB。

在已安装 OpenPI/LeRobot 的独立训练环境中，从 `XPolicyLab/policy/Pi_05/openpi` 执行：

```bash
python scripts/process_data.py w1_cookery_001 w1_finals_real w1_g2 joint --fps 30
```

`--fps 30` 会核对 HDF5 的实际频率，不会把 50 Hz 数据虚假改标成 30 Hz。后续将最终数据集 repo_id 写入 `pi05_w1_g2_full`，对最终混合集计算 W1 自己的归一化统计，再启动训练；默认的 `w1_g2_final` 是占位名称。实机与仿真来源分别标记，训练/验证按原始 episode 分组，不把相邻帧或同一轨迹的分段拆到不同集合。

本次交付验证仿真、采集和读取接口；模型训练另行运行。实际验收结果见 `validation/`，示例视频见 `examples/`。

## 本版验收结果

- 独立安装目录从初始场景采集完成 1 条 seed 0 完整示范，2894 帧、30 Hz、14 维、三路 640×480 RGB-D；总时长 96.54 秒。
- 实际反馈中两臂同时运动 26.53 秒；预测最小双臂净间隙 16.14 mm，要求至少 12 mm。双臂接触与严重碰撞均为 0。
- 左臂食材阶段末端高频残差 RMS 0.001 mm（250 Hz 反馈减去 31 点三次 Savitzky–Golay 趋势）。
- 右臂食材阶段末端高频残差 RMS 0.001 mm（250 Hz 反馈减去 31 点三次 Savitzky–Golay 趋势）。
- 有效翻炒 2.556 秒，实际接触 2.340 秒，方向反转 3 次，食材最大位移 0.67 cm；锅铲释放归位、四食材稳定及双臂归位全部通过。
- 原生 IDX/TSF、ZIP CRC、HDF5 独立检查和 π0.5 全部 2894 帧读取通过；在线米制开度/0.05 rad 限幅接口验证、28 项任务测试和 7 项转换测试通过。重复安装无待修改文件。

实际末端搬运累计转角（SO(3)，含中途转动）：

| 阶段 | 累计转角 ° |
| --- | ---: |
| GREEN_PEPPER_TRANSFER | 0.52 |
| EGGPLANT_TRANSFER | 17.65 |
| SMALL_PEPPER_TRANSFER | 0.33 |
| POTATO_TRANSFER | 5.78 |
| STIR_APPROACH | 19.61 |
| STIR_LOWER | 0.31 |

视频：[完整流程 4×](examples/complete_workflow_4x.mp4)、[双臂抓取与放置 1×](examples/dual_food_pick_place_1x.mp4)、[锅铲翻炒归位 1×](examples/tool_grasp_stir_return_1x.mp4)。

实际轨迹及双臂活动图见 [dual_motion.png](examples/dual_motion.png)。本次仅验收 1 个种子；尚未进行多种子成功率评估或模型训练。高频残差是轨迹平滑度指标，不代表实机无振动。详细结果见 `validation/acceptance_report.json` 与 `motion_quality.json`。
