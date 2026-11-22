# 双 W1/G2 做饭仿真环境

当前入口为 `w1_finals_real_v5`，包版本为 `w1-cookery-stir-v1`。此目录是 RoboTwin 2.0 的覆盖包，包含运行所需的 W1 模型、处理后的锅与食材资产、任务脚本、相机标定参考和采集接口。原始实机视频、ZIP、训练 HDF5 不进入本仓库。

## 场景与任务

- 两条黑色 W1 六轴机械臂，各带 G2；底座中心左右相距 0.60 m，桌面高度 0.74 m。
- 锅位于中央；锅铲位于锅右侧，柄朝前，**初始背面朝上、工作面朝下**，由真实碰撞接触自然静置。
- 食材为两颗青椒、土豆、茄子。锅、锅铲、青椒、茄子使用用户提供的模型，土豆使用 RoboTwin 资产。
- 流程：右臂小青椒入锅 → 左臂大青椒入锅 → 右臂土豆入锅 → 左臂茄子入锅 → 左臂避让 → 右臂抓锅铲 → 小幅往返推动 → 锅铲放回桌面 → 双臂归位、夹爪闭合。

翻炒按用户确认的标准判断：锅内真实接触任一种食材、可见食材位移、小幅往返推动，累计有效运动至少 2 秒。实现中要求被接触食材至少移动 5 mm、推动方向至少反转一次（有效行程至少 8 mm）。计时仅包括锅内往返阶段、工具速度大于 3 mm/s 且最近 0.5 秒内发生过食材接触的运动；静止等待不计时。记录实际接触时长，**不要求 45° 转角或两次翻面**。最终还检查所有食材完整留在锅内、受支撑且连续静止至少 1 秒、锅铲释放归位及无严重碰撞。

锅的 26 cm 是用户提供的待核实尺寸，内径/外径含义未确定。本版按实机画面拟合，锅体外宽约 30.0 cm；资产清单保留估计来源。食材尺寸及腕相机安装也是图像拟合值。前相机采用实机内参及外参；当前锅固定，实机中锅会移动。质量、摩擦与驱动参数为仿真参数，尚未进行实机动力学辨识。

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
python scripts/preview_w1_current.py --config w1_finals_real_v5 --output data/w1_preview_001

# 30 Hz reset/step 接口的保持动作检查；此输出不属于成功示范
python scripts/example_w1_policy_collection.py --config w1_finals_real_v5 \
  --steps 30 --record-dir data/w1_api_smoke_001 --report data/w1_api_smoke_001_report.json

# 判定器和接触过滤测试
python -m unittest discover -s scripts/tests -v
python -m unittest discover -s XPolicyLab/policy/Pi_05/tests -v
```

预览默认输出 `current_scene_1x.mp4` 和 `preview_report.json`。检查报告中的任务结果；有画面并不表示整条任务通过。预览为 15 FPS 合成画面，训练记录使用下文的三路独立 30 Hz RGB-D。

## 采集一条完整示范

```bash
python scripts/collect_w1_real_style.py --config w1_finals_real_v5 \
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

只使用 `collection_summary.json` 中 accepted 列表对应的成功示范。失败候选保留用于排查，转换器会拒绝 `success=false` 的仿真记录。不要把预览、保持动作检查、失败候选或旧版本数据直接混入本批训练集。

原生机器人轨迹约 250 Hz；三路 RGB-D 为 640×480、30 Hz。RGB 使用 H.264，深度是 FFV1 保存的 uint16 毫米。统一 HDF5 在共同的真实时间轴上重采样到 **30 Hz、14 维**，保留源相机帧号和时间戳。

## VLA 接口与数据约定

```python
from scripts.w1_sim_interface import W1Simulation

sim = W1Simulation("w1_finals_real_v5", control_hz=30)
try:
    obs = sim.reset(seed=0)
    action = obs["state"].copy()  # 在这里替换为 VLA 输出
    obs, reward, terminated, truncated, info = sim.step(action)
finally:
    sim.close()
```

动作与状态顺序为 `[左 q1…q6, 左开度, 右 q1…q6, 右开度]`；关节单位 rad，动作是绝对关节目标，夹爪为 0…1 的总开度比例。观测包含 `state[14]`、`images` 三路 RGB、`depths` 三路 uint16 毫米深度和时间戳。物理频率 250 Hz，接口每次推进 8 或 9 步，实现 30 Hz 控制。

原生实机主手与从手夹爪行程不同：**state = 从手宽度 / 0.072，action = 主手宽度 / 0.0479**。原生 EE 四元数为机械臂底座坐标系的 xyzw；仿真物体真值为世界坐标的 wxyz。时间戳保留 float64。仿真 effort 为无效零占位，不能作为力矩监督。实机 `master_qpos` 是遥操主手编码器参考，不能当作已经确认的从手网络命令。

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

- 原工作区完整采集通过：1 条物理示范，5924 帧，30 Hz、14 维、三路 RGB-D。
- 有效往返运动 2.448 秒，实际食材接触 2.140 秒，推动方向反转 5 次，食材最大位移约 3.29 cm；严重碰撞事件为 0。
- 四个食材完整入锅、自然稳定、锅铲释放归位和双臂归位检查全部通过。初始工作面法向 z 分量约 -0.998，验证锅铲背面朝上。
- 原生 IDX/TSF、ZIP CRC 和 HDF5 独立检查通过；π0.5 读取器实际读取全部 5924 帧通过。
- 干净 RoboTwin/XPolicyLab 基线安装后，30 次 reset/step 保持检查和完整状态机均通过；仿真目录没有原始实机 episode。
- 12 项任务/碰撞判定测试及 7 项 π0.5 数据转换测试通过。重复安装预检没有待修改项目。

本示范约 197.54 秒，实机样例约 106.43 秒；保守规划和接触阶段降速使仿真更慢，未做录像重定时。左右末端累计路径分别约 3.27/6.36 m，实机为 3.68/6.47 m。完整对比及当前近似保留在 `validation/trajectory_comparison.json`，后续扩充训练数据时应结合实机数据评估执行节奏。
