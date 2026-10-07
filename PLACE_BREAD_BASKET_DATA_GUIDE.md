# 面包入篮任务：专家行为与训练数据交接

本文随 `robot-cookery` 仓库发布；代码文件保留 RoboTwin 原有相对路径，需合并到团队成员已安装的 RoboTwin 2.0 根目录，并非独立可运行仓库。本文供数据采集和模型训练团队共用。**当前修复版动作**同时作用于 `bread_competition_v3` 的三面包局和可选的固定三面包 `bread_competition_v3_three_r2`：第三次放置后不再强制上抬/回原位，第二次放置后空载机械臂的世界坐标避让上抬为 0.10 m。旧 500 局数据已移到 `data/bread_competition_v3_500/`；当前 `data/bread_competition_v3/` 是修复后正在生成的新批次。不同动作版本的预规划轨迹不可混用。

关键文件：

- [任务与专家动作](envs/place_bread_basket.py)
- [当前混合数量配置：500 局，1–3 面包](env_cfg/task_config/bread_competition_v3.yml)
- [可选固定三面包配置：100 局、独立新目录](env_cfg/task_config/bread_competition_v3_three_r2.yml)
- [历史 50 局参数快照](env_cfg/task_config/bread_competition_v3_50.yml)（根据当时保留的配置重建；如需严格溯源仍应核对原始运行日志）
- [历史固定三面包配置](env_cfg/task_config/bread_competition_v3_three.yml)（目录中已有不同动作版本的种子和规划轨迹）
- [单次可视化入口](scripts/preview_task.py)
- RoboTwin 基础仓库自带的 `collect_data.sh`、`scripts/collect_data.py`、`envs/utils/pkl2hdf5.py`（本覆盖仓库不重复存放）

## 任务场景与专家行为

篮子位于桌面中央；当前 `v3` 混合数量配置每局随机生成 1–3 个面包，`v3_three_r2` 固定生成 3 个。两者均关闭桌面杂物，随机化背景、光照和桌高。每个面包独立地以 80% 概率从方形/块状模型 `base1/base5/base6` 中抽样，否则从 `base0/base3` 中抽样；因此一局的方形比例并非固定 80%。

| 初始布局 | 抓取、悬停和放置顺序 |
| --- | --- |
| 1 个 | 对应侧机械臂抓取、抬起、悬停 2 秒、放入篮中并张开夹爪；没有额外回原位动作。 |
| 2 个，同侧 | 同侧机械臂按桌面前后位置依次抓放；每次抓起后悬停 2 秒。 |
| 2 个，分处两侧 | 双臂同时抓取和抬起，共同悬停 2 秒；左臂先放并抬高、回初始位；再等待 **1 秒仿真时间**，右臂才开始放置。 |
| 3 个，同侧 | 同侧机械臂按桌面前后位置逐个抓放；前两次放置后竖直避让 0.15 m 并回初始位，以便继续抓取；第三次放置后夹爪张开并结束，不再强制上抬/回原位。 |
| 3 个，分处两侧 | 两臂先同时抓取；拥有两个面包的一侧先放第一个，并在另一臂放第二个时撤离、回初始位、抓同侧剩余面包。第二臂放置后竖直避让 0.10 m；第三个面包悬停 2 秒后由第一臂放置，同时第二臂回原位。第三次放置后第一臂张开夹爪并结束，不再强制上抬/回原位。同一时刻只有一臂放置；此分支**没有**两面包分侧时的额外 1 秒交接等待。 |

每次抓取抬起后才开始 2 秒悬停计时。悬停检查要求面包保持高于初始高度 3 cm，且相对对应末端执行器的位移不超过 3 cm。两臂同时抓取时，共享一次 2 秒悬停；后续第三个面包另悬停 2 秒。

放置目标点随顺序升高：第一个采用篮子的原始功能点；第二个增加第一个面包的估算竖直厚度与 `bread_stack_clearance=0.005 m`；第三个再累计第二个的厚度和间隙。厚度取放置规划前、面包在桌上姿态下的包围盒估计值，不是实时接触测量值。每局实际偏移记录在 `scene_info.json` 的 `bread_placement_height_biases` 中。

## 成功判定的实际边界

采集的预规划阶段需要 `plan_success=True`、`check_success=True`，并通过关节范围检查。Aloha URDF 中若关节限制是 `[-10, 10] rad` 这类占位值，采集脚本另以 ±π 筛除绕圈规划。`check_success` 当前要求：生成数量符合目标；每个面包都通过悬停检查；所有面包中心在篮中心的 x/y 各 ±5 cm 范围内、z 高于桌面阈值；双夹爪均张开。采集回放结束后还会再次断言 `check_success`。**没有面包高度上限**，所以堆叠较高本身不会直接导致该判定失败。

这**不是**完整几何/赛事判定：目前没有严格检查面包整个碰撞体完全入篮、转移途中从未掉落、面包间无穿透/挤压或 30 秒上限。关节范围检查针对**规划轨迹**，并非逐步实测关节反馈；`plan_success` 与最终位置判定也可能不一致。训练数据发布前应抽看视频，尤其是三面包堆叠、两臂交接及篮边缘案例。

## 单次预览和正式采集

在 RoboTwin 仓库根目录、已激活 `RoboTwin` Conda 环境的本机图形桌面终端运行：

```bash
# 当前混合数量 v3：预览可强制三面包，正式采集按配置随机生成 1–3 个。
bash preview_task.sh place_bread_basket bread_competition_v3 0 3 15
bash collect_data.sh place_bread_basket bread_competition_v3 0

# 可选：固定三面包，使用独立的新配置名和数据目录。
bash preview_task.sh place_bread_basket bread_competition_v3_three_r2 0 3 15
bash collect_data.sh place_bread_basket bread_competition_v3_three_r2 0
```

参数顺序分别是 `任务 配置 GPU [面包数量|random] [种子]`（预览）和 `任务 配置 GPU`（采集）。预览只执行一局，不保存训练数据；播放完会保留窗口，关闭窗口退出。采集先寻找并保存成功种子/预规划轨迹，再按相同种子回放生成 HDF5 和视频；失败种子会继续重试，所以实际尝试次数可超过配置的 `episode_num`。当前 v3 与固定三面包配置均为 `render_freq: 0`；**数据回放阶段脚本也强制设为 0**。历史 50 局参数快照为 `render_freq: 20`，只影响找种子阶段的窗口显示。当前 v3 正在本机运行时，不要再启动第二个采集进程。

旧 `v3_500` 批次中三面包占比较低。当前 v3 新批次目标为 500 局混合数量成功轨迹；可选的固定三面包 r2 配置以 100 局为目标。专家规划仍可能因第三次放置姿态不可达、关节超限或物理碰撞而失败；同一种子在不同运行中也可能得到不同结果。采集沿用原判定、只保留通过的轨迹，不应把失败样本强行标成功。若第一批验证质量可用，再按需要提高新配置的 `episode_num` 继续采集，并保存最终配置快照。

**当前 `bread_competition_v3` 的种子与 `_traj_data` 是修复后生成的，可以按当前脚本继续回放；旧 `v3_500` 和 `v3_three` 的轨迹则不可混入。** 后两者可能由不同动作版本生成，路径数量及顺序已变化。`v3_three_r2` 是需要固定三面包补采时的独立新目录，从空种子集开始。

## 交付目录和训练字段

当前混合数量新数据目录：`data/bread_competition_v3/place_bread_basket/aloha_agilex/`；本机该目录正在回放生成 HDF5。可选固定三面包新目录：`data/bread_competition_v3_three_r2/place_bread_basket/aloha_agilex/`，截至本文更新尚未启动。旧 500 局数据位于 `data/bread_competition_v3_500/place_bread_basket/aloha_agilex/`；另有历史 `v3_50` 和 `v3_three` 目录。各目录分别打包并标注对应代码版本，不要直接覆盖或混写。

```text
seed.txt                         成功种子编号；不是训练样本
_traj_data/episodeN.pkl         预规划双臂关节路径；不是最终训练样本
data/episode_0000000.hdf5       正式轨迹
video/episode_0000000.mp4       头部相机视频，用于人工抽检
instruction/episode_0000000.json 该局 seen/unseen 语言指令
scene_info.json                 按 episode_N 索引的面包数量、模型、放置偏移及悬停结果
```

HDF5 是 XPolicyLab/RoboDojo 风格，不是直接可供所有模型读取的 LeRobot 数据集。主要字段：

| HDF5 路径 | 内容 |
| --- | --- |
| `vision/cam_head/colors`、`vision/cam_left_wrist/colors`、`vision/cam_right_wrist/colors`、`vision/cam_third_view/colors` | 实际 HDF5 有四路 240×320 RGB，存储为编码后的 image bits，需用 `XPolicyLab.utils.process_data.decode_image_bit` 解码。没有深度或点云。 |
| `state/left_arm_joint_states`、`state/right_arm_joint_states` | 各 6 维关节**驱动目标值**，并非实测关节反馈。 |
| `state/left_ee_joint_states`、`state/right_ee_joint_states` | 各 1 维归一化夹爪值。按左臂 6 + 左夹爪 1 + 右臂 6 + 右夹爪 1 拼接为 14 维。 |
| `action/` 下同名字段 | 下一采样帧的上述值：写入器使用 `state = values[:-1]`、`action = values[1:]`；并非单独记录的专家动作命令。 |
| `state/*_ee_poses`、`action/*_ee_poses` | 双臂末端位姿；相邻帧对齐方式相同。 |
| `instructions`、`additional_info/frequency` | `instructions` 是该局 `seen` 指令列表的 JSON 字符串；频率字段当前写入 15。 |

`data_type.third_view: false` 控制单独的 observer 图像路径，并不关闭静态 `front_camera`；写入器把后者映射为 `cam_third_view`。默认 LeRobot 转换示例只映射头部和双腕三路图像，如需第四路须由训练侧扩展转换。`instruction/episode_*.json` 还包含 `unseen` 指令；训练侧应明确使用哪组语言，不要在评估 `unseen` 泛化时把它们混入训练。视频及 HDF5 由 `save_freq: 15` 的采样过程生成，悬停/交接等待也会被采样。**不要仅凭 `frequency=15` 推断每两帧严格相隔 1/15 秒**：动作段按仿真步抽样且段末可能额外采样，文件没有逐帧时间戳；有精确时序要求的训练须先核对或重建时间轴。

## 训练侧接入与验收

XPolicyLab 适配器可消费上述 HDF5。使用 LeRobot 数据管线（例如 pi0.5）时，先选定与训练代码匹配的 LeRobot 版本，再转换；不要直接把原始 HDF5 当作 LeRobot 输入。以下示例只在当前 v3 数据完成验收后运行；转换脚本会重建同名输出目录，所以须使用新的 `repo_id`，不要覆盖已有转换结果：

```bash
# 在已安装对应 LeRobot 版本的环境运行；确保 HF_LEROBOT_HOME 所在磁盘有足够空间。
python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "bread_competition_v3.place_bread_basket.aloha_agilex" \
  --repo_id place_bread_basket_v3_current_aloha_agilex --max_episode 500
```

若模型要求 LeRobot v3.0，改用 `transform_lerobot_v30_format.py` 并换一个 `repo_id`。转换后的常用键为 `observation.state`、`action`、`observation.images.cam_high`、`cam_left_wrist`、`cam_right_wrist`。模型自己的动作归一化、图像裁剪/尺寸、语言字段映射、训练配方仍需由训练团队确认；不可假定 pi0.5 对原始格式即插即用。按 episode/seed 划分训练与验证集，避免同一轨迹的帧跨集合泄漏，并统计 1/2/3 面包比例。

交付前至少确认 `data/*.hdf5`、`video/*.mp4`、`instruction/*.json` 数量一致，逐局 `state`/`action`/四路图像的时间维度相同，`scene_info.json` 覆盖所有 episode，且抽看视频无明显碰撞、漏抓或堆叠异常。可用下面的只读检查查看当前产出和首个 HDF5 的主要维度：

```bash
python - <<'PY'
from pathlib import Path
import h5py
root = Path('data/bread_competition_v3/place_bread_basket/aloha_agilex')
for subdir, suffix in [('data', '*.hdf5'), ('video', '*.mp4'), ('instruction', '*.json')]:
    files = sorted((root / subdir).glob(suffix))
    print(subdir, len(files))
files = sorted((root / 'data').glob('*.hdf5'))
if files:
    with h5py.File(files[0], 'r') as h:
        for key in ('state/left_arm_joint_states', 'action/left_arm_joint_states',
                    'vision/cam_head/colors', 'vision/cam_left_wrist/colors',
                    'vision/cam_right_wrist/colors', 'vision/cam_third_view/colors'):
            print(key, h[key].shape)
PY
```

**`seed.txt` 数量不等于已完成 HDF5 数量；HDF5 存在也不自动等于该局已通过验收。** 当前回放代码先写文件、后执行成功断言，断言失败时可能留下异常样本。交付前须结合采集日志、`scene_info.json` 的 `bread_hold_success`、逐局数量检查及视频抽检建立最终白名单。发布时一并提供本配置、任务脚本及对应 Git 提交/未提交改动说明，以便复现；不要只传 HDF5。

## 团队共享与版本对应

代码放在 GitHub：当前混合数量批次的最小移植集合是 `envs/place_bread_basket.py`、`env_cfg/task_config/bread_competition_v3.yml`、`description/task_instruction/place_bread_basket.json`、`preview_task.sh`、`scripts/preview_task.py`、本文和 README；若要固定三面包补采，再加入 `bread_competition_v3_three_r2.yml`。`v3_50`、旧 `v3_three`、`demo_clean.yml`、`demo_randomized.yml`、旧 v2 配置和 `XPolicyLab` 子模块指针变化均不在本次最小移植集合内。团队成员应在相同 RoboTwin 基线上安装依赖与模型资产，然后用上面的单次预览命令验证。

数据与代码分开交付：RoboTwin 基仓库的 `.gitignore` 忽略 `data/` 和 `assets/`；本覆盖仓库不包含训练数据，也不应把 HDF5 直接加入 Git。等采集结束、日志与视频验收完成、确认 `data/`、`video/`、`instruction/` 的 episode 编号一致后，再从 RoboTwin 根目录执行以下打包命令。不要在 `.cache/` 或 HDF5 正在写入时打包。

```bash
mkdir -p data/releases
tar -cf data/releases/place_bread_basket_v3_current_aloha_agilex.tar \
  -C data/bread_competition_v3/place_bread_basket/aloha_agilex \
  data video instruction scene_info.json seed.txt
(cd data/releases && sha256sum place_bread_basket_v3_current_aloha_agilex.tar \
  > place_bread_basket_v3_current_aloha_agilex.tar.sha256)
```

将 `.tar`、`.sha256` 和对应 Git 提交号交给团队（共享存储由团队自行选择）。`_traj_data/` 是可选的复现附件，不是训练输入；`.cache/` 不应交付。接收方在归档文件所在目录运行 `sha256sum -c place_bread_basket_v3_current_aloha_agilex.tar.sha256`，再将包解压到**空目录**；例如在仓库根目录运行：

```bash
mkdir -p data/bread_competition_v3/place_bread_basket/aloha_agilex
tar -xf /path/to/place_bread_basket_v3_current_aloha_agilex.tar \
  -C data/bread_competition_v3/place_bread_basket/aloha_agilex
```

接收方应检查上面的文件数量与 HDF5 字段；训练集只使用人工验收后的 episode 白名单。视频用于质检，训练主要读取 HDF5 和语言指令。由于这些数据可能包含视觉上成功但程序判定失败的轨迹，不能只按文件存在与否自动认定成功。

交付时记录每种面包数量的样本数；当前 v3 混合数量批次中三面包占比仍可能低于场景随机生成比例，训练团队应据此决定是否用 `v3_three_r2` 补采，而非假设数据集均衡。本文不固定写入随采集变化的样本数量，以交付时的验收清单为准。
