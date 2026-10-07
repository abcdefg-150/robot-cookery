# 青椒入篮任务交付包

此目录中的 `robotwin_overlay/` 按 RoboTwin 的相对路径排列。它包含青椒资产、任务专家程序、采集配置和语言描述，以及任务使用的篮子 `base0` 资产。基于 RoboTwin 源码版本 `ea8b211` 整理；`place_pepper_basket.py` 等任务文件是本项目新增文件。

## 文件结构

```text
robotwin_overlay/
├── assets/objects/my_pepper/
│   ├── visual/base0.glb            # 青椒外观
│   ├── collision/base0.glb         # 青椒碰撞网格
│   └── model_data0.json            # 缩放与抓取位姿
├── assets/objects/076_breadbasket/
│   ├── visual/base0.glb            # 任务所用篮子外观
│   ├── collision/base0.glb         # 篮子碰撞网格
│   └── model_data0.json            # 篮子放置目标点
├── envs/place_pepper_basket.py      # 专家动作与成功判定
├── env_cfg/task_config/place_pepper_basket.yml
├── description/task_instruction/place_pepper_basket.json
└── description/objects_description/
    ├── my_pepper/base0.json
    └── 076_breadbasket/base0.json
```

## 安装到队友自己的 RoboTwin

先确认目标是队友自己的 RoboTwin 根目录。以下命令会覆盖目标中同名文件，尤其是已有的 `076_breadbasket/base0`；如果改过篮子资产，请先备份或逐项比较。

```bash
cp -a /path/to/robot-cookery/robotwin_overlay/. /path/to/RoboTwin/
```

复制完成后，在 RoboTwin 根目录确认 `envs/place_pepper_basket.py`、`assets/objects/my_pepper/model_data0.json`、`env_cfg/task_config/place_pepper_basket.yml` 都存在。需要仓库原有的 SAPIEN、机器人、`scripts/collect_data.py` 等基础环境；此包不包含完整 RoboTwin 或 Conda 环境。

## 采集专家数据

在已安装 RoboTwin 依赖的环境中，从 RoboTwin 根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_data.py place_pepper_basket place_pepper_basket
```

配置目前要求 `episode_num: 10`，即采到 10 条成功轨迹，而不是只尝试 10 次；失败时会继续换 seed。根据显卡情况调整 `CUDA_VISIBLE_DEVICES`。数据会写到 `data/place_pepper_basket/place_pepper_basket/aloha_agilex/`，其中 `data/` 子目录为 HDF5 轨迹，`instruction/` 为语言指令。需要先少量试跑时，可把配置中的 `episode_num` 改成 `1`。

目前专家程序使用右臂，青椒初始姿态由 `qpos` 控制为竖直，并在局部 Y 轴周围随机转向。青椒原始 `.glb` 边长约 1.8 个模型单位，`model_data0.json` 用 `scale: [0.03, 0.03, 0.03]` 将其缩到约 5–6 厘米；不要在重新导出已缩小的 `.glb` 后再次施加同一缩放。青椒的 visual 与 collision 目前是相同网格，可先用于复现；若物理抓取不稳定，应制作更简单的碰撞网格并重新验证。

此交付包不包含运行生成的 `data/`、`.cache/`、视频、checkpoint，也不包含某个特定 policy 的训练配置。采集得到 HDF5 后，按队友所选模型的 RoboTwin/XPolicyLab 数据转换与训练流程继续。成功率尚未作为此交付包的验收指标，请先观察少量回合的抓取、放置和 `check_success()` 结果。
