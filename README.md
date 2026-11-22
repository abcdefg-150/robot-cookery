# RoboTwin 比赛仿真环境

- **双 W1/G2 做饭任务**：环境、资产、安装器、采集接口和使用说明见 [w1/README.md](w1/README.md)。当前稳定配置为 `w1_finals_real_v6`，食材保持抓取姿态直接入锅，锅铲抓起后小幅调整进入翻炒；初始背面朝上，翻炒按真实接触、食材位移及至少 2 秒的小幅往返运动验收。
- **面包入篮任务**：保留原有覆盖文件和说明，见下文及 [面包入篮数据指南](PLACE_BREAD_BASKET_DATA_GUIDE.md)。

W1 模块包含运行所需的处理后模型资产；原始实机录制和训练数据独立保存，不进入本仓库。

# 面包入篮比赛任务

面包模块共享 `place_bread_basket` 的任务覆盖文件和交接文档，不包含 RoboTwin 本体、模型资产或训练数据。相对路径与 RoboTwin 根目录一致；团队成员需先安装 RoboTwin 2.0，再将这些文件合并到自己的仓库。本地开发基线提交为 `ee3d076fb61c4542e9be6c16f3af5ecd67d4a6f7`；其他基线需验证兼容性。

## 文件清单与安装

当前需要同步的代码和配置：

- `envs/place_bread_basket.py`
- `env_cfg/task_config/bread_competition_v3.yml`（当前 1–3 面包混合采集，使用修复后的任务脚本）
- `env_cfg/task_config/bread_competition_v3_three_r2.yml`（可选：固定三面包独立补采）
- `description/task_instruction/place_bread_basket.json`
- `preview_task.sh`、`scripts/preview_task.py`

历史 500 局数据已移动到 RoboTwin 的 `data/bread_competition_v3_500/`；当前 `data/bread_competition_v3/` 是修复后新生成、仍在回放的批次。仓库另保留 `bread_competition_v3_50.yml`、`bread_competition_v3_three.yml` 用于旧批次溯源。

在本仓库根目录执行（请将目标路径改为自己的 RoboTwin 安装路径）：

```bash
cp --parents envs/place_bread_basket.py \
  env_cfg/task_config/bread_competition_v3.yml \
  env_cfg/task_config/bread_competition_v3_three_r2.yml \
  description/task_instruction/place_bread_basket.json \
  scripts/preview_task.py preview_task.sh \
  /path/to/RoboTwin
```

完成后，在 RoboTwin 根目录预览当前混合数量批次；若需另采固定三面包，再使用独立的 r2 配置：

```bash
bash preview_task.sh place_bread_basket bread_competition_v3 0 3 15
# 当前 v3 采集正在运行时，不要再启动第二个采集进程。
bash collect_data.sh place_bread_basket bread_competition_v3 0

bash preview_task.sh place_bread_basket bread_competition_v3_three_r2 0 3 15
bash collect_data.sh place_bread_basket bread_competition_v3_three_r2 0
```

`v3_three_r2` 写入独立的 `data/bread_competition_v3_three_r2/`，不会读取旧 `bread_competition_v3_three` 的种子与规划轨迹。当前 `v3` 的新种子/轨迹可配当前脚本继续回放；**不要把 `_500` 或旧 `v3_three` 的 `_traj_data` 混入其中**。动作逻辑、数据格式、验收边界和归档方式详见 [面包入篮数据交接指南](PLACE_BREAD_BASKET_DATA_GUIDE.md)。训练数据须单独共享，不进入本 Git 仓库。
