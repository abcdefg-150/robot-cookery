"""Preview one expert episode in the SAPIEN viewer without collecting data."""

import argparse
import importlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "env_cfg" / "task_config"
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name")
    parser.add_argument("task_config", help="YAML config name, without .yml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bread-count", type=int, choices=(1, 2, 3))
    parser.add_argument("--render-freq", type=int, default=20)
    options = parser.parse_args()

    if options.seed < 0:
        parser.error("--seed must be non-negative")
    if options.render_freq <= 0:
        parser.error("--render-freq must be positive")
    if options.bread_count is not None and options.task_name != "place_bread_basket":
        parser.error("--bread-count is only supported for place_bread_basket")

    config_path = CONFIG_DIR / f"{options.task_config}.yml"
    if not config_path.is_file():
        parser.error(f"config not found: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    with (CONFIG_DIR / "_embodiment_config.yml").open(encoding="utf-8") as file:
        embodiments = yaml.safe_load(file)
    embodiment = config["embodiment"]
    if len(embodiment) != 1:
        parser.error("this preview command currently supports single-type dual-arm embodiments")
    robot_file = str((ROOT / embodiments[embodiment[0]]["file_path"]).resolve())
    with (Path(robot_file) / "config.yml").open(encoding="utf-8") as file:
        robot_config = yaml.safe_load(file)

    config.update(
        task_name=options.task_name,
        task_config=options.task_config,
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        left_embodiment_config=robot_config,
        right_embodiment_config=robot_config,
        dual_arm_embodied=True,
        embodiment_name=embodiment[0],
        render_freq=options.render_freq,
        need_plan=True,
        save_data=False,
        collect_data=False,
    )
    if options.bread_count is not None:
        config["bread_count_range"] = [options.bread_count, options.bread_count]

    task_class = getattr(importlib.import_module(f"envs.{options.task_name}"), options.task_name)
    task = task_class()
    try:
        task.setup_demo(now_ep_num=0, seed=options.seed, **config)
        info = task.play_once()
        print(f"seed={options.seed}, plan_success={task.plan_success}, check_success={task.check_success()}", flush=True)
        if isinstance(info, dict):
            for key in ("bread_count", "bread_placement_height_biases", "bread_hold_success"):
                if key in info:
                    print(f"{key}: {info[key]}", flush=True)
        viewer = getattr(task, "viewer", None)
        if viewer is not None and not viewer.closed:
            print("Episode finished. Close the viewer window to exit.", flush=True)
            viewer.loop()
    finally:
        task.close_env()
        viewer = getattr(task, "viewer", None)
        if viewer is not None and not viewer.closed:
            viewer.close()


if __name__ == "__main__":
    main()
