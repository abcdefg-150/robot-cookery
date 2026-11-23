"""Run W1 physics in the RoboTwin Python environment for Dexbotic evaluation."""

import argparse
from contextlib import contextmanager
from multiprocessing.connection import Connection
from pathlib import Path
import subprocess
import sys

import numpy as np


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
GRIPPERS = [6, 13]
VIDEO_FPS = 15
VIDEO_STEP_INTERVAL = 2


def observation_packet(observation, gripper_scale):
    state = observation["state"].copy()
    state[GRIPPERS] /= gripper_scale
    images = []
    for name in CAMERAS:
        image = observation["images"][name]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB uint8 image from {name}")
        images.append((image.shape, image.tobytes()))
    return {"state": state.tolist(), "images": images, "phase": observation["phase"]}


@contextmanager
def front_video(path, first_frame):
    height, width, _ = first_frame.shape
    command = [
        "ffmpeg", "-nostdin", "-n", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(VIDEO_FPS), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p", "-g", str(VIDEO_FPS),
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-flush_packets", "1",
        str(path),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            encoder.stdin.write(first_frame.tobytes())
            yield encoder.stdin
        finally:
            try:
                encoder.stdin.close()
                returncode = encoder.wait(timeout=30)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
                raise
            if returncode:
                raise RuntimeError(f"FFmpeg failed to save {path} (exit code {returncode})")


def capture_front(simulator):
    task = simulator.task
    task._update_render()
    camera = task.cameras.static_camera_list[task.cameras.head_camera_id]
    camera.take_picture()
    colors = camera.get_picture("Color")[..., :3]
    return np.uint8(np.clip(colors * 255, 0, 255))


def run_episode(connection, simulator, seed, gripper_scale, max_steps, video_path=None):
    observation = simulator.reset(seed=seed)
    if video_path is None:
        return rollout(connection, simulator, observation, seed, gripper_scale, max_steps)
    with front_video(video_path, observation["images"]["cam_head"]) as video:
        report = rollout(connection, simulator, observation, seed, gripper_scale, max_steps, video)
    report["video_file"] = str(video_path)
    report["video_fps"] = VIDEO_FPS
    return report


def rollout(connection, simulator, observation, seed, gripper_scale, max_steps, video=None):
    lower = np.full(14, -np.inf)
    upper = np.full(14, np.inf)
    for side, offset in (("left", 0), ("right", 7)):
        pose_ik = getattr(simulator.task.robot, f"{side}_planner").pose_ik
        lower[offset:offset + 6] = pose_ik.lower
        upper[offset:offset + 6] = pose_ik.upper

    joint_limit_actions = 0
    policy_calls = 0
    reason = "step_limit"
    success = False
    while simulator.control_steps < max_steps:
        connection.send({"kind": "observation", **observation_packet(observation, gripper_scale)})
        actions = np.asarray(connection.recv(), dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) == 0 or not np.isfinite(actions).all():
            raise ValueError("Expected a nonempty finite action chunk of shape [N, 14]")
        policy_calls += 1
        for action in actions[:max_steps - simulator.control_steps]:
            action[GRIPPERS] *= gripper_scale
            bounded = np.clip(action, lower, upper)
            joint_limit_actions += int(np.any(bounded != action))
            try:
                _, _, success, truncated, _ = simulator.step(bounded, observe=False)
            except RuntimeError:
                collision_quality = simulator.task.collision_quality
                if not collision_quality.terminate_on_severe or not collision_quality.severe:
                    raise
                if video is not None:
                    video.write(capture_front(simulator).tobytes())
                reason = "severe_collision"
                break
            if video is not None and (
                simulator.control_steps % VIDEO_STEP_INTERVAL == 0 or success or truncated
                or simulator.control_steps == max_steps
            ):
                video.write(capture_front(simulator).tobytes())
            if success or truncated:
                reason = "success" if success else "step_limit"
                break
        if success or reason == "severe_collision":
            break
        if truncated or simulator.control_steps >= max_steps:
            break
        observation = simulator.observation()

    return {
        "seed": seed,
        "success": bool(success),
        "reason": reason,
        "steps": simulator.control_steps,
        "policy_calls": policy_calls,
        "joint_limit_actions": joint_limit_actions,
        "phase": simulator.task.fsm_state,
        "task_checks": simulator.policy_monitor.info()["success_checks"],
        "stir_metrics": simulator.task.stir_monitor.report(),
        "collision": simulator.task.collision_quality.report(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-gripper-unit", choices=("metres", "fraction"), required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    from scripts.w1_recording import SLAVE_GRIPPER_STROKE_M
    from scripts.w1_sim_interface import W1Simulation

    scale = SLAVE_GRIPPER_STROKE_M if args.policy_gripper_unit == "fraction" else 1.0
    with Connection(args.fd) as connection:
        connection.send({"kind": "ready"})
        while True:
            try:
                request = connection.recv()
            except EOFError:
                break
            if request is None:
                break
            simulation = W1Simulation(args.config)
            try:
                video_path = Path(request["video_path"]) if request.get("video_path") else None
                result = run_episode(
                    connection, simulation, request["seed"], scale, request["max_steps"], video_path
                )
            finally:
                simulation.close()
            connection.send({"kind": "result", **result})


if __name__ == "__main__":
    main()
