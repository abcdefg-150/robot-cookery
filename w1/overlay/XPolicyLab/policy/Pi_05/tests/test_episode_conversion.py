"""Run with unittest in the RoboTwin env; LeRobot/JAX are not needed for loading."""

import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

from XPolicyLab.utils.process_data import encode_image_bit

SCRIPT = Path(__file__).resolve().parents[1] / "openpi/scripts/process_data.py"
spec = importlib.util.spec_from_file_location("pi05_process_data", SCRIPT)
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


class EpisodeConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "episode.hdf5"
        with h5py.File(self.path, "w") as episode:
            for group in ("state", "action"):
                for side in ("left", "right"):
                    value = 0.1 if group == "state" else 0.7
                    episode[f"{group}/{side}_arm_joint_states"] = np.full((3, 6), value, np.float32)
                    episode[f"{group}/{side}_ee_joint_states"] = np.full((3, 1), value, np.float32)
            episode["instructions"] = json.dumps(["put the pepper in the pot", "move the pepper"])
            episode["additional_info/frequency"] = 50
            rgb = np.zeros((8, 12, 3), np.uint8)
            rgb[..., 0] = 240
            encoded = encode_image_bit(rgb)
            for camera in converter.CAMERA_ALIASES:
                episode[f"vision/{camera}/colors"] = np.asarray([encoded] * 3)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_sim_shapes_actions_language_and_rgb(self):
        data = converter.load_data(self.path)
        self.assertEqual(data["state"].shape, (3, 14))
        np.testing.assert_allclose(data["action"], 0.7)  # must not shift measured state
        self.assertEqual(data["instructions"][0], "put the pepper in the pot")
        self.assertEqual(data["frequency"], 50)
        rgb = data["images"]["cam_high"]
        self.assertEqual(rgb.shape, (3, 480, 640, 3))
        self.assertGreater(rgb[0, 0, 0, 0], 200)
        self.assertLess(rgb[0, 0, 0, 2], 20)

    def test_one_dimensional_grippers_and_vector_instructions(self):
        with h5py.File(self.path, "a") as episode:
            for group in ("state", "action"):
                for side in ("left", "right"):
                    key = f"{group}/{side}_ee_joint_states"
                    del episode[key]
                    episode[key] = np.zeros(3)
            del episode["instructions"]
            episode["instructions"] = np.asarray(["whole trajectory"], dtype=h5py.string_dtype())
        data = converter.load_data(self.path)
        self.assertEqual(data["state"].shape, (3, 14))
        self.assertEqual(data["instructions"], ["whole trajectory"])

    def test_rejects_camera_length_mismatch(self):
        with h5py.File(self.path, "a") as episode:
            key = "vision/cam_head/colors"
            image = episode[key][0]
            del episode[key]
            episode[key] = np.asarray([image])
        with self.assertRaisesRegex(ValueError, "frames"):
            converter.load_data(self.path)

    def test_rejects_nan_actions(self):
        with h5py.File(self.path, "a") as episode:
            episode["action/left_arm_joint_states"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            converter.load_data(self.path)

    def _run_cli(self, fps):
        folder = Path(self.temp.name) / "data/bench/task/w1_g2/data"
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / "episode_0000000.hdf5"
        self.path.replace(destination)
        with h5py.File(destination, "a") as episode:
            episode["additional_info/frequency"][()] = 30
        dataset = MagicMock()
        with patch.object(converter, "ROOT_PATH", Path(self.temp.name)), \
             patch.object(converter, "load_yaml", return_value={"config": {"robot": "dual_w1_g2"}, "observation": {"collect_freq": 50}}), \
             patch.object(converter, "get_robot_action_dim_info", return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}), \
             patch.object(converter, "create_empty_dataset", return_value=dataset) as create, \
             patch.object(sys, "argv", ["convert", "bench", "task", "w1_g2", "joint", "--fps", str(fps)]):
            converter.main()
        return dataset, create

    def test_explicit_30hz_does_not_change_legacy_50hz_config(self):
        dataset, create = self._run_cli(30)
        self.assertEqual(create.call_args.kwargs["fps"], 30)
        self.assertEqual(dataset.add_frame.call_count, 3)
        dataset.save_episode.assert_called_once()

    def test_root_and_joint_dimensions_resolve_from_parent_workspace(self):
        self.assertEqual(converter.ROOT_PATH, SCRIPT.resolve().parents[5])
        info = converter.get_robot_action_dim_info("w1_g2")
        self.assertEqual(sum(info["arm_dim"]) + sum(info["ee_dim"]), 14)

    def test_frequency_override_cannot_relabel_30hz_as_50hz(self):
        with self.assertRaisesRegex(RuntimeError, "Conversion stopped"):
            self._run_cli(50)


if __name__ == "__main__":
    unittest.main()
