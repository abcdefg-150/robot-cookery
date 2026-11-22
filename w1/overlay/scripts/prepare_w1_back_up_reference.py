"""Derive a back-up spatula scene without changing the original meshes or v3."""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    assets = yaml.safe_load((ROOT / 'env_cfg/w1_real_reference_assets_v3.yml').read_text())
    tool = json.loads((ROOT / assets['spatula']['metadata']).read_text())
    rotation = Rotation.from_euler('xyz', [math.pi, 0, tool['rest_yaw']]).as_matrix()
    vertices = [trimesh.load(ROOT / assets['spatula']['visual'], force='mesh').vertices]
    for part in tool['parts']:
        if 'collision_mesh' in part:
            vertices.append(trimesh.load(ROOT / part['collision_mesh'], force='mesh').vertices)
        else:
            corners = np.array(list(itertools.product([-1, 1], repeat=3)))
            vertices.append(corners * part['half_size'] + part['center'])
    points = np.concatenate(vertices)
    height = -float((points @ rotation.T)[:, 2].min()) + .0015
    # +local Z is the working-face normal in the existing imported tool frame.
    # Back-up means this normal points down, and the TCP approaches from above.
    grasp = np.array([.020, 0, -.006])
    tool.update(rest_roll=math.pi, rest_pitch=0, rest_height=height,
                initial_surface='back_up', working_face_normal_tool_local=[0, 0, 1],
                grasp_point_tool_local=grasp.tolist(), grasp_offset=(rotation @ grasp).tolist(),
                rest_height_basis='visual and collision lower envelope + 1.5 mm before natural settling')
    metadata = ROOT / 'assets/objects/w1_real_image_v1/spatula_metadata_back_up_v4.json'
    metadata.write_text(json.dumps(tool, indent=2) + '\n')
    assets['spatula']['metadata'] = str(metadata.relative_to(ROOT))
    manifest = ROOT / 'env_cfg/w1_real_reference_assets_v4.yml'
    manifest.write_text(yaml.safe_dump(assets, sort_keys=False))
    config = yaml.safe_load((ROOT / 'env_cfg/task_config/w1_finals_real_v3.yml').read_text())
    config['asset_manifest'] = str(manifest.relative_to(ROOT))
    (ROOT / 'env_cfg/task_config/w1_finals_real_v4.yml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps({'config': 'w1_finals_real_v4', 'spatula_initial_surface': 'back_up',
                      'initial_rest_height_m': height, 'xy_and_yaw_preserved': True}, indent=2))


if __name__ == '__main__':
    prepare()
