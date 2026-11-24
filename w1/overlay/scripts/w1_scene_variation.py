"""Seeded W1 collection layouts, camera perturbations and fixed episode lighting."""
from functools import lru_cache
import json

import numpy as np
from transforms3d.euler import euler2mat
import trimesh


FOOD_PAIRS = (("green_pepper", "small_pepper"), ("eggplant", "potato"))
TABLE_XY = np.array([[-.60, -.311], [.60, .215]])


@lru_cache(maxsize=16)
def mesh_hull(path):
    mesh = trimesh.load(path, force="mesh")
    return mesh.convex_hull.vertices, mesh.bounds.mean(axis=0)


def footprint(root, asset, rotation_rpy, center=None):
    hull, midpoint = mesh_hull(root / asset.get("collision", asset["visual"]))
    if center is None:
        center = midpoint if asset.get("center_visual") else np.zeros(3)
    return ((hull - center) * asset.get("scale", 1) @ euler2mat(*rotation_rpy).T)[:, :2]


def bounds(points, xy, yaw):
    points = points @ euler2mat(0, 0, yaw)[:2, :2].T + xy
    return np.stack((points.min(axis=0), points.max(axis=0)))


def separate(first, second, margin):
    return bool(np.any(first[1] + margin <= second[0]) or np.any(second[1] + margin <= first[0]))


def _sample_layout(root, assets, options, rng):
    food_hulls = {name: footprint(root, assets[name], assets[name]["rotation_rpy"])
                  for pair in FOOD_PAIRS for name in pair}
    pan = assets["pan"]
    pan_hull = footprint(root, pan, pan["rotation_rpy"], np.asarray(pan["center_local"]))
    tool = assets["spatula"]
    tool_meta = json.loads((root / tool["metadata"]).read_text())
    tool_hull = footprint(root, tool, [tool_meta.get("rest_roll", 0), tool_meta.get("rest_pitch", 0), 0])
    margin = float(options["prop_clearance_m"])

    def xy(region):
        return rng.uniform(*np.asarray(region).T)

    for _ in range(250):
        pan_xy = xy(options["pan_xy"])
        pan_yaw = np.radians(rng.choice(options["pan_handle_yaw_deg"]) +
                             rng.uniform(-options["pan_yaw_jitter_deg"], options["pan_yaw_jitter_deg"]))
        tool_xy = xy(options["spatula_xy"])
        tool_yaw = tool_meta["rest_yaw"] + np.radians(rng.uniform(
            -options["spatula_yaw_jitter_deg"], options["spatula_yaw_jitter_deg"]))
        pan_box = bounds(pan_hull, pan_xy, pan_yaw)
        tool_box = bounds(tool_hull, tool_xy, tool_yaw)
        if not separate(pan_box, tool_box, margin):
            continue
        boxes = [pan_box, tool_box]
        pairs = [tuple(rng.permutation(pair)) if options["swap_pair_arms"] else pair for pair in FOOD_PAIRS]
        bands = {side: rng.permutation(2) for side in ("left", "right")}
        foods = {}
        for index, pair in enumerate(pairs):
            for side, name in zip(("left", "right"), pair):
                y_range = options["food_y_bands_m"][bands[side][index]]
                position = xy([options["food_x_m"][side], y_range])
                yaw = np.radians(rng.uniform(-options["food_yaw_jitter_deg"],
                                             options["food_yaw_jitter_deg"]))
                bounds_xy = bounds(food_hulls[name], position, yaw)
                if all(separate(bounds_xy, box, margin) for box in boxes):
                    boxes.append(bounds_xy)
                    foods[name] = {"xy_m": position.tolist(), "yaw_rad": float(yaw), "arm": side}
        if len(foods) == 4 and all(np.all(box[0] >= TABLE_XY[0] + margin) and
                                    np.all(box[1] <= TABLE_XY[1] - margin) for box in boxes):
            return {
                "foods": foods, "pair_foods": [list(pair) for pair in pairs],
                "pan": {"xy_m": pan_xy.tolist(), "yaw_rad": float(pan_yaw)},
                "spatula": {"xy_m": tool_xy.tolist(), "yaw_rad": float(tool_yaw)},
            }
    raise ValueError("Could not sample a separated W1 tabletop layout within the configured regions")


def _sample_camera(rng, camera_options):
    def perturbation(prefix):
        shift = np.asarray(camera_options[f"{prefix}_translation_m"])
        rotation = np.asarray(camera_options[f"{prefix}_rpy_deg"])
        return {"translation_m": rng.uniform(-shift, shift).tolist(),
                "rpy_deg": rng.uniform(-rotation, rotation).tolist(),
                "focal_scale": float(rng.uniform(*camera_options[f"{prefix}_focal_scale"]))}

    return {"front": {**perturbation("front"), "reference_index": int(rng.integers(2))},
            **{side: perturbation("wrist") for side in ("left", "right")}}


def _sample_lighting(rng, options):
    tint = rng.uniform(-options["tint"], options["tint"])
    rgb = np.array([1.0 + tint, 1.0, 1.0 - tint])
    tilt = rng.uniform(-.65, .65, size=2)
    return {"ambient_light": (rng.uniform(*options["ambient"]) * rgb).tolist(),
            "direction_lights": [[[float(tilt[0]), float(tilt[1]), -1.0],
                                  (rng.uniform(*options["key"]) * rgb).tolist()]],
            "point_lights": [
                [[float(x + rng.uniform(-.25, .25)), float(rng.uniform(-.45, .45)), 1.8],
                 (rng.uniform(*options["fill"]) * rgb).tolist()]
                for x in (1.0, -1.0)
            ]}


def sample_scene(root, assets, options, seed):
    layout_rng, camera_rng, light_rng = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(3))
    return {**_sample_layout(root, assets, options["layout"], layout_rng), "seed": int(seed),
            "cameras": _sample_camera(camera_rng, options["cameras"]),
            "lighting": _sample_lighting(light_rng, options["lighting"])}


def perturb_optical_pose(matrix, sample):
    result = np.array(matrix, dtype=float, copy=True)
    result[:3, 3] += sample["translation_m"]
    result[:3, :3] = result[:3, :3] @ euler2mat(*np.radians(sample["rpy_deg"]))
    return result
