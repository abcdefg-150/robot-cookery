"""Metric loaders for replaceable W1 finals assets and the native skillet."""

import json
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d
import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_asset_config(path="env_cfg/w1_finals_assets.yml"):
    with (ROOT / path).open() as file:
        return yaml.safe_load(file)


def build_vegetable(scene, kind, xy, table_z, config):
    cfg = config[kind]
    scale = float(cfg["scale"])
    visual = ROOT / cfg["visual"]
    collision = ROOT / cfg["collision"]
    mesh = trimesh.load(collision, force="mesh")
    rotation = t3d.euler.euler2mat(*cfg["rotation_rpy"])
    center = mesh.bounds.mean(axis=0) if cfg["center_visual"] else np.zeros(3)
    vertices = (mesh.vertices - center) * scale @ rotation.T
    half = np.maximum(np.abs(vertices.min(axis=0)), np.abs(vertices.max(axis=0)))
    offset = -rotation @ (center * scale)
    local_pose = sapien.Pose(offset, t3d.quaternions.mat2quat(rotation))
    builder = scene.create_actor_builder()
    material = scene.create_physical_material(0.9, 0.8, 0)
    add_collision = (builder.add_multiple_convex_collisions_from_file if cfg.get("collision_mode") == "multiple"
                     else builder.add_convex_collision_from_file)
    add_collision(str(collision), pose=local_pose, scale=[scale] * 3, material=material)
    builder.add_visual_from_file(str(visual), pose=local_pose, scale=[scale] * 3)
    builder.set_initial_pose(sapien.Pose([*xy, table_z + half[2] + 0.001]))
    entity = builder.build(name=kind)
    body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    old_mass=float(body.mass);old_inertia=body.inertia.copy()
    body.mass = float(cfg["mass"])
    if cfg.get("rescale_inertia_to_mass",False):body.inertia=old_inertia*(body.mass/old_mass)
    # Approximate rolling resistance of the small whole toy props. Coulomb
    # friction alone does not dissipate ideal pure rolling on a pan floor.
    body.linear_damping = float(cfg.get("linear_damping", 0.3))
    body.angular_damping = float(cfg.get("angular_damping", 2.0))
    visual_mesh = trimesh.load(visual, force="mesh")
    hull_vertices = (visual_mesh.convex_hull.vertices - center) * scale @ rotation.T
    return entity, body, half, hull_vertices


def build_pan(scene, center, table_z, config):
    cfg = config["pan"]
    scale = np.asarray(cfg["scale"] if isinstance(cfg["scale"], list) else [cfg["scale"]] * 3)
    rotation = t3d.euler.euler2mat(*cfg["rotation_rpy"])
    offset = -rotation @ (np.asarray(cfg["center_local"]) * scale)
    local = sapien.Pose(offset, t3d.quaternions.mat2quat(rotation))
    builder = scene.create_actor_builder()
    friction = scene.create_physical_material(1.0, 0.9, 0.0)
    # Static cookware can use its exact concave surface. Adjacent VHACD hulls
    # introduce internal contact seams which can trap a small prop against a
    # blade. Keep the old convex loader available for existing datasets.
    collision_loader = (builder.add_nonconvex_collision_from_file
                        if cfg.get("collision_mode") == "static_mesh"
                        else builder.add_multiple_convex_collisions_from_file)
    collision_loader(str(ROOT / cfg["collision"]), pose=local, scale=scale.tolist(), material=friction)
    builder.add_visual_from_file(str(ROOT / cfg["visual"]), pose=local, scale=scale.tolist())
    builder.set_initial_pose(sapien.Pose([*center, table_z]))
    return builder.build_static(name="skillet"), cfg


def build_spatula(scene, position, config):
    cfg = config["spatula"]
    metadata = json.loads((ROOT / cfg["metadata"]).read_text())
    rest_rotation = t3d.euler.euler2mat(metadata.get("rest_roll", 0),
                                        metadata.get("rest_pitch", 0), metadata.get("rest_yaw", 0))
    rest_pose = sapien.Pose(position, t3d.quaternions.mat2quat(rest_rotation))
    builder = scene.create_actor_builder()
    friction = scene.create_physical_material(1.0, 1.0, 0)
    contact = {"patch_radius": float(metadata.get("contact_patch_radius", 0)),
               "min_patch_radius": float(metadata.get("min_contact_patch_radius", 0))}
    for part in metadata["parts"]:
        if part.get("collision_mesh"):
            builder.add_convex_collision_from_file(str(ROOT / part["collision_mesh"]), material=friction, **contact)
        else:
            builder.add_box_collision(pose=sapien.Pose(part["center"]), half_size=part["half_size"], material=friction, **contact)
    builder.add_visual_from_file(str(ROOT / cfg["visual"]))
    builder.set_initial_pose(rest_pose)
    entity = builder.build(name="spatula")
    body = entity.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    old_mass=float(body.mass);old_inertia=body.inertia.copy()
    body.mass = metadata["mass_kg"]
    if metadata.get("rescale_inertia_to_mass",False):body.inertia=old_inertia*(body.mass/old_mass)
    # An explicit stand defines the designated tool-return region.
    stand = scene.create_actor_builder()
    rest_height = float(metadata.get("rest_height", 0.056))
    post_height = float(metadata.get("stand_post_height", 0.050))
    for x in metadata.get("stand_post_x", [-0.045, 0.045]) if metadata.get("use_stand", True) else []:
        local = sapien.Pose([x, 0, -rest_height + post_height/2])
        stand.add_box_collision(pose=local, half_size=[0.012, 0.022, post_height/2])
        stand.add_box_visual(pose=local, half_size=[0.012, 0.022, post_height/2], material=[0.12, 0.25, 0.4, 1])
    mat_offset = metadata.get("return_center_offset", [0.0325, 0])
    mat_half = metadata.get("return_half_size", [0.15, 0.075])
    if metadata.get("use_stand", True):
        stand.add_box_visual(pose=sapien.Pose([*mat_offset, -rest_height + 0.001]), half_size=[*mat_half, 0.001],
                             material=[0.12, 0.28, 0.5, 1])
        stand.set_initial_pose(rest_pose)
        stand.build_static(name="spatula_return_stand")
    vertices = trimesh.load(ROOT / cfg["visual"], force="mesh").convex_hull.vertices
    return entity, body, vertices
