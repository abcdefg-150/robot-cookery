"""Geodesic rotation checks: quaternion sign and Euler wrapping invariant."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_distance_deg(initial, target):
    rotations=Rotation.from_matrix(np.asarray([initial,target]))
    return float(np.degrees((rotations[0].inv()*rotations[1]).magnitude()))


def rotation_path_metrics(matrices):
    rotations=Rotation.from_matrix(np.asarray(matrices))
    net=float(np.degrees((rotations[0].inv()*rotations[-1]).magnitude()))
    path=float(np.degrees((rotations[:-1].inv()*rotations[1:]).magnitude()).sum()) if len(rotations)>1 else 0.0
    return {'net_rotation_deg':net,'angular_path_deg':path,'excess_rotation_deg':max(0.0,path-net)}
