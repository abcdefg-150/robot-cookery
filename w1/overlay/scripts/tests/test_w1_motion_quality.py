import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.w1_motion_quality import rotation_distance_deg, rotation_path_metrics


class RotationPathTests(unittest.TestCase):
    def test_full_turn_is_detected_despite_equal_endpoints(self):
        matrices=Rotation.from_euler('z',np.linspace(0,360,361),degrees=True).as_matrix()
        metrics=rotation_path_metrics(matrices)
        self.assertAlmostEqual(metrics['net_rotation_deg'],0)
        self.assertAlmostEqual(metrics['excess_rotation_deg'],360)

    def test_shortest_rotation_crossing_euler_wrap_is_allowed(self):
        matrices=Rotation.from_euler('z',np.linspace(179,181,21),degrees=True).as_matrix()
        metrics=rotation_path_metrics(matrices)
        self.assertAlmostEqual(metrics['net_rotation_deg'],2)
        self.assertAlmostEqual(metrics['excess_rotation_deg'],0)

    def test_out_and_back_rotation_is_detected(self):
        matrices=Rotation.from_euler('x',[0,30,60,30,0],degrees=True).as_matrix()
        self.assertAlmostEqual(rotation_path_metrics(matrices)['excess_rotation_deg'],120)

    def test_single_pose_has_no_motion(self):
        metrics=rotation_path_metrics(np.eye(3)[None])
        self.assertEqual(metrics,{'net_rotation_deg':0.0,'angular_path_deg':0.0,'excess_rotation_deg':0.0})

    def test_distance_is_geodesic(self):
        self.assertAlmostEqual(rotation_distance_deg(np.eye(3),Rotation.from_euler('z',350,degrees=True).as_matrix()),10)


if __name__=='__main__':unittest.main()
