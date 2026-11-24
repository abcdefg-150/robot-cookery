import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.w1_dual_trajectory import smooth_trajectory,stretch_trajectory
from scripts.w1_dual_collision import food_spheres,SphereKinematics


class DualTrajectoryTests(unittest.TestCase):
    def test_quiescent_endpoints_and_motion_bounds(self):
        path=smooth_trajectory([[0.,0.],[.1,-.2],[.3,-.1]],np.array([.5,.5]),np.array([1.,1.]),jerk_limits=8.)
        np.testing.assert_allclose(path['position'][[0,-1]],[[0,0],[.3,-.1]],atol=1e-12)
        np.testing.assert_allclose(path['velocity'][[0,-1]],0,atol=1e-12)
        np.testing.assert_allclose(path['acceleration'][[0,-1]],0,atol=1e-12)
        self.assertLessEqual(abs(path['velocity']).max(),.5)
        self.assertLessEqual(abs(path['acceleration']).max(),1.)
        self.assertLessEqual(abs(path['jerk']).max(),8.)

    def test_synchronization_cannot_speed_up_an_arm(self):
        path=smooth_trajectory([[0.],[.4]],np.array([.5]),np.array([1.]))
        stretched=stretch_trajectory(path,path['duration_s']*2)
        self.assertLessEqual(abs(stretched['velocity']).max(),.25)
        # The denser stretched grid may sample the exact peak between the
        # original control samples; compare with an interpolation tolerance.
        self.assertAlmostEqual(abs(stretched['velocity']).max(),abs(path['velocity']).max()/2,delta=1e-5)
        np.testing.assert_allclose(stretched['position'][[0,-1]],path['position'][[0,-1]])
        np.testing.assert_allclose(stretched['acceleration'][[0,-1]],0,atol=1e-12)
        with self.assertRaises(ValueError):stretch_trajectory(path,path['duration_s']/2)

    def test_food_cover_contains_convex_shape(self):
        rng=np.random.default_rng(19)
        vertices=rng.uniform([-0.08,-.025,-.02],[.08,.025,.02],(100,3));spheres=food_spheres(vertices)
        weights=rng.dirichlet(np.ones(100),size=500)
        points=np.r_[vertices,weights@vertices]
        distances=np.linalg.norm(points[:,None,:]-spheres[None,:,:3],axis=2)-spheres[None,:,3]
        self.assertTrue(np.all(distances.min(axis=1)<=1e-9))

    def test_world_frame_base_transforms_are_respected(self):
        root=Path(__file__).resolve().parents[2]/'assets/embodiments/w1-g2-black-real-v2'
        model=SphereKinematics(root/'w1_g2.urdf',root/'collision_spheres.yml')
        q={f'joint{i}':np.zeros(2) for i in range(1,7)}
        a,_=model.forward(q,np.eye(4));base=np.eye(4);base[:3,3]=[.6,.1,0]
        b,_=model.forward(q,base)
        np.testing.assert_allclose(b[:,:,:3]-a[:,:,:3],np.broadcast_to([.6,.1,0],a[:,:,:3].shape),atol=1e-12)


if __name__=='__main__':unittest.main()
