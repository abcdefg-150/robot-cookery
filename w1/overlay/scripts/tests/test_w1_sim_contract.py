import unittest
from types import SimpleNamespace as N
from unittest.mock import patch
import numpy as np
from scripts.w1_sim_interface import W1Simulation


class OnlineContractTests(unittest.TestCase):
    def setUp(self):
        self.commands=[];self.openings=[]
        planner=lambda:N(pose_ik=N(lower=np.full(6,-3.),upper=np.full(6,3.)))
        robot=N(left_planner=planner(),right_planner=planner(),
                set_arm_joints=lambda q,v,s:self.commands.append((s,np.asarray(q).copy())),
                set_gripper=lambda w,s,gripper_eps:self.openings.append((s,w)))
        self.sim=W1Simulation();self.sim.task=N(robot=robot,_wait=lambda seconds:None)
        self.sim.policy_monitor=N(success=False,info=lambda:{'success':False})
        self.sim.observation=lambda:{}
        self.action=np.zeros(14)

    def step(self):
        with patch('scripts.w1_sim_interface.joint_packet',return_value={'joint_positions':np.zeros(6)}):
            return self.sim.step(self.action)

    def test_opening_metres_become_internal_width_fractions(self):
        self.action[[6,13]]=[.036,.072];self.step()
        self.assertEqual(self.openings,[('left',.5),('right',1.)])

    def test_joint_changes_are_limited_in_both_directions(self):
        self.action[:6]=.2;self.action[7:13]=-.2;self.step()
        np.testing.assert_allclose(self.commands[0][1],.05)
        np.testing.assert_allclose(self.commands[1][1],-.05)

    def test_right_limit_error_does_not_apply_left_commands(self):
        self.action[7]=4.
        with self.assertRaises(ValueError):self.step()
        self.assertFalse(self.commands);self.assertFalse(self.openings)

    def test_openings_are_clipped_to_physical_stroke(self):
        self.action[[6,13]]=[-.1,.2];self.step()
        self.assertEqual(self.openings,[('left',0.),('right',1.)])


if __name__=='__main__':unittest.main()
