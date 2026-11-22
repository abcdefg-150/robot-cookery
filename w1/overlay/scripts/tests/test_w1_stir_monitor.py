import unittest
import numpy as np
from scripts.w1_stir_monitor import StirMonitor


class StirMonitorTest(unittest.TestCase):
    def run_motion(self,seconds=3,contact=True,inside=True,food_motion=.01,reciprocal=True):
        monitor=StirMonitor();monitor.begin([1,0],{'potato':np.zeros(3)})
        for time in np.arange(0,seconds,.004):
            x=.025*np.sin(2*np.pi*time) if reciprocal else .025*time
            monitor.observe(.004,[x,0],inside,{'potato'} if contact else set(),{'potato':[food_motion,0,0]})
        return monitor
    def test_two_seconds_of_reciprocal_motion_and_visible_contact_pass(self):
        self.assertTrue(self.run_motion().success)
    def test_short_motion_fails(self):self.assertFalse(self.run_motion(seconds=1.9).success)
    def test_one_way_motion_fails(self):self.assertFalse(self.run_motion(reciprocal=False).success)
    def test_no_contact_fails(self):self.assertFalse(self.run_motion(contact=False).success)
    def test_no_visible_food_motion_fails(self):self.assertFalse(self.run_motion(food_motion=.001).success)
    def test_outside_pan_fails(self):self.assertFalse(self.run_motion(inside=False).success)
    def test_static_contact_does_not_count_waiting(self):
        monitor=StirMonitor();monitor.begin([1,0],{'potato':np.zeros(3)})
        for _ in range(1000):monitor.observe(.004,[0,0],True,{'potato'},{'potato':[.01,0,0]})
        self.assertFalse(monitor.success);self.assertEqual(monitor.motion_s,0)


if __name__=='__main__':unittest.main()
