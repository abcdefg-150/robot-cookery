import unittest
from types import SimpleNamespace as N
import numpy as np
from scripts.w1_collision_quality import CollisionQuality

def task(pair,impulse,phase,intent="green_pepper"):
 contact=N(bodies=[N(entity=N(name=x)) for x in pair],points=[N(impulse=np.array([impulse,0,0]))])
 return N(scene=N(get_contacts=lambda:[contact],get_timestep=lambda:.004),robot_link_names={"g2_left_link","g2_right_link","link6"},foods={"green_pepper":None},grasp_intent=intent,fsm_state=phase,_record_step=250)

class ContactFilterTest(unittest.TestCase):
 def test_intended_clamp_during_release_is_not_an_impact(self):
  q=CollisionQuality();q.observe(task(["g2_left_link","green_pepper"],.08,"GREEN_PEPPER_RELEASE"));self.assertEqual(q.report()["severe_event_count"],0)
 def test_robot_pan_contact_is_rejected(self):
  with self.assertRaisesRegex(RuntimeError,"strong contact"):CollisionQuality().observe(task(["link6","skillet"],.02,"TURN_1_APPROACH"))
 def test_can_record_severe_contact_without_terminating(self):
  q=CollisionQuality(terminate_on_severe=False)
  q.observe(task(["link6","skillet"],.02,"TURN_1_APPROACH"))
  self.assertEqual(q.report()["severe_event_count"],1)
  self.assertEqual(q.severe[0]["limit_Ns"],.01)
 def test_air_approach_and_controlled_stir_are_distinguished(self):
  with self.assertRaises(RuntimeError):CollisionQuality().observe(task(["spatula","skillet"],.001,"TURN_1_APPROACH"))
  q=CollisionQuality();q.observe(task(["spatula","skillet"],.01,"TURN_1_SCOOP"));self.assertFalse(q.severe)
 def test_wrong_food_contact_is_rejected(self):
  with self.assertRaises(RuntimeError):CollisionQuality().observe(task(["g2_left_link","green_pepper"],.02,"POTATO_GRASP","potato"))
 def test_stir_withdraw_allows_only_bounded_tool_pan_contact(self):
  q=CollisionQuality();q.observe(task(['spatula','skillet'],.01,'STIR_WITHDRAW'));self.assertFalse(q.severe)
  with self.assertRaises(RuntimeError):CollisionQuality().observe(task(['spatula','skillet'],.03,'STIR_WITHDRAW'))
 def test_dual_intents_do_not_allow_the_other_arms_food(self):
  t=task(['g2_left_link','green_pepper'],.08,'PAIR_1_GRASP',None)
  body=t.scene.get_contacts()[0].bodies[0]
  t.robot=N(left_entity=N(get_links=lambda:[body]),right_entity=N(get_links=lambda:[]))
  t.grasp_intents={'left':'green_pepper','right':'potato'}
  t.arm_fsm_state={'left':'GREEN_PEPPER_TRANSFER','right':'POTATO_RELEASE'}
  t.fsm_state='PAIR_1_RIGHT_LOWER'
  CollisionQuality().observe(t)
  t.grasp_intents={'left':'potato','right':'green_pepper'}
  with self.assertRaises(RuntimeError):CollisionQuality().observe(t)
 def test_even_light_contact_between_arms_is_rejected(self):
  t=task(['g2_left_link','g2_right_link'],1e-5,'PAIR_1_TRANSFER',None)
  bodies=t.scene.get_contacts()[0].bodies
  t.robot=N(left_entity=N(get_links=lambda:[bodies[0]]),right_entity=N(get_links=lambda:[bodies[1]]))
  q=CollisionQuality()
  with self.assertRaises(RuntimeError):q.observe(t)
  self.assertEqual(q.inter_arm_contact_count,1)
 def test_policy_closed_gripper_can_grasp_without_expert_intent(self):
  t=task(["g2_left_link","green_pepper"],.15,"GREEN_PEPPER_POLICY",None)
  body=t.scene.get_contacts()[0].bodies[0]
  opening=[0.0]
  entity=N(get_links=lambda:[body],get_active_joints=lambda:[N(name='g2_left_joint',get_drive_target=lambda:opening)])
  t.robot=N(left_entity=entity,right_entity=N(get_links=lambda:[]))
  t.policy_observer=N(lifted={'spatula':False})
  q=CollisionQuality();q.observe(t);self.assertFalse(q.severe)
  opening[0]=.072
  with self.assertRaises(RuntimeError):CollisionQuality().observe(t)
if __name__=="__main__":unittest.main()
