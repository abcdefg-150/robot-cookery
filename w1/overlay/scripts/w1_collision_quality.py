"""Reject unintended strong contacts throughout a simulated demonstration."""
import numpy as np


class CollisionQuality:
    def __init__(self):
        self.severe=[];self.peak={};self.contact_steps=0

    @staticmethod
    def policy_grasp(task,contact,other,robot):
        if getattr(task,"policy_observer",None) is None:return False
        if not other or not other<=set(task.foods)|{"spatula"}:return False
        if not all(n in {"g2_left_link","g2_right_link"} for n in robot):return False
        for body in contact.bodies:
            if body.entity.name not in robot:continue
            side=next((s for s in ("left","right") if body in getattr(task.robot,f"{s}_entity").get_links()),None)
            if side is None:return False
            joints=getattr(task.robot,f"{side}_entity").get_active_joints()
            target=sum(float(j.get_drive_target()[0]) for j in joints if j.name in {"g2_left_joint","g2_right_joint"})
            if target>=.065:return False
        return True

    def observe(self,task):
        for contact in task.scene.get_contacts():
            names=[body.entity.name for body in contact.bodies]
            pair=sorted(names);key="|".join(pair)
            impulse=sum(float(np.linalg.norm(p.impulse)) for p in contact.points)
            self.peak[key]=max(self.peak.get(key,0.0),impulse)
            if impulse<1e-7:continue
            self.contact_steps+=1
            robot=[name for name in names if name in task.robot_link_names]
            limit=None
            if robot:
                other=set(names)-task.robot_link_names
                intended=(other and other<={getattr(task,"grasp_intent",None)} and
                          all(name in {"g2_left_link","g2_right_link"} for name in robot) and
                          ("GRASP" in task.fsm_state or "DESCEND" in task.fsm_state or "RELEASE" in task.fsm_state
                           or "LIFT" in task.fsm_state or "TRANSFER" in task.fsm_state
                           or "TURN" in task.fsm_state or "STIR" in task.fsm_state or "SPATULA_RETURN" in task.fsm_state))
                intended=intended or self.policy_grasp(task,contact,other,robot)
                limit=.25 if intended else .01
            elif set(names)=={"spatula","skillet"}:
                intended_turn=task.fsm_state.startswith("TURN_") and any(x in task.fsm_state for x in ("INSERT","SCOOP","WITHDRAW"))
                intended_turn=intended_turn or task.fsm_state in {'STIR_LOWER','STIR_PUSH','STIR_REVERSE','STIR_WITHDRAW'}
                observer=getattr(task,"policy_observer",None)
                intended_turn=intended_turn or bool(observer is not None and task.fsm_state.startswith("TURN_") and observer.lifted["spatula"])
                intended_turn=intended_turn or bool(observer is not None and task.fsm_state=='STIR_POLICY' and observer.lifted['spatula'])
                limit=.025 if intended_turn else .0001
            elif set(names)&set(task.foods):
                limit=.08
            if limit is not None and impulse>limit:
                event={"phase":task.fsm_state,"time_s":task._record_step*task.scene.get_timestep(),
                       "bodies":pair,"impulse_Ns":impulse,"limit_Ns":limit}
                self.severe.append(event)
                raise RuntimeError(f"Unintended strong contact: {event}")

    def report(self):
        return {"severe_event_count":len(self.severe),"severe_events":self.severe,
                "peak_pair_impulse_Ns":dict(self.peak),"contact_steps":self.contact_steps,
                "thresholds_note":"Engineering simulation filters, not measured hardware force limits."}
