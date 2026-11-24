"""Reject unintended strong contacts throughout a simulated demonstration."""
import numpy as np


class CollisionQuality:
    def __init__(self, *, terminate_on_severe=True):
        self.terminate_on_severe = terminate_on_severe
        self.severe=[];self.peak={};self.contact_steps=0;self.inter_arm_contact_count=0

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
                intents={getattr(task,"grasp_intent",None)}
                phases=[task.fsm_state]
                if getattr(task,'grasp_intents',{}):
                    intents=set();phases=[]
                    for body in contact.bodies:
                        if body.entity.name not in robot:continue
                        for side in ('left','right'):
                            if body in getattr(task.robot,f'{side}_entity').get_links():
                                intents.add(task.grasp_intents.get(side))
                                phases.append(getattr(task,'arm_fsm_state',{}).get(side,task.fsm_state));break
                intended=(other and other<=intents and
                          all(name in {"g2_left_link","g2_right_link"} for name in robot) and
                          any(any(word in phase for word in ('GRASP','DESCEND','RELEASE','LIFT','TRANSFER','TURN','STIR','SPATULA_RETURN')) for phase in phases))
                intended=intended or self.policy_grasp(task,contact,other,robot)
                limit=.25 if intended else .01
                if len(robot)>1 and hasattr(task,'robot'):
                    sides=set()
                    for body in contact.bodies:
                        for side in ('left','right'):
                            if body in getattr(task.robot,f'{side}_entity').get_links():sides.add(side);break
                    if len(sides)>1:
                        self.inter_arm_contact_count+=1;limit=1e-7
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
                if self.terminate_on_severe:
                    raise RuntimeError(f"Unintended strong contact: {event}")

    def report(self):
        return {"severe_event_count":len(self.severe),"severe_events":self.severe,
                "peak_pair_impulse_Ns":dict(self.peak),"contact_steps":self.contact_steps,
                "inter_arm_contact_count":self.inter_arm_contact_count,
                "thresholds_note":"Engineering simulation filters, not measured hardware force limits."}
