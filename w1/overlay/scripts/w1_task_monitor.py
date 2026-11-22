"""Physical success observer usable without running the expert FSM."""
import math
import numpy as np
import transforms3d as t3d


class W1TaskMonitor:
    def __init__(self,task):
        self.stir_task=task.stir_task;self.stir_monitor=task.stir_monitor
        self.initial={name:actor.pose.p.copy() for name,actor in task.foods.items()}
        self.initial["spatula"]=task.spatula.pose.p.copy()
        self.lifted={name:False for name in self.initial};self.events=[];self.window=None
        self.release_seen=False;self.stable_steps=0;self.success=False;self.checks={}

    @staticmethod
    def inside(task,kind):
        actor=task.foods[kind];points=task.food_vertices[kind]@t3d.quaternions.quat2mat(actor.pose.q).T+actor.pose.p
        return bool(np.all(np.linalg.norm(points[:,:2]-task.pot_center[:2],axis=1)<task.pot_radius+.003)
                    and task.pot_floor-.004<points[:,2].min()<task.pot_rim+.006)

    def observe(self,task):
        inside={name:self.inside(task,name) for name in task.foods}
        contact=False;robot_impulse=0.0;held=set()
        for value in task.scene.get_contacts():
            names={body.entity.name for body in value.bodies}
            impulse=sum(float(np.linalg.norm(p.impulse)) for p in value.points)
            if names&{"g2_left_link","g2_right_link"} and impulse>1e-7:held.update(names&set(self.initial))
            if names=={"green_pepper","spatula"} and impulse>1e-8:contact=True
            if "green_pepper" in names and names&task.robot_link_names:robot_impulse=max(robot_impulse,impulse)
        for name,actor in {**task.foods,"spatula":task.spatula}.items():
            if name in held and actor.pose.p[2]>self.initial[name][2]+.07:self.lifted[name]=True
        if task.stir_task and not task.stir_monitor.active and not task.stir_monitor.success and self.lifted['spatula'] and all(inside.values()):
            task.stir_monitor.begin(task.spatula.pose.to_transformation_matrix()[:2,0],{k:v.pose.p for k,v in task.foods.items()})
        up=t3d.quaternions.quat2mat(task.pepper.pose.q)[:,2]
        if self.window is None and contact and all(inside.values()) and self.lifted["spatula"] and len(self.events)<2:
            self.window={"up":up.copy(),"position":task.pepper.pose.p.copy(),"contact_steps":0,
                         "peak_angle":0.0,"peak_lift":0.0,"peak_displacement":0.0,"robot_contact_max_impulse":0.0,"clear_steps":0}
        angle=0.0
        if self.window is not None:
            window=self.window
            angle=math.degrees(math.acos(float(np.clip(up@window["up"],-1,1))))
            window["peak_angle"]=max(window["peak_angle"],angle)
            window["peak_lift"]=max(window["peak_lift"],float(task.pepper.pose.p[2]-window["position"][2]))
            window["peak_displacement"]=max(window["peak_displacement"],float(np.linalg.norm(task.pepper.pose.p[:2]-window["position"][:2])))
            window["contact_steps"]+=int(contact);window["robot_contact_max_impulse"]=max(window["robot_contact_max_impulse"],robot_impulse)
            tool_points=task.tool_vertices@t3d.quaternions.quat2mat(task.spatula.pose.q).T+task.spatula.pose.p
            clear=tool_points[:,2].min()>task.pot_rim+.01
            window["clear_steps"]=window["clear_steps"]+1 if clear else 0
            if window["clear_steps"]>=round(.15/task.scene.get_timestep()):
                valid=(window["contact_steps"]>=2 and window["peak_angle"]>=math.degrees(task.flip_min_angle)
                    and (window["peak_lift"]>=.004 or window["peak_displacement"]>=.012)
                    and window["robot_contact_max_impulse"]<1e-5 and all(inside.values()))
                if valid:self.events.append({"cycle":len(self.events)+1,"valid":True,"peak_angle_deg":window["peak_angle"],
                    "peak_lift":window["peak_lift"],"peak_displacement":window["peak_displacement"],
                    "contact_steps":window["contact_steps"],"robot_contact_max_impulse":window["robot_contact_max_impulse"]})
                self.window=None
        task.flip_events=list(self.events);task.last_tool_contact=contact;task.current_turn_angle=angle
        task.pick_checks=dict(self.lifted)
        entity=getattr(task.robot,f"{task.tool_arm}_entity");names=[j.name for j in entity.get_active_joints()]
        width=sum(float(entity.get_qpos()[names.index(n)]) for n in ("g2_left_joint","g2_right_joint"))
        returned=task._tool_in_region()
        work_done=task.stir_monitor.success if task.stir_task else len(self.events)==2
        if work_done and returned and width>.028:self.release_seen=True
        task.tool_returned=bool(self.release_seen and returned)
        supported={"skillet"}
        for _ in task.foods:
            for names,positions in task.support_edges.items():
                if any(n in supported for n in names) and all(np.linalg.norm(task.foods[n].pose.p-p)<.008 for n,p in positions.items()):supported.update(names)
        parked=all(np.linalg.norm(np.asarray(getattr(task.robot,f"get_{side}_ee_pose")()[:3])-task.home_targets[side][:3])<.025 for side in ("left","right"))
        self.checks={"all_picks_verified":all(self.lifted.values()),"all_foods_inside_pan":all(inside.values()),
            "no_severe_collisions":not task.collision_quality.severe,
            "two_effective_turns":len(self.events)==2,"tool_returned_and_released":task.tool_returned,
            "foods_supported":set(task.foods)<=supported,"both_arms_parked":bool(parked),
            "foods_stationary":all(np.linalg.norm(b.linear_velocity)<.04 and np.linalg.norm(b.angular_velocity)<.4 for b in task.food_bodies.values())}
        if task.stir_task:
            self.checks.pop('two_effective_turns')
            self.checks['effective_reciprocal_stirring']=task.stir_monitor.success
        self.stable_steps=self.stable_steps+1 if all(self.checks.values()) else 0
        self.success=self.stable_steps>=round(1/task.scene.get_timestep())
        pending=next((k for k in ("small_pepper","green_pepper","potato","eggplant") if not inside[k]),None)
        if pending:task.fsm_state=f"{pending.upper()}_POLICY"
        elif not self.lifted["spatula"]:task.fsm_state="SPATULA_PICKUP_POLICY"
        elif not work_done:task.fsm_state='STIR_POLICY' if task.stir_task else f"TURN_{len(self.events)+1}_POLICY"
        elif not task.tool_returned:task.fsm_state="SPATULA_RETURN_POLICY"
        else:task.fsm_state="FINAL_SETTLE_POLICY"
        if self.success:task.full_completed=True;task.fsm_state="DONE"

    def info(self):
        return {"success":self.success,"success_checks":dict(self.checks),"pick_checks":dict(self.lifted),
                "flip_events":list(self.events),"effective_flip_count":len(self.events),"stable_steps":self.stable_steps,
                "task_judgement_version":"stir_contact_motion_v1" if self.stir_task else "two_turns_45deg_v1",
                "stir_metrics":self.stir_monitor.report()}
