"""Paired food pickup, concurrent transport, and coordinated pan access."""
import math
import numpy as np
import transforms3d as t3d
from .w1_dual_trajectory import smooth_trajectory,stretch_trajectory
from .w1_dual_collision import DualCollisionChecker


class DualFoodCoordinator:
    def __init__(self,task):
        self.task=task;self.checker=DualCollisionChecker(task);self.executed=[];self.gripper_checks=[]

    def _q(self,side):
        entity=getattr(self.task.robot,f'{side}_entity');names=[j.name for j in entity.get_active_joints()]
        planner=getattr(self.task.robot,f'{side}_planner')
        joints=entity.get_active_joints()
        # Join successive command segments at the previous drive target.
        # Restarting at lagging feedback introduces a tiny position step and
        # a large numerical acceleration at every phase boundary.
        return np.asarray([float(joints[names.index(n)].get_drive_target()[0]) for n in planner.active_joints_name])

    def _phase(self,label,states):
        task=self.task;task._transition(label)
        for side,state in states.items():
            if state!=task.arm_fsm_state[side]:
                task.arm_fsm_state[side]=state
                task.arm_fsm_transitions[side].append({'state':state,'time':task._record_step*task.scene.get_timestep()})

    def _validate(self,side,result):
        import torch
        planner=getattr(self.task.robot,f'{side}_planner');positions=result['position']
        if np.any(positions<planner.pose_ik.lower) or np.any(positions>planner.pose_ik.upper):return False
        if np.max(abs(result['velocity'])/np.asarray(planner.trajectory_velocity_limits))>1.0001:return False
        if 'acceleration' in result and np.max(abs(result['acceleration'])/np.asarray(planner.trajectory_acceleration_limits))>1.0001:return False
        if 'jerk' in result and np.max(abs(result['jerk']))>8.001:return False
        for part in np.array_split(positions,max(1,int(np.ceil(len(positions)/256)))):
            query=torch.as_tensor(part,dtype=torch.float32,device='cuda').unsqueeze(1)
            if not planner.motion_gen.rollout_fn.rollout_constraint(query,use_batch_env=False).feasible.all().item():return False
        if not self.task._path_clear(side,positions):return False
        result['all_control_samples_collision_checked']=True
        return True

    def _plan(self,side,tcp,start=None):
        from envs.robot.cartesian_trajectory import cartesian_knots
        task=self.task;planner=getattr(task.robot,f'{side}_planner');base=planner.robot_origion_pose
        virtual=np.r_[tcp[:3,3]-tcp[:3,:3]@[.12,0,0],t3d.quaternions.mat2quat(tcp[:3,:3])]
        ee=task.robot._trans_from_gripper_to_endlink(virtual,arm_tag=side)
        p,q=planner._trans_from_world_to_base(np.r_[base.p,base.q],np.r_[ee.p,ee.q]);p=np.asarray(p)+planner.frame_bias
        current=self._q(side) if start is None else np.asarray(start)
        def parameterize(knots):
            return smooth_trajectory(knots,np.asarray(planner.trajectory_velocity_limits),
                                     np.asarray(planner.trajectory_acceleration_limits),jerk_limits=8.)
        knots=cartesian_knots(planner.pose_ik,current,p,q)
        if knots is not None:
            result=parameterize(knots)
            if self._validate(side,result):return result
        selected=None
        def accept(solution):
            nonlocal selected
            candidate=parameterize(np.stack([current,solution]))
            if self._validate(side,candidate):selected=candidate;return True
            return False
        planner.pose_ik.solve(p,q,current,accept=accept,max_restarts=0 if start is not None else 32)
        return selected

    def _tcp(self,position,rotation):
        result=np.eye(4);result[:3,:3]=rotation;result[:3,3]=position
        return result

    def _object_tcp(self,side,position,rotation):
        return self._tcp(position,rotation)@np.linalg.inv(self.task.held_frames[side])

    def _object_plan(self,kind,side,position):
        current=self.task.foods[kind].pose.to_transformation_matrix()[:3,:3].copy()
        rotations=[current]
        for angle in (.1,.2,.3,.4):
            rotations.extend(t3d.euler.euler2mat(*r)@current for r in ((angle,0,0),(-angle,0,0),(0,angle,0),(0,-angle,0)))
        for rotation in rotations:
            plan=self._plan(side,self._object_tcp(side,position,rotation))
            if plan is not None:return plan,rotation
        raise RuntimeError(f'No smooth low-rotation object path: {kind}')

    def _execute(self,label,plans,states):
        task=self.task;self._phase(label,states)
        if any(plan is None for plan in plans.values()):raise RuntimeError(f'Unreachable dual stage {label}: {[s for s,p in plans.items() if p is None]}')
        duration=max(plan['duration_s'] for plan in plans.values())
        paired={}
        for side in ('left','right'):
            if side in plans:paired[side]=stretch_trajectory(plans[side],duration)
            else:
                n=int(round(duration/.004))+1
                paired[side]={'status':'Success','position':np.repeat(self._q(side)[None],n,axis=0),
                              'velocity':np.zeros((n,6)),'duration_s':duration}
        for side in plans:
            if not self._validate(side,paired[side]):raise RuntimeError(f'Synchronized path invalid: {label}/{side}')
        endpoint_clearance=self.checker.clearance(paired['left']['position'][-1:],paired['right']['position'][-1:])
        if endpoint_clearance<self.checker.margin:
            raise RuntimeError(f'Dual sweep collision: {label}, endpoint clearance={endpoint_clearance:.4f}m')
        selected=None;best=-float('inf')
        # A time offset retains each arm's smooth path and its zero-velocity
        # endpoints. Never execute either half until the paired sweep passes.
        for delay,delayed in [(0.,'right'),(.2,'right'),(.2,'left'),(.4,'right'),(.4,'left'),(.8,'right'),(.8,'left')]:
            if delay and len(plans)<2:break
            shift=int(round(delay/.004));candidate={}
            for side in ('left','right'):
                source=paired[side];before=shift if side==delayed else 0;after=shift-before
                candidate[side]=dict(source,position=np.pad(source['position'],((before,after),(0,0)),mode='edge'),
                                     velocity=np.pad(source['velocity'],((before,after),(0,0)),mode='constant'))
                for field in ('acceleration','jerk'):
                    if field in source:candidate[side][field]=np.pad(source[field],((before,after),(0,0)),mode='constant')
                candidate[side]['duration_s']=(len(candidate[side]['position'])-1)*.004
            clearance=self.checker.clearance(candidate['left']['position'],candidate['right']['position'])
            best=max(best,clearance)
            if clearance>=self.checker.margin:selected=candidate;break
        if selected is None:raise RuntimeError(f'Dual sweep collision: {label}, best clearance={best:.4f}m')
        event={'stage':label,'time_s':task._record_step*task.scene.get_timestep(),
               'duration_s':len(selected['left']['position'])*.004,'start_delay_s':delay,
               'delayed_arm':delayed if delay else None,'minimum_sphere_clearance_m':clearance,
               'command_joint_jerk_limit_rad_s3':8.}
        event['command_peaks']={side:{'velocity_rad_s':float(np.max(abs(paired[side]['velocity']))),
                                     'acceleration_rad_s2':float(np.max(abs(paired[side]['acceleration']))),
                                     'jerk_rad_s3':float(np.max(abs(paired[side]['jerk'])))} for side in plans}
        event['both_arms_moving_s']=float(np.sum((np.linalg.norm(selected['left']['velocity'],axis=1)>.01)&
                                                     (np.linalg.norm(selected['right']['velocity'],axis=1)>.01))*.004)
        self.executed.append(event);print(f'[W1 dual] {event}',flush=True)
        for side in ('left','right'):getattr(task,f'{side}_joint_path').append(selected[side])
        task.take_dense_action({'left_arm':selected['left'],'right_arm':selected['right'],'left_gripper':None,'right_gripper':None})

    def _move(self,label,targets,states):
        plans={side:self._plan(side,pose) for side,pose in targets.items()}
        self._execute(label,plans,states)

    def _grippers(self,label,openings,states):
        from envs.utils import ArmTag
        self._phase(label,states);q={s:self._q(s)[None] for s in ('left','right')}
        gap=self.checker.clearance(q['left'],q['right'],open_gripper=any(v>.5 for v in openings.values()))
        if gap<self.checker.margin:raise RuntimeError(f'Unsafe paired finger opening: {label} {gap:.4f}m')
        self.gripper_checks.append({'stage':label,'minimum_sphere_clearance_m':gap})
        actions=[]
        for side,value in openings.items():
            joints=getattr(self.task.robot,f'{side}_entity').get_active_joints()
            width=sum(float(j.get_drive_target()[0]) for j in joints if j.name in ('g2_left_joint','g2_right_joint'))
            if abs(width-.072*value)<1e-5:continue
            actions.append(self.task.open_gripper(ArmTag(side)) if value>.5 else self.task.close_gripper(ArmTag(side)))
        if actions:
            self.task.move(*actions);self.task._wait(.18)

    def _pair(self,index,left,right,left_xy,right_xy):
        task=self.task;names={'left':left,'right':right};configs={}
        for side,kind in names.items():configs[side]=task._select_grasp(kind,side)
        task.grasp_intent=None;task.grasp_intents=dict(names)
        start={s:task.foods[k].pose.p.copy() for s,k in names.items()}
        rotations={s:t3d.euler.euler2mat(*configs[s]['grasp_rotation_rpy']) for s in names}
        label=lambda s:s.upper()
        states=lambda phase:{s:f'{k.upper()}_{phase}' for s,k in names.items()}
        self._move(f'PAIR_{index}_APPROACH',{s:self._tcp(start[s]+configs[s]['approach_offset'],rotations[s]) for s in names},states('APPROACH'))
        self._grippers(f'PAIR_{index}_OPEN',{'left':1.,'right':1.},states('APPROACH'))
        start={s:task.foods[k].pose.p.copy() for s,k in names.items()}
        self._move(f'PAIR_{index}_DESCEND',{s:self._tcp(start[s]+configs[s]['grasp_offset'],rotations[s]) for s in names},states('DESCEND'))
        self._grippers(f'PAIR_{index}_GRASP',{'left':0.,'right':0.},states('GRASP'))
        lift_start={}
        for side,kind in names.items():
            task.held_entity[side]=task.foods[kind];task._remember_grasp(side,task.foods[kind]);lift_start[side]=task.foods[kind].pose.p.copy()
        self._move(f'PAIR_{index}_LIFT',{s:self._object_tcp(s,lift_start[s]+configs[s]['lift_vector'],task.foods[names[s]].pose.to_transformation_matrix()[:3,:3]) for s in names},states('LIFT'))
        task._wait(.2)
        for side,kind in names.items():
            task.pick_checks[kind]=bool(task.foods[kind].pose.p[2]>lift_start[side][2]+.07)
            if not task.pick_checks[kind]:raise RuntimeError(f'Dual physical grasp failed: {kind}')
            task._remember_grasp(side,task.foods[kind])
        staging=task.pot_center[:2]+[-.14,-.035]
        destinations={'left':np.r_[staging,task.pot_rim+.065],'right':np.r_[right_xy,task.pot_rim+.065]}
        plans={}
        from .w1_motion_quality import rotation_distance_deg
        for side,kind in names.items():
            initial=task.foods[kind].pose.to_transformation_matrix()[:3,:3]
            plans[side],rotation=self._object_plan(kind,side,destinations[side])
            task.transfer_plans[kind]={'orientation_change_deg':rotation_distance_deg(initial,rotation),'yaw_constrained':False,
                                       'target_xy':np.asarray(left_xy if side=='left' else right_xy).tolist()}
        self._execute(f'PAIR_{index}_TRANSFER',plans,states('TRANSFER'))
        task._wait(.1)
        p,r=task._food_release_target(right,'right',right_xy)
        self._move(f'PAIR_{index}_RIGHT_LOWER',{'right':self._object_tcp('right',p,r)},
                   {'right':f'{right.upper()}_RELEASE'})
        self._grippers(f'PAIR_{index}_RIGHT_RELEASE',{'right':1.},{'right':f'{right.upper()}_RELEASE'})
        task.held_entity['right']=None;task.grasp_intents.pop('right');task._wait(.2)
        release_p,release_r=task._food_release_target(left,'left',left_xy,require_reachable=False)
        left_plan,enter_rotation=self._object_plan(left,'left',np.r_[left_xy,max(task.pot_rim+.065,release_p[2]+.025)])
        release_p,release_r=task._food_release_target(left,'left',left_xy,rotation=enter_rotation,require_reachable=False)
        future_lower=None
        for extra in (0.,.01,.02,.03,.04,.05,.06):
            candidate=release_p+[0,0,extra]
            future_lower=self._plan('left',self._object_tcp('left',candidate,release_r),start=left_plan['position'][-1])
            if future_lower is not None:
                task.open_gripper_clearance=True
                try:opening_clear=task._path_clear('left',future_lower['position'][-1:])
                finally:task.open_gripper_clearance=False
                if not opening_clear:future_lower=None
            if future_lower is not None:release_p=candidate;break
        if future_lower is None:raise RuntimeError(f'No left release lookahead path: pair {index}')
        release_minimum=float(release_p[2])
        for retreat in self._retreat_plans('right'):
            stationary=np.repeat(retreat['position'][-1:],len(future_lower['position']),axis=0)
            if self.checker.clearance(future_lower['position'],stationary)<self.checker.margin:continue
            if self.checker.clearance(future_lower['position'][-1:],stationary[-1:],open_gripper=True)<self.checker.margin:continue
            try:
                self._execute(f'PAIR_{index}_LEFT_ENTER_RIGHT_RETREAT',{'left':left_plan,'right':retreat},
                              {'left':f'{left.upper()}_TRANSFER','right':f'{right.upper()}_RETREAT'})
                break
            except RuntimeError as error:
                if not str(error).startswith('Dual sweep collision:'):raise
        else:raise RuntimeError(f'No paired pan-entry and retreat corridor: pair {index}')
        p,r=task._food_release_target(left,'left',left_xy,require_reachable=False)
        p[2]=max(float(p[2]),release_minimum);task.placement_release_heights[left]=float(p[2])
        self._move(f'PAIR_{index}_LEFT_LOWER',{'left':self._object_tcp('left',p,r)},
                   {'left':f'{left.upper()}_RELEASE'})
        self._grippers(f'PAIR_{index}_LEFT_RELEASE',{'left':1.},{'left':f'{left.upper()}_RELEASE'})
        task.held_entity['left']=None;task.grasp_intents.pop('left');task._wait(.2)
        self._execute(f'PAIR_{index}_LEFT_RETREAT',{'left':self._retreat_plan('left')},{'left':f'{left.upper()}_RETREAT'})
        task._wait(.3)
        for kind in names.values():
            if not task._food_inside(kind):raise RuntimeError(f'Dual food not settled inside pan: {kind}')

    def _retreat_plan(self,side):
        result=next(self._retreat_plans(side),None)
        if result is None:raise RuntimeError(f'No smooth retreat: {side}')
        return result

    def _retreat_plans(self,side):
        sign=-1 if side=='left' else 1;actual=self.task._tcp_matrix(side)
        for offset in ([0,0,.08],[sign*.04,0,.08],[sign*.08,0,.08],[sign*.08,-.04,.08],
                       [sign*.04,-.04,.04],[sign*.06,-.04,.04],[0,0,.04],
                       [sign*.08,-.04,.04],[sign*.10,-.04,.04],[sign*.12,-.04,.04],
                       [sign*.04,-.06,.025],[0,-.04,.04],[0,0,.065]):
            candidate=actual.copy();candidate[:3,3]+=offset
            result=self._plan(side,candidate)
            if result is not None:yield result
        # A released hand can tilt slightly toward the base to leave room
        # above the pan without an unnecessary wrist flip or a high lift.
        for tilt in (.15,.2,.3):
            candidate=actual.copy();candidate[:3,3]+=[0,-.08,.06]
            candidate[:3,:3]=t3d.euler.euler2mat(tilt,0,0)@actual[:3,:3]
            result=self._plan(side,candidate)
            if result is not None:yield result

    def run(self):
        task=self.task;center=task.pot_center[:2]
        pairs = (('green_pepper', 'small_pepper'), ('eggplant', 'potato'))
        if task.scene_variation is not None:
            pairs = task.scene_variation["pair_foods"]
        self._pair(1, *pairs[0], center+[-.025,-.025], center+[.060,-.025])
        self._pair(2, *pairs[1], center+[-.055,-.015], center+[.030,.065])
        targets={}
        for side in ('left','right'):
            home=task.home_targets[side];r=t3d.quaternions.quat2mat(home[3:])
            targets[side]=self._tcp(np.asarray(home[:3])+r@[.12,0,0],r)
        park_states = {side: f"{name.upper()}_PARK" for side, name in zip(("left", "right"), pairs[1])}
        self._move('PAIR_FINAL_PARK', targets, park_states)
        self._grippers('PAIR_FINAL_CLOSE',{'left':0.,'right':0.},{'left':'IDLE','right':'IDLE'})
        task.grasp_intents={}

    def report(self):
        return {'mode':'paired_pickup_and_transport_coordinated_pan_release','stages':self.executed,
                'minimum_predicted_clearance_m':min(e['minimum_sphere_clearance_m'] for e in self.executed+self.gripper_checks),
                'gripper_clearance_checks':self.gripper_checks,
                'required_clearance_m':self.checker.margin,
                'kinematics_max_link_error_m':self.checker.kinematics_max_link_error_m,
                'both_arms_moving_s':sum(e['both_arms_moving_s'] for e in self.executed),
                'intermediate_home_returns':0,'timing':'C2 joint splines, quintic phase, jerk bounded, stretch-only synchronization'}
