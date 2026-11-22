"""Real-reference W1 task: four props first, right-hand tool, table return."""
import json
import math
from pathlib import Path
import numpy as np
import sapien
import transforms3d as t3d
from .w1_finals import w1_finals
from .utils import ArmTag
from .utils.finals_assets import build_vegetable

ROOT=Path(__file__).resolve().parents[1]


class w1_finals_real(w1_finals):
    def setup_demo(self, **kwargs):
        self.record_sink=None
        self.reference_episode=kwargs.get("real_episode","data/real_w1/episode_0001")
        self.camera_reference_file=kwargs.get('camera_reference_file')
        self.native_meta_template=kwargs.get('native_meta_template')
        fit_file=kwargs.get("wrist_image_fit")
        self.wrist_image_fit=json.loads((ROOT/fit_file).read_text()) if fit_file else None
        self.camera_render_mode=kwargs.get("render_mode","raster")
        self.contact_offset=float(kwargs.get("small_object_contact_offset_m",.002))
        self.object_solver_iterations=kwargs.get("small_object_solver_iterations",[32,8])
        self.final_stability_min_s=float(kwargs.get("final_stability_min_s",1.0))
        self.final_settle_timeout_s=float(kwargs.get("final_settle_timeout_s",15.0))
        self.final_stability_steps=0
        self.held_entity={};self.grasp_intent=None
        self.policy_observer=None
        self.selected_grasps={}
        self.stir_task=bool(kwargs.get("stir_task",False))
        from scripts.w1_stir_monitor import StirMonitor
        self.stir_monitor=StirMonitor(kwargs.get("stir_min_motion_s",2.0))
        self.stir_plan=None
        super().setup_demo(**kwargs)
        self.open_before_descend=True
        for side in ("left","right"):
            for link in getattr(self.robot,f"{side}_entity").get_links():
                if link.name in {"g2_base_link","g2_left_link","g2_right_link"}:
                    for shape in link.get_collision_shapes():shape.contact_offset=self.contact_offset
            for joint in getattr(self.robot,f"{side}_entity").get_active_joints():
                if joint.name in {"g2_left_joint","g2_right_joint"}:
                    joint.set_drive_property(stiffness=getattr(self.robot,f"{side}_gripper_stiffness"),
                        damping=getattr(self.robot,f"{side}_gripper_damping"),force_limit=50)
        from scripts.w1_collision_quality import CollisionQuality
        self.collision_quality=CollisionQuality()
        for side in ("left","right"):
            getattr(self.robot,f"{side}_planner").trajectory_accept=lambda path,side=side:self._path_clear(side,path)

    def setup_scene(self, **kwargs):
        super().setup_scene(**kwargs)
        # Base_Task calls setup_scene() without forwarding task kwargs.
        if self.camera_render_mode=="raster":sapien.render.set_camera_shader_dir("default")
        elif self.camera_render_mode!="rt":raise ValueError(f"Unknown camera renderer: {self.camera_render_mode}")

    def create_table_and_wall(self, table_xy_bias=[0,0],table_height=.74):
        from .utils.create_actor import create_box,create_table
        self.table_xy_bias=table_xy_bias;self.wall_texture=self.table_texture=None
        self.wall=create_box(self.scene,sapien.Pose([0,1.5,1.5]),half_size=[4,.15,1.5],
                             color=(.005,.006,.005),name="wall",is_static=True)
        # Approximate rear panels inferred from initial RGB-D, outside the
        # reachable workspace. These provide the real views' grey/black split.
        self.rear_panel=create_box(self.scene,sapien.Pose([.885,1.025,.60]),half_size=[.615,.025,.50],
                                  color=(.12,.14,.15),name="rear_panel",is_static=True)
        self.left_rear_panel=create_box(self.scene,sapien.Pose([-2.7,.9,1.0]),half_size=[.55,.025,1.0],
                                       color=(.12,.14,.15),name="left_rear_panel",is_static=True)
        self.rear_floor=create_box(self.scene,sapien.Pose([0,1.5,.005]),half_size=[4,2,.005],
                                  color=(.005,.006,.005),name="rear_floor",is_static=True)
        # Real image board boundaries back-project to y=-.311..+.215 m.
        self.table=create_table(self.scene,sapien.Pose([0,-.048,table_height+self.table_z_bias]),
            length=1.2,width=.526,height=table_height+self.table_z_bias,thickness=.05,is_static=True)

    def load_camera(self, **kwargs):
        super().load_camera(**kwargs)
        if self.camera_reference_file:
            calib=json.loads((ROOT/self.camera_reference_file).read_text())['calibration']
        else:
            meta=json.loads((ROOT/self.reference_episode/"meta.json").read_text())
            calib=meta["sensors"]["cam_front"]["calibration"]
        optical=np.asarray(calib["extrinsics"]["left_arm"])
        base=self.robot.left_entity_origion_pose.to_transformation_matrix()
        axes=np.eye(4);axes[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]]
        head=self.cameras.static_camera_list[self.cameras.head_camera_id]
        head.entity.set_pose(sapien.Pose(base@optical@axes))
        k=np.asarray(calib["intrinsics"]["camera"])
        for camera in (head,self.cameras.left_camera,self.cameras.right_camera):
            camera.set_perspective_parameters(.01,10,k[0,0],k[1,1],k[0,2],k[1,2],0)

    def load_actors(self):
        super().load_actors()
        kind="small_pepper"
        actor,body,half,vertices=build_vegetable(self.scene,kind,self.assets[kind]["initial_xy"],self.table_top,self.assets)
        self.foods[kind]=actor;self.food_bodies[kind]=body;self.food_half[kind]=half;self.food_vertices[kind]=vertices
        for value in [*self.food_bodies.values(),self.spatula_body]:
            value.solver_position_iterations=self.object_solver_iterations[0]
            value.solver_velocity_iterations=self.object_solver_iterations[1]
            for shape in value.get_collision_shapes():shape.contact_offset=self.contact_offset
        for shape in self.pot.find_component_by_type(sapien.physx.PhysxRigidStaticComponent).get_collision_shapes():
            shape.contact_offset=self.contact_offset

    def _update_render(self):
        super()._update_render()
        if self.wrist_image_fit is not None:
            optical_to_sapien=np.eye(4)
            optical_to_sapien[:3,:3]=[[0,-1,0],[0,0,-1],[1,0,0]]
            for side in ("left","right"):
                flange=getattr(self.robot,f"{side}_entity").find_link_by_name("link6").pose.to_transformation_matrix()
                mount=np.asarray(self.wrist_image_fit[side]["flange_from_camera_optical"])
                getattr(self.cameras,f"{side}_camera").entity.set_pose(sapien.Pose(flange@mount@optical_to_sapien))
            self.scene.update_render()

    def _observe_physics(self):
        if hasattr(self,"collision_quality"):self.collision_quality.observe(self)
        super()._observe_physics()
        if self.stir_task and self.stir_monitor.active:
            matrix=self.spatula.pose.to_transformation_matrix()
            blade=matrix[:3,:3]@np.asarray(self.tool_metadata['blade_center'])+matrix[:3,3]
            local=self.tool_vertices[self.tool_vertices[:,0]>.12]
            points=local@matrix[:3,:3].T+matrix[:3,3]
            inside=(np.linalg.norm(blade[:2]-self.pot_center[:2])<self.pot_radius-.003 and
                    points[:,2].min()<self.pot_rim+.015 and points[:,2].max()>self.pot_floor-.004 and
                    self.fsm_state in {'STIR_PUSH','STIR_REVERSE','STIR_POLICY'})
            contacted=set()
            for contact in self.scene.get_contacts():
                names={body.entity.name for body in contact.bodies}
                if 'spatula' in names and any(np.linalg.norm(p.impulse)>1e-7 for p in contact.points):contacted.update(names&set(self.foods))
            self.stir_monitor.observe(self.scene.get_timestep(),blade[:2],inside,contacted,{k:v.pose.p for k,v in self.foods.items()})
        if self.fsm_state=="FINAL_SETTLE":
            stable=all(np.linalg.norm(b.linear_velocity)<.04 and np.linalg.norm(b.angular_velocity)<.4 for b in self.food_bodies.values())
            self.final_stability_steps=self.final_stability_steps+1 if stable else 0
        if self.policy_observer is not None:self.policy_observer.observe(self)
        if self.record_sink is not None:self.record_sink.observe(self)

    def _tcp_reachable(self,arm,position,rotation):
        import torch
        target=np.r_[np.asarray(position)-rotation@[.12,0,0],t3d.quaternions.mat2quat(rotation)]
        ee=self.robot._trans_from_gripper_to_endlink(target,arm_tag=arm)
        planner=getattr(self.robot,f"{arm}_planner");base=planner.robot_origion_pose
        p,q=planner._trans_from_world_to_base(np.r_[base.p,base.q],np.r_[ee.p,ee.q]);p=np.asarray(p)+planner.frame_bias
        entity=getattr(self.robot,f"{arm}_entity");names=[j.name for j in entity.get_active_joints()]
        current=entity.get_qpos()[[names.index(n) for n in planner.active_joints_name]]
        def accept(values):
            query=torch.as_tensor(values,dtype=torch.float32,device="cuda").reshape(1,1,-1)
            valid=bool(planner.motion_gen.rollout_fn.rollout_constraint(query,use_batch_env=False).feasible.all().item())
            return valid and (planner.trajectory_accept is None or planner.trajectory_accept(np.asarray(values)[None]))
        return planner.pose_ik.solve(p,q,current,accept=accept) is not None

    def _pick(self,name,arm):
        self.grasp_intent=name;self.held_entity[str(arm)]=None
        config=self.tool_metadata if name=="spatula" else self.assets[name]
        if name=="spatula":
            # The thin handle otherwise receives much less normal force than
            # the wider foods under the same position controller. Keep the
            # same 50 N actuator cap while increasing the handle grip gain.
            for joint in getattr(self.robot,f"{arm}_entity").get_active_joints():
                if joint.name in {"g2_left_joint","g2_right_joint"}:
                    joint.set_drive_property(stiffness=6000,damping=200,force_limit=50)
        entity=self.spatula if name=="spatula" else self.foods[name]
        sign=1 if arm=="left" else -1
        vector=np.asarray(config.get("lift_vector",[sign*.07,-.07,.13]))
        if name!="spatula":
            config["lift_vector"]=vector.tolist();config["approach_offset"]=vector.tolist();config["lift_segments"]=1
        approach=np.asarray(config.get("approach_offset",vector));offset=np.asarray(config.get("grasp_offset",[0,0,0]))
        if name=="spatula" and "grasp_point_tool_local" in config:
            offset=entity.pose.to_transformation_matrix()[:3,:3]@np.asarray(config["grasp_point_tool_local"])
        vertical=np.array([[0,-1,0],[0,0,1],[-1,0,0]],dtype=float)
        candidates=[]
        if name=="spatula":
            axis=entity.pose.to_transformation_matrix()[:3,0]
            yaw=math.atan2(axis[1],axis[0])
            candidates.append(t3d.euler.euler2mat(0,0,yaw)@vertical)
        for field in ("preferred_grasp_rotation_rpy","real_grasp_rotation_rpy","grasp_rotation_rpy"):
            if field in config:candidates.append(t3d.euler.euler2mat(*config[field]))
        if name!="spatula" and config.get("image_fitted_dimensions_not_measured"):
            points=self.food_vertices[name]@entity.pose.to_transformation_matrix()[:3,:3].T
            _,axes=np.linalg.eigh(np.cov(points[:,:2].T));axis=axes[:,-1];yaw=math.atan2(axis[1],axis[0])
            reference=t3d.euler.euler2mat(*config["real_grasp_rotation_rpy"])
            aligned=[t3d.euler.euler2mat(0,0,yaw+k*np.pi)@vertical for k in (0,1)]
            aligned.sort(key=lambda r:-np.trace(reference.T@r))
            candidates=[reference,*aligned,*candidates]
        candidates.extend(t3d.euler.euler2mat(tilt,0,0)@vertical for tilt in (.35,.6,.8,0.0))
        if name=="spatula":
            candidates.insert(0,t3d.euler.euler2mat(*config["grasp_rotation_rpy"]))
            candidates=candidates+[r@t3d.euler.euler2mat(np.pi,0,0) for r in candidates]
            # G2 closes along virtual TCP z. Clamp across the handle, not
            # along its long axis, when the real tool rests on the table.
            candidates=[r for r in candidates if abs(r[:,2]@axis)<.65]
        start=entity.pose.p.copy()
        for extra_height in (-.006,-.004,-.002,0.0,.002,.004,.006,.008):
            for rotation in candidates:
                adjusted=offset+np.array([0,0,extra_height])
                if name!="spatula":
                    from scipy.spatial import ConvexHull
                    from scipy.optimize import linprog
                    world=self.food_vertices[name]@entity.pose.to_transformation_matrix()[:3,:3].T+start
                    local=(world-start-adjusted)@rotation
                    equations=ConvexHull(local).equations
                    bounds=[(-.065,.006),(-.032,.032),(None,None)]
                    low=linprog([0,0,1],A_ub=equations[:,:3],b_ub=-equations[:,3],bounds=bounds,method="highs")
                    high=linprog([0,0,-1],A_ub=equations[:,:3],b_ub=-equations[:,3],bounds=bounds,method="highs")
                    if not low.success or not high.success or high.x[2]-low.x[2]>.0705:continue
                    adjusted=adjusted+.5*(high.x[2]+low.x[2])*rotation[:,2]
                targets=(start+approach,start+adjusted) if name=="spatula" else (start+approach,start+adjusted,start+adjusted+vector)
                if all(self._tcp_reachable(arm,p,rotation) for p in targets):
                    config["grasp_rotation_rpy"]=t3d.euler.mat2euler(rotation)
                    config["grasp_offset"]=adjusted.tolist()
                    self.selected_grasps[name]={"rotation_rpy":list(config["grasp_rotation_rpy"]),"lift_vector":vector.tolist(),"grasp_offset":adjusted.tolist()}
                    print(f"[W1 real grasp] {name}: {self.selected_grasps[name]}",flush=True)
                    result=super()._pick(name,arm)
                    self.held_entity[str(arm)]=entity
                    return result
        raise RuntimeError(f"No collision-free approach/grasp/lift for {name}: {start}")

    def _place_food(self,kind,arm,xy):
        arm=ArmTag(arm);self.active_arm=arm;food=self.foods[kind]
        rotation=t3d.euler.euler2mat(.6,0,0)@food.pose.to_transformation_matrix()[:3,:3]
        if kind in {"green_pepper","eggplant","small_pepper"}:
            points=self.food_vertices[kind]@rotation.T
            _,axes=np.linalg.eigh(np.cov(points[:,:2].T));axis=axes[:,-1]
            yaw=np.pi/2-math.atan2(axis[1],axis[0])
            for angle in (yaw,yaw+np.pi):
                candidate=t3d.euler.euler2mat(0,0,angle)@rotation
                if self._object_goal_reachable(arm,[*xy,self.pot_rim+.065],candidate):rotation=candidate;break
        self._transition(f"{kind.upper()}_TRANSFER")
        self._move_object(arm,food,[*xy,self.pot_rim+.065],rotation)
        self._wait(.10)
        # Rounded props can rotate inside a real contact grasp during transfer.
        # Re-observe that grip before computing a low placement; retaining the
        # initial rigid transform can command the food below the pan surface.
        self._remember_grasp(arm,food)
        rotation=food.pose.to_transformation_matrix()[:3,:3].copy()
        profile=self.assets["pan"]["collision_approximation"]
        points=self.food_vertices[kind]@rotation.T
        radius=np.linalg.norm(points[:,:2]+np.asarray(xy)-self.pot_center[:2],axis=1)
        height=np.interp(radius,np.r_[0,profile["radii_m"]],np.r_[self.assets["pan"]["floor_height"],profile["heights_m"]])
        minimum=self.table_top+float(np.max(height-points[:,2]))+.012
        # Clear the actual settled foods before opening the fingers. The
        # previous pan-only height could press a held potato into a pepper.
        from scipy.spatial import ConvexHull
        from scipy.optimize import linprog
        held_hull=ConvexHull(points+np.r_[xy,0]).equations
        for other,actor in self.foods.items():
            if other==kind:continue
            world=self.food_vertices[other]@actor.pose.to_transformation_matrix()[:3,:3].T+actor.pose.p
            fixed_hull=ConvexHull(world).equations
            # x/y are shared; z of the held shape and settled object vary
            # independently. Max(z_other-z_held) is the first nonoverlap height.
            a=np.r_[np.c_[held_hull[:,:3],np.zeros(len(held_hull))],
                    np.c_[fixed_hull[:,:2],np.zeros(len(fixed_hull)),fixed_hull[:,2]]]
            b=-np.r_[held_hull[:,3],fixed_hull[:,3]]
            result=linprog([0,0,1,-1],A_ub=a,b_ub=b,bounds=[(None,None)]*4,method="highs")
            if result.success:minimum=max(minimum,float(-result.fun)+.008)
        release=None
        for z in np.arange(minimum,max(minimum+.025,self.pot_rim+.15),.005):
            if self._release_goal_reachable(arm,[*xy,z],rotation):release=float(z);break
        if release is None:raise RuntimeError(f"No collision-free low release pose: {kind}")
        self.placement_release_heights[kind]=release
        self._transition(f"{kind.upper()}_RELEASE")
        self._move_object(arm,food,[*xy,release],rotation)
        self.move(self.open_gripper(arm));self._wait(.25)
        self.held_entity[str(arm)]=None
        self._transition(f"{kind.upper()}_RETREAT")
        pose=self._tcp_matrix(arm);sign=-1 if arm=="left" else 1
        retreat=None
        for offset in ([0,0,.04],[sign*.04,-.04,.04],[sign*.06,-.05,.02],[0,-.06,.04],[0,0,.065]):
            candidate=pose[:3,3]+offset
            if self._tcp_reachable(arm,candidate,pose[:3,:3]):retreat=candidate;break
        if retreat is None:raise RuntimeError(f"No reachable open-gripper retreat: {kind}")
        self._tcp_move(retreat,orientation=pose[:3,:3].copy())
        self._wait(.6)
        if not self._food_inside(kind):raise RuntimeError(f"{kind} did not settle inside pan")
        self.held_entity[str(arm)]=None;self.grasp_intent=None

    def _release_goal_reachable(self,arm,position,rotation,**kwargs):
        self.open_gripper_clearance=True
        try:return self._object_goal_reachable(arm,position,rotation,**kwargs)
        finally:self.open_gripper_clearance=False

    def _path_clear(self,side,path):
        """Check gripper and held mesh sweeps against the fitted solid pan."""
        from scipy.spatial.transform import Rotation
        import trimesh
        if not hasattr(self,"_gripper_hulls"):
            self._gripper_hulls={}
            for name in ("g2_base_link","g2_left_link","g2_right_link"):
                mesh=trimesh.load(ROOT/f"w1_urdf/meshes/{name}.STL",force="mesh").convex_hull
                vertices,_=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=.008,max_iter=6)
                self._gripper_hulls[name]=vertices
        pan=self.assets["pan"];profile=pan.get("collision_approximation")
        if profile is None:return True
        radii=np.r_[0,profile["radii_m"]];heights=np.r_[pan["floor_height"],profile["heights_m"]]
        def intersects(points,margin):
            local=points-self.pot_center
            radius=np.linalg.norm(local[...,:2],axis=-1)
            height=np.interp(radius,radii,heights)
            outer=profile.get("outer_radius_m",radii[-1]+.002)
            return bool(np.any((radius<outer+max(margin,0))&(local[...,2]>-.005)&(local[...,2]<height+margin)))
        planner=getattr(self.robot,f"{side}_planner");base=getattr(self.robot,f"{side}_entity_origion_pose").to_transformation_matrix()
        entity=getattr(self.robot,f"{side}_entity");joints=entity.get_active_joints();names=[j.name for j in joints]
        q=entity.get_qpos();widths=[q[names.index(n)] for n in ("g2_left_joint","g2_right_joint")]
        if getattr(self,"open_gripper_clearance",False):widths=[.036,.036]
        goal_width=sum(float(joints[names.index(n)].get_drive_target()[0]) for n in ("g2_left_joint","g2_right_joint"))
        held=self.held_entity.get(side) if goal_width<.065 else None
        hand=[]
        rleft=Rotation.from_euler("y",-np.pi/2).as_matrix();rright=Rotation.from_euler("xyz",[np.pi,-np.pi/2,0]).as_matrix()
        hand.extend((self._gripper_hulls["g2_base_link"]+[0,0,.0955]).tolist())
        for name,r,opening in (("g2_left_link",rleft,widths[0]),("g2_right_link",rright,widths[1])):
            hand.extend(((self._gripper_hulls[name]+[0,0,opening])@r.T+[0,0,.166]).tolist())
        hand=np.asarray(hand)
        for values in path:
            flange=base@planner.pose_ik.forward(values)
            if intersects(hand@flange[:3,:3].T+flange[:3,3],.006):return False
            if held is not None and side in self.held_frames:
                tcp=np.eye(4);tcp[:3,:3]=flange[:3,:3]@Rotation.from_euler("y",-np.pi/2).as_matrix()
                tcp[:3,3]=flange[:3,3]+flange[:3,:3]@[0,0,.2466]
                pose=tcp@self.held_frames[side]
                if held is self.spatula:
                    points=self.tool_vertices@pose[:3,:3].T+pose[:3,3]
                    if ((self.fsm_state.startswith("TURN_") and any(x in self.fsm_state for x in ("INSERT","SCOOP","WITHDRAW"))) or
                        self.fsm_state in {"STIR_LOWER","STIR_PUSH","STIR_REVERSE"}):
                        handle=self.tool_vertices[:,0]<.10
                        if intersects(points[handle],.001) or intersects(points[~handle],-.002):return False
                    elif intersects(points,.002):return False
                else:
                    kind=next(k for k,v in self.foods.items() if v is held)
                    if intersects(self.food_vertices[kind]@pose[:3,:3].T+pose[:3,3],0):return False
        return True

    def _return_tool(self):
        arm=ArmTag(self.tool_arm);self.active_arm=arm
        rest=self.tool_initial_rest_pose[:3,3];rest_rotation=self.tool_initial_rest_pose[:3,:3]
        self._transition("SPATULA_RETURN_CLEAR")
        clear=None
        for lift in (.13,.11,.09,.16):
            candidate=rest+[0,0,lift]
            if self._object_goal_reachable(arm,candidate,rest_rotation):clear=candidate;break
        if clear is None:raise RuntimeError('No reachable tool return transfer pose')
        self._move_tool(arm,clear,rest_rotation)
        rotations=[rest_rotation]
        rotations.extend(rest_rotation@t3d.euler.euler2mat(0,pitch,0) for pitch in (.1,-.1,.2,-.2,.3,-.3))
        rotations.extend(rest_rotation@t3d.euler.euler2mat(roll,0,0) for roll in (.1,-.1,.2,-.2,.3,-.3))
        selected=None
        for extra_height in (0,.005,.015,.03,.05):
            for rotation in rotations:
                points=self.tool_vertices@rotation.T
                goal=np.r_[rest[:2],self.table_top+.002-points[:,2].min()+extra_height]
                if self._release_goal_reachable(arm,goal,rotation,limit_margin=.015):
                    for lift in (.10,.08,.06):
                        approach=goal+[0,0,lift]
                        if self._object_goal_reachable(arm,approach,rotation,limit_margin=.015):
                            selected=(goal,approach,rotation);break
                if selected is not None:break
            if selected is not None:break
        if selected is None:raise RuntimeError("No reachable table-return pose for the observed tool grip")
        goal,approach,rotation=selected
        self.return_release_pose={"position":goal.tolist(),"rotation_rpy":list(t3d.euler.mat2euler(rotation))}
        self._transition("SPATULA_RETURN_APPROACH");self._move_tool(arm,approach,rotation)
        self._transition("SPATULA_RETURN_LOWER");self._move_tool(arm,goal,rotation)
        self._transition("SPATULA_RETURN_RELEASE");self.move(self.open_gripper(arm));self._wait(.4)
        self._transition("SPATULA_RETURN_RETREAT")
        tcp=self._tcp_matrix(arm);orientation=tcp[:3,:3].copy();retreat=None
        for offset in ([0,0,.07],[.04,-.04,.05],[0,0,.05]):
            candidate=tcp[:3,3]+offset
            if self._tcp_reachable(arm,candidate,orientation):retreat=candidate;break
        if retreat is None:raise RuntimeError('No reachable released-tool retreat')
        self._tcp_move(retreat,orientation=orientation);self._wait(.8)
        self.tool_returned=self._tool_in_region()
        self.held_entity[str(arm)]=None;self.grasp_intent=None
        if not self.tool_returned:raise RuntimeError(f"Tool did not settle in table return region: {self.spatula.pose.p}")

    def _park(self, side, label):
        arm=ArmTag(side)
        self._transition(label)
        self.move(self.move_to_pose(arm,target_pose=self.home_targets[side]))
        if not self.plan_success:raise RuntimeError(f"Park failed: {side}")
        self.move(self.close_gripper(arm))

    def _clear_tool_workspace(self):
        # Rotate the idle left arm away from the tool's approach corridor;
        # keep its original recorded home target for final return.
        self._transition("TOOL_CLEARANCE_PARK")
        q=np.asarray(self.robot.left_homestate).copy();q[0]=0
        flange=self.robot.left_entity_origion_pose.to_transformation_matrix()@self.robot.left_planner.pose_ik.forward(q)
        rotation=flange[:3,:3]@t3d.euler.euler2mat(0,-np.pi/2,0)
        target=np.r_[flange[:3,3]+rotation@[.1266,0,0],t3d.quaternions.mat2quat(rotation)]
        self.move(self.move_to_pose(ArmTag("left"),target_pose=target))
        if not self.plan_success:raise RuntimeError("Left arm tool-clearance park failed")

    def _stir(self):
        """Small reciprocal pushes; judge contacts and visible motion for 2 s."""
        arm=ArmTag(self.tool_arm);self.active_arm=arm
        tip=np.asarray(self.tool_metadata['blade_tip']);selected=None
        self._transition('STIR_SEARCH')
        for kind in ('small_pepper','green_pepper','potato','eggplant'):
            food=self.foods[kind]
            for heading in (.8,.4,0.,-.4,-.8,-1.2,-1.6):
                flat=t3d.euler.euler2mat(0,0,heading);direction=flat[:,0]
                for pitch in (.55,.7,.85):
                    rotation=flat@t3d.euler.euler2mat(0,pitch,0)@t3d.euler.euler2mat(np.pi,0,0)
                    for extra in (0.,.01,.02,.03):
                        lip=np.r_[food.pose.p[:2]-.030*direction[:2],food.pose.p[2]+extra]
                        root=lip-rotation@tip;end=root+.035*direction
                        if self._object_goal_reachable(arm,root,rotation) and self._object_goal_reachable(arm,end,rotation):
                            selected=(kind,heading,pitch,extra,root,end,rotation,direction);break
                    if selected is not None:break
                if selected is not None:break
            if selected is not None:break
        if selected is None:raise RuntimeError('No reachable collision-free reciprocal stirring corridor')
        kind,heading,pitch,extra,root,end,rotation,direction=selected
        self.stir_plan={'target_food':kind,'heading_rad':heading,'pitch_rad':pitch,'height_offset_m':extra,
                        'start_root':root.tolist(),'end_root':end.tolist(),'stroke_m':.035}
        print(f'[W1 stir plan] {self.stir_plan}',flush=True)
        self._transition('STIR_APPROACH')
        hover,hover_rotation=self._tool_hover_target(arm,(root+rotation@tip)[:2],[rotation])
        self._move_tool(arm,hover,hover_rotation)
        self.stir_monitor.begin(direction,{k:v.pose.p for k,v in self.foods.items()})
        planner=getattr(self.robot,f'{arm}_planner')
        velocity=planner.trajectory_velocity_limits;acceleration=planner.trajectory_acceleration_limits
        planner.trajectory_velocity_limits=(np.asarray(velocity)*.25).tolist()
        planner.trajectory_acceleration_limits=(np.asarray(acceleration)*.25).tolist()
        try:
            self._transition('STIR_LOWER');self._move_tool(arm,root,rotation)
            for cycle in range(6):
                self._transition('STIR_PUSH');self._move_tool(arm,end,rotation)
                self._transition('STIR_REVERSE');self._move_tool(arm,root,rotation)
                print(f'[W1 stir] cycle={cycle+1} {self.stir_monitor.report()}',flush=True)
                if self.stir_monitor.success:break
        finally:
            planner.trajectory_velocity_limits=velocity;planner.trajectory_acceleration_limits=acceleration
        self._transition('STIR_WITHDRAW')
        hover,hover_rotation=self._tool_hover_target(arm,(root+rotation@tip)[:2],[rotation])
        self._move_tool(arm,hover,hover_rotation)
        if not self.stir_monitor.success:raise RuntimeError(f'Insufficient reciprocal stirring: {self.stir_monitor.report()}')
        self._transition('STIR_VERIFIED');self.stir_monitor.active=False

    def play_once(self):
        self._wait(.8)
        for side in ("left","right"):self.move(self.close_gripper(ArmTag(side)))
        self._recording=True
        try:
            for kind,side,offset in (("small_pepper","right",[.060,-.025]),("green_pepper","left",[.005,-.035]),
                                      ("potato","right",[.030,.065]),("eggplant","left",[-.055,-.015])):
                self._pick(kind,side)
                self._place_food(kind,side,self.pot_center[:2]+offset)
                self._park(side,f"{kind.upper()}_PARK")
            self._clear_tool_workspace()
            self._pick("spatula",self.tool_arm)
            self._wait(.4);self._remember_grasp(self.tool_arm,self.spatula)
            if self.stir_task:self._stir()
            else:self._turn(1);self._turn(2)
            self._return_tool()
            self._park(self.tool_arm,"FINAL_PARK")
            self._park("left","FINAL_LEFT_PARK")
            self._transition("FINAL_SETTLE");self._wait(2.5)
            for _ in range(math.ceil(self.final_settle_timeout_s/.1)):
                if self.final_stability_steps*self.scene.get_timestep()>=self.final_stability_min_s:break
                self._wait(.1)
            if self.final_stability_steps*self.scene.get_timestep()<self.final_stability_min_s:
                raise RuntimeError("Foods did not reach the required continuous stationary dwell")
            self.full_completed=True;self._transition("DONE");self._wait(.1)
        finally:self._recording=False
        self.info.update(task_scope="real_reference_four_props_v1",active_arm="both",asset_manifest=self.asset_manifest,
            reference_episode=self.reference_episode,proxy_assets=False,tool_arm=self.tool_arm,
            fsm_transitions=self.transitions,pick_checks=self.pick_checks,lift_verified=all(self.pick_checks.values()),
            flip_events=self.flip_events,flip_min_angle_deg=math.degrees(self.flip_min_angle),tool_tracking=self.tool_tracking,
            selected_grasps=self.selected_grasps,
            final_stable_duration_s=self.final_stability_steps*self.scene.get_timestep(),
            collision_quality=self.collision_quality.report(),
            task_judgement_version='stir_contact_motion_v1' if self.stir_task else 'two_turns_45deg_v1',
            stir_metrics=self.stir_monitor.report(),stir_plan=self.stir_plan,
            pan_geometry={"center":self.pot_center.tolist(),"inner_radius":self.pot_radius,"floor":self.pot_floor,"rim":self.pot_rim},
            tool_return_region={"tool_rest":self.tool_rest.tolist(),"center_xy":self.tool_return_center.tolist(),"half_size_xy":self.tool_return_half},
            limitations=["Pan fixed in this initial profile; real pan slides/rotates.","Wrist camera pose is an approximate RGB-D fit; front intrinsics borrowed.","Idle left arm moves outward for tool clearance; not present in the real recording."])
        self.check_success();return self.info

    def check_success(self):
        supported={"skillet"}
        for _ in range(len(self.foods)):
            for names,positions in self.support_edges.items():
                if any(n in supported for n in names) and all(np.linalg.norm(self.foods[n].pose.p-p)<.008 for n,p in positions.items()):supported.update(names)
        checks={"plan_success":bool(self.plan_success),"completed":self.full_completed,
            "all_picks_verified":len(self.pick_checks)==5 and all(self.pick_checks.values()),
            "two_effective_turns":len(self.flip_events)==2 and all(e["valid"] for e in self.flip_events),
            "tool_returned":bool(self.tool_returned and self._tool_in_region()),
            **{f"{kind}_inside_pan":self._food_inside(kind) for kind in self.foods},
            "foods_supported":set(self.foods)<=supported,
            "foods_stationary":all(np.linalg.norm(b.linear_velocity)<.04 and np.linalg.norm(b.angular_velocity)<.4 for b in self.food_bodies.values()),
            "foods_stable_dwell":self.final_stability_steps*self.scene.get_timestep()>=self.final_stability_min_s,
            "no_severe_collisions":not self.collision_quality.severe,
            "both_arms_parked":all(np.linalg.norm(np.asarray(getattr(self.robot,f"get_{s}_ee_pose")()[:3])-self.home_targets[s][:3])<.015 for s in ("left","right"))}
        if self.stir_task:
            checks.pop('two_effective_turns')
            checks['effective_reciprocal_stirring']=self.stir_monitor.success
        self.info["success_checks"]=checks
        print(f"[W1 real-reference check] {checks}",flush=True)
        return bool(all(checks.values()))
