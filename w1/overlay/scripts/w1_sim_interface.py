"""Programmatic dual-W1 interface and physical 250 Hz / RGB-D 30 Hz recorder.

Action/state: [left q1..q6 (rad), left total opening (m),
               right q1..q6 (rad), right total opening (m)].
Native JSONL widths are metres; quaternion order is xyzw in each arm base.
"""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.w1_recording import IndexedJSONL,write_tsf,MASTER_GRIPPER_STROKE_M,SLAVE_GRIPPER_STROKE_M

MAX_JOINT_STEP_RAD = 0.05


def task_configuration(name, need_plan=True):
    cfg=yaml.safe_load((ROOT/f"env_cfg/task_config/{name}.yml").read_text())
    registry=yaml.safe_load((ROOT/"env_cfg/task_config/_embodiment_config.yml").read_text())
    for side,robot in zip(("left","right"),cfg["embodiment"][:2]):
        folder=(ROOT/registry[robot]["file_path"]).resolve()
        cfg[f"{side}_robot_file"]=str(folder)
        cfg[f"{side}_embodiment_config"]=yaml.safe_load((folder/"config.yml").read_text())
    cfg.update(task_name="w1_finals_real",task_config=name,dual_arm_embodied=False,embodiment_dis=cfg["embodiment"][2],
        embodiment_name=cfg["embodiment"][0],need_plan=need_plan,save_data=False,collect_data=False,render_freq=0)
    return cfg


def joint_packet(task,side):
    entity=getattr(task.robot,f"{side}_entity")
    joints=entity.get_active_joints();names=[j.name for j in joints];values=entity.get_qpos()
    q=np.asarray([values[names.index(f"joint{i}")] for i in range(1,7)])
    width=float(sum(values[names.index(n)] for n in ("g2_left_joint","g2_right_joint")))
    targets=np.asarray([joints[names.index(f"joint{i}")].get_drive_target()[0] for i in range(1,7)])
    grip_target=sum(float(joints[names.index(n)].get_drive_target()[0]) for n in ("g2_left_joint","g2_right_joint"))
    base=getattr(task.robot,f"{side}_entity_origion_pose").to_transformation_matrix()
    flange=np.linalg.inv(base)@entity.find_link_by_name("link6").pose.to_transformation_matrix()
    fixed=np.array([[0,-1,0],[0,0,-1],[1,0,0]])
    def ee(matrix,offset):
        return np.r_[matrix[:3,3]+matrix[:3,:3]@[0,0,offset],Rotation.from_matrix(matrix[:3,:3]@fixed).as_quat()].tolist()
    planner=getattr(task.robot,f"{side}_planner")
    reference=planner.pose_ik.forward(targets)
    return {"joint_positions":q.tolist(),"ee_positions":ee(flange,.2466),"gripper":width,
            "effort":[0.0]*7,"master_qpos":np.r_[targets,grip_target/SLAVE_GRIPPER_STROKE_M*MASTER_GRIPPER_STROKE_M].tolist(),
            "master_ee_positions":ee(reference,.1466)}


class NativeRecorder:
    def __init__(self,directory,task,episode_id=0):
        self.directory=Path(directory)
        if self.directory.exists():raise FileExistsError(self.directory)
        self.directory.joinpath("sensors").mkdir(parents=True)
        self.actions={s:IndexedJSONL(self.directory/f"actions/{s}_arm.jsonl") for s in ("left","right")}
        self.car=IndexedJSONL(self.directory/"actions/car.jsonl")
        self.truth=IndexedJSONL(self.directory/"extra/ground_truth.jsonl")
        self.epoch=time.time();self.frames=0;self.physics_samples=0;self.times=[];self.encoders={};self.closed=False
        for camera in ("cam_front","cam_left","cam_right"):
            for kind,extension,pix,codec in (("rgb","mp4","rgb24","libx264"),("depth","mkv","gray16le","ffv1")):
                file=self.directory/f"sensors/{camera}_{kind}.{extension}"
                command=["ffmpeg","-y","-v","error","-f","rawvideo","-pix_fmt",pix,"-video_size","640x480",
                         "-framerate","30","-i","-","-an","-c:v",codec,"-threads","1"]
                command+=(["-preset","fast","-crf","18","-pix_fmt","yuv420p"] if kind=="rgb" else ["-level","3","-pix_fmt","gray16le"])
                self.encoders[(camera,kind)]=(file,subprocess.Popen(command+[str(file)],stdin=subprocess.PIPE))
        source=json.loads((ROOT/task.native_meta_template if task.native_meta_template else ROOT/task.reference_episode/"meta.json").read_text())
        self.meta=source
        self.meta.update(created_at=self.epoch,finished_at=None)
        self.meta["robot_meta"]={"robot_type":"dos_w1","serial":"simulation","description":"Black dual W1/G2, 0.60 m base separation"}
        self.meta["task_meta"]={"task_name":"w1_finals_real","episode_id":int(episode_id),"prompt":
            "Put both green peppers, the potato and the eggplant in the pan, turn the contents twice with the spatula, and return the spatula.","tags":["simulation","real_reference_v1"]}
        self.meta["extra"]={"source":"simulation","physics_hz":250,"camera_hz":30,"depth_unit":"mm","depth_invalid_value":0,
            "task_judgement_version":"stir_contact_motion_v1" if task.stir_task else "two_turns_45deg_v1",
            "gripper_unit":"total_width_m","maximum_width_m":.072,"arm_joint_unit":"radian","ee_quaternion_order":"xyzw",
            "master_gripper_unit":"master_width_m","master_maximum_width_m":.0479,
            "master_to_slave_gripper_scale":.072/.0479,
            "action_reference_source":"simulated_drive_target","effort_valid":False,
            "effort_note":"Zero placeholders; no measured motor torque interface is available. Do not use as torque supervision.",
            "master_tcp_offset_m":.1466,"slave_tcp_offset_m":.2466,"timestamp_domain":"simulated_epoch_seconds",
            "wrist_calibration":"approximate RGB/depth landmark fit" if task.wrist_image_fit else "provisional CAD mount; not measured real wrist calibration",
            "camera_renderer":task.camera_render_mode,
            "ground_truth_pose_frame":"simulation_world","ground_truth_quaternion_order":"wxyz",
            "small_object_contact_offset_m":task.contact_offset,
            "small_object_solver_iterations":task.object_solver_iterations,
            "required_final_stationary_dwell_s":task.final_stability_min_s,
            "minimal_rotation":task.minimal_rotation,"rotation_path_slack_deg":task.rotation_path_slack_deg,
            "paired_foods":task.paired_foods,
            "asset_manifest":task.assets,"spatula_metadata":task.tool_metadata,
            "pan_mouth_diameter_m":task.assets["pan"].get("mouth_diameter_m"),
            "pan_size_basis":task.assets["pan"].get("diameter_measurement_basis","previous nominal dimensions"),
            "pan_outer_diameter_m":task.assets["pan"].get("simulated_outer_diameter_m"),
            "gripper_drive_force_limit_N":50,
            "gripper_food_stiffness":3000,"gripper_tool_stiffness":6000,
            "static_layout":{"pan_xy":task.pot_center[:2].tolist(),"spatula_rest_xy":task.tool_rest[:2].tolist(),"spatula_yaw":task.tool_metadata["rest_yaw"]},
            "base_separation_m":.6,"success":False}
        for entry in self.meta["actions"].values():
            if "timestamp" in entry["schema"]:entry["schema"]["timestamp"]["dtype"]="float64"
        for name in self.meta["sensors"]:
            self.meta["sensors"][name]["camera"][0]["width"]=640
            self.meta["sensors"][name]["camera"][0]["height"]=480
        if task.stir_task:
            self.meta['task_meta']['prompt']='Put both green peppers, the potato and the eggplant in the pan, stir with small back-and-forth pushes for at least two seconds, and return the spatula.'
        if task.paired_foods:
            self.meta['task_meta']['prompt']='Use both arms to pick and place the vegetables into the pan, then stir with small back-and-forth spatula pushes for at least two seconds and return the spatula.'
            self.meta['task_meta']['tags'].append('paired_dual_arm')

    def observe(self,task):
        relative=task._record_step*task.scene.get_timestep();stamp=self.epoch+relative
        self.physics_samples+=1
        for side,file in self.actions.items():file.write({**joint_packet(task,side),"timestamp":stamp})
        if self.physics_samples%5==0:self.car.write({"v":0.0,"w":0.0,"timestamp":stamp})
        if relative*30+1e-8 < self.frames+1:return
        task._update_render();task.cameras.update_picture()
        cameras={"cam_front":task.cameras.static_camera_list[task.cameras.head_camera_id],
                 "cam_left":task.cameras.left_camera,"cam_right":task.cameras.right_camera}
        for name,camera in cameras.items():
            if self.frames==0:
                extrinsic=np.eye(4)
                extrinsic[:3,:]=camera.get_extrinsic_matrix()[:3,:]
                world_optical=np.linalg.inv(extrinsic)
                calibration={"intrinsics":{"camera":camera.get_intrinsic_matrix().tolist(),"distortion":[[0.0]*5]},
                             "calibration_source":"simulation_geometry"}
                if name=="cam_front":
                    calibration["extrinsics"]={f"{s}_arm":(np.linalg.inv(getattr(task.robot,f"{s}_entity_origion_pose").to_transformation_matrix())@world_optical).tolist() for s in ("left","right")}
                else:
                    side="left" if name=="cam_left" else "right"
                    flange=getattr(task.robot,f"{side}_entity").find_link_by_name("link6").pose.to_transformation_matrix()
                    calibration["extrinsics"]={"flange_from_camera_optical":(np.linalg.inv(flange)@world_optical).tolist()}
                    calibration["real_mount_measured"]=False
                self.meta["sensors"][name]["calibration"]=calibration
            rgba=camera.get_picture("Color")
            rgb=np.uint8(np.clip(rgba[...,:3]*255,0,255))
            position=camera.get_picture("Position");depth=-position[...,2]
            valid=np.isfinite(depth)&(depth>0)&(depth<10)&(rgba[...,3]>.5)
            depth=np.where(valid,np.rint(depth*1000),0).clip(0,65535).astype("<u2")
            self.encoders[(name,"rgb")][1].stdin.write(rgb.tobytes())
            self.encoders[(name,"depth")][1].stdin.write(depth.tobytes())
        self.times.append(stamp);self.frames+=1
        poses={f"{name}_pose":np.r_[actor.pose.p,actor.pose.q].tolist() for name,actor in task.foods.items()}
        poses.update(spatula_pose=np.r_[task.spatula.pose.p,task.spatula.pose.q].tolist(),pan_pose=np.r_[task.pot.pose.p,task.pot.pose.q].tolist())
        contacts=[]
        for contact in task.scene.get_contacts():
            impulse=sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
            if impulse>1e-7:contacts.append({"bodies":sorted({body.entity.name for body in contact.bodies}),"impulse_Ns":impulse})
        self.truth.write({"timestamp":stamp,"fsm_state":task.fsm_state,"arm_fsm_state":dict(task.arm_fsm_state),"poses":poses,"contacts":contacts,
            "effective_flip_count":sum(e["valid"] for e in task.flip_events),"turn_angle_deg":task.current_turn_angle,
            "tool_pepper_contact":task.last_tool_contact,"tool_returned":task.tool_returned,
            "stir_success":task.stir_monitor.success,"stir_motion_duration_s":task.stir_monitor.motion_s,
            "stir_contact_duration_s":task.stir_monitor.contact_s,"stir_reversals":task.stir_monitor.reversals,
            "stir_max_food_displacement_m":task.stir_monitor.peak_food_motion})
        if self.frames%300==0:print(f"[W1 native] RGB-D={self.frames}, physics={self.physics_samples}, phase={task.fsm_state}",flush=True)

    def close(self,info=None,success=False):
        if self.closed:return
        self.closed=True
        for file in [*self.actions.values(),self.car,self.truth]:file.close()
        failures=[]
        for _,(path,process) in self.encoders.items():
            process.stdin.close()
            if process.wait():failures.append(str(path))
            if path.exists():write_tsf(path,self.times)
        self.meta["finished_at"]=self.epoch+(self.physics_samples/250)
        self.meta["extra"].update(success=bool(success),physics_samples=self.physics_samples,rgbd_frames=self.frames)
        (self.directory/"meta.json").write_text(json.dumps(self.meta,indent=2)+"\n")
        if info is not None:(self.directory/"scene_info.json").write_text(json.dumps(info,indent=2)+"\n")
        if failures:raise RuntimeError(f"Video encoder failures: {failures}")


class W1Simulation:
    """Minimal reset/step API; no network or physical-robot side effects."""
    def __init__(self,config="w1_finals_real_v7",control_hz=30):
        self.config_name=config;self.control_hz=control_hz;self.task=None;self.recorder=None;self.control_steps=0

    def reset(self,seed=0,record_dir=None,need_plan=True,episode_id=0):
        if self.task is not None:self.close()
        cls=getattr(importlib.import_module("envs.w1_finals_real"),"w1_finals_real")
        self.task=cls();self.configuration=task_configuration(self.config_name,need_plan)
        self.task.setup_demo(now_ep_num=0,seed=seed,**self.configuration)
        for side in ("left","right"):self.task.robot.set_gripper(0.0,side,gripper_eps=0)
        self.task._wait(.4)
        self.control_steps=0
        from scripts.w1_task_monitor import W1TaskMonitor
        self.policy_monitor=W1TaskMonitor(self.task);self.task.policy_observer=self.policy_monitor
        if record_dir is not None:self.recorder=NativeRecorder(record_dir,self.task,episode_id);self.task.record_sink=self.recorder
        return self.observation()

    def observation(self):
        task=self.task;task._update_render();task.cameras.update_picture()
        packets={s:joint_packet(task,s) for s in ("left","right")}
        state = np.concatenate([
            np.r_[packet["joint_positions"], np.clip(packet["gripper"], 0, SLAVE_GRIPPER_STROKE_M)]
            for packet in packets.values()
        ])
        images={"cam_head":task.cameras.static_camera_list[task.cameras.head_camera_id],
                "cam_left_wrist":task.cameras.left_camera,"cam_right_wrist":task.cameras.right_camera}
        depths={}
        for name,camera in images.items():
            depth=-camera.get_picture("Position")[...,2]
            valid=np.isfinite(depth)&(depth>0)&(depth<10)&(camera.get_picture("Color")[...,3]>.5)
            depths[name]=np.where(valid,np.rint(depth*1000),0).clip(0,65535).astype(np.uint16)
        return {"state":state.astype(np.float32),"images":{name:np.uint8(np.clip(c.get_picture("Color")[...,:3]*255,0,255)) for name,c in images.items()},
                "depths":depths,"depth_unit":"mm",
                "timestamp":task._record_step/250,"phase":task.fsm_state}

    def step(self,action):
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected finite 14-D action")

        commands = []
        for side, offset in (("left", 0), ("right", 7)):
            target = action[offset:offset + 6]
            pose_ik = getattr(self.task.robot, f"{side}_planner").pose_ik
            if np.any(target < pose_ik.lower) or np.any(target > pose_ik.upper):
                raise ValueError(f"{side} joint target exceeds source URDF limits")
            current = np.asarray(joint_packet(self.task, side)["joint_positions"])
            joint_target = current + np.clip(
                target - current, -MAX_JOINT_STEP_RAD, MAX_JOINT_STEP_RAD
            )
            opening = np.clip(action[offset + 6], 0, SLAVE_GRIPPER_STROKE_M)
            commands.append((side, joint_target, float(opening / SLAVE_GRIPPER_STROKE_M)))

        for side, joint_target, opening_fraction in commands:
            self.task.robot.set_arm_joints(joint_target, np.zeros(6), side)
            self.task.robot.set_gripper(opening_fraction, side, gripper_eps=0)
        self.task._recording=True;self.control_steps+=1
        desired=round(self.control_steps*250/self.control_hz)
        steps=desired-round((self.control_steps-1)*250/self.control_hz)
        self.task._wait(steps/250)
        success=self.policy_monitor.success
        return self.observation(),float(success),success,self.control_steps>=9000,{"control_hz":self.control_hz,"physics_steps":steps,**self.policy_monitor.info()}

    def expert(self):
        self.task.policy_observer=None
        info=self.task.play_once();success=self.task.check_success()
        if self.recorder:self.recorder.close(info,success);self.task.record_sink=None
        return info,success

    def close(self):
        if self.recorder:
            info=self.policy_monitor.info() if getattr(self,"policy_monitor",None) else None
            if info is not None and self.task is not None:
                task=self.task
                info.update(task_scope="real_reference_four_props_v1",tool_arm=task.tool_arm,
                    pan_geometry={"center":task.pot_center.tolist(),"inner_radius":task.pot_radius,"floor":task.pot_floor,"rim":task.pot_rim},
                    tool_return_region={"tool_rest":task.tool_rest.tolist(),"center_xy":task.tool_return_center.tolist(),"half_size_xy":task.tool_return_half},
                    limitations=["Pan fixed in this initial profile; real pan slides/rotates.","Wrist camera pose is an approximate RGB-D fit; front intrinsics borrowed.","Idle left arm moves outward for tool clearance; not present in the real recording."])
            self.recorder.close(info,bool(info and info["success"]));self.recorder=None
        if self.task:self.task.close_env();self.task=None
