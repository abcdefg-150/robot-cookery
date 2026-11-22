"""First finals subtask: physical pick-and-place with a deterministic expert FSM.

Pepper and pot use procedural proxy geometry until the competition assets and
measured dimensions are available. No attachment, teleport, or pose reset is
used during the demonstration.
"""

import math
import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import ArmTag


class w1_chili_pot(Base_Task):
    def setup_demo(self, **kwargs):
        self.fsm_state = "SETUP"
        self.transitions = []
        self.lift_verified = False
        self.completed = False
        self._recording = False
        self._from_physics_step = False
        super()._init_task_env_(**kwargs)
        self._record_step = 0
        self._physics_step = self.scene.step

        def step_and_record():
            self._physics_step()
            if self._recording:
                self._record_step += 1
                self._observe_physics()
                if self._record_step % self.save_freq == 0:
                    self._from_physics_step = True
                    try:
                        self._take_picture()
                    finally:
                        self._from_physics_step = False
        self.scene.step = step_and_record

    def _observe_physics(self):
        pass

    def load_actors(self):
        self.active_arm = ArmTag("left" if np.random.randint(2) == 0 else "right")
        sign = -1 if self.active_arm == "left" else 1
        x = sign * np.random.uniform(0.17, 0.22)
        y = np.random.uniform(-0.16, -0.12)
        self.pepper_radius = 0.012
        self.pepper_half_length = 0.035
        self.table_top = 0.74 + self.table_z_bias
        self.pot_center = np.array([0.0, 0.015, self.table_top])
        self.pot_radius = 0.11
        self.pot_floor = self.table_top + 0.012
        self.pot_rim = self.table_top + 0.082
        builder = self.scene.create_actor_builder()
        material = self.scene.create_physical_material(1.0, 1.0, 0.0)
        builder.add_capsule_collision(radius=self.pepper_radius, half_length=self.pepper_half_length, material=material)
        builder.add_capsule_visual(radius=self.pepper_radius, half_length=self.pepper_half_length,
                                   material=sapien.render.RenderMaterial(base_color=[0.8, 0.025, 0.015, 1]))
        builder.add_capsule_visual(pose=sapien.Pose([0.049, 0, 0]), radius=0.004, half_length=0.009,
                                   material=sapien.render.RenderMaterial(base_color=[0.05, 0.4, 0.05, 1]))
        builder.set_initial_pose(sapien.Pose([x, y, self.table_top + self.pepper_radius + 0.001]))
        self.pepper = builder.build(name="chili_proxy")
        self.pepper_body = self.pepper.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        self.pepper_body.mass = 0.025

        # A shallow round basin, built from a floor and 32 wall segments.
        builder = self.scene.create_actor_builder()
        dark = sapien.render.RenderMaterial(base_color=[0.08, 0.09, 0.1, 1], metallic=0.3, roughness=0.5)
        floor_pose = sapien.Pose([0, 0, 0.006], [math.sqrt(0.5), 0, math.sqrt(0.5), 0])
        builder.add_cylinder_collision(pose=floor_pose, radius=self.pot_radius, half_length=0.006)
        builder.add_cylinder_visual(pose=floor_pose, radius=self.pot_radius, half_length=0.006, material=dark)
        for index in range(32):
            angle = index * 2 * math.pi / 32
            radius = self.pot_radius + 0.004
            pose = sapien.Pose([radius * math.cos(angle), radius * math.sin(angle), 0.047],
                               t3d.euler.euler2quat(0, 0, angle))
            half_size = [0.004, radius * math.tan(math.pi / 32) + 0.001, 0.035]
            builder.add_box_collision(pose=pose, half_size=half_size)
            builder.add_box_visual(pose=pose, half_size=half_size, material=dark)
        builder.add_box_visual(pose=sapien.Pose([0.19, 0, 0.047]), half_size=[0.08, 0.018, 0.012], material=dark)
        builder.set_initial_pose(sapien.Pose(self.pot_center))
        self.pot = builder.build_static(name="pot_proxy")

    def _take_picture(self):
        # Base actions capture extra frames at motion boundaries. This task
        # samples once per five physics steps instead, including the waits.
        if self._from_physics_step:
            super()._take_picture()

    def get_obs(self):
        result = super().get_obs()
        result["joint_targets"] = {key: np.asarray(value).copy() for key, value in result["joint_action"].items()}
        for side in ("left", "right"):
            entity = getattr(self.robot, f"{side}_entity")
            joints = entity.get_active_joints()
            values = entity.get_qpos()
            result["joint_action"][f"{side}_arm"] = np.asarray([
                values[joints.index(joint)] for joint in getattr(self.robot, f"{side}_arm_joints")])
            result["joint_action"][f"{side}_gripper"] = float(
                values[[joint.name for joint in joints].index("g2_left_joint")] / 0.036)
        result["timestamp"] = self._record_step * self.scene.get_timestep()
        result["fsm_state"] = self.fsm_state
        return result

    def _transition(self, state):
        self.fsm_state = state
        self.transitions.append({"state": state, "time": self._record_step * self.scene.get_timestep()})
        print(f"[W1 FSM] {state}", flush=True)

    def _wait(self, seconds):
        for _ in range(round(seconds / self.scene.get_timestep())):
            for entity in (self.robot.left_entity, self.robot.right_entity):
                entity.set_qf(entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True))
            self.scene.step()

    def _tcp_move(self, position, tilt=0.0, tilt_y=0.0, orientation=None):
        # RoboTwin's action pose is 12 cm behind the physical grasp center.
        rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
        rotation = t3d.euler.euler2mat(tilt, tilt_y, 0) @ rotation
        if orientation is not None:
            rotation = np.asarray(orientation)
        position = np.asarray(position) - rotation @ [0.12, 0, 0]
        pose = position.tolist() + t3d.quaternions.mat2quat(rotation).tolist()
        self.move(self.move_to_pose(self.active_arm, target_pose=pose))
        if not self.plan_success:
            raise RuntimeError(f"Planning failed in {self.fsm_state}")

    def play_once(self):
        self._recording = True
        try:
            position = self.pepper.pose.p.copy()
            initial_z = float(position[2])
            self._transition("APPROACH")
            self._tcp_move([position[0], position[1], initial_z + 0.15])
            self._transition("DESCEND")
            self._tcp_move(position)
            self._transition("GRASP")
            self.move(self.close_gripper(self.active_arm))
            self._wait(0.2)
            self._transition("LIFT")
            self._tcp_move([position[0], position[1], initial_z + 0.15])
            self._wait(0.3)
            self.lift_verified = bool(self.pepper.pose.p[2] > initial_z + 0.08)
            if not self.lift_verified:
                raise RuntimeError("Physical grasp failed: pepper did not lift")
            self._transition("TRANSFER")
            self._tcp_move([*self.pot_center[:2], initial_z + 0.15], tilt=0.6)
            self._transition("LOWER")
            self._tcp_move([*self.pot_center[:2], self.pot_rim + 0.055], tilt=0.6)
            self._transition("RELEASE")
            self.move(self.open_gripper(self.active_arm))
            self._transition("RETREAT")
            self._tcp_move([self.pot_center[0], self.pot_center[1] - 0.10, self.pot_rim + 0.12], tilt=0.6)
            self._transition("SETTLE")
            self._wait(1.0)
            self.completed = True
            self._transition("DONE")
            self._wait(0.1)
        finally:
            self._recording = False
        self.info.update(active_arm=str(self.active_arm), fsm_transitions=self.transitions,
                         lift_verified=self.lift_verified, proxy_assets=True,
                         task_scope="finals_chili_into_pot_only")
        self.check_success()
        return self.info

    def check_success(self):
        pose = self.pepper.pose
        axis = t3d.quaternions.quat2mat(pose.q)[:, 0]
        endpoints = pose.p[None, :] + np.array([-1, 1])[:, None] * self.pepper_half_length * axis
        inside = np.all(np.linalg.norm(endpoints[:, :2] - self.pot_center[:2], axis=1) + self.pepper_radius < self.pot_radius)
        above_floor = np.min(endpoints[:, 2]) - self.pepper_radius > self.pot_floor - 0.004
        below_rim = np.max(endpoints[:, 2]) + self.pepper_radius < self.pot_rim
        stationary = np.linalg.norm(self.pepper_body.linear_velocity) < 0.03
        checks = dict(plan_success=bool(self.plan_success), completed=bool(self.completed),
                      lift_verified=bool(self.lift_verified), inside=bool(inside), above_floor=bool(above_floor),
                      below_rim=bool(below_rim), stationary=bool(stationary),
                      grippers_open=bool(self.is_left_gripper_open() and self.is_right_gripper_open()))
        if hasattr(self, "info"):
            self.info["success_checks"] = checks
            self.info["final_pepper_position"] = pose.p.tolist()
        success = all(checks.values())
        if self.completed and not success:
            print(f"[W1 check] arm={self.active_arm} position={pose.p} checks={checks}", flush=True)
        return success
