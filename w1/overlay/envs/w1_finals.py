"""Full W1 finals FSM: green pepper, two physical turns, tool return, potato, eggplant."""

import json
import math
from pathlib import Path
import numpy as np
import sapien
import transforms3d as t3d

from .w1_chili_pot import w1_chili_pot
from .utils import ArmTag
from .utils.finals_assets import load_asset_config, build_vegetable, build_pan, build_spatula


class w1_finals(w1_chili_pot):
    def setup_demo(self, **kwargs):
        self.asset_manifest = kwargs.get("asset_manifest", "env_cfg/w1_finals_assets.yml")
        self.tool_motion_profile = kwargs.get("tool_motion_profile")
        self.tool_arm = kwargs.get("tool_arm", "left")
        self.pan_position_xy = kwargs.get("pan_position_xy", [0.0, -0.04])
        self.tool_rest_xy = kwargs.get("tool_rest_xy", [-0.21, -0.255])
        self.tool_cartesian_steps = max(1, int(kwargs.get("tool_cartesian_steps", 1)))
        self.flip_min_angle = math.radians(float(kwargs.get("flip_min_angle_deg", 45)))
        self.flip_events = []
        self.flip_monitor = None
        self.pick_checks = {}
        self.held_frames = {}
        self.tool_returned = False
        self.full_completed = False
        self.debug_frames = kwargs.get("debug_frames")
        self.stage = 0
        self.last_tool_contact = False
        self.current_turn_angle = 0.0
        self.support_edges = {}
        self.tool_tracking = []
        self.placement_release_heights = {}
        self.debug_trace = [] if kwargs.get("debug_trace", False) else None
        super().setup_demo(**kwargs)
        # A finite fingertip contact patch models torsional friction for round
        # handles. Zero-radius point contacts otherwise let them hinge freely.
        self.gripper_contact_patch = float(kwargs.get("gripper_contact_patch_radius", 0))
        if self.gripper_contact_patch:
            for side in ("left", "right"):
                for link in getattr(self.robot, f"{side}_entity").get_links():
                    if link.name in ("g2_left_link", "g2_right_link"):
                        for shape in link.get_collision_shapes():
                            shape.patch_radius = self.gripper_contact_patch
                            shape.min_patch_radius = self.gripper_contact_patch/2
        self.home_targets = {side: getattr(self.robot, f"get_{side}_ee_pose")() for side in ("left", "right")}
        self.robot_link_names = {link.name for side in ("left", "right")
                                 for link in getattr(self.robot, f"{side}_entity").get_links()}
        if self.debug_frames:
            self.debug_overhead = self.scene.add_camera("finals_debug_overhead", 320, 240, math.radians(37), 0.01, 10)
            transform = np.eye(4)
            transform[:3, :3] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
            transform[:3, 3] = [0, -0.02, 1.78]
            self.debug_overhead.entity.set_pose(sapien.Pose(transform))

    def load_actors(self):
        self.assets = load_asset_config(self.asset_manifest)
        self.table_top = 0.74 + self.table_z_bias
        self.pot_center = np.array([*self.pan_position_xy, self.table_top])
        self.pot, pan = build_pan(self.scene, self.pot_center[:2], self.table_top, self.assets)
        self.pot_radius = float(pan["inner_radius"])
        self.pot_floor = self.table_top + float(pan["floor_height"])
        self.pot_rim = self.table_top + float(pan["rim_height"])
        self.foods = {}
        self.food_bodies = {}
        self.food_half = {}
        self.food_vertices = {}
        jitter = np.random.uniform(-0.004, 0.004, (3, 2))
        for index, (kind, xy) in enumerate([
            ("green_pepper", [-0.20, -0.12]), ("potato", [0.19, -0.15]), ("eggplant", [0.28, -0.10]),
        ]):
            xy = self.assets[kind].get("initial_xy", xy)
            actor, body, half, vertices = build_vegetable(self.scene, kind, np.asarray(xy) + jitter[index], self.table_top, self.assets)
            self.foods[kind], self.food_bodies[kind], self.food_half[kind] = actor, body, half
            self.food_vertices[kind] = vertices
        self.pepper = self.foods["green_pepper"]
        self.pepper_body = self.food_bodies["green_pepper"]
        root = Path(__file__).resolve().parents[1]
        self.tool_metadata = json.loads((root / self.assets["spatula"]["metadata"]).read_text())
        self.tool_motion_profile = self.tool_motion_profile or self.tool_metadata.get("motion_profile", "raised_handle")
        self.tool_rest = np.array([*self.tool_rest_xy, self.table_top + self.tool_metadata.get("rest_height", 0.056)])
        self.tool_return_rotation = t3d.euler.euler2mat(0, 0, self.tool_metadata.get("rest_yaw", 0))[:2,:2]
        self.tool_return_offset = np.asarray(self.tool_metadata.get("return_center_offset", [0.0325, 0]))
        self.tool_return_center = self.tool_rest[:2] + self.tool_return_rotation @ self.tool_return_offset
        self.tool_return_half = self.tool_metadata.get("return_half_size", [0.15, 0.075])
        self.spatula, self.spatula_body, self.tool_vertices = build_spatula(self.scene, self.tool_rest, self.assets)
        self.active_arm = ArmTag("left")

    def _transition(self, state):
        self.stage += 1
        super()._transition(state)
        if self.debug_frames:
            from pathlib import Path
            folder = Path(self.debug_frames)
            folder.mkdir(parents=True, exist_ok=True)
            self.save_camera_rgb(str(folder / f"{self.stage:02d}_{state}.png"), "head_camera")
            from PIL import Image
            self.scene.update_render()
            self.debug_overhead.take_picture()
            rgb = np.uint8(np.clip(self.debug_overhead.get_picture("Color")[..., :3]*255, 0, 255))
            Image.fromarray(rgb).save(folder / f"{self.stage:02d}_{state}_overhead.png")

    def _tcp_matrix(self, arm):
        pose = getattr(self.robot, f"get_{arm}_tcp_pose")()
        return sapien.Pose(pose[:3], pose[3:]).to_transformation_matrix()

    def _wait(self, seconds):
        if self.tool_motion_profile == "curved_wok" and hasattr(self, "robot"):
            # Hold actuator positions at waypoints. A trimmed planner path may
            # end with a nonzero velocity target; leaving that target active
            # makes the wrist drift while the FSM observes the grip.
            for side in ("left", "right"):
                for joint in getattr(self.robot, f"{side}_entity").get_active_joints():
                    joint.set_drive_velocity_target(0)
        super()._wait(seconds)

    def _remember_grasp(self, arm, entity):
        self.held_frames[str(arm)] = np.linalg.inv(self._tcp_matrix(arm)) @ entity.pose.to_transformation_matrix()

    def _move_object(self, arm, entity, position, rotation=None):
        transform = entity.pose.to_transformation_matrix().copy()
        transform[:3, 3] = position
        if rotation is not None:
            transform[:3, :3] = rotation
        tcp = transform @ np.linalg.inv(self.held_frames[str(arm)])
        virtual = tcp[:3, 3] - tcp[:3, :3] @ [0.12, 0, 0]
        target = virtual.tolist() + t3d.quaternions.mat2quat(tcp[:3, :3]).tolist()
        self.move(self.move_to_pose(arm, target_pose=target))
        if not self.plan_success:
            paths = self.left_joint_path if arm == "left" else self.right_joint_path
            raise RuntimeError(f"Motion planning failed: {self.fsm_state}; target={target}; result={paths[-1] if paths else None}")
        if entity is self.spatula:
            if self.tool_motion_profile == "curved_wok":
                self._wait(0.15)
            actual = entity.pose.to_transformation_matrix()
            error = np.linalg.norm(actual[:3, 3] - transform[:3, 3])
            rotation_error = math.degrees(math.acos(float(np.clip((np.trace(transform[:3, :3].T @ actual[:3, :3]) - 1) / 2, -1, 1))))
            self.tool_tracking.append({"phase": self.fsm_state, "position_error_m": float(error),
                                       "rotation_error_deg": float(rotation_error),
                                       "tip_actual": (actual[:3,:3] @ self.tool_metadata.get("blade_tip", [0.145,0,-0.0545]) + actual[:3,3]).tolist(),
                                       "tip_target": (transform[:3,:3] @ self.tool_metadata.get("blade_tip", [0.145,0,-0.0545]) + transform[:3,3]).tolist()})
            print(f"[W1 tool tracking] phase={self.fsm_state} position_error={error:.4f}m rotation_error={rotation_error:.1f}deg", flush=True)
            if self.tool_motion_profile == "curved_wok":
                if error > 0.04 or rotation_error > 45:
                    raise RuntimeError(f"Tool grasp lost during {self.fsm_state}: {error:.3f} m, {rotation_error:.1f} deg")
                # Re-observe the physical grip after each waypoint. A small
                # handle rotation otherwise leaves the long tip centimetres
                # from its target for every following motion. No attachment
                # or object pose reset is used; replay keeps the same calls.
                self._remember_grasp(arm, entity)

    def _move_tool(self, arm, position, rotation):
        """Keep a held tool close to a Cartesian path through the open bowl."""
        if self.tool_motion_profile != "curved_wok" or self.tool_cartesian_steps == 1:
            return self._move_object(arm, self.spatula, position, rotation)
        from scipy.spatial.transform import Rotation, Slerp
        initial = self.spatula.pose.to_transformation_matrix()
        orientations = Slerp([0, 1], Rotation.from_matrix(np.stack([initial[:3,:3], rotation])))
        for alpha in np.linspace(0, 1, self.tool_cartesian_steps+1)[1:]:
            p = (1-alpha)*initial[:3,3] + alpha*np.asarray(position)
            waypoint_rotation = orientations(alpha).as_matrix()
            if self.fsm_state == "TURN_2_SCOOP":
                # Finite friction permits a few degrees of handle motion.
                # Keep a joint-limit margin using the newly observed grip,
                # pivoting at the lip so the food-contact target stays fixed.
                tip = np.asarray(self.tool_metadata["blade_tip"])
                lip = p + waypoint_rotation @ tip
                for extra_pitch in self.tool_metadata.get("scoop_pitch_adjustments", (0.0, 0.05, 0.10, 0.15, 0.20)):
                    candidate = waypoint_rotation @ t3d.euler.euler2mat(0, extra_pitch, 0)
                    candidate_position = lip - candidate @ tip
                    if self._object_goal_reachable(arm, candidate_position, candidate, limit_margin=0.05):
                        p, waypoint_rotation = candidate_position, candidate
                        break
                else:
                    raise RuntimeError(f"No reachable scoop waypoint with joint margin: {self.fsm_state}")
            self._move_object(arm, self.spatula, p, waypoint_rotation)

    def _object_goal_reachable(self, arm, position, rotation, limit_margin=0, max_restarts=16):
        """Query bounded IK and collision feasibility without moving anything."""
        import torch
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, position
        tcp = transform @ np.linalg.inv(self.held_frames[str(arm)])
        virtual = tcp[:3, 3] - tcp[:3, :3] @ [0.12, 0, 0]
        target = virtual.tolist() + t3d.quaternions.mat2quat(tcp[:3, :3]).tolist()
        ee = self.robot._trans_from_gripper_to_endlink(target, arm_tag=str(arm))
        planner = getattr(self.robot, f"{arm}_planner")
        origin = planner.robot_origion_pose
        p, q = planner._trans_from_world_to_base(np.r_[origin.p, origin.q], np.r_[ee.p, ee.q])
        p = np.asarray(p) + np.asarray(planner.frame_bias)
        entity = getattr(self.robot, f"{arm}_entity")
        names = [joint.name for joint in entity.get_active_joints()]
        current = entity.get_qpos()[[names.index(name) for name in planner.active_joints_name]]
        def feasible(values):
            if (np.any(values < planner.pose_ik.lower + limit_margin)
                    or np.any(values > planner.pose_ik.upper - limit_margin)):
                return False
            query = torch.tensor(values, dtype=torch.float32, device="cuda").reshape(1, 1, -1)
            valid = planner.motion_gen.rollout_fn.rollout_constraint(query, use_batch_env=False).feasible.all().item()
            callback = getattr(planner, "trajectory_accept", None)
            return valid and (callback is None or callback(np.asarray(values)[None]))
        return planner.pose_ik.solve(p, q, current, accept=feasible,max_restarts=max_restarts) is not None

    def _tool_hover_target(self, arm, lip_xy, rotations):
        """Clear the entire rim while respecting the observed physical grip."""
        tip = np.asarray(self.tool_metadata["blade_tip"])
        preferred = float(self.tool_metadata.get("hover_clearance", 0.03))
        current = self.spatula.pose.to_transformation_matrix()[:3,:3]
        alternatives = [current @ t3d.euler.euler2mat(0, pitch, 0) for pitch in (0, 0.15, 0.30)]
        heading = float(self.assets["green_pepper"].get("turn_heading", math.pi/2))
        flat = t3d.euler.euler2mat(0, 0, heading)
        alternatives.extend(flat @ t3d.euler.euler2mat(0, pitch, 0) @ t3d.euler.euler2mat(roll, 0, 0)
                            for pitch in (1.2, 1.35) for roll in (0.3, 0.6, -0.3))
        for rotation in [*rotations, *alternatives]:
            local = self.tool_vertices @ rotation.T
            for clearance in dict.fromkeys([preferred, 0.020, 0.012]):
                position = np.r_[np.asarray(lip_xy) - (rotation @ tip)[:2],
                                  self.pot_rim + clearance - local[:, 2].min()]
                if self._object_goal_reachable(arm, position, rotation):
                    print(f"[W1 tool hover] phase={self.fsm_state} rim_clearance={clearance:.3f}m", flush=True)
                    return position, rotation
        raise RuntimeError(f"No collision-free tool hover pose within W1 limits: {self.fsm_state}")

    def _pick(self, name, arm):
        arm = ArmTag(arm)
        self.active_arm = arm
        entity = self.spatula if name == "spatula" else self.foods[name]
        start = entity.pose.p.copy()
        self._transition(f"{name.upper()}_APPROACH")
        lift_vector = np.array([0, 0.08, 0.08]) if name == "spatula" else np.array([0, 0, 0.13])
        grasp_cfg = self.tool_metadata if name == "spatula" else self.assets[name]
        lift_vector = np.asarray(grasp_cfg.get("lift_vector", lift_vector))
        approach = np.asarray(grasp_cfg.get("approach_offset", lift_vector))
        orientation = (t3d.euler.euler2mat(*grasp_cfg["grasp_rotation_rpy"])
                       if "grasp_rotation_rpy" in grasp_cfg else None)
        grasp_tilt = grasp_cfg.get("grasp_tilt_y", 0.0)
        self._tcp_move(start + approach, tilt_y=grasp_tilt, orientation=orientation)
        if getattr(self, "open_before_descend", False):
            self.move(self.open_gripper(arm))
        if grasp_cfg.get("refresh_grasp_pose", False):
            start = entity.pose.p.copy()
        if name == "spatula":
            self.tool_initial_rest_pose = entity.pose.to_transformation_matrix().copy()
        self._transition(f"{name.upper()}_DESCEND")
        grasp_offset = (self.tool_metadata.get("grasp_offset", [0, 0, -0.015]) if name == "spatula"
                        else self.assets[name].get("grasp_offset", [0, 0, 0]))
        grasp_position = start + grasp_offset
        self._tcp_move(grasp_position, tilt_y=grasp_tilt, orientation=orientation)
        self._transition(f"{name.upper()}_GRASP")
        self.move(self.close_gripper(arm))
        self._wait(0.2)
        self._remember_grasp(arm, entity)
        start = entity.pose.p.copy()  # Lift from the physically recentered grip.
        self._transition(f"{name.upper()}_LIFT")
        segments = int((self.tool_metadata if name == "spatula" else self.assets[name]).get("lift_segments", 1))
        lift_rotation = entity.pose.to_transformation_matrix()[:3, :3].copy()
        for segment in range(1, segments+1):
            self._move_object(arm, entity, start + lift_vector*segment/segments, lift_rotation)
        self._wait(0.2)
        lifted = bool(entity.pose.p[2] > start[2] + 0.07)
        self.pick_checks[name] = lifted
        if not lifted:
            raise RuntimeError(f"Physical grasp failed for {name}: {entity.pose.p}")
        self._remember_grasp(arm, entity)

    def _place_food(self, kind, arm, xy):
        arm = ArmTag(arm)
        self.active_arm = arm
        food = self.foods[kind]
        # Tilt toward the workspace to stay inside the W1 flange workspace.
        rotation = t3d.euler.euler2mat(0.6, 0, 0) @ food.pose.to_transformation_matrix()[:3, :3]
        self._transition(f"{kind.upper()}_TRANSFER")
        self._move_object(arm, food, [*xy, self.pot_rim + 0.085], rotation)
        self._transition(f"{kind.upper()}_RELEASE")
        release_z = (self.pot_floor + 0.030 if self.tool_motion_profile == "curved_wok" and kind == "green_pepper"
                     else self.pot_rim + 0.035)
        if self.tool_motion_profile == "curved_wok" and kind == "green_pepper":
            # Use the lowest collision-free, reachable release height instead
            # of assuming the old shallow-pan pose is valid for a deep wok.
            for height in np.arange(release_z, self.pot_rim + 0.045, 0.010):
                if self._object_goal_reachable(arm, [*xy, height], rotation):
                    release_z = float(height)
                    break
            print(f"[W1 placement] {kind} release_z={release_z:.4f}", flush=True)
        self.placement_release_heights[kind] = float(release_z)
        self._move_object(arm, food, [*xy, release_z], rotation)
        self.move(self.open_gripper(arm))
        self._wait(0.25)
        self._transition(f"{kind.upper()}_RETREAT")
        self._tcp_move([xy[0], xy[1] - 0.08, self.pot_rim + 0.12], tilt=0.6)
        self._wait(0.6)
        if not self._food_inside(kind):
            raise RuntimeError(f"{kind} did not settle inside skillet: {food.pose.p}")

    def _food_inside(self, kind):
        entity = self.foods[kind]
        half = self.food_half[kind]
        # Test the collision hull itself; rotated AABB corners falsely extend
        # below the pan floor even when a rounded vegetable rests correctly.
        corners = self.food_vertices[kind]
        rotation = t3d.quaternions.quat2mat(entity.pose.q)
        points = corners @ rotation.T + entity.pose.p
        inside = bool(np.all(np.linalg.norm(points[:, :2] - self.pot_center[:2], axis=1) < self.pot_radius + 0.003)
                    and points[:, 2].min() > self.pot_floor - 0.004
                    and points[:, 2].min() < self.pot_rim + 0.006)
        if not inside:
            print(f"[W1 containment] {kind} half={half} z={points[:,2].min(),points[:,2].max()} "
                  f"allowed={self.pot_floor-.004,self.pot_rim+.003} radial={np.linalg.norm(points[:,:2]-self.pot_center[:2],axis=1).max()} limit={self.pot_radius+.003}", flush=True)
        return inside

    def _observe_physics(self):
        self.last_tool_contact = False
        contacts = self.scene.get_contacts()
        if self.debug_trace is not None:
            impulses = dict(tool_pepper=0., pan_pepper=0., robot_pepper=0., tool_pan=0.)
            other_names = []
            for contact in contacts:
                names = {body.entity.name for body in contact.bodies}
                impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
                if names == {"spatula", "green_pepper"}: impulses["tool_pepper"] += impulse
                elif names == {"skillet", "green_pepper"}: impulses["pan_pepper"] += impulse
                elif names == {"skillet", "spatula"}: impulses["tool_pan"] += impulse
                elif "green_pepper" in names and names != {"green_pepper", "table"}:
                    impulses["robot_pepper"] += impulse
                    if impulse > 1e-6:
                        other_names.extend(sorted(names-{"green_pepper"}))
                    if impulse > .01 and self.fsm_state.startswith("TURN_"):
                        print(f"[W1 unexpected food contact] phase={self.fsm_state} bodies={names} impulse={impulse:.4f} Ns", flush=True)
            self.debug_trace.append({
                "time": self._record_step*self.scene.get_timestep(), "phase": self.fsm_state,
                "green_pose": np.r_[self.pepper.pose.p, self.pepper.pose.q].copy(),
                "green_velocity": self.pepper_body.linear_velocity.copy(),
                "green_angular_velocity": self.pepper_body.angular_velocity.copy(),
                "tool_pose": np.r_[self.spatula.pose.p, self.spatula.pose.q].copy(),
                "tool_velocity": self.spatula_body.linear_velocity.copy(),
                "tool_angular_velocity": self.spatula_body.angular_velocity.copy(),
                "other_contact": ",".join(sorted(set(other_names))), **impulses})
        allowed = {"skillet", *self.foods}
        for contact in contacts:
            names = {body.entity.name for body in contact.bodies}
            if len(names) == 2 and names <= allowed and contact.points:
                self.support_edges[tuple(sorted(names))] = {
                    name: self.foods[name].pose.p.copy() for name in names if name in self.foods}
        monitor = self.flip_monitor
        if monitor is None:
            self.current_turn_angle = 0.0
            return
        up = t3d.quaternions.quat2mat(self.pepper.pose.q)[:, 2]
        angle = math.acos(float(np.clip(np.dot(up, monitor["initial_up"]), -1, 1)))
        self.current_turn_angle = math.degrees(angle)
        monitor["peak_angle"] = max(monitor["peak_angle"], angle)
        monitor["peak_lift"] = max(monitor["peak_lift"], float(self.pepper.pose.p[2] - monitor["initial_position"][2]))
        monitor["peak_displacement"] = max(monitor["peak_displacement"], float(np.linalg.norm(self.pepper.pose.p[:2] - monitor["initial_position"][:2])))
        for contact in contacts:
            names = {body.entity.name for body in contact.bodies}
            if "green_pepper" in names and names & self.robot_link_names:
                impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
                monitor["robot_contact_max_impulse"] = max(monitor["robot_contact_max_impulse"], impulse)
            if (names == {"spatula", "green_pepper"} and contact.points
                    and (self.tool_motion_profile != "curved_wok" or any(np.linalg.norm(point.impulse) > 1e-8 for point in contact.points))):
                self.last_tool_contact = True
        if self.last_tool_contact:
            monitor["contact_steps"] += 1

    def save_debug_trace(self, path):
        if self.debug_trace:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {key: np.asarray([row[key] for row in self.debug_trace]) for key in self.debug_trace[0]}
            np.savez_compressed(path, **arrays)
            print(f"Saved physical debug trace: {path} ({len(self.debug_trace)} steps)", flush=True)

    def _turn(self, cycle):
        arm = ArmTag(self.tool_arm)
        pepper = self.pepper.pose.p.copy()
        # Roll across the pepper's short cross-section rather than pushing its
        # long axis. Start near the front of the pan so both strokes stay inside.
        heading = float(self.assets["green_pepper"].get("turn_heading", math.pi / 2))
        flat_rotation = t3d.euler.euler2mat(0, 0, heading)
        direction = flat_rotation[:, 0]
        insertion_distance = 0.145
        root = np.r_[pepper[:2], self.pot_floor + 0.057] - insertion_distance * direction
        entering_rotation = flat_rotation
        scoop_rotation = flat_rotation @ t3d.euler.euler2mat(0, -0.85, 0)
        push_distance = insertion_distance - 0.115
        if self.tool_motion_profile == "curved_wok":
            # The user tool has no raised handle. Pitch the real blade down
            # into the deeper wok while keeping the grip above its near rim.
            pitch = float(self.tool_metadata.get("enter_pitch", 0.70))
            roll = self.tool_metadata.get("enter_roll", 0)
            if cycle == 2:
                roll = self.tool_metadata.get("second_enter_roll", roll)
            entering_rotation = flat_rotation @ t3d.euler.euler2mat(0, pitch, 0) @ t3d.euler.euler2mat(roll, 0, 0)
            scoop_pitch = self.tool_metadata.get("scoop_pitch", 0.10)
            if cycle == 2:
                scoop_pitch = self.tool_metadata.get("second_scoop_pitch", scoop_pitch)
            scoop_roll = self.tool_metadata.get("second_scoop_roll", 0) if cycle == 2 else self.tool_metadata.get("scoop_roll",0)
            scoop_rotation = flat_rotation @ t3d.euler.euler2mat(0, scoop_pitch, 0) @ t3d.euler.euler2mat(scoop_roll, 0, 0)
            tip = np.asarray(self.tool_metadata["blade_tip"])
            working = pepper + self.pepper.pose.to_transformation_matrix()[:3, :3] @ [-0.020, 0, 0]
            if cycle == 1 and "contact_x_max_from_pan" in self.tool_metadata:
                # Use the left side of the broad blade to keep the grip in
                # the left W1 workspace after the curved prop rolls sideways.
                working[0] = min(working[0], self.pot_center[0] + self.tool_metadata["contact_x_max_from_pan"])
            if cycle == 1 and "contact_x_min_from_pan" in self.tool_metadata:
                working[0] = max(working[0], self.pot_center[0] + self.tool_metadata["contact_x_min_from_pan"])
            clearance = self.tool_metadata.get("tip_clearance", 0.002)
            if cycle == 2:
                clearance = self.tool_metadata.get("second_tip_clearance", clearance)
            root = np.r_[working[:2], self.pot_floor + clearance] - 0.020*direction - entering_rotation @ tip
            push_distance = self.tool_metadata.get("push_distance", 0.050)
            if cycle == 2:
                push_distance = self.tool_metadata.get("second_push_distance", push_distance)
            if cycle == 2 and direction[1] > 0.9:
                push_distance = min(0.050, max(0.010, -0.130-root[1]))
            insertion=None
            for extra in np.arange(0,.065,.005):
                candidate=root+[0,0,extra]
                if self._object_goal_reachable(arm,candidate,entering_rotation) and self._object_goal_reachable(arm,candidate+push_distance*direction,entering_rotation):
                    insertion=candidate
                    print(f"[W1 tool insertion] cycle={cycle} fitted_surface_lift={extra:.3f}m",flush=True)
                    break
            if insertion is None:raise RuntimeError(f"No collision-free insertion and push pose: turn {cycle}")
            root=insertion
            lip=root+push_distance*direction+entering_rotation@tip
            preferred_lift=self.tool_metadata.get("second_scoop_lift",.004) if cycle==2 else self.tool_metadata.get("scoop_lift",.004)
            advance=self.tool_metadata.get("second_scoop_advance",-.005) if cycle==2 else self.tool_metadata.get("scoop_advance",-.005)
            scoop_target=None
            for lift_value in dict.fromkeys([preferred_lift,.025,.012,.006]):
                for pitch_value in dict.fromkeys([self.tool_metadata.get("second_scoop_pitch",.1) if cycle==2 else self.tool_metadata.get("scoop_pitch",.1),.6,.8,1.0]):
                    roll_value=self.tool_metadata.get("second_scoop_roll",0) if cycle==2 else self.tool_metadata.get("scoop_roll",0)
                    candidate_rotation=flat_rotation@t3d.euler.euler2mat(0,pitch_value,0)@t3d.euler.euler2mat(roll_value,0,0)
                    candidate=lip+advance*direction+[0,0,lift_value]-candidate_rotation@tip
                    if self._object_goal_reachable(arm,candidate,candidate_rotation):
                        scoop_target=(candidate,candidate_rotation)
                        print(f"[W1 tool scoop] cycle={cycle} pitch={pitch_value:.2f} lift={lift_value:.3f}m",flush=True)
                        break
                if scoop_target is not None:break
            if scoop_target is None:raise RuntimeError(f"No collision-free scoop pose: turn {cycle}")
        self._transition(f"TURN_{cycle}_APPROACH")
        hover_rotation = flat_rotation
        if self.tool_motion_profile == "curved_wok":
            hover_rotation = flat_rotation @ t3d.euler.euler2mat(0, self.tool_metadata.get("hover_pitch", 0), 0)
            hover, hover_rotation = self._tool_hover_target(
                arm, (root + entering_rotation @ tip)[:2], [hover_rotation, entering_rotation])
            # Turn the blade while it is above the rim. Rotating an already
            # inserted long tool across the wall knocks it out of the grip.
            self._move_tool(arm, hover, hover_rotation)
        else:
            self._move_object(arm, self.spatula, root + [0, 0, 0.055], entering_rotation)
        self.flip_monitor = {
            "cycle": cycle, "initial_up": t3d.quaternions.quat2mat(self.pepper.pose.q)[:, 2].copy(),
            "initial_position": pepper.copy(), "peak_angle": 0.0, "peak_lift": 0.0,
            "peak_displacement": 0.0, "contact_steps": 0,
            "robot_contact_max_impulse": 0.0,
            "start_time": self._record_step * self.scene.get_timestep(),
        }
        self._transition(f"TURN_{cycle}_INSERT")
        self._move_tool(arm, root, entering_rotation)
        self._transition(f"TURN_{cycle}_SCOOP")
        self._move_tool(arm, root + push_distance * direction, entering_rotation)
        scoop_position = root + (push_distance - 0.005) * direction + [0, 0, 0.008]
        if self.tool_motion_profile == "curved_wok":
            # Rock around the blade lip, not the handle. Rotating this 26 cm
            # straight tool around its grip sweeps the blade through the bowl
            # and requests an unnecessarily extreme wrist configuration.
            lip = root + push_distance*direction + entering_rotation @ tip
            lift = self.tool_metadata.get("second_scoop_lift", 0.004) if cycle == 2 else self.tool_metadata.get("scoop_lift", 0.004)
            advance = self.tool_metadata.get("second_scoop_advance", -0.005) if cycle == 2 else self.tool_metadata.get("scoop_advance", -0.005)
            scoop_position = lip + advance*direction + [0, 0, lift] - scoop_rotation @ tip
            scoop_position,scoop_rotation=scoop_target
        self._move_tool(arm, scoop_position, scoop_rotation)
        self._wait(0.15)
        self._transition(f"TURN_{cycle}_WITHDRAW")
        if self.tool_motion_profile == "curved_wok":
            hover, hover_rotation = self._tool_hover_target(
                arm, (root + entering_rotation @ tip)[:2], [hover_rotation, entering_rotation])
            self._move_tool(arm, hover, hover_rotation)
        else:
            self._move_object(arm, self.spatula, root - 0.015 * direction + [0, 0, 0.060], entering_rotation)
        self._wait(0.6)
        monitor = self.flip_monitor
        self.flip_monitor = None
        valid = bool(monitor["contact_steps"] >= 2 and monitor["peak_angle"] >= self.flip_min_angle
                     and (monitor["peak_lift"] >= 0.004 or monitor["peak_displacement"] >= 0.012)
                     and (self.tool_motion_profile != "curved_wok" or monitor["robot_contact_max_impulse"] < 1e-5)
                     and self._food_inside("green_pepper"))
        event = {key: value for key, value in monitor.items() if key not in ("initial_up", "initial_position")}
        event["peak_angle_deg"] = math.degrees(event.pop("peak_angle"))
        event["initial_up"] = monitor["initial_up"].tolist()
        event["initial_position"] = monitor["initial_position"].tolist()
        event["final_position"] = self.pepper.pose.p.tolist()
        event["end_time"] = self._record_step * self.scene.get_timestep()
        event["valid"] = valid
        self.flip_events.append(event)
        print(f"[W1 turn] {event}", flush=True)
        if not valid:
            raise RuntimeError(f"Turn {cycle} did not meet physical contact/rotation criteria")
        self._transition(f"TURN_{cycle}_VERIFIED")
        self._wait(0.1)

    def _return_tool(self):
        arm = ArmTag(self.tool_arm)
        self.active_arm = arm
        rest = self.tool_rest
        rotation = np.eye(3)
        approach = self.tool_rest + [0, 0.08, 0.08]
        if self.tool_motion_profile == "curved_wok":
            rest = self.tool_initial_rest_pose[:3,3]
            rotation = self.tool_initial_rest_pose[:3,:3]
            approach = rest + self.tool_metadata.get("lift_vector", [0, 0, 0.115])
            # Clear the bowl laterally at the verified hover height before
            # lowering or turning the long blade toward its stand.
            self._transition("SPATULA_RETURN_CLEAR")
            clear = self.spatula.pose.p.copy()
            clear[:2] = approach[:2]
            self._move_tool(arm, clear, self.spatula.pose.to_transformation_matrix()[:3,:3].copy())
        self._transition("SPATULA_RETURN_APPROACH")
        self._move_tool(arm, approach, rotation)
        self._transition("SPATULA_RETURN_LOWER")
        self._move_tool(arm, rest + [0, 0, 0.003], rotation)
        self._transition("SPATULA_RETURN_RELEASE")
        self.move(self.open_gripper(arm))
        self._wait(0.3)
        self._transition("SPATULA_RETURN_RETREAT")
        self._tcp_move(self.tool_rest + [0, 0.08, 0.08])
        self._wait(0.3)
        self.tool_returned = self._tool_in_region()
        if not self.tool_returned:
            raise RuntimeError(f"Spatula missed return region: {self.spatula.pose.p}")
        self._transition(f"{str(arm).upper()}_ARM_PARK")
        self.move(self.move_to_pose(arm, target_pose=self.home_targets[str(arm)]))

    def _tool_in_region(self):
        rotation = t3d.quaternions.quat2mat(self.spatula.pose.q)
        points = self.tool_vertices @ rotation.T + self.spatula.pose.p
        local_xy = (points[:, :2] - self.tool_rest[:2]) @ self.tool_return_rotation
        return bool(np.all(np.abs(local_xy - self.tool_return_offset) < self.tool_return_half)
                    and points[:, 2].min() >= self.table_top - 0.004
                    and points[:, 2].max() <= self.table_top + 0.15
                    and np.linalg.norm(self.spatula_body.linear_velocity) < 0.04
                    and np.linalg.norm(self.spatula_body.angular_velocity) < 0.3)

    def get_obs(self):
        observation = super().get_obs()
        observation["task_observation"] = {
            **{f"{kind}_pose": np.r_[entity.pose.p, entity.pose.q] for kind, entity in self.foods.items()},
            "spatula_pose": np.r_[self.spatula.pose.p, self.spatula.pose.q],
            "effective_flip_count": sum(event["valid"] for event in self.flip_events),
            "stage_index": self.stage,
            "tool_returned": self.tool_returned,
            "tool_pepper_contact": self.last_tool_contact,
            "turn_angle_deg": self.current_turn_angle,
            "turn_monitor_active": self.flip_monitor is not None,
        }
        return observation

    def play_once(self):
        if self.tool_motion_profile == "curved_wok":
            # Imported curved props roll to a stable resting orientation before
            # the first grasp target is measured. This is natural settling.
            self._wait(0.8)
        self._recording = True
        try:
            self._pick("green_pepper", "left")
            self._place_food("green_pepper", "left", self.pot_center[:2] + self.assets["green_pepper"].get("pan_offset_xy", [0, 0]))
            self._pick("spatula", "left")
            if self.tool_motion_profile == "curved_wok":
                self._wait(0.4)
                self._remember_grasp("left", self.spatula)
            self._turn(1)
            self._turn(2)
            self._return_tool()
            self._pick("potato", "right")
            self._place_food("potato", "right", self.pot_center[:2] + [-0.04, -0.025])
            self._pick("eggplant", "right")
            self._place_food("eggplant", "right", self.pot_center[:2] + [0.04, 0.025])
            self._transition("FINAL_SETTLE")
            self._wait(2.5)
            self.full_completed = True
            self._transition("DONE")
            self._wait(0.1)
        finally:
            self._recording = False
        self.info.update(active_arm="both", task_scope="finals_full_three_vegetables",
                         proxy_assets=any(cfg.get("procedural", False) for cfg in self.assets.values()),
                         asset_manifest=self.asset_manifest, all_vegetables_target="skillet",
                         fsm_transitions=self.transitions, pick_checks=self.pick_checks,
                         lift_verified=all(self.pick_checks.values()), flip_events=self.flip_events,
                         flip_min_angle_deg=math.degrees(self.flip_min_angle), tool_tracking=self.tool_tracking,
                         placement_release_heights=self.placement_release_heights,
                         contact_parameters={"gripper_patch_radius_m": self.gripper_contact_patch,
                                             "tool_patch_radius_m": self.tool_metadata.get("contact_patch_radius", 0),
                                             "tool_min_patch_radius_m": self.tool_metadata.get("min_contact_patch_radius", 0)})
        self.info["pan_geometry"] = {"center": self.pot_center.tolist(), "inner_radius": self.pot_radius,
                                     "floor": self.pot_floor, "rim": self.pot_rim}
        self.info["tool_return_region"] = {"center_xy": self.tool_return_center.tolist(),
                                           "half_size_xy": self.tool_return_half,
                                           "tool_rest": self.tool_rest.tolist()}
        self.check_success()
        return self.info

    def check_success(self):
        supported = {"skillet"}
        for _ in range(3):
            for names, positions in self.support_edges.items():
                if any(name in supported for name in names) and all(
                        np.linalg.norm(self.foods[name].pose.p - position) < 0.008
                        for name, position in positions.items()):
                    supported.update(names)
        checks = {
            "plan_success": bool(self.plan_success), "completed": bool(self.full_completed),
            "all_picks_verified": len(self.pick_checks) == 4 and all(self.pick_checks.values()),
            "two_effective_turns": len(self.flip_events) == 2 and all(e["valid"] for e in self.flip_events),
            "tool_returned": bool(self.tool_returned and self._tool_in_region()),
            **{f"{kind}_inside_pan": self._food_inside(kind) for kind in self.foods},
            "foods_stationary": all(np.linalg.norm(body.linear_velocity) < 0.04
                                    and np.linalg.norm(body.angular_velocity) < 0.4 for body in self.food_bodies.values()),
            "foods_supported": set(self.foods) <= supported,
            "grippers_open": bool(self.is_left_gripper_open() and self.is_right_gripper_open()),
        }
        if hasattr(self, "info"):
            self.info["success_checks"] = checks
            self.info["final_velocities"] = {
                name: {"linear": float(np.linalg.norm(body.linear_velocity)),
                       "angular": float(np.linalg.norm(body.angular_velocity))}
                for name, body in self.food_bodies.items()}
        print(f"[W1 finals check] {checks}", flush=True)
        if not checks["foods_stationary"]:
            print(f"[W1 velocities] {self.info['final_velocities']}", flush=True)
        return bool(all(checks.values()))
