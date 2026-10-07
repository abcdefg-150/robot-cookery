# 任务说明：
# 使用单臂（右臂）抓取放置在右侧区域的一个青椒

import numpy as np

from ._base_task import Base_Task
from .utils import *


class place_pepper_basket(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.basket = rand_create_actor(
            scene=self,
            modelname="076_breadbasket",
            model_id=0,
            xlim=[0.0, 0.0],
            ylim=[-0.2, -0.2],
            qpos=[0.5, 0.5, 0.5, 0.5],
            convex=True,
        )

        self.pepper = rand_create_actor(
            scene=self,
            modelname="my_pepper",
            model_id=0,
            xlim=[0.18, 0.25],
            ylim=[-0.10, 0.05],
            zlim=[0.775, 0.775],
            qpos=[0.70710678, 0.70710678, 0, 0],  # 模型局部 Y 轴朝上
            rotate_rand=True,  # 保持竖直，随机改变水平朝向
            rotate_lim=[0, np.pi / 6, 0],  # 以后开启时绕局部 Y 轴随机转向
            convex=True,
        )

        self.pepper.set_mass(0.02)
        self.pepper_start_z = self.pepper.get_pose().p[2]
        self.pepper_was_lifted = False

        self.add_prohibit_area(self.pepper, padding=0.04)
        self.add_prohibit_area(self.basket, padding=0.05)

    def play_once(self):
        arm = ArmTag("right")

        self.move(
            self.grasp_actor(
                self.pepper,
                arm_tag=arm,
                pre_grasp_dis=0.07,
            )
        )

        self.move(self.move_by_displacement(arm_tag=arm, z=0.10))
        self.pepper_was_lifted = self.pepper.get_pose().p[2] > self.pepper_start_z + 0.02

        # Keep the pepper's grasped orientation while translating it into the basket.
        basket_target = np.asarray(self.basket.get_functional_point(0)[:3], dtype=float)
        pepper_height = self.pepper.config["extents"][1] * self.pepper.config["scale"][1]
        target_center = basket_target + np.array([0.0, 0.0, pepper_height / 2 + 0.005])
        displacement = target_center - self.pepper.get_pose().p
        above_basket = np.asarray(self.robot.get_right_ee_pose(), dtype=float)
        above_basket[:3] += displacement + np.array([0.0, 0.0, 0.08])
        self.move(self.move_to_pose(arm_tag=arm, target_pose=above_basket))
        self.move(self.move_by_displacement(arm_tag=arm, z=-0.08))
        self.move(self.open_gripper(arm_tag=arm))

        self.move(self.move_by_displacement(arm_tag=arm, z=0.10))

        self.info["info"] = {
            "{A}": "076_breadbasket/base0",
            "{B}": "my_pepper/base0",
            "{a}": "right",
        }
        return self.info

    def check_success(self):
        pepper_p = self.pepper.get_pose().p
        basket_p = self.basket.get_pose().p

        near_basket_center = np.all(
            np.abs(pepper_p[:2] - basket_p[:2]) < 0.05
        )
        touches_basket = self.check_actors_contact(
            self.pepper.get_name(), self.basket.get_name()
        )
        released = self.robot.is_right_gripper_open()

        return near_basket_center and self.pepper_was_lifted and touches_basket and released
