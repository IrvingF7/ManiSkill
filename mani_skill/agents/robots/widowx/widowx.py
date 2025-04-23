import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


# TODO (stao) (xuanlin): model it properly based on real2sim
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250S(BaseAgent):
    uid = "widowx250s"
    urdf_path = f"{ASSET_DIR}/robots/widowx/wx250s.urdf"
    urdf_config = dict()

    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["left_finger_link"]
        self.finger2_link = self.robot.links_map["right_finger_link"]

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)
    
    def _after_init(self):
        super()._after_init()
        
        self.base_link = [x for x in self.robot.get_links() if x.name == "base_link"][0]
        self.ee_link = [x for x in self.robot.get_links() if x.name == "ee_gripper_link"][0]

    @property
    def base_pose(self):
        return self.base_link.pose
    
    @property
    def ee_pose(self):
        return self.ee_link.pose


    @property
    def gripper_closedness(self) -> np.ndarray:
        """
        Returns:
            closedness: np.ndarray of shape [batch, 1], values in [0, 1],
                        where 0 means fully open and 1 means fully closed.
        """
        # get_qpos() -> (B, 7), get_qlimits() -> (B, 7, 2)
        qpos = self.robot.get_qpos()           # torch.Tensor, shape (B, 7)
        qlimits = self.robot.get_qlimits()     # torch.Tensor, shape (B, 7, 2)

        # pick off the last two joints (the gripper fingers)
        finger_qpos = qpos[:, -2:]             # shape (B, 2)
        finger_qlim = qlimits[:, -2:, :]       # shape (B, 2, 2)

        # split limits: upper & lower each (B, 2)
        lower = finger_qlim[..., 0]
        upper = finger_qlim[..., 1]

        # compute per-finger closedness: (upper − pos) ÷ (upper − lower)
        closedness = (upper - finger_qpos) / (upper - lower)  # shape (B, 2)

        # average fingers, keep a singleton dim -> (B, 1)
        mean_closedness = closedness.mean(dim=1, keepdim=True)

        # clamp to ≥ 0 (you could also do `.clamp(0.0, 1.0)` if you want an upper bound)
        return mean_closedness.clamp_min(0.0)
