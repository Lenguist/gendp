# cube_pick_env.py (in sapien_env/rl_env)
from functools import cached_property
from typing import Optional
import numpy as np
import sapien.core as sapien
import transforms3d
from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.sim_env.cube_env import CubePickEnv
from sapien_env.rl_env.para import ARM_INIT
from sapien_env.utils.common_robot_utils import (
    generate_free_robot_hand_info,
    generate_arm_robot_hand_info,
    generate_panda_info,
)

class CubePickRLEnv(CubePickEnv, BaseRLEnv):
    """
    RL wrapper: table + cube + robot, with full HangMug analog.
    """
    def __init__(self,
                 use_gui=False,
                 frame_skip=5,
                 robot_name="panda",
                 constant_object_state=False,
                 object_pose_noise=0.01,
                 **renderer_kwargs):
        # 1) Build sim (table + cube)
        super().__init__(use_gui=use_gui,
                         frame_skip=frame_skip,
                         use_ray_tracing=False,
                         **renderer_kwargs)
        # 2) Attach robot
        self.setup(robot_name)

        self.constant_object_state = constant_object_state
        self.object_pose_noise = object_pose_noise

        # 3) Link parsing
        if self.is_robot_free:
            info = generate_free_robot_hand_info()[robot_name]
        elif self.is_xarm:
            info = generate_arm_robot_hand_info()[robot_name]
        elif self.is_panda:
            info = generate_panda_info()[robot_name]
        else:
            raise NotImplementedError
        self.palm_link_name = info.palm_name
        self.palm_link = next(
            link for link in self.robot.get_links()
            if link.get_name() == self.palm_link_name)
        # finger_tip_names = ["panda_leftfinger", "panda_rightfinger"]
        names = [l.get_name() for l in self.robot.get_links()]
        # self.finger_tip_links = [
        #     self.robot.get_links()[names.index(n)]
        #     for n in finger_tip_names
        # ]
        # Record cube pose
        self.object_episode_init_pose = sapien.Pose()

    def get_oracle_state(self):
        """
        Expert state: robot qpos, cube pose, vel, ang, cube-in-palm, vertical cosine.
        """
        robot_qpos_vec = self.robot.get_qpos()
        obj_pose = (self.object_episode_init_pose if self.constant_object_state
                    else self.cube.get_pose())
        object_pose_vec = np.concatenate([obj_pose.p, obj_pose.q])
        # orientation's world z-axis
        z_axis = obj_pose.to_transformation_matrix()[:3, 2]
        theta_cos = np.dot(np.array([0, 0, 1]), z_axis)
        palm_pose = self.palm_link.get_pose()
        object_in_palm = obj_pose.p - palm_pose.p
        v = self.cube.get_velocity()
        w = self.cube.get_angular_velocity()
        return np.concatenate([
            robot_qpos_vec,
            object_pose_vec,
            v, w,
            object_in_palm,
            np.array([theta_cos])
        ])

    def get_robot_state(self):
        """
        Robot-only state: qpos + palm position.
        """
        robot_qpos_vec = self.robot.get_qpos()
        palm_pose = self.palm_link.get_pose()
        return np.concatenate([robot_qpos_vec, palm_pose.p])

    def get_reward(self, action):
        """
        +1 if cube lifted above 0.10 m, else 0.
        """
        z = self.cube.get_pose().p[2]
        return float(z > 0.10)

    def reset(self,
              *,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None):
        # 1) Home robot (xarm, trossen, panda, else) exactly as HangMug
        if self.is_xarm:
            qpos = np.zeros(self.robot.dof)
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos,
                                    transforms3d.euler.euler2quat(0, 0, 0))
        elif getattr(self, 'is_trossen_arm', False):
            qpos = np.zeros(self.robot.dof)
            qpos[self.arm_dof:] = [0.021, -0.021]
            arm_qpos = self.robot_info.arm_init_qpos
            qpos[:self.arm_dof] = arm_qpos
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = ARM_INIT + self.robot_info.root_offset
            init_pose = sapien.Pose(init_pos,
                                    transforms3d.euler.euler2quat(0, 0, 0))
        elif self.is_panda:
            qpos = self.robot_info.arm_init_qpos.copy()
            self.robot.set_qpos(qpos)
            self.robot.set_drive_target(qpos)
            init_pos = np.array([0.0, -0.5, 0.0])
            init_ori = transforms3d.euler.euler2quat(0, 0, np.pi/2)
            init_pose = sapien.Pose(init_pos, init_ori)
        else:
            init_pose = sapien.Pose(
                np.array([-0.4, 0, 0.2]),
                transforms3d.euler.euler2quat(0, np.pi/2, 0)
            )
        self.robot.set_pose(init_pose)
        # 2) Sim reset + stabilize
        self.reset_internal()
        for _ in range(100):
            self.robot.set_qf(
                self.robot.compute_passive_force(external=False)
            )
            self.scene.step()
        # 3) Place cube + optional noise
        self.object_episode_init_pose = self.cube.get_pose()
        rand_pos = self.np_random.randn(3)*self.object_pose_noise
        rand_ori = transforms3d.euler.euler2quat(*(
            self.np_random.randn(3)*self.object_pose_noise
        ))
        new_pose = self.object_episode_init_pose * sapien.Pose(rand_pos, rand_ori)
        self.cube.set_pose(new_pose)
        # Return obs & oracle to match signature
        return self.get_observation(), self.get_oracle_state()

    @cached_property
    def obs_dim(self):
        if not self.use_visual_obs:
            # robot dof + (pose 7 + vel 3 + ang 3 + in_palm 3 + theta 1)
            return self.robot.dof + 7 + 3 + 3 + 3 + 1
        else:
            return len(self.get_robot_state())

    def set_init(self, init_states):
        # Load cube initial pose
        init_pose = sapien.Pose.from_transformation_matrix(init_states[0])
        self.cube.set_pose(init_pose)

    def is_done(self):
        return self.current_step >= self.horizon

    @cached_property
    def horizon(self):
        return 10000