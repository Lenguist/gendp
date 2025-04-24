#!/usr/bin/env python3
import os
import cv2
import numpy as np
import transforms3d
import sapien.core as sapien
from omegaconf import OmegaConf
import hydra

from sapien_env.rl_env.mug_collect_env import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.gui_base import GUIBase  # no need to import the camera dicts here
from gendp.common.data_utils import save_dict_to_hdf5
from gendp.common.kinematics_utils import KinHelper


def stack_dict(dic):
    # stack list of numpy arrays into a single numpy array inside a nested dict
    for key, item in dic.items():
        if isinstance(item, dict):
            dic[key] = stack_dict(item)
        elif isinstance(item, list):
            dic[key] = np.stack(item, axis=0)
    return dic


def transform_action_from_world_to_robot(action: np.ndarray, pose: sapien.Pose):
    # :param action: (7,) in world frame. action[:3]=xyz, action[3:6]=euler, action[6]=gripper
    # :param pose: robot base pose in world frame
    # :return: same action expressed in robot frame
    action_mat = np.eye(4)
    action_mat[:3, :3] = transforms3d.euler.euler2mat(
        action[3], action[4], action[5]
    )
    action_mat[:3, 3] = action[:3]
    # transform into robot frame
    action_mat_in_robot = np.linalg.inv(pose.to_transformation_matrix()) @ action_mat
    out = np.zeros(7)
    out[:3] = action_mat_in_robot[:3, 3]
    out[3:6] = transforms3d.euler.mat2euler(
        action_mat_in_robot[:3, :3], axes="sxyz"
    )
    out[6] = action[6]
    return out


def task_to_cfg(task, manip_obj=None):
    if task == "cube_pick":
        cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.rl_env.cube_pick_env.CubePickRLEnv",
                "use_gui": True,
                "frame_skip": 10,
                "robot_name": "xarm7",
                "use_visual_obs": False,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                "_target_": "sapien_env.teleop.cube_pick_scripted_policy.SingleArmPolicy",
            }
        )
    else:
        raise ValueError(f"Unknown task {task}")
    return cfg, policy_cfg


def main_env(episode_idx, dataset_dir, headless, mode, task_name, manip_obj=None):
    os.makedirs(dataset_dir, exist_ok=True)
    cfg, policy_cfg = task_to_cfg(task_name, manip_obj=manip_obj)

    # save config
    with open(os.path.join(dataset_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f.name)

    # instantiate env & IK helper
    env: BaseRLEnv = hydra.utils.instantiate(cfg)
    kin_helper = KinHelper(robot_name=env.robot_name) #kinematics util helper

    env.seed(episode_idx)
    env.reset()
    arm_dof = env.arm_dof

    # setup cameras
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=headless)
    # no extra loops — GUIBase already creates exactly 4 calibrated cameras
    if not gui.headless:
        # optionally adjust the interactive viewer (not the mounted cameras)
        gui.viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi / 2)
        gui.viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    env.scene.step()

    dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
    scripted_policy = hydra.utils.instantiate(policy_cfg)

    # prepare data buffers
    init_poses = env.get_init_poses()
    data_dict = {
        "observations": {
            "joint_pos": [],
            "joint_vel": [],
            "full_joint_pos": [],
            "robot_base_pose_in_world": [],
            "ee_pos": [],
            "ee_vel": [],
            "images": {},
        },
        "joint_action": [],
        "cartesian_action": [],
        "info": {"init_poses": init_poses},
    }
    for cam in gui.cams:
        data_dict["observations"]["images"][f"{cam.name}_color"] = []
        data_dict["observations"]["images"][f"{cam.name}_depth"] = []
        data_dict["observations"]["images"][f"{cam.name}_intrinsic"] = []
        data_dict["observations"]["images"][f"{cam.name}_extrinsic"] = []

    attr_dict = {"sim": True}
    config_dict = {"observations": {"images": {}}}
    for cam in gui.cams:
        config_dict["observations"]["images"][f"{cam.name}_color"] = {
            "chunks": (1, cam.height, cam.width, 3),
            "compression": "gzip",
            "compression_opts": 9,
            "dtype": "uint8",
        }
        config_dict["observations"]["images"][f"{cam.name}_depth"] = {
            "chunks": (1, cam.height, cam.width),
            "compression": "gzip",
            "compression_opts": 9,
            "dtype": "uint16",
        }

    # main loop
    reward = 0.0
    while True:
        # build an arm-only action (no gripper DOF)
        action = np.zeros(arm_dof)

        cart_action, quit = scripted_policy.single_trajectory(
            env, env.palm_link.get_pose(), mode=mode
        )
        if quit:
            break

        # drop the gripper component, only transform xyz+orientation
        cart_action = cart_action[:6]
        cart_rob = transform_action_from_world_to_robot(
            np.concatenate([cart_action, [0.0]]),  # pad to length 7
            env.robot.get_pose(),
        )[:6]

        # solve IK for arm joints
        try:
            qsol = kin_helper.compute_ik_sapien(
                env.robot.get_qpos(), cart_rob, pose_fmt="euler"
            )
        except RuntimeError:
            print("=== IK FAILURE ===")
            print("Target (robot frame xyz+euler):", cart_rob)
            raise

        action[:] = qsol[:arm_dof]

        # step the environment
        obs, reward, done, _ = env.step(action)
        rgbs, depths = gui.render(depth=True)

        # record
        qpos = env.robot.get_qpos()
        qvel = env.robot.get_qvel()
        data_dict["observations"]["joint_pos"].append(qpos.copy())
        data_dict["observations"]["joint_vel"].append(qvel.copy())
        data_dict["observations"]["full_joint_pos"].append(qpos.copy())
        data_dict["observations"]["robot_base_pose_in_world"].append(
            env.robot.get_pose().to_transformation_matrix()
        )

        ee_p = env.palm_link.get_pose().p
        ee_e = transforms3d.euler.quat2euler(env.palm_link.get_pose().q, axes="sxyz")
        ee_pos = np.concatenate([ee_p, ee_e])
        ee_vel = np.concatenate(
            [env.palm_link.get_velocity(), env.palm_link.get_angular_velocity()]
        )
        data_dict["observations"]["ee_pos"].append(ee_pos)
        data_dict["observations"]["ee_vel"].append(ee_vel)

        data_dict["joint_action"].append(action.copy())
        data_dict["cartesian_action"].append(cart_action.copy())

        for idx, cam in enumerate(gui.cams):
            data_dict["observations"]["images"][f"{cam.name}_color"].append(rgbs[idx])
            data_dict["observations"]["images"][f"{cam.name}_depth"].append(depths[idx])
            data_dict["observations"]["images"][f"{cam.name}_intrinsic"].append(
                cam.get_intrinsic_matrix()
            )
            data_dict["observations"]["images"][f"{cam.name}_extrinsic"].append(
                cam.get_extrinsic_matrix()
            )

    # finalize
    if reward < 1:
        print(f"Failed at episode {episode_idx}")
    data_dict = stack_dict(data_dict)
    save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)

    if not gui.headless:
        gui.viewer.close()
        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "episode_idx", help="random seed for the episode"
    # )
    # parser.add_argument(
    #     "dataset_dir",
    #     default = "dataset-test",
    #     help="directory to save the dataset"
    # )
    # parser.add_argument(
    #     "task_name",
    #     default = "cube_pick",
    #     help="task name: hang_mug, mug_collect, pen_insertion, cube_pick"
    # )
    # parser.add_argument(
    #     "--headless",
    #     action="store_true",
    #     help="whether to run in headless mode"
    # )
    # parser.add_argument(
    #     "--obj_name",
    #     default=None,
    #     help="manipulated object name (for mug tasks)"
    # )
    # parser.add_argument(
    #     "--mode",
    #     default="straight",
    #     help="mode for scripted policy"
    # )
    # args = parser.parse_args()

    main_env(
        episode_idx=0,
        dataset_dir="dataset-test",
        headless=False,
        mode="straight",
        task_name="cube_pick",
        manip_obj=None,
    )
