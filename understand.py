import os

import cv2
import numpy as np
import transforms3d
import sapien.core as sapien
from omegaconf import OmegaConf
import hydra

from sapien_env.rl_env.mug_collect_env import BaseRLEnv
from sapien_env.sim_env.constructor import add_default_scene_light
from sapien_env.gui.gui_base import GUIBase, DEFAULT_TABLE_TOP_CAMERAS, YX_TABLE_TOP_CAMERAS
from gendp.common.data_utils import save_dict_to_hdf5
from gendp.common.kinematics_utils import KinHelper
from datetime import datetime

import time

def stack_dict(dic):
    # stack list of numpy arrays into a single numpy array inside a nested dict
    for key, item in dic.items():
        if isinstance(item, dict):
            dic[key] = stack_dict(item)
        elif isinstance(item, list):
            dic[key] = np.stack(item, axis=0)
    return dic

def transform_action_from_world_to_robot(action : np.ndarray, pose : sapien.Pose):
    # :param action: (7,) np.ndarray in world frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # :param pose: sapien.Pose of the robot base in world frame
    # :return: (7,) np.ndarray in robot frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # transform action from world to robot frame
    action_mat = np.zeros((4,4))
    action_mat[:3,:3] = transforms3d.euler.euler2mat(action[3], action[4], action[5])
    action_mat[:3,3] = action[:3]
    action_mat[3,3] = 1
    action_mat_in_robot = np.matmul(np.linalg.inv(pose.to_transformation_matrix()),action_mat)
    action_robot = np.zeros(7)
    action_robot[:3] = action_mat_in_robot[:3,3]
    action_robot[3:6] = transforms3d.euler.mat2euler(action_mat_in_robot[:3,:3],axes='sxyz')
    action_robot[6] = action[6]
    return action_robot

def task_to_cfg(task, robot_name, manip_obj=None):
    if task == 'cube_pick':
        cfg = OmegaConf.create({
            '_target_': 'sapien_env.rl_env.cube_pick_env.CubePickRLEnv',
            'use_gui': True,
            'frame_skip': 10,
            'robot_name': robot_name, # "xarm7_with_gripper",
            'use_visual_obs': False,
        })
        # reuse an existing scripted policy so it just runs some arm motion
        policy_cfg = OmegaConf.create({
            '_target_': 'sapien_env.teleop.cube_pick_scripted_policy.SingleArmPolicy',
        })
    else:
        raise ValueError(f'Unknown task {task}')
    return cfg, policy_cfg

def main_env(episode_idx, dataset_dir, headless, mode, task_name, manip_obj=None):
    # initialize env
    os.system(f'mkdir -p {dataset_dir}')
    robot_name = "xarm7_with_gripper"
    print(f"\n=== INITIALIZING ENV ({episode_idx}) ===")
    print(f"USING ROBOT {robot_name}")
    kin_helper = KinHelper(robot_name=robot_name)

    cfg, policy_cfg = task_to_cfg(task_name, manip_obj=manip_obj)
    with open(os.path.join(dataset_dir, 'config.yaml'), 'w') as f:
        OmegaConf.save(cfg, f.name)

    # instantiate & reset
    env: BaseRLEnv = hydra.utils.instantiate(cfg)
    env.seed(episode_idx)
    obs, _ = env.reset(), None
    arm_dof = env.arm_dof
    print(f"arm_dof = {arm_dof}\n")

    # ─── INITIAL WORLD‐FRAME STATES ────────────────────────────────────────
    base_pose_W = env.robot.get_pose().to_transformation_matrix()
    cube_pose_W = env.cube.get_pose().to_transformation_matrix()
    ee_pose_W   = env.palm_link.get_pose().to_transformation_matrix()

    print("### World‐frame (W) ###")
    print(f"  Robot base position W: {np.round(base_pose_W[:3,3],3)}")
    print(f"  Cube        position W: {np.round(cube_pose_W[:3,3],3)}")
    print(f"  EE          position W: {np.round(ee_pose_W[:3,3],3)}\n")

    # ─── TRANSFORM FORMULA ─────────────────────────────────────────────────
    print("Transformation formula to robot frame (B):")
    print("  T_B_target = inv(T_W_base) @ T_W_target\n")

    # ─── INITIAL ROBOT‐FRAME STATES ────────────────────────────────────────
    T_W_base_inv = np.linalg.inv(base_pose_W)
    base_pose_B = T_W_base_inv @ base_pose_W
    cube_pose_B = T_W_base_inv @ cube_pose_W
    ee_pose_B   = T_W_base_inv @ ee_pose_W

    print("### Robot‐frame (B) ###")
    print(f"  Robot base in B: {np.round(base_pose_B[:3,3],3)}  # should be ~[0,0,0]")
    print(f"  Cube        in B: {np.round(cube_pose_B[:3,3],3)}")
    print(f"  EE          in B: {np.round(ee_pose_B[:3,3],3)}\n")

    # ─── INITIAL JOINT STATES & IK COMPARISON ──────────────────────────────
    qpos_init = env.robot.get_qpos()
    print(f"Initial qpos (joint space): {np.round(qpos_init,3)}")

    # we want to see what IK would do to reach the current EE in B
    # extract the robot‐frame ee target as a 7‐vector [x,y,z, r,p,y, grip]
    ee_xyz_B = ee_pose_B[:3,3]
    ee_rpy_B = transforms3d.euler.mat2euler(ee_pose_B[:3,:3], axes='sxyz')
    ee_grip  = qpos_init[arm_dof]     # current gripper  
    target_cart_B = np.concatenate([ee_xyz_B, ee_rpy_B, [ee_grip]])

    ik_qpos = kin_helper.compute_ik_sapien(qpos_init, target_cart_B)
    print(f"IK‐computed qpos to hold EE at its current B‐pose: {np.round(ik_qpos,3)}")

    diff = ik_qpos - qpos_init
    print(f"Difference (IK_qpos – initial_qpos): {np.round(diff,3)}\n")
    # ────────────────────────────────────────────────────────────────────────
    
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer, headless=headless)
    for name, params in YX_TABLE_TOP_CAMERAS.items():
        if 'rotation' in params:
            gui.create_camera_from_pos_rot(**params)
        else:
            gui.create_camera(**params)
    if not gui.headless:
        gui.viewer.set_camera_rpy(r=0, p=-0.5, y=np.pi/2)
        gui.viewer.set_camera_xyz(x=0, y=0.5, z=0.5)
    scene = env.scene
    scene.step()

    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    scripted_policy = hydra.utils.instantiate(policy_cfg)
    init_poses = env.get_init_poses()

    data_dict = {
        'observations': {
            'joint_pos': [],
            'joint_vel': [],
            'full_joint_pos': [],
            'robot_base_pose_in_world': [],
            'ee_pos': [],
            'ee_vel': [],
            'images': {},
        },
        'joint_action': [],
        'cartesian_action': [],
        'info': {'init_poses': init_poses}
    }

    cams = gui.cams
    for cam in cams:
        for typ in ['color', 'depth', 'intrinsic', 'extrinsic']:
            data_dict['observations']['images'][f'{cam.name}_{typ}'] = []

    attr_dict = {'sim': True}
    config_dict = {'observations': {'images': {}}}
    for cam in gui.cams:
        color_save_kwargs = {
            'chunks': (1, cam.height, cam.width, 3),
            'compression': 'gzip', 'compression_opts': 9, 'dtype': 'uint8'}
        depth_save_kwargs = {
            'chunks': (1, cam.height, cam.width),
            'compression': 'gzip', 'compression_opts': 9, 'dtype': 'uint16'}
        config_dict['observations']['images'][f'{cam.name}_color'] = color_save_kwargs
        config_dict['observations']['images'][f'{cam.name}_depth'] = depth_save_kwargs

    timesteps = 0
    print("\n=== BEGINNING ROLLOUT ===\n")
    while True:
        action = np.zeros(arm_dof + 1)
        cartisen_action, quit = scripted_policy.single_trajectory(env, env.palm_link.get_pose(), mode=mode)

        cube_pos = np.round(env.cube.get_pose().p, 3)
        ee_world = np.round(env.palm_link.get_pose().p, 3)
        desired_ee_world = np.round(cartisen_action[:3], 3)

        print(f"[Step {timesteps}] Cube position (world): {cube_pos} (3x1)")
        print(f"[Step {timesteps}] EE position (world): {ee_world} (3x1)")
        print(f"[Step {timesteps}] Target EE position (world): {desired_ee_world} (3x1)")

        cartisen_action_in_rob = transform_action_from_world_to_robot(
            cartisen_action, env.robot.get_pose()
        )
        ee_robot_target = np.round(cartisen_action_in_rob[:3], 3)
        print(f"[Step {timesteps}] Target EE position (robot): {ee_robot_target} (3x1)")

        if quit:
            print("Exiting rollout loop.")
            break

        ik_result = kin_helper.compute_ik_sapien(env.robot.get_qpos()[:], cartisen_action_in_rob)
        print(f"[Step {timesteps}] IK result (1x{len(ik_result)}): {np.round(ik_result, 3)}")

        action[:arm_dof] = ik_result[:arm_dof]
        action[arm_dof] = cartisen_action_in_rob[6]
        print(f"[Step {timesteps}] Final action (1x{len(action)}): {np.round(action, 3)}")

        obs, reward, done, _ = env.step(action[:arm_dof + 1])
        rgbs, depths = gui.render(depth=True)

        qpos = env.robot.get_qpos()
        qvel = env.robot.get_qvel()
        data_dict['observations']['joint_pos'].append(qpos[:-1])
        data_dict['observations']['joint_vel'].append(qvel[:-1])
        data_dict['observations']['full_joint_pos'].append(qpos)
        data_dict['observations']['robot_base_pose_in_world'].append(env.robot.get_pose().to_transformation_matrix())

        ee_translation = env.palm_link.get_pose().p
        ee_rotation = transforms3d.euler.quat2euler(env.palm_link.get_pose().q, axes='sxyz')
        ee_gripper = qpos[arm_dof]
        ee_pos = np.concatenate([ee_translation, ee_rotation, [ee_gripper]])
        ee_vel = np.concatenate([env.palm_link.get_velocity(),
                                 env.palm_link.get_angular_velocity(),
                                 qvel[arm_dof:arm_dof + 1]])
        data_dict['observations']['ee_pos'].append(ee_pos)
        data_dict['observations']['ee_vel'].append(ee_vel)
        data_dict['joint_action'].append(action.copy())
        data_dict['cartesian_action'].append(cartisen_action.copy())

        print(f"[Step {timesteps}] EE pos (1x7): {np.round(ee_pos, 3)}")
        print(f"[Step {timesteps}] EE vel (1x7): {np.round(ee_vel, 3)}\n")

        for cam_idx, cam in enumerate(gui.cams):
            data_dict['observations']['images'][f'{cam.name}_color'].append(rgbs[cam_idx])
            data_dict['observations']['images'][f'{cam.name}_depth'].append(depths[cam_idx])
            data_dict['observations']['images'][f'{cam.name}_intrinsic'].append(cam.get_intrinsic_matrix())
            data_dict['observations']['images'][f'{cam.name}_extrinsic'].append(cam.get_extrinsic_matrix())

        timesteps += 1

    if reward < 1:
        print(f"Failed at episode {episode_idx} (reward: {reward})")

    print(f"Total timesteps: {timesteps}")
    data_dict = stack_dict(data_dict)
    save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)
    print(f"Saved dataset to: {dataset_path}")

    if not gui.headless:
        gui.viewer.close()
        cv2.destroyAllWindows()
    env.close()

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('episode_idx', help='random seed for the episode')
    # parser.add_argument('dataset_dir', help='directory to save the dataset')
    # parser.add_argument('task_name', help='task name, including hang_mug, mug_collect, pen_insertion')
    # parser.add_argument('--headless', action='store_true', help='whether to run in headless mode')
    # parser.add_argument('--obj_name', default=None, help='manipulated object name. The full list is shown in YX_DEFAULT_SCALE at sapien_env/sapien_env/utils/yx_object_utils.py')
    # parser.add_argument('--mode', default='straight', help='mode for scripted policy. Examples are shown in generate_trajectory() at sapien_env/sapien_env/teleop/mug_collect_scripted_policy.py')
    # args = parser.parse_args()

    dataset_dir = os.path.join('datasets', datetime.now().strftime('%Y%m%d_%H%M%S'))
    main_env(episode_idx=0,
             dataset_dir=dataset_dir,
             headless=False,
             mode="straight",
             manip_obj=None,
             task_name="cube_pick")



