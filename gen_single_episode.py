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
import csv

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

def task_to_cfg(task, manip_obj=None):
    if task == 'hang_mug':
        cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.rl_env.hang_mug_env.HangMugRLEnv',
                'use_gui': True,
                'robot_name': 'panda',
                'frame_skip': 10,
                'use_visual_obs': False,
                'manip_obj': 'nescafe_mug' if manip_obj is None else manip_obj,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.teleop.hang_mug_scripted_policy.SingleArmPolicy',
            }
        )
    elif task == 'mug_collect':
        cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.rl_env.mug_collect_env.MugCollectRLEnv',
                'use_gui': True,
                'robot_name': 'panda',
                'frame_skip': 10,
                'use_visual_obs': False,
                'manip_obj': 'pepsi' if manip_obj is None else manip_obj,
                'randomness_level': 'half'
            }
        )
        policy_cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.teleop.mug_collect_scripted_policy.SingleArmPolicy',
            }
        )
    elif task == 'pen_insertion':
        cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.rl_env.pen_insertion_env.PenInsertionRLEnv',
                'use_gui': True,
                'robot_name': 'panda',
                'frame_skip': 10,
                'use_visual_obs': False,
                'manip_obj': 'pencil' if manip_obj is None else manip_obj,
            }
        )
        policy_cfg = OmegaConf.create(
            {
                '_target_': 'sapien_env.teleop.pen_insertion_scripted_policy.SingleArmPolicy',
            }
        )
    elif task == 'cube_pick':
        cfg = OmegaConf.create({
            '_target_': 'sapien_env.rl_env.cube_pick_env.CubePickRLEnv',
            'use_gui': True,
            'frame_skip': 10,
            'robot_name': "xarm7_with_gripper", # "xarm7_with_gripper",
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
    # print(f"arm_dof = {arm_dof}\n")

    # ─── INITIAL WORLD‐FRAME STATES ────────────────────────────────────────
    base_pose_W = env.robot.get_pose().to_transformation_matrix()
    cube_pose_W = env.cube.get_pose().to_transformation_matrix()
    ee_pose_W   = env.palm_link.get_pose().to_transformation_matrix()

    # print("### World‐frame (W) ###")
    # print(f"  Robot base position W: {np.round(base_pose_W[:3,3],3)}")
    # print(f"  Cube        position W: {np.round(cube_pose_W[:3,3],3)}")
    # print(f"  EE          position W: {np.round(ee_pose_W[:3,3],3)}\n")

    # # ─── TRANSFORM FORMULA ─────────────────────────────────────────────────
    # print("Transformation formula to robot frame (B):")
    # print("  T_B_target = inv(T_W_base) @ T_W_target\n")

    # ─── INITIAL ROBOT‐FRAME STATES ────────────────────────────────────────
    T_W_base_inv = np.linalg.inv(base_pose_W)
    base_pose_B = T_W_base_inv @ base_pose_W
    cube_pose_B = T_W_base_inv @ cube_pose_W
    ee_pose_B   = T_W_base_inv @ ee_pose_W

    # print("### Robot‐frame (B) ###")
    # print(f"  Robot base in B: {np.round(base_pose_B[:3,3],3)}  # should be ~[0,0,0]")
    # print(f"  Cube        in B: {np.round(cube_pose_B[:3,3],3)}")
    # print(f"  EE          in B: {np.round(ee_pose_B[:3,3],3)}\n")

    # # ─── INITIAL JOINT STATES & IK COMPARISON ──────────────────────────────
    qpos_init = env.robot.get_qpos()
    print(f"Initial qpos (joint space): {np.round(qpos_init,3)}")

    # we want to see what IK would do to reach the current EE in B
    # extract the robot‐frame ee target as a 7‐vector [x,y,z, r,p,y, grip]
    ee_xyz_B = ee_pose_B[:3,3]
    ee_rpy_B = transforms3d.euler.mat2euler(ee_pose_B[:3,:3], axes='sxyz')
    ee_grip  = qpos_init[arm_dof]     # current gripper  
    target_cart_B = np.concatenate([ee_xyz_B, ee_rpy_B, [ee_grip]])

    ik_qpos = kin_helper.compute_ik_sapien(qpos_init, target_cart_B)
    # print(f"IK‐computed qpos to hold EE at its current B‐pose: {np.round(ik_qpos,3)}")

    # diff = ik_qpos - qpos_init
    # print(f"Difference (IK_qpos – initial_qpos): {np.round(diff,3)}\n")
    # # ────────────────────────────────────────────────────────────────────────
    
    # Setup viewer and camera
    add_default_scene_light(env.scene, env.renderer)
    gui = GUIBase(env.scene, env.renderer,headless=headless)
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
    
    timesteps = 0

    csv_log_path = os.path.join(dataset_dir, f'log_{episode_idx}.csv')
    csv_file = open(csv_log_path, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        "timestep",
        "ee_pos_world_x", "ee_pos_world_y", "ee_pos_world_z",
        "ee_rpy_world_r", "ee_rpy_world_p", "ee_rpy_world_y",
        "qpos_env", "qpos_ik", "delta_ik",
        "desired_ee_pos_world_x", "desired_ee_pos_world_y", "desired_ee_pos_world_z",
        "qpos_target", "qpos_post", "delta_post"
    ])

    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    
    scripted_policy = hydra.utils.instantiate(policy_cfg)
    
    # set up data saving hyperparameters
    init_poses = env.get_init_poses()
    data_dict = {
        'observations': 
            {'joint_pos': [],
             'joint_vel': [],
             'full_joint_pos': [], # this is to compute FK
             'robot_base_pose_in_world': [],
             'ee_pos': [],
             'ee_vel': [],
            #  'finger_pos': {},
             'images': {},},
        'joint_action': [],
        'cartesian_action': [],
        'info':
            {'init_poses': init_poses}
    }
    # finger_names = ['left_finger_link','right_finger_link']
    # for finger in finger_names:
    #     data_dict['observations']['finger_pos'][finger] = []
    cams = gui.cams
    for cam in cams:
        data_dict['observations']['images'][f'{cam.name}_color'] = []
        data_dict['observations']['images'][f'{cam.name}_depth'] = []
        data_dict['observations']['images'][f'{cam.name}_intrinsic'] = []
        data_dict['observations']['images'][f'{cam.name}_extrinsic'] = []
    attr_dict = {
        'sim': True,
    }
    config_dict = {
        'observations':
            {
                'images': {}
            }
    }
    for cam_idx, cam in enumerate(gui.cams):
        color_save_kwargs = {
            'chunks': (1, cam.height, cam.width, 3), # (1, 480, 640, 3)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint8',
        }
        depth_save_kwargs = {
            'chunks': (1, cam.height, cam.width), # (1, 480, 640)
            'compression': 'gzip',
            'compression_opts': 9,
            'dtype': 'uint16',
        }
        config_dict['observations']['images'][f'{cam.name}_color'] = color_save_kwargs
        config_dict['observations']['images'][f'{cam.name}_depth'] = depth_save_kwargs

    while timesteps < 1000:
        print(f"\n==== STEP {timesteps} ====")
        # re-initialize action each step
        action = np.zeros(arm_dof+1)
        # ─── Before control update ─────────────────────────────────────
        # 1) EE current pose in world
        ee_pose_W = env.palm_link.get_pose().to_transformation_matrix()
        pos_W = np.round(ee_pose_W[:3,3], 3)
        rpy_W = np.round(transforms3d.euler.mat2euler(ee_pose_W[:3,:3], axes='sxyz'), 3)
        # print("Initial:")
        # print(f"  EE pose W: pos={pos_W}, rpy={rpy_W}")

        # 2) EE in robot frame
        base_W = env.robot.get_pose().to_transformation_matrix()
        ee_pose_B = np.linalg.inv(base_W) @ ee_pose_W
        pos_B = np.round(ee_pose_B[:3,3], 3)
        rpy_B = np.round(transforms3d.euler.mat2euler(ee_pose_B[:3,:3], axes='sxyz'), 3)
        # print(f"  EE pose B: pos={pos_B}, rpy={rpy_B}")

        # 3) qpos from env
        qpos_env = np.round(env.robot.get_qpos(), 3)
        # print(f"  qpos_env: {qpos_env}")

        # 4) IK‐computed qpos for that same EE pose
        target_cart_B = np.concatenate([pos_B, rpy_B, [qpos_env[arm_dof]]])
        qpos_ik_init = np.round(kin_helper.compute_ik_sapien(qpos_env, target_cart_B), 3)
        # print(f"  qpos_ik (hold current EE): {qpos_ik_init}")

        # 5) difference
        diff_init = np.round(qpos_ik_init - qpos_env, 3)
        # print(f"  Δqpos (ik − env): {diff_init}\n")

        # ─── Compute new action ─────────────────────────────────────────
        cartisen_action, quit = scripted_policy.single_trajectory(
            env, env.palm_link.get_pose(), mode=mode
        )
        # print("Next step requested:")
        # print(f"  cartisen_action (W): {np.round(cartisen_action, 3)}")

        # 6) Desired EE in W + B
        des_W = np.round(cartisen_action[:3], 3)
        # print(f"  Desired EE pos W: {des_W}")
        des_B = np.round(transform_action_from_world_to_robot(
            cartisen_action, env.robot.get_pose()
        )[:3], 3)
        # print(f"  Desired EE pos B: {des_B}")

        # 7) IK → qpos_target
        target_cart_B = np.concatenate([
            des_B,
            np.round(cartisen_action[3:6], 3),
            [cartisen_action[6]]
        ])
        qpos_target = np.round(kin_helper.compute_ik_sapien(qpos_env, target_cart_B), 3)
        print(f"  qpos_target (IK): {qpos_target}\n")

        if quit:
            break

        # existing sim‐step
        action[:arm_dof]    = qpos_target[:arm_dof]
        action[arm_dof:]    = cartisen_action[6]
        obs, reward, done, _ = env.step(action[:arm_dof+1])
        rgbs, depths        = gui.render(depth=True)

        # ─── After env.step ───────────────────────────────────────────────
        qpos_post = np.round(env.robot.get_qpos(), 3)
        delta_post = np.round(qpos_post - qpos_target, 3)
        # print("After env.step():")
        # print(f"  qpos_post: {qpos_post}")
        # print(f"  Δqpos post-step (post − target): {delta_post}")

        # your existing data collection:
        data_dict['observations']['joint_pos'].append(env.robot.get_qpos()[:-1])
        data_dict['observations']['joint_vel'].append(env.robot.get_qvel()[:-1])
        data_dict['observations']['full_joint_pos'].append(env.robot.get_qpos())
        data_dict['observations']['robot_base_pose_in_world'].append(
            env.robot.get_pose().to_transformation_matrix())
        ee_translation = env.palm_link.get_pose().p
        ee_rotation    = transforms3d.euler.quat2euler(
            env.palm_link.get_pose().q, axes='sxyz')
        ee_gripper     = env.robot.get_qpos()[arm_dof]
        ee_pos         = np.concatenate([ee_translation, ee_rotation, [ee_gripper]])
        ee_vel         = np.concatenate([
            env.palm_link.get_velocity(),
            env.palm_link.get_angular_velocity(),
            env.robot.get_qvel()[arm_dof:arm_dof+1]
        ])
        data_dict['observations']['ee_pos'].append(ee_pos)
        data_dict['observations']['ee_vel'].append(ee_vel)
        data_dict['joint_action'].append(action.copy())
        data_dict['cartesian_action'].append(cartisen_action.copy())
        for cam_idx, cam in enumerate(gui.cams):
            data_dict['observations']['images'][f'{cam.name}_color'].append(rgbs[cam_idx])
            data_dict['observations']['images'][f'{cam.name}_depth'].append(depths[cam_idx])
            data_dict['observations']['images'][f'{cam.name}_intrinsic'].append(
                cam.get_intrinsic_matrix())
            data_dict['observations']['images'][f'{cam.name}_extrinsic'].append(
                cam.get_extrinsic_matrix())

        writer.writerow([
                            timesteps,
                            *pos_W, *rpy_W,
                            qpos_env.tolist(),
                            qpos_ik_init.tolist(),
                            diff_init.tolist(),
                            *des_W,
                            qpos_target.tolist(),
                            qpos_post.tolist(),
                            delta_post.tolist()
                        ])

        timesteps += 1

    if reward < 1:
        print("Failed at episode {}".format(episode_idx))
    data_dict = stack_dict(data_dict)
    save_dict_to_hdf5(data_dict, config_dict, dataset_path, attr_dict=attr_dict)
    if not gui.headless:
        gui.viewer.close()
        cv2.destroyAllWindows()
    csv_file.close()
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
