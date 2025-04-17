import numpy as np
from pyquaternion import Quaternion
import transforms3d.euler
import sapien.core as sapien

from sapien_env.rl_env.cube_pick_env import CubePickRLEnv

class SingleArmPolicy:
    """
    Scripted straight-line pick policy for CubePickRLEnv.
    """
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.trajectory = None

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        curr_q = Quaternion(curr_quat)
        next_q = Quaternion(next_quat)
        quat = Quaternion.slerp(curr_q, next_q, t_frac).elements
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def single_trajectory(self, env: CubePickRLEnv, ee_link_pose, mode='straight'):
        # Generate trajectory once
        if self.step_count == 0:
            self.generate_trajectory(env, ee_link_pose, mode)

        # Pop to current waypoint
        if self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)
        # If done
        if len(self.trajectory) == 0:
            return None, True
        # Interpolate to next
        next_wp = self.trajectory[0]
        xyz, quat, gripper = self.interpolate(self.curr_waypoint, next_wp, self.step_count)

        # Optional noise
        if self.inject_noise:
            xyz += np.random.uniform(-0.01, 0.01, size=xyz.shape)

        self.step_count += 1
        # Build action: [xyz(3), euler(3), gripper(1)]
        action = np.zeros(7)
        euler = transforms3d.euler.quat2euler(quat, axes='sxyz')
        action[0:3] = xyz
        action[3:6] = euler
        action[6] = gripper
        return action, False

    def generate_trajectory(self, env: CubePickRLEnv, ee_link_pose, mode='straight'):
        # world pose of cube
        cube_pose = env.cube.get_pose()
        # Define key poses
        pre_grasp = cube_pose.p + np.array([0.0, 0.0, 0.2])
        grasp = cube_pose.p + np.array([0.0, 0.0, 0.05])
        leave = cube_pose.p + np.array([0.0, 0.0, 0.3])
        # Keep orientation constant (current ee orientation)
        quat = ee_link_pose.q
        # Trajectory waypoints (t, xyz, quat, gripper)
        if mode == 'straight':
            self.trajectory = [
                {'t': 0,   'xyz': ee_link_pose.p, 'quat': quat, 'gripper': 0.09},
                {'t': 20,  'xyz': pre_grasp,       'quat': quat, 'gripper': 0.09},
                {'t': 60,  'xyz': grasp,           'quat': quat, 'gripper': 0.09},
                {'t': 80,  'xyz': grasp,           'quat': quat, 'gripper': 0.00},
                {'t': 100, 'xyz': grasp,           'quat': quat, 'gripper': 0.00},
                {'t': 140, 'xyz': pre_grasp,       'quat': quat, 'gripper': 0.00},
                {'t': 200, 'xyz': leave,           'quat': quat, 'gripper': 0.00},
                {'t': 220, 'xyz': leave,           'quat': quat, 'gripper': 0.00},
            ]
        else:
            raise RuntimeError(f"Mode '{mode}' not implemented for cube pick.")
