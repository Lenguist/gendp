import csv
import numpy as np
from pyquaternion import Quaternion
import transforms3d.euler
import sapien.core as sapien
from sapien_env.rl_env.cube_pick_env import CubePickRLEnv

class SingleArmPolicy:
    """
    Replay a pre-recorded EE trajectory from CSV, skipping waypoints
    and holding each one for multiple steps.

    Args:
      inject_noise (bool): jitter the positions slightly each step
      csv_path    (str):  full path to your episode_XXXX.csv
      skip        (int):  take only every `skip`th row from the CSV
      hold        (int):  repeat each waypoint for `hold` simulation steps
    """
    GRIP_OPEN = 0.1
    GRIP_CLOSED = 0.65

    def __init__(self,
                 inject_noise: bool = False,
                 csv_path: str = "/home/maksymbondarenko/Desktop/gendp/episode_csvs/episode_0000.csv",
                 skip: int = 1,
                 hold: int = 1):
        self.inject_noise = inject_noise
        self.csv_path     = csv_path
        self.skip         = max(1, skip)
        self.hold         = max(1, hold)
        self.step_count   = 0
        self.trajectory   = []
        self.curr_waypoint = None

    @staticmethod
    def interpolate(curr_wp, next_wp, t):
        frac = (t - curr_wp["t"]) / (next_wp["t"] - curr_wp["t"])
        xyz  = curr_wp['xyz'] + (next_wp['xyz'] - curr_wp['xyz']) * frac
        q0   = Quaternion(curr_wp['quat'])
        q1   = Quaternion(next_wp['quat'])
        quat = Quaternion.slerp(q0, q1, frac).elements
        grip = curr_wp['gripper'] + (next_wp['gripper'] - curr_wp['gripper']) * frac
        return xyz, quat, grip

    def single_trajectory(self,
                          env: CubePickRLEnv,
                          ee_link_pose,
                          mode='straight'):
        # 1) Build once on the very first call
        if self.step_count == 0:
            self._build_trajectory(env, ee_link_pose)

        # 2) Advance waypoint when we've hit its timestamp
        if self.trajectory and self.trajectory[0]['t'] == self.step_count:
            self.curr_waypoint = self.trajectory.pop(0)

        # 3) Done?
        if not self.trajectory:
            return None, True

        # 4) Interpolate toward the next waypoint
        next_wp = self.trajectory[0]
        xyz, quat, grip = self.interpolate(self.curr_waypoint, next_wp, self.step_count)

        if self.inject_noise:
            xyz += np.random.uniform(-0.01, 0.01, size=xyz.shape)

        action = np.zeros(7, dtype=np.float32)
        action[0:3] = xyz
        action[3:6] = transforms3d.euler.quat2euler(quat, axes='sxyz')
        action[6]   = grip

        self.step_count += 1
        return action, False

    def _build_trajectory(self,
                          env: CubePickRLEnv,
                          ee_link_pose,
                          mode='straight'):
        # --- load and downsample CSV points ---
        raw_pts = []
        with open(self.csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                x,y,z = map(float, row[1:4])
                raw_pts.append(np.array([x,y,z],dtype=np.float32))

        if not raw_pts:
            raise RuntimeError(f"No points found in {self.csv_path}")

        pts = raw_pts[::self.skip]

        # --- compute offset so we start from the current EE ---
        actual_start = ee_link_pose.p
        offset       = actual_start - pts[0]

        # --- constant orientation & gripper value ---
        quat0 = ee_link_pose.q
        grip0 = float(env.robot.get_qpos()[env.arm_dof])

        # --- build timed waypoints ---
        self.trajectory = []
        # first “current” waypoint
        self.curr_waypoint = {
            't'      : 0,
            'xyz'    : pts[0] + offset,
            'quat'   : quat0,
            'gripper': grip0
        }

        # then each downsampled point, held for `hold` steps
        for idx, p in enumerate(pts):
            t_stamp = idx * self.hold
            self.trajectory.append({
                't'      : t_stamp,
                'xyz'    : p + offset,
                'quat'   : quat0,
                'gripper': grip0
            })

        print(f"Loaded trajectory ({len(pts)} pts, skip={self.skip}, hold={self.hold})")
        for wp in self.trajectory:
            print(f"  t={wp['t']:4d} → xyz={wp['xyz']}")

