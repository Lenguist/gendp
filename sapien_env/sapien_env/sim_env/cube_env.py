# cube_env.py (in sapien_env/sim_env)
import numpy as np
import sapien.core as sapien
import transforms3d.euler
from sapien_env.sim_env.base import BaseSimulationEnv

class CubePickEnv(BaseSimulationEnv):
    def __init__(self, use_gui=True, frame_skip=5, use_ray_tracing=True, **renderer_kwargs):
        # 1) Initialize base sim (engine, renderer)
        super().__init__(
            use_gui=use_gui,
            frame_skip=frame_skip,
            use_ray_tracing=use_ray_tracing,
            **renderer_kwargs
        )
        # 2) Create the Sapien scene
        scene_cfg = sapien.SceneConfig()
        self.scene = self.engine.create_scene(config=scene_cfg)
        self.scene.set_timestep(0.004)

        # 3) Add a table
        self.table = self.create_table(
            table_height=0.6,
            table_half_size=[0.5, 0.7, 0.025]
        )

        # 4) Build a 0.06 m white cube
        self.cube_side = 0.03
        cube_side = self.cube_side
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(half_size=[cube_side, cube_side, cube_side], color=[1, 1, 1])
        builder.add_box_collision(half_size=[cube_side, cube_side, cube_side])
        self.cube = builder.build("white_cube")

    def reset_env(self):
        # Place cube at random X/Y on table (Z = half-height = 0.05)
        x = self.np_random.uniform(0.1, 0.2)
        y = self.np_random.uniform(-0.2, -0.1)
        pos = np.array([x, y, self.cube_side])  # z = half-height
        quat = transforms3d.euler.euler2quat(0, 0, 0)
        self.cube.set_pose(sapien.Pose(pos, quat))

    def put_cube_at_pos(self, x, y):
        # Place cube at random X/Y on table (Z = half-height = 0.05)
        pos = np.array([x, y, self.cube_side])  # z = half-height
        quat = transforms3d.euler.euler2quat(0, 0, 0)
        self.cube.set_pose(sapien.Pose(pos, quat))

    def get_init_poses(self):
        # For HDF5 logger: return cube's initial transform
        return np.stack([self.cube.get_pose().to_transformation_matrix()])