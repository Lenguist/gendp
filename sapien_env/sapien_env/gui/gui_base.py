# File: sapien_env/gui/gui_base.py

from typing import List, Dict, Callable, Union

import cv2
import numpy as np
import sapien.core as sapien
import torch.utils.dlpack
from sapien.core import Pose
from sapien.core.pysapien import renderer as R
from sapien.utils import Viewer
from sapien_env.utils.render_scene_utils import add_mesh_to_renderer

# Your four calibrated camera poses
YX_TABLE_TOP_CAMERAS = {
    "235422302222": dict(
        position=np.array([-0.30624305,  0.29016147,  0.64103958]),
        rotation=np.array([ 0.765023271273258, -0.538866041309784, -0.155966151460057, -0.316286805814317]),
        name="cam_235422302222",
    ),
    "239222300740": dict(
        position=np.array([-0.06265331, -0.07071282,  1.03064751]),
        rotation=np.array([ 0.749170836880919, -0.442580422730515,  0.231968764735815,  0.434805840312387]),
        name="cam_239222300740",
    ),
    "239222303153": dict(
        position=np.array([ 0.13656804, -0.07055096,  0.94455591]),
        rotation=np.array([ 0.410502491166933, -0.218048756674483,  0.411110745063491,  0.784174980314850]),
        name="cam_239222303153",
    ),
    "239222303404": dict(
        position=np.array([ 0.11300572,  0.10515995,  0.48421512]),
        rotation=np.array([ 0.349777290835052, -0.092179765600059, -0.307208868157601, -0.880216705678400]),
        name="cam_239222303404",
    ),
}

# A default top‑view camera constant for compatibility; picks the first calibrated view
DEFAULT_TABLE_TOP_CAMERAS = {
    "cam_235422302222": YX_TABLE_TOP_CAMERAS["235422302222"]
}

def depth_to_vis_depth(depth):
    # :param depth: depth image in np.uint16
    # :return: visualized depth image in np.uint8
    depth_f = depth.astype(np.float32) / 1000.0
    norm = (depth_f - depth_f.min()) / (depth_f.max() - depth_f.min())
    vis = np.clip(norm * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)

class GUIBase:
    def __init__(
        self,
        scene: sapien.Scene,
        renderer: Union[sapien.VulkanRenderer, sapien.KuafuRenderer],
        resolution=(640, 480),
        window_scale=0.5,
        headless=False,
    ):
        # Basic setup
        self.scene = scene
        self.renderer = renderer
        self.headless = headless
        self.resolution = resolution
        self.window_scale = window_scale
        self.cams: List[sapien.CameraEntity] = []
        self.cam_mounts: List[sapien.ActorBase] = []
        self.keydown_map: Dict[str, Callable] = {}

        # Ray tracing context
        self.use_ray_tracing = isinstance(renderer, sapien.KuafuRenderer)
        if not self.use_ray_tracing:
            self.context: R.Context = renderer._internal_context
            self.render_scene: R.Scene = scene.get_renderer_scene()._internal_scene
            self.nodes: List[R.Node] = []
        self.sphere_nodes: Dict[str, List[R.Node]] = {}
        self.sphere_model: Dict[str, R.Model] = {}

        # Viewer
        if not self.use_ray_tracing and not headless:
            self.viewer = Viewer(renderer)
            self.viewer.set_scene(scene)
            self.viewer.toggle_axes(False)
            self.viewer.toggle_camera_lines(False)
            self.viewer.set_camera_xyz(-0.3, 0, 0.5)
            self.viewer.set_camera_rpy(0, -1.4, 0)

        # Visualization material
        self.viz_mat_hand = self.renderer.create_material()
        self.viz_mat_hand.set_base_color(np.array([0.96, 0.75, 0.69, 1]))
        self.viz_mat_hand.set_specular(0)
        self.viz_mat_hand.set_metallic(0.8)
        self.viz_mat_hand.set_roughness(0)

        # --- Instantiate exactly 4 calibrated cameras ---
        for params in YX_TABLE_TOP_CAMERAS.values():
            self.create_camera_from_pos_rot(
                position=params["position"],
                rotation=params["rotation"],
                name=params["name"],
            )

    def create_camera(self, position, look_at_dir, right_dir, name):
        builder = self.scene.create_actor_builder()
        builder.set_mass_and_inertia(1e-2, Pose(np.zeros(3)), np.ones(3)*1e-4)
        mount = builder.build_static(name=f"{name}_mount")
        cam = self.scene.add_mounted_camera(
            name, mount, Pose(),
            width=self.resolution[0], height=self.resolution[1],
            fovy=0.9, fovx=0.9, near=0.1, far=10,
        )

        # Orientation from look_at and right_dir
        look = look_at_dir / np.linalg.norm(look_at_dir)
        right = right_dir - np.dot(right_dir, look)*look
        right = right / np.linalg.norm(right)
        up = np.cross(look, -right)
        mat = np.stack([look, -right, up, position], axis=1)
        tf = np.vstack([mat, [0,0,0,1]])
        mount.set_pose(Pose.from_transformation_matrix(tf))

        self.cams.append(cam)
        self.cam_mounts.append(mount)

    def create_camera_from_pos_rot(self, position, rotation, name):
        builder = self.scene.create_actor_builder()
        builder.set_mass_and_inertia(1e-2, Pose(np.zeros(3)), np.ones(3)*1e-4)
        mount = builder.build_static(name=f"{name}_mount")
        cam = self.scene.add_mounted_camera(
            name, mount, Pose(),
            width=self.resolution[0], height=self.resolution[1],
            fovy=0.9, fovx=0.9, near=0.1, far=10,
        )

        # Directly apply given pose
        pose = sapien.Pose(p=position, q=rotation)
        mount.set_pose(pose)

        self.cams.append(cam)
        self.cam_mounts.append(mount)

    def _fetch_all_views(self, use_bgr=False, render_depth=False):
        views = []
        depths = [] if render_depth else None
        for cam in self.cams:
            cam.take_picture()
            rgb = (np.clip(cam.get_float_texture("Color")[..., :3], 0, 1)*255).astype(np.uint8)
            if use_bgr:
                rgb = rgb[..., ::-1]
            views.append(rgb)
            if render_depth:
                depth = (-cam.get_float_texture("Position")[..., 2]*1000).astype(np.uint16)
                depths.append(depth)
        return (views, depths) if render_depth else views

    def take_single_view(self, camera_name: str, use_bgr=False, render_depth=False):
        import cupy
        for cam in self.cams:
            if cam.get_name() == camera_name:
                cam.take_picture()
                rgb = np.clip(cupy.asnumpy(cupy.from_dlpack(cam.get_dl_tensor("Color")))[..., :3], 0, 1)*255
                rgb = rgb.astype(np.uint8)
                if use_bgr:
                    rgb = rgb[..., ::-1]
                if render_depth:
                    depth = (-cupy.asnumpy(cupy.from_dlpack(cam.get_dl_tensor("Position")))[..., 2]*1000).astype(np.uint16)
                    return rgb, depth
                return rgb
        raise RuntimeError(f"Camera name not found: {camera_name}")

    def render(self, horizontal=True, depth=False):
        self.scene.update_render()
        if not self.headless:
            self.viewer.render()
            for k, act in self.keydown_map.items():
                if self.viewer.window.key_down(k):
                    act()
        if depth:
            views, depths = self._fetch_all_views(use_bgr=True, render_depth=True)
        else:
            views = self._fetch_all_views(use_bgr=True, render_depth=False)

        # Tile and display
        pad = np.ones([views[0].shape[0], 200, 3], dtype=np.uint8)*255 if horizontal else np.ones([200, views[0].shape[1], 3], dtype=np.uint8)*255
        tiles = [views[0]]
        depth_tiles = [depth_to_vis_depth(depths[0])] if depth else None
        for i in range(1, len(views)):
            tiles += [pad, views[i]]
            if depth:
                depth_tiles += [pad, depth_to_vis_depth(depths[i])]
        axis = 1 if horizontal else 0
        img = np.concatenate(tiles, axis=axis)
        h, w = img.shape[:2]
        resized = cv2.resize(img, (int(w*self.window_scale), int(h*self.window_scale)))
        cv2.imshow("Monitor", resized)
        if depth:
            dimg = np.concatenate(depth_tiles, axis=axis)
            dh, dw = dimg.shape[:2]
            dres = cv2.resize(dimg, (int(dw*self.window_scale), int(dh*self.window_scale)))
            cv2.imshow("Depth", dres)
        cv2.waitKey(1)
        return (views, depths) if depth else views

    def close(self):
        for cam in self.cams:
            self.scene.remove_camera(cam)
        self.cams = []
        self.scene = None
