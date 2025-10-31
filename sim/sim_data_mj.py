import mujoco 
from mujoco import mjtJoint, mjtObj
import math, os, numpy as np, imageio.v3 as iio 
from typing import NamedTuple
import tyro 
from dataclasses import dataclass, field
# from gymnasium.envs.mujoco.mujoco_env import MujocoEnv


@dataclass(slots=True)
class joint_info:
        jnt_id: int 
        qpos_adr: int 
        limit: tuple[float, float]

@dataclass(slots=True) # slots help with memory usage?
class camera_spec:
        pos: np.ndarray 
        lookat: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))
        up: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1.]))
        fov: float = 60.0
        near: float = 0.1
        far: float = 4.0
        aspect: float = 1.0
        
        def apply_to_freecam(self, renderer_cam):
                v = self.pos - self.lookat
                dist = float(np.linalg.norm(v) + 1e-12)
                az   = math.degrees(math.atan2(v[1], v[0]))  # yaw around z 
                el   = math.degrees(math.asin(v[2] / dist))  # pitch 

                renderer_cam.lookat[:] = self.lookat
                renderer_cam.distance  = dist
                renderer_cam.azimuth   = az
                renderer_cam.elevation = el
                # MuJoCo handles intrinsics; depth = meters 

class SimEnv:
        """
        - load model
        - keep joint info (names, qpos addresses, limits)
        - sample/apply joint configurations
        - generate RGB-D from N cameras around the robot
        """
        def __init__(
                self, 
                model_path: str, 
                dof: int,
                radius: float,
                num_cameras: int,
                cam_angle_deg: float = 20.0,
                width: int = 800,
                height: int = 800,
                settle_physics: bool = False,
                settle_steps: int = 120        
        ):

           self.model = mujoco.MjModel.from_xml_path(model_path)
           self.data = mujoco.MjData(self.model)

           self.w = width 
           self.h = height 
           self.renderer = mujoco.Renderer(self.model, height=self.h, width=self.w)
           self.renderer.enable_freecamera()  # manually set up free camera pose 
           
           self.dof = dof
           self.settle_physics = settle_physics
           self.settle_steps = settle_steps

           # map for hinge joints 
           self.joint_map = self._build_joint_map()
           # deterministic joint order (based on pybullet joint list)
           self.joint_names = list(self.joint_map.keys())
           # the first "dof" joints are controlled
           self.dof_names = self.joint_names[:self.dof]

           # for the actuated subset 
           self.joint_limits = np.array(
                  [self.joint_map[name].limit for name in self.dof_names]
           ) 

           self.cameras = self._make_cameras(radius, num_cameras, cam_angle_deg)
           # set the simulation forward / consistent 
           mujoco.mj_forward(self.model, self.data)


        def _build_joint_map(self) -> dict[str, joint_info]:
               """Iterate all joints, pick hinge (revolute) joints, save their MuJoCo indices, qpos addresses, and limits."""
               joint_map: dict[str, joint_info] = {}
               for j in range(self.model.njnt):
                        if self.model.jnt_type[j] != mjtJoint.mjJNT_HINGE:
                                continue
                        name = mujoco.mj_id2name(self.model, mjtObj.mjOBJ_JOINT, j)
                        qadr = self.model.jnt_qposadr[j]

                        if self.model.jnt_limited[j]:
                                lo, hi = self.model.jnt_range[j]  # radians
                        else:
                                # if unlimited joints then fall back within a safe range 
                                lo, hi = -np.pi, np.pi
                        
                        joint_map[name] = joint_info(
                                jnt_id=j, 
                                qpos_adr=qadr, 
                                limit=(float(lo), float(hi))
                        )
               return joint_map
        
        def _make_cameras(self, radius: float, num_cameras: int, cam_angle_deg: float):
                """sample camera poses on a ring/ hemisphere around the robot"""
                if num_cameras < 20:
                        theta = np.linspace(0, 2 * np.pi, num_cameras, endpoint = False)
                        phi = np.deg2rad(cam_angle_deg) * np.ones_like(theta)
                else:
                        theta = np.random.rand(num_cameras) * 2 * np.pi
                        phi = np.random.rand(num_cameras) * (0.5 * np.pi)
                
                xs = radius * np.cos(theta) * np.cos(phi)
                ys = radius * np.sin(theta) * np.cos(phi)
                zs = radius * np.sin(phi)

                cams = []
                for x, y, z in zip(xs, ys, zs):
                        cams.append(
                                camera_spec(pos = np.array([x, y, z], dtype = float),)
                        )
                return cams 
        
        def set_joint_positions(self, commands: np.ndarray):
                """teleport the joints to target angles (radians), update mj_forward, return dict of final joint positions"""
                # set commanded joints 
                for val, jname in zip(commands, self.dof_names):
                        info = self.joint_map[jname]
                        lo, hi = info.limit 
                        self.data.qpos[info.qpos_adr] = float(np.clip(val, lo, hi))
                
                mujoco.mj_forward(self.model, self.data)

                # dict of final joint positions (including the non commanded ones)
                joint_positions = {}
                for jname in self.joint_names:
                        info = self.joint_map[jname]
                        joint_positions[jname] = float(self.data.qpos[info.qpos_adr])
                return joint_positions
        
        def render_camera(self, camera_index: int):
                """set mujoco free camera to cameras[camera_index] then return (rgb, depth)"""
                cam = self.cameras[camera_index]
                cam.apply_to_freecam(self.renderer.camera)
                self.renderer.update_scene(self.data)

                rgb = self.renderer.render() # (h, w, 3)
                depth = self.renderer.read_depth() # (h, w), float32 in meters 
                
                return rgb, depth, cam # camera return for base knowledge 

        def reset(self):
                mujoco.mj_resetData(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)

def angle_list(num_step, dof, joint_limits, seed_i = 0):
        np.random.seed(seed_i)
        jl_deg = joint_limits * 180 / np.pi 
        start = (jl_deg[:,0] + jl_deg[:, 1]) / 2.0 
        angles = [] 

        for _ in range(num_step):
                tgt = [np.random.uniform(lo, hi) for (lo, hi) in jl_deg]
                angles.append(np.deg2rad(tgt))
        return np.asarray(angles)

def save_rgb_depth(out_dir: str, step_id: int, cam_id: int, rgb, depth_m):
        os.makedirs(out_dir, exist_ok=True)
        iio.imwrite(os.path.join(out_dir, f"{step_id:04d}_cam{cam_id:02d}_rgb.png"), rgb)
        depth_mm = np.clip(depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        iio.imwrite(os.path.join(out_dir, f"{step_id:04d}_cam{cam_id:02d}_depth.png"), depth_mm)

@dataclass
class Config:
        # model + env 
        model_path: str 
        dof: int = 6 
        radius: float = 1.5 
        num_cameras: int = 8 
        cam_angle_deg: float = 20.0 
        width: int = 800
        height: int = 800
        settle_physics: bool = False
        settle_steps: int = 120

        # dataset
        out_dir: str = "data_rgbd"
        num_steps: int = 10
        seed: int = 0

def main(cfg: Config):
    env = SimEnv(
        model_path=cfg.model_path,
        dof=cfg.dof,
        radius=cfg.radius,
        num_cameras=cfg.num_cameras,
        cam_angle_deg=cfg.cam_angle_deg,
        width=cfg.width,
        height=cfg.height,
        settle_physics=cfg.settle_physics,
        settle_steps=cfg.settle_steps,
    )

    a_list = angle_list(
        num_step=cfg.num_steps,
        dof=cfg.dof,
        joint_limits=env.joint_limits,
        seed_i=cfg.seed,
    )

    for step_id in range(cfg.num_steps):
        env.set_joint_positions(a_list[step_id])
        for cam_id in range(cfg.num_cameras):
            rgb, depth, cam = env.render_camera(cam_id)
            save_rgb_depth(cfg.out_dir, step_id, cam_id, rgb, depth)

if __name__ == "__main__":
    main(tyro.cli(Config))
        
"""python sim_data_mj.py \
  --model-path ../menagerie/franka_fr3/scene.xml \
  --dof 6 --radius 1.5 --num-cameras 8 \
  --width 640 --height 480 \
  --out-dir data_rgbd --num-steps 10 --seed 0
"""