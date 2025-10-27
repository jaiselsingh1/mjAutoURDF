import mujoco 
from mujoco import mjtJoint, mjtObj
import math, os, numpy as np
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
        
        