import mujoco 
import math, os, numpy as np
from typing import NamedTuple
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

class SimEnv(MujocoEnv):
        metadata = {"render_modes":["human", "rgb_array"], "render_fps": 50}

        def __init__(self, ):


# set the base of the robot statically
# legged robot (quadreped); we set free motion would be cool 