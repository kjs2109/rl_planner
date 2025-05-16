import os 

import torch 
import numpy as np 
from shapely.geometry import LinearRing, Polygon, Point  
from typing import OrderedDict

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
SEED=42 


################# vehicle ################# 
WHEEL_BASE = 1.9*1.1  # wheelbase
FRONT_HANG = 0.32*1.1  # front hang length
REAR_HANG = 0.32*1.1  # rear hang length
LENGTH = WHEEL_BASE+FRONT_HANG+REAR_HANG
WIDTH = (0.1+1.465+0.1)*1.1  # width 

VehicleBox = LinearRing([
    (-REAR_HANG, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE, -WIDTH/2), 
    (FRONT_HANG + WHEEL_BASE,  WIDTH/2),
    (-REAR_HANG,  WIDTH/2)])

Obs_WHEEL_BASE = 2.8  
Obs_FRONT_HANG = 0.96  # front hang length
Obs_REAR_HANG = 0.93 
Obs_LENGTH = Obs_WHEEL_BASE+Obs_FRONT_HANG+Obs_REAR_HANG
Obs_WIDTH = 1.94  

ObsVehicleBox = LinearRing([
    (-Obs_REAR_HANG, -Obs_WIDTH/2), 
    (Obs_FRONT_HANG + Obs_WHEEL_BASE, -Obs_WIDTH/2), 
    (Obs_FRONT_HANG + Obs_WHEEL_BASE,  Obs_WIDTH/2),
    (-Obs_REAR_HANG,  Obs_WIDTH/2)])

COLOR_POOL = [
    (30, 144, 255, 255), # dodger blue
    (255, 127, 80, 255), # coral
    (255, 215, 0, 255) # gold
]

VALID_SPEED = [-2.5, 2.5]
VALID_STEER = [-0.4887, 0.4887] 
# VALID_ACCEL = [-1.0, 1.0]
# VALID_ANGULAR_SPEED = [-0.5, 0.5]

NUM_STEP = 10
STEP_LENGTH = 0.05


################# env ################# 
FPS = 100
K = 12 # the render scale
WIN_W = 600
WIN_H = 800

# camera 
USE_IMG = True
R = 2
OBS_W = 256*R 
OBS_H = 256*R

# lidar 
USE_LIDAR = True
LIDAR_RANGE = 20.0
LIDAR_NUM = 200
ORIGIN = Point((0,0))

# config 
BG_COLOR = (255, 255, 255, 255)
START_COLOR = (100, 149, 237, 255)
DEST_COLOR = (69, 139, 0, 255)
OBSTACLE_COLOR = (255, 0, 0, 255)
NON_DRIVABLE_COLOR = (200, 200, 200, 255)
RENDER_TRAJ = True
TRAJ_COLOR_HIGH = (10, 10, 200, 255)
TRAJ_COLOR_LOW = (10, 10, 10, 10)
TRAJ_RENDER_LEN = 20
TRAJ_COLORS = list(map(tuple,np.linspace(\
    np.array(TRAJ_COLOR_LOW), np.array(TRAJ_COLOR_HIGH), TRAJ_RENDER_LEN, endpoint=True, dtype=np.uint8)))

# constraints 
TOLERANT_TIME = 200
MAX_DIST_TO_DEST = 30

# reward 
REWARD_RATIO = 0.1
REWARD_WEIGHT = OrderedDict({'time_cost':1,
                            'rs_dist_reward':0,
                            'dist_reward':5,
                            'angle_reward':0,
                            'box_union_reward':10,})


################# model ################# 
GAMMA = 0.98 
BATCH_SIZE = 8192 
LR = 0.00001 
TAU = 0.1 
MAX_TRAIN_STEP = 1000000 
ORTHOGONAL_INIT = True 
LR_DECAY = False 
UPDATE_IMG_ENCODE = False 

C_CONV = [4, 8,] 
SIZE_FC = [256] 

ATTENTION_CONFIG = {
    'depth': 1, 
    'heads': 8, 
    'dim_head': 32, 
    'mlp_dim': 128, 
    'hidden_dim': 128 
}
USE_ATTENTION = True 

# action mask 
USE_ACTION_MASK = False 
PRECISION = 10
step_speed = 1
discrete_actions = []
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, step_speed])
for i in np.arange(VALID_STEER[-1], -(VALID_STEER[-1] + VALID_STEER[-1]/PRECISION), -VALID_STEER[-1]/PRECISION):
    discrete_actions.append([i, -step_speed])
N_DISCRETE_ACTION = len(discrete_actions)

ACTOR_CONFIGS = {
    'n_modal':2+int(USE_IMG),
    'lidar_shape':LIDAR_NUM,
    'target_shape':5,
    'action_mask_shape':N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape':(3,64*R,64*R) if USE_IMG else None,
    'output_size':2,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'img_conv_layers':C_CONV,
    'img_linear_layers':SIZE_FC,
    'k_img_conv':3,
    'orthogonal_init':True,
    'use_tanh_output':True,
    'use_tanh_activate':True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}

CRITIC_CONFIGS = {
    'n_modal':2+int(USE_IMG),
    'lidar_shape':LIDAR_NUM,
    'target_shape':5,
    'action_mask_shape':N_DISCRETE_ACTION if USE_ACTION_MASK else None,
    'img_shape':(3,64*R,64*R) if USE_IMG else None,
    'output_size':1,
    'embed_size':128,
    'hidden_size':256,
    'n_hidden_layers':3,
    'n_embed_layers':2,
    'img_conv_layers':C_CONV,
    'img_linear_layers':SIZE_FC,
    'k_img_conv':3,
    'orthogonal_init':True,
    'use_tanh_output':False,
    'use_tanh_activate':True,
    'attention_configs': ATTENTION_CONFIG if USE_ATTENTION else None,
}