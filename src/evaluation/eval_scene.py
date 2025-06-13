import os
import time 
import argparse 

import numpy as np 
import torch 
from torch.utils.tensorboard import SummaryWriter 

from model.agent.sac_agent import SACAgent as SAC 
from model.agent.agent import PlanningAgent 
from env.campus_env_base import CampusEnvBase 
from env.env_wrapper import CampusEnvWrapper 
from configs import (SEED, ACTOR_CONFIGS, CRITIC_CONFIGS,)
from evaluation.eval_utils import eval_agent


if __name__=="__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str, default=None) 
    parser.add_argument('--map_path', type=str, default='../data/lanelet2_map/carla_lanelet2_map_v1.osm') 
    parser.add_argument('--trajectory_path', type=str, default='../data/trajectory/carla_trajectory_data_v1.json')
    parser.add_argument('--eval_episode', type=int, default=1000)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()

    checkpoint_path = args.ckpt_path
    print('ckpt path: ',checkpoint_path)
    verbose = args.verbose

    if args.visualize: 
        raw_env = CampusEnvBase(fps=100, map_path=args.map_path, trajectory_path=args.trajectory_path)
    else:
        raw_env = CampusEnvBase(fps=100, render_mode="rgb_array", map_path=args.map_path, trajectory_path=args.trajectory_path)
    env = CampusEnvWrapper(raw_env) 

    relative_path = '.'
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = relative_path+'/log/eval/%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    configs_file = os.path.join(save_path, 'configs.txt')
    with open(configs_file, 'w') as f:
        f.write(str(checkpoint_path))

    writer = SummaryWriter(save_path)
    print("You can track the training process by command 'tensorboard --logdir %s'" % save_path)

    env.action_space.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    actor_params = ACTOR_CONFIGS
    critic_params = CRITIC_CONFIGS
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
    print('observation_space:',env.observation_space)

    rl_agent = SAC(configs) 
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print('load pre-trained model!')

    eval_episode = args.eval_episode 
    with torch.no_grad(): 
        eval_agent(env, rl_agent, episode=eval_episode, log_path=save_path, multi_mode=False, mode='normal_parking') 


    