import numpy as np 
from gym import Wrapper 

from env.campus_env_base import CampusEnvBase 
from env.vehicle import Status 

from configs import REWARD_WEIGHT, REWARD_RATIO


def reward_shaping(*args):
    # info = {'reward_info': reward_info, 'status': status} 
    obs, reward_info, status, info = args 
    if status == Status.CONTINUE: 
        reward = 0 
        for reward_type in REWARD_WEIGHT.keys():
            reward += REWARD_WEIGHT[reward_type]*reward_info[reward_type] 
    elif status == Status.OUTBOUND: 
        reward = -50 
    elif status == Status.OUTTIME: 
        reward = -5 
    elif status == Status.ARRIVED: 
        reward = 50 
    elif status == Status.COLLIDED: 
        reward = -10 
    else:
        print(status) 
        print('Never reach here !!!') 
    reward *= REWARD_RATIO 
    info['status'] = status 
    return obs, reward, status, info 

def action_rescale(action, action_space, raw_action_range=(-1,1), explore=True, epsilon=0.0):
    action = np.clip(action, *raw_action_range)
    action = action * (action_space.high - action_space.low) / 2 + (action_space.high + action_space.low) / 2
    if explore and np.random.random() < epsilon:
        action = action_space.sample()
    return action

def observation_rescale(obs): 
    if obs['img'] is not None: 
        obs['img'] = obs['img'].transpose(2, 0, 1) 
    return obs 


class CampusEnvWrapper(Wrapper):
    def __init__(self, env, action_func=action_rescale, reward_func=reward_shaping, observation_func=observation_rescale):
        super().__init__(env) 
        self.reward_func = reward_func 
        self.action_func = action_func 
        self.obs_func = observation_func 
        self.observation_shape = {k:self.env.observation_space[k].shape for k in self.env.observation_space} 
        if 'img' in self.observation_shape: 
            w, h, c = self.observation_shape['img'] 
            self.observation_shape['img'] = (c, w, h) 

    def step(self, action=None):
        if action is None: 
            return self.obs_func(self.env.step()[0])
        
        action = self.action_func(action, self.env.action_space) 
        returns = self.env.step(action) 
        obs, reward, status, info = self.reward_func(*returns) 
        obs = self.obs_func(obs) 
        done = False if status == Status.CONTINUE else True 

        return obs, reward, done, info 
    
    def reset(self, *args):
        obs = self.env.reset(*args) 
        return self.obs_func(obs)