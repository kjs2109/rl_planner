import math
import copy 
import numpy as np 
from enum import Enum 
from typing import Callable, List 
from shapely.geometry import Point, LinearRing 
from env.object_base import State 

from configs import (WHEEL_BASE, STEP_LENGTH, NUM_STEP, 
                     VALID_SPEED, VALID_STEER)


class Status(Enum):
    CONTINUE = 1 
    ARRIVED = 2 
    COLLIDED = 3 
    OUTBOUND = 4 
    OUTTIME = 5 
    

class KSModel(object):
    def __init__(self, wheel_base:float, step_len:float, n_step:int, speed_range:list, angle_range:list):
        self.wheel_base = wheel_base 
        self.step_len = step_len       # 0.05s 
        self.n_step = n_step      
        self.speed_range = speed_range 
        self.angle_range = angle_range
        self.mini_iter = 20 

    def step(self, state:State, action:list, step_time:int) -> State: 
        new_state = copy.deepcopy(state) 
        x, y = new_state.loc.x, new_state.loc.y 
        steer, speed = action 
        new_state.steering = steer 
        new_state.speed = speed 
        new_state.speed = np.clip(new_state.speed, self.speed_range[0], self.speed_range[1]) 
        new_state.steering = np.clip(new_state.steering, self.angle_range[0], self.angle_range[1]) 

        for _ in range(step_time):  # 1 step = 0.05s 
            for _ in range(self.mini_iter):  # dt = 0.0025s
                x += new_state.speed * np.cos(new_state.heading) * self.step_len/self.mini_iter 
                y += new_state.speed * np.sin(new_state.heading) * self.step_len/self.mini_iter
                new_state.heading += new_state.speed * np.tan(new_state.steering) / self.wheel_base * self.step_len/self.mini_iter 

        new_state.loc = Point(x, y) 
        return new_state 
    

class Vehicle(object):
    def __init__(
            self, 
            wheel_base=WHEEL_BASE,
            step_len=STEP_LENGTH,
            n_step=NUM_STEP, 
            speed_range=VALID_SPEED, 
            angle_range=VALID_STEER
        ):
        self.initial_state:State = None 
        self.box:LinearRing = None 
        self.trajectory:List[State] = [] 
        self.kinetic_model:Callable = KSModel(wheel_base, step_len, n_step, speed_range, angle_range) 
        self.color = (30, 144, 255, 255), # dodger blue 
        self.v_max = None 
        self.v_min = None 

    def reset(self, initial_state:State):
        self.initial_state = initial_state 
        self.state = self.initial_state 
        self.v_max = self.initial_state.speed 
        self.v_min = self.initial_state.speed 

        self.box = self.state.create_box() 
        self.trajectory.clear() 
        self.trajectory.append(self.state) 
        self.tmp_trajectory = self.trajectory.copy() 

    def step(self, action:np.ndarray, step_time:int):
        prev_info = copy.deepcopy((self.state, self.box, self.v_max, self.v_min)) 
        self.state = self.kinetic_model.step(self.state, action, step_time) 

        self.box = self.state.create_box()
        self.trajectory.append(self.state) 
        self.tmp_trajectory.append(self.state) 
        self.v_max = self.state.speed if self.state.speed > self.v_max else self.v_max 
        self.v_min = self.state.speed if self.state.speed < self.v_min else self.v_min 

        return prev_info 

    def retreat(self, prev_info):
        self.state, self.box, self.v_max, self.v_min = prev_info 
        self.trajectory.pop(-1) 

    def get_ego_box(self): 
        return State((0, 0, math.pi/2)).create_box(mode='ego_vehicle')  


