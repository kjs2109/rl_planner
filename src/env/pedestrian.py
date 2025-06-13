import numpy as np
from typing import Tuple, Union, List
from shapely.geometry import Point, LinearRing
from env.object_base import State 
from utils import random_gaussian_num


class Pedestrian:
    def __init__(self, uncertainty, step_len=0.5, speed_range=(0.0, 1.0), alpha=0.1):  
        self.dt = step_len
        self.speed_range = speed_range
        self.speed_std   = 1.0 * uncertainty
        self.heading_std = 0.5 * uncertainty
        self.alpha = alpha 

        self.initial_state = None
        self.state         = None
        self.trajectory    = []

        self._filt_speed   = None
        self._filt_heading = None
    
    def reset(self, initial_state:State):
        self.initial_state = initial_state
        self.state = initial_state
        self.trajectory.clear()
        self.trajectory.append(self.state)

        self._filt_speed   = initial_state.speed
        self._filt_heading = initial_state.heading
        return self

    def step(self, step_time=1):
        prev_state = self.state
        x, y = prev_state.loc.x, prev_state.loc.y

        raw_speed = random_gaussian_num(prev_state.speed,
                                        std=self.speed_std,
                                        clip_low=self.speed_range[0],
                                        clip_high=self.speed_range[1])

        raw_heading = random_gaussian_num(prev_state.heading,
                                          std=self.heading_std,
                                          clip_low=-np.pi,
                                          clip_high=np.pi)

        self._filt_speed   = self.alpha * self._filt_speed + (1 - self.alpha) * raw_speed 
        self._filt_heading = self.alpha * self._filt_heading + (1 - self.alpha) * raw_heading

        for _ in range(step_time):
            x += self._filt_speed * np.cos(self._filt_heading) * self.dt
            y += self._filt_speed * np.sin(self._filt_heading) * self.dt

        new_state   = State([x, y, self._filt_heading, self._filt_speed, 0.0])
        self.state  = new_state
        self.trajectory.append(new_state)
        return prev_state
    
    def create_box(self) -> LinearRing:
        radius = 0.3  # typical pedestrian radius
        circle = Point(self.state.loc.x, self.state.loc.y).buffer(radius, resolution=8)
        return LinearRing(circle.exterior.coords)