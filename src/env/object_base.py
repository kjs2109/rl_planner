import numpy as np 
from shapely.affinity import affine_transform 
from shapely.geometry import LinearRing, Point
from shapely.geometry.base import BaseGeometry 
from configs import VehicleBox, ObsVehicleBox


class Area(object): 
    def __init__(self, shape:BaseGeometry = None, subtype:str = None, color:float = None):
        self.shape = shape 
        self.subtype = subtype 
        self.color = color 

    def get_shape(self):
        return np.array(self.shape.coords) 


class State: 
    def __init__(self, raw_state:list): 
        self.loc:Point = Point(raw_state[:2]) 
        self.heading:float = raw_state[2] 
        if len(raw_state) == 3: 
            self.speed:float = 0.0 
            self.steering:float = 0.0 
        else:
            self.speed:float = raw_state[3] 
            self.steering:float = raw_state[4] 

    def create_box(self, mode='ego_vehicle') -> LinearRing: 
        cos_theta = np.cos(self.heading) 
        sin_theta = np.sin(self.heading) 
        mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y] 
        if mode == 'ego_vehicle': 
            box = affine_transform(VehicleBox, mat) 
        elif mode == 'obs_vehicle':
            box = affine_transform(ObsVehicleBox, mat)
        elif mode == 'pedestrian': 
            circle = Point(self.loc.x, self.loc.y).buffer(1.0)
            box = LinearRing(circle.exterior.coords)
        else: 
            raise ValueError("Invalid mode. Choose 'ego_vehicle', 'obs_vehicle', or 'pedestrian'.")
        return  box
    
    def get_pose(self):
        return (self.loc.x, self.loc.y, self.heading) 