import os 
import math 
from math import pi
import numpy as np 
import random as rand 

from typing import List
from shapely.geometry import Polygon, LinearRing, Point, LineString, MultiLineString 
from env.vehicle import State 
from env.map_base import Area 
from env.lanelet2_map_parser import LaneletMapParser 
from env.trajectory_parser import TrajectoryParser 
from shapely.affinity import rotate, translate 
from shapely.ops import unary_union

from utils import DebugVisualizer
from utils import random_uniform_num, random_gaussian_num

DEBUG = False

class CampusMap(object): 
    def __init__(self, lanelet_map_path, trajectory_path):
        self.start:State = None 
        self.dest:State = None 
        self.start_box:LinearRing = None 
        self.dest_box:LinearRing = None 
        self.xmin, self.xmax = 0, 0 
        self.ymin, self.ymax = 0, 0 
        self.obstacles:List[Area] = [] 
        self.zoomed_non_drivable_area:List[Area] = []
        self.zoomed_drivable_area:List[Area] = []
        if DEBUG: 
            self.visualizer = DebugVisualizer(save_path='./log/figure/')

        if lanelet_map_path is None and trajectory_path is None:
            raise ValueError("lanelet_map_path and trajectory_path cannot be None") 
        else: 
            self.map_parser = LaneletMapParser(lanelet_map_path) 
            self.trajectory_parser = TrajectoryParser(trajectory_path)
            self.non_drivable_area = self.map_parser.get_non_drivable_area()

    def reset(self, case_id=None, scene_info=None): 
        self.zoomed_non_drivable_area = [] 
        self.zoomed_drivable_area = []
        if case_id is not None: 
            if scene_info['mode'] == 'normal': 
                scene = self.generate_normal_scene(case_id)
            elif scene_info['mode'] == 'parking':
                scene = self.generate_parking_scene(case_id)
            elif scene_info['mode'] == 'normal_parking':
                cx, cy, yaw = self.trajectory_parser.get_ori_center_point(case_id)
                parking_union = unary_union(self.map_parser.parking_lots)
                if parking_union.contains(Point(cx, cy)):
                    scene = self.generate_parking_scene(case_id) 
                else:
                    scene = self.generate_normal_scene(case_id)
        else: 
            scene = self.generate_simulator_scene(scene_info) 

        start = scene['start'] 
        dest = scene['dest']
        obstacles = scene['obstacles']
        zoomed_non_drivable_area = scene['zoomed_non_drivable_area']
        zoomed_drivable_area = scene['zoomed_drivable_area']

        self.start = State(start+[0, 0]) 
        self.start_box = self.start.create_box() 
        self.dest = State(dest+[0, 0]) 
        self.dest_box = self.dest.create_box() 

        self.xmin = np.floor(min(self.start.loc.x, self.dest.loc.x) - 10)
        self.xmax = np.ceil(max(self.start.loc.x, self.dest.loc.x) + 10)
        self.ymin = np.floor(min(self.start.loc.y, self.dest.loc.y) - 10)
        self.ymax = np.ceil(max(self.start.loc.y, self.dest.loc.y) + 10)

        # obstacles 
        self.obstacles = list([Area(shape=obs, subtype="obstacle", color=(150, 150, 150, 255)) for obs in obstacles])
        
        # non-drivable area 
        if zoomed_non_drivable_area is not None:
            if isinstance(zoomed_non_drivable_area, Polygon):
                self.zoomed_non_drivable_area.append(Area(shape=zoomed_non_drivable_area, subtype="non_drivable", color=(220, 220, 220, 255)))
            else:  
                for poly in zoomed_non_drivable_area.geoms:
                    self.zoomed_non_drivable_area.append(Area(shape=poly, subtype="non_drivable", color=(220, 220, 220, 255)))

        # drivable area 
        if zoomed_drivable_area is not None:
            if isinstance(zoomed_drivable_area, Polygon):
                self.zoomed_drivable_area.append(Area(shape=zoomed_drivable_area, subtype="drivable", color=(150, 150, 150, 255)))
            else:  
                for poly in zoomed_drivable_area.geoms:
                    self.zoomed_drivable_area.append(Area(shape=poly, subtype="drivable", color=(100, 100, 100, 255)))
        
        self.n_obstacle = len(self.obstacles)

        return self.start
    
    def sample_points_on_boundary(self, geometry, n_points=15):
        """
        Given a polygon or multipolygon, return sampled points from its boundary.
        """
        boundary = geometry.boundary
        points = []
        random_n_points = rand.randint(3, n_points)
        if isinstance(boundary, LineString):
            for i in np.linspace(0, boundary.length, n_points):
                points.append(boundary.interpolate(i))
        elif isinstance(boundary, MultiLineString):
            for line in boundary.geoms:
                for i in np.linspace(0, line.length, n_points // len(boundary.geoms)):
                    points.append(line.interpolate(i))
        return points
    
    def _map_to_center(self, ego_point, target_point): 
        ego_x, ego_y, ego_yaw = ego_point
        target_x, target_y, target_yaw = target_point
        dyaw = target_yaw - ego_yaw 
        pt = translate(Point(target_x, target_y), xoff=-ego_x, yoff=-ego_y) 
        rotated_pt = rotate(pt, pi/2-ego_yaw, origin=(0, 0), use_radians=True) 
        return (rotated_pt.x, rotated_pt.y, dyaw+pi/2)   

    def generate_simulator_scene(self, scene_info): 
        generate_success = True

        initial_point = scene_info['start'] 
        goal_point = scene_info['dest'] 
        detected_obstacles = scene_info['obstacles'] 

        zoomed_drivable_area, zoomed_non_drivable_area = self.map_parser.get_zoomed_area(center=initial_point, zoom_width=60, zoom_height=90)

        if DEBUG: 
            self.visualizer.figure_init(title="Simulator Scene", xlim=(-30, 30), ylim=(-40, 40))
            self.visualizer.draw_polygon(zoomed_non_drivable_area, color='lightgray', edgecolor="gray", alpha=1.0)

        # 1. start_point 
        start_x, start_y, start_yaw = (0, 0, pi/2)
        car_rb, car_rf, car_lf, car_lb = list(State([start_x, start_y, start_yaw, 0, 0]).create_box().coords)[:-1]
        start_box = LinearRing((car_rb, car_rf, car_lf, car_lb))
        print('start done.', end=' ')

        if DEBUG:
            self.visualizer.draw_linear_ring(start_box, color='blue', edgecolor="blue")
            self.visualizer.draw_arrow(start_x, start_y, start_yaw)

        # 2. obstacles 
        obstacles = []
        boundary_points = self.sample_points_on_boundary(zoomed_drivable_area, n_points=10)

        if detected_obstacles is not None:
            for obj_point in detected_obstacles:
                obj_x, obj_y, obj_yaw = self._map_to_center(initial_point, obj_point) 
                car_rb, car_rf, car_lf, car_lb = list(State([obj_x, obj_y, obj_yaw, 0, 0]).create_box(mode='obs_vehicle').coords)[:-1]
                obs_box = Polygon((car_rb, car_rf, car_lf, car_lb))

                if obs_box.intersects(start_box):
                    continue
                obstacles.append(obs_box)
                if DEBUG:
                    self.visualizer.draw_polygon(obs_box, color='dimgray', edgecolor="dimgray")
        else:
            for pt in boundary_points:
                offset_x = pt.x + random_uniform_num(-0.5, 0.5)
                offset_y = pt.y + random_uniform_num(-0.5, 0.5)
                yaw = np.random.rand() * pi * 2
                obs_box = Polygon(State([offset_x, offset_y, yaw, 0, 0]).create_box())

                if obs_box.intersects(start_box):
                    continue
                if any(obs_box.intersects(obs) for obs in obstacles):
                    continue

                if zoomed_drivable_area and not zoomed_drivable_area.is_empty:
                    inter_area = obs_box.intersection(zoomed_drivable_area).area
                    overlab_area = inter_area / obs_box.area
                    if overlab_area < 0.3:
                        obstacles.append(obs_box)
                        if DEBUG:
                            self.visualizer.draw_linear_ring(obs_box, color='dimgray', edgecolor="dimgray")
        print('obstacle done.', end=' ')

        # 3. goal_point 
        dest_x, dest_y, dest_yaw = self._map_to_center(initial_point, goal_point) 
        car_rb, car_rf, car_lf, car_lb = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
        dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))
        print('dest done.')

        if DEBUG:
            self.visualizer.draw_linear_ring(dest_box, color='green', edgecolor='green')
            self.visualizer.draw_arrow(dest_x, dest_y, dest_yaw)
            if generate_success:
                self.visualizer.save()
            self.visualizer.clear() 

        scene = {
            'start': [start_x, start_y, start_yaw],
            'dest': [dest_x, dest_y, dest_yaw],
            'obstacles': obstacles,
            'zoomed_non_drivable_area': zoomed_non_drivable_area,
            'zoomed_drivable_area': zoomed_drivable_area
        }
        return scene
    
    def generate_normal_scene(self, case_id=0):
        generate_success = True

        ori_center_point = self.trajectory_parser.get_ori_center_point(case_id)
        trajectory = self.trajectory_parser.get_processed_trajectory(case_id) 
        zoomed_drivable_area, zoomed_non_drivable_area = self.map_parser.get_zoomed_area(center=ori_center_point, zoom_width=60, zoom_height=90)
        if DEBUG: 
            self.visualizer.figure_init(title="Normal Scene", xlim=(-30, 30), ylim=(-40, 40))
            self.visualizer.draw_polygon(zoomed_non_drivable_area, color='lightgray', edgecolor="gray", alpha=1.0) 

        # 1. start_point
        while True:
            start_x = 0 
            start_y = random_uniform_num(-1.0, 2.0)
            start_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12) 
            car_rb, car_rf, car_lf, car_lb = list(State([start_x, start_y, start_yaw, 0, 0]).create_box().coords)[:-1]
            start_box = LinearRing((car_rb, car_rf, car_lf, car_lb))

            if zoomed_non_drivable_area.intersects(start_box):
                continue
            break
        print('start done.', end=' ')

        if DEBUG:
            self.visualizer.draw_linear_ring(start_box, color='blue', edgecolor="blue") 
            self.visualizer.draw_arrow(start_x, start_y, start_yaw) 

        # 2. obstacles
        obstacles = []
        boundary_points = self.sample_points_on_boundary(zoomed_drivable_area, n_points=10)

        for pt in boundary_points:
            obs_x = pt.x + random_uniform_num(-0.5, 0.5)
            obs_y = pt.y + random_uniform_num(-0.5, 0.5)
            yaw = np.random.rand() * pi * 2
            obs_box = Polygon(State([obs_x, obs_y, yaw, 0, 0]).create_box('obs_vehicle'))

            if obs_box.intersects(start_box):
                continue
            if any(obs_box.intersects(obs) for obs in obstacles):
                continue

            if zoomed_drivable_area and not zoomed_drivable_area.is_empty:
                inter_area = obs_box.intersection(zoomed_drivable_area).area
                overlab_area = inter_area / obs_box.area
                if overlab_area < 0.3:
                    obstacles.append(obs_box)
                    if DEBUG:
                        self.visualizer.draw_polygon(obs_box, color='dimgray', edgecolor='dimgray') 
        print('obstacle done.', end=' ')

        # 3. goal_point
        dest_box_valid = False 
        while not dest_box_valid:
            for traj_p in trajectory[::-1]:
                _dest_x, _dest_y, _dest_yaw = traj_p 
                dest_x = _dest_x  
                dest_y = random_uniform_num(_dest_y-0.5, _dest_y+0.5)
                dest_yaw = random_gaussian_num(_dest_yaw, pi/36, -2*pi, 2*pi)  # pi/6
                car_rb, car_rf, car_lf, car_lb = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
                dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))

                if dest_box.intersects(start_box):
                    continue
                if any(dest_box.intersects(obs) for obs in obstacles):
                    continue
                if zoomed_non_drivable_area.intersects(dest_box):
                    continue
                dest_box_valid = True
                break
        print('dest done.')

        if DEBUG:
            self.visualizer.draw_linear_ring(dest_box, color='green', edgecolor='green') 
            self.visualizer.draw_arrow(dest_x, dest_y, dest_yaw) 
            if generate_success:
                self.visualizer.save()
            self.visualizer.clear()

        scene = {
            'start': [start_x, start_y, start_yaw],
            'dest': [dest_x, dest_y, dest_yaw],
            'obstacles': obstacles,
            'zoomed_non_drivable_area': zoomed_non_drivable_area,
            'zoomed_drivable_area': zoomed_drivable_area
        }
        return scene 

    def generate_parking_scene(self, case_id=0):
        generate_success = True
        ori_center_point = self.trajectory_parser.get_ori_center_point(case_id)
        trajectory = self.trajectory_parser.get_processed_trajectory(case_id) 
        zoomed_drivable_area, zoomed_non_drivable_area, lanelet_area = self.map_parser.get_parking_drivable_area(
            center=ori_center_point, zoom_width=60, zoom_height=90)

        if not zoomed_drivable_area:
            raise ValueError("Drivable area could not be generated")

        if DEBUG:
            self.visualizer.figure_init(title="Parking Scene", xlim=(-30, 30), ylim=(-40, 40))
            self.visualizer.draw_polygon(zoomed_non_drivable_area, color='gray', edgecolor="gray", alpha=1.0) 
            self.visualizer.draw_polygon(lanelet_area, color='whitesmoke', edgecolor="black", alpha=0.6)

        # 1. start_point 
        while True:
            start_yaw = random_gaussian_num(pi/2, pi/36, pi*5/12, pi*7/12)  # pi/36
            start_x = 0 # random_uniform_num(0, 1.0)
            start_y = random_uniform_num(-1.0, 2.0)
            car_rb, car_rf, car_lf, car_lb = list(State([start_x, start_y, start_yaw, 0, 0]).create_box().coords)[:-1]
            start_box = LinearRing((car_rb, car_rf, car_lf, car_lb))

            if zoomed_non_drivable_area.intersects(start_box):
                continue
            break
        print('start done.', end=' ')

        if DEBUG:
            self.visualizer.draw_linear_ring(start_box, color='blue', edgecolor="blue") 
            self.visualizer.draw_arrow(start_x, start_y, start_yaw) 

        # 2. obstacles
        obstacles = []
        parking_areas = self.map_parser.get_parking_area(ori_center_point)
        for parking_poly in parking_areas:
            if not parking_poly.is_valid or parking_poly.is_empty or np.random.rand() < 0.5:
                continue

            min_rect = parking_poly.minimum_rotated_rectangle
            coords = list(min_rect.exterior.coords)[:-1]
            if len(coords) < 2:
                continue

            center = min_rect.centroid
            vec = np.array(coords[1]) - np.array(coords[0])
            yaw = np.arctan2(vec[1], vec[0]) + pi/2

            vehicle_length = 4.5
            vehicle_width = 2.0
            dx = vehicle_length / 2
            dy = vehicle_width / 2
            local_box = np.array([[ dx,  dy], [ dx, -dy], [-dx, -dy], [-dx,  dy]])
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rotated_box = [tuple(R @ pt + np.array([center.x, center.y])) for pt in local_box]
            obs_box = Polygon(rotated_box)

            if obs_box.intersects(start_box) or any(obs_box.intersects(o) for o in obstacles):
                continue
            if zoomed_non_drivable_area.intersects(obs_box):
                continue

            obstacles.append(obs_box)

            if DEBUG:
                self.visualizer.draw_polygon(obs_box, color='dimgray', edgecolor='dimgray') 
        print('obstacle done.', end=' ')

        # 3. goal_point 
        dest_box_valid = False 
        while not dest_box_valid:
            for traj_p in trajectory[::-1]:
                _dest_x, _dest_y, _dest_yaw = traj_p 
                dest_yaw = random_gaussian_num(_dest_yaw, pi/36, -2*pi, 2*pi)
                dest_x = _dest_x
                dest_y = random_uniform_num(_dest_y-0.5, _dest_y+0.5)
                car_rb, car_rf, car_lf, car_lb = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
                dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))

                if dest_box.intersects(start_box):
                    continue
                if any(dest_box.intersects(obs) for obs in obstacles):
                    continue
                if zoomed_non_drivable_area.intersects(dest_box):
                    continue
                dest_box_valid = True
                break
        print('dest done.')

        if DEBUG:
            self.visualizer.draw_linear_ring(dest_box, color='green', edgecolor='green')  
            self.visualizer.draw_arrow(dest_x, dest_y, dest_yaw) 
            if generate_success:
                self.visualizer.save()
            self.visualizer.clear()

        scene = {
            'start': [start_x, start_y, start_yaw],
            'dest': [dest_x, dest_y, dest_yaw],
            'obstacles': obstacles,
            'zoomed_non_drivable_area': zoomed_non_drivable_area,
            'zoomed_drivable_area': zoomed_drivable_area
        }
        return scene


    def _flip_box_orientation(self, target_state:State):
        x, y, heading = target_state.get_pos()
        center = np.mean(target_state.create_box().coords[:-1], axis=0)
        new_x = 2*center[0] - x
        new_y = 2*center[1] - y
        heading = heading + np.pi
        return State([new_x, new_y, heading])
    
    def flip_dest_orientation(self,):
        self.dest = self._flip_box_orientation(self.dest)
        self.dest_box = self.dest.create_box()

    def flip_start_orientation(self,):
        self.start = self._flip_box_orientation(self.start)
        self.start_box = self.start.create_box() 


if __name__ == "__main__":
    # map_path = '/media/k/part11/workspace/rl_planner/data/lanelet2_map/campus_lanelet2_map.osm' 
    # trajectory_path = "/media/k/part11/workspace/rl_planner/data/trajectory/synced_trajectory_odometry_v1.json"
    # campus_map = CampusMap(map_path, trajectory_path)
    # for i in range(445):
    #     campus_map.generate_normal_scene(i) 
    map_path = '/media/k/part11/workspace/rl_planner/data/lanelet2_map/parking_lot_lanelet2_map_v1.osm'
    trajectory_path = '/media/k/part11/workspace/rl_planner/data/trajectory/parking_lot_trajectory_odometry.json' 
    campus_map = CampusMap(map_path, trajectory_path) 
    # for i in range(156):
    #     campus_map.generate_parking_scene(i)
    campus_map.generate_parking_scene(40)