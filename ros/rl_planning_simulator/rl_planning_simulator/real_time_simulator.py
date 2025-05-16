import os
import math 
import time 
import rclpy 
import pygame
import threading
import numpy as np
from rclpy.node import Node 
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup 
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped 
from tf_transformations import euler_from_quaternion 
from autoware_planning_msgs.msg import Trajectory
from autoware_perception_msgs.msg import PredictedObjects
from nav_msgs.msg import Odometry
from collections import deque
from shapely.affinity import affine_transform, translate, rotate  
from shapely.geometry import Polygon, Point, LinearRing 

from environment.map_base import Area
from environment.agent_simulator import AgentSimulator 
from environment.vehicle import State, Vehicle
from configs import (
    WIN_W, WIN_H, K, 
    NON_DRIVABLE_COLOR, BG_COLOR, OBSTACLE_COLOR, START_COLOR, DEST_COLOR,)


class RealTimeSimulator(Node): 
    def __init__(self): 
        super().__init__('real_time_simulator') 
 
        self.start_center = None 
        self.dest_center = None 
        self.goal_center = None 
        self.matrix = self.coord_transform_matrix() 

        self.create_timer(0.1, self.timer_callback) 
        self.create_subscription(PoseStamped, '/planning/mission_planning/goal', self.goal_callback, 1) 
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 1) 
        self.create_subscription(Trajectory, '/planning/scenario_planning/trajectory', self.trajectory_callback, 10)
        self.create_subscription(Odometry, '/localization/kinematic_state', self.odometry_callback, 10)
        self.create_subscription(PredictedObjects, '/perception/object_recognition/objects', self.predicted_objects_callback, 10)

        self.trajectory_buffer = deque(maxlen=20)
        self.odometry_buffer = deque(maxlen=20)
        self.predicted_objects_buffer = deque(maxlen=20)

        pygame.init()
        self.screen = pygame.display.set_mode((600, 800))
        pygame.display.set_caption("Trajectory Visualization")
        self.clock = pygame.time.Clock()

        self.simulator = AgentSimulator(
            map_path='/media/k/part11/workspace/rl_planner/data/lanelet2_map/campus_lanelet2_map_v1.osm',
            agent_path='/media/k/part11/workspace/rl_planner/src/log/ckpt/exp4_SAC_best.pt'
        )
        
        self.rl_mode = False
        self.rl_trajectory = [] 
        self.simulate_thread = threading.Thread(target=self.simulate_loop, daemon=True)
        self.trajectory_lock = threading.Lock() 
        self.simulate_thread.start()

    def coord_transform_matrix(self) -> list:
        k = K
        ego_x = 0 
        ego_y = 0
        bx = WIN_W / 2 - k * ego_x
        by = (WIN_H * 2 / 3) + k * ego_y  
        self.k = k
        return [k, 0, 0, -k, bx, by] 

    def goal_callback(self, msg:PoseStamped):
        orientation = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.goal_center = (msg.pose.position.x, msg.pose.position.y, yaw)
        self.get_logger().info(f'Goal received: {self.goal_center}') 

    def initialpose_callback(self, msg:PoseWithCovarianceStamped): 
        orientation = msg.pose.pose.orientation 
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w]) 
        self.start_center = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self.get_logger().info(f'Initialized: {self.start_center}') 

    def _get_quaternion_yaw(self, orientation):
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    def odometry_callback(self, msg:Odometry):
        now = self.get_clock().now().nanoseconds
        odom_pos = msg.pose.pose.position 
        odom_yaw = self._get_quaternion_yaw(msg.pose.pose.orientation)
        odom_data = (odom_pos.x, odom_pos.y, odom_yaw)
        self.odometry_buffer.append((now, odom_data))

    def trajectory_callback(self, msg:Trajectory):
        now = self.get_clock().now().nanoseconds
        traj_data = [] 
        for traj_point in msg.points:
            traj_pos = traj_point.pose.position 
            traj_yaw = self._get_quaternion_yaw(traj_point.pose.orientation)
            traj_data.append((traj_pos.x, traj_pos.y, traj_yaw))
        self.dest_center = (traj_data[-1][0], traj_data[-1][1], traj_data[-1][2])
        self.trajectory_buffer.append((now, traj_data))

    def predicted_objects_callback(self, msg:PredictedObjects):
        now = self.get_clock().now().nanoseconds
        obs_data = [] 
        for obj in msg.objects: 
            obj_x = obj.kinematics.initial_pose_with_covariance.pose.position.x 
            obj_y = obj.kinematics.initial_pose_with_covariance.pose.position.y
            obj_yaw = self._get_quaternion_yaw(obj.kinematics.initial_pose_with_covariance.pose.orientation)
            obs_data.append((obj_x, obj_y, obj_yaw))
        self.predicted_objects_buffer.append((now, obs_data))
        
    def sync_data(self):
        odom_data, traj_data, obs_data = None, None, None 
        if self.odometry_buffer:
            odom_time, odom_data = self.odometry_buffer[-1]

            if self.trajectory_buffer:
                traj_time, traj_data = self.trajectory_buffer[-1]
                time_diff_sec = abs(traj_time - odom_time) / 1e9
                if time_diff_sec < 0.1:
                    traj_data = self.trajectory_buffer[-1][1] if self.trajectory_buffer else None

            if self.predicted_objects_buffer:
                obs_time, obs_data = self.predicted_objects_buffer[-1]
                time_diff_sec = abs(obs_time - odom_time) / 1e9
                if time_diff_sec < 0.1:
                    obs_data = self.predicted_objects_buffer[-1][1] if self.predicted_objects_buffer else None
                
        return odom_data, traj_data, obs_data 
    
    def _coord_transform_poly(self, object) -> list: 
        if isinstance(object, Area): 
            if isinstance(object.shape, Polygon):
                transformed = affine_transform(object.shape, self.matrix) 
                return list(transformed.exterior.coords)
            elif isinstance(object.shape, LinearRing):
                transformed = affine_transform(object.shape, self.matrix) 
                return list(transformed.coords)
        elif isinstance(object, Polygon):
            transformed = affine_transform(object, self.matrix) 
            return list(transformed.exterior.coords)
        elif isinstance(object, LinearRing):
            transformed = affine_transform(object, self.matrix) 
            return list(transformed.coords)
    
    def _coord_transform_point(self, x: float, y: float) -> tuple:
        a, b, d, e, xoff, yoff = self.matrix  # self.coord_transform_matrix()
        x_new = a * x + b * y + xoff
        y_new = d * x + e * y + yoff
        return int(x_new), int(y_new)
        
    def _map_coord_transform_base(self, base_point, target_point): 
        base_x, base_y, base_yaw = base_point
        target_x, target_y, target_yaw = target_point
        dyaw = target_yaw - base_yaw 
        pt = translate(Point(target_x, target_y), xoff=-base_x, yoff=-base_y) 
        rotated_pt = rotate(pt, math.pi/2-base_yaw, origin=(0, 0), use_radians=True) 
        return (rotated_pt.x, rotated_pt.y, dyaw+math.pi/2)  

    def _ego_coord_transform_map(self, base_pose, target_pose):
        base_x, base_y, base_yaw = base_pose
        target_x, target_y, target_yaw = target_pose
        rotated_pt = rotate(Point(target_x, target_y), base_yaw - math.pi/2, origin=(0, 0), use_radians=True)
        map_pt = translate(rotated_pt, xoff=base_x, yoff=base_y)
        yaw = target_yaw - math.pi/2 + base_yaw
        return (map_pt.x, map_pt.y, yaw)

    def _listen_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rclpy.shutdown()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    self.rl_mode = True  if self.rl_mode == False else False
                if self.rl_mode: 
                    print("RL mode activated") 
                else: 
                    print("RL mode deactivated")
    
    def timer_callback(self): 
        odom_data, traj_data, obs_data = self.sync_data()  # 맵 좌표 기준 데이터

        self._listen_events() 

        if self.start_center and self.goal_center:
            if odom_data: 
                self.curr_pose = odom_data
                
                self.processed_traj = [] 
                if traj_data is not None: 
                    for traj_point in traj_data[::5]: 
                        point = self._map_coord_transform_base(self.curr_pose, traj_point)
                        self.processed_traj.append(point)

                self.processed_obs = [] 
                if obs_data is not None: 
                    for obs_point in obs_data: 
                        pose = self._map_coord_transform_base(self.curr_pose, obs_point)
                        self.processed_obs.append(Area(shape=State(pose).create_box(mode='obs_vehicle'), subtype="obstacle"))

                drivable_area, non_drivable_area, lanelet_area = self.simulator.get_scene(self.curr_pose, zoom_width=60, zoom_height=80)
                self.processed_areas = []
                if non_drivable_area is not None:
                    if isinstance(non_drivable_area, Polygon):
                        self.processed_areas.append(Area(shape=non_drivable_area, subtype="non_drivable"))
                    else:  
                        for poly in non_drivable_area.geoms:
                            self.processed_areas.append(Area(shape=poly, subtype="non_drivable"))

                self._render(self.screen, self.processed_areas, self.processed_traj, self.processed_obs)
                    
        pygame.display.flip()
        self.clock.tick(10) 

    def _render(self, surface:pygame.Surface, area_data, traj_data, obs_data): 
        surface.fill(BG_COLOR) 
        
        for area in area_data:
            pygame.draw.polygon(surface, NON_DRIVABLE_COLOR, self._coord_transform_poly(area))
        for obstacle in obs_data: 
            pygame.draw.polygon(surface, OBSTACLE_COLOR, self._coord_transform_poly(obstacle))
        for traj_point in traj_data: 
            x, y, yaw = traj_point 
            pygame.draw.circle(surface, (0, 0, 255), self._coord_transform_point(x, y), 3)
            
        if self.start_center:
            start_center = self._map_coord_transform_base(self.curr_pose, self.start_center)
            pygame.draw.polygon(surface, START_COLOR, self._coord_transform_poly(State(start_center).create_box()), width=1) 
        if self.goal_center:
            goal_center = self._map_coord_transform_base(self.curr_pose, self.goal_center)
            pygame.draw.polygon(surface, DEST_COLOR, self._coord_transform_poly(State(goal_center).create_box()), width=1)
        
        pygame.draw.polygon(surface, (30, 144, 255), self._coord_transform_poly(State((0, 0, math.pi/2)).create_box()))
        
        for traj_point in self.rl_trajectory[10::3]: 
            x, y, yaw = self._map_coord_transform_base(self.curr_pose, traj_point)
            pygame.draw.circle(surface, (0, 255, 0), self._coord_transform_point(x, y), 3)
    
    def simulate_loop(self):                    
        while rclpy.ok():
            if self.rl_mode:
                # goal_center = self._map_coord_transform_base(self.curr_pose, self.goal_center)
                dest_center = self._map_coord_transform_base(self.curr_pose, self.dest_center)
                self.simulator.reset(self.screen, self.processed_areas, self.processed_obs, self.curr_pose, dest_center)
                rl_trajectory = self.simulator.run()  
                with self.trajectory_lock: 
                    self.rl_trajectory = rl_trajectory
            else: 
                with self.trajectory_lock:
                    self.rl_trajectory = []
            time.sleep(0.5)
        

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        node.destroy_node()
        rclpy.shutdown()