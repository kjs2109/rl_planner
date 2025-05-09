import time
import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped 
from tf_transformations import euler_from_quaternion 
from autoware_planning_msgs.msg import Trajectory
from autoware_perception_msgs.msg import PredictedObjects
from nav_msgs.msg import Odometry
from collections import deque

from env.lanelet2_map_parser import LaneletMapParser
from env.campus_env_base import CampusEnvBase
from env.vehicle import Status

import math 
import numpy as np
from numpy.random import randn
import random

def random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = randn()*std + mean
    return np.clip(rand_num, clip_low, clip_high)

def random_uniform_num(clip_low, clip_high):
    rand_num = random()*(clip_high - clip_low) + clip_low
    return rand_num

def sample_straight_forward_action():
        steer = random_gaussian_num(mean=0, std=0.3, clip_low=-2*math.pi, clip_high=2*math.pi)  
        speed = random_gaussian_num(mean=1.5, std=0.3, clip_low=0.5, clip_high=2.5)  
        return np.array([steer, speed], dtype=np.float32)

class PlanningSimulator(Node): 
    def __init__(self):
        super().__init__('rl_planning_simulator_node')
        
        self.start_center = None 
        self.goal_center = None 
        self.map_path = '/media/k/part11/workspace/rl_planner_mod/data/lanelet2_map/parking_lot_lanelet2_map_v1.osm' 
        self.trajectory_path = "/media/k/part11/workspace/rl_planner_mod/data/trajectory/synced_trajectory_odometry_v1.json"
        self.map_parser = LaneletMapParser(osm_path='/media/k/part11/workspace/rl_planner_mod/data/lanelet2_map/parking_lot_lanelet2_map_v1.osm')

        self.create_subscription(PoseStamped, '/planning/mission_planning/goal', self.goal_callback, 1) 
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 1) 
        self.create_subscription(PredictedObjects, '/perception/object_recognition/objects', self.predicted_objects_callback, 1)

        self.create_timer(1.0, self.timer_callback)

        self.create_subscription(Trajectory, '/planning/scenario_planning/trajectory', self.trajectory_callback, 10)
        self.create_subscription(Odometry, '/localization/kinematic_state', self.odometry_callback, 10)

        self.trajectory_buffer = deque(maxlen=20)
        self.odometry_buffer = deque(maxlen=20)
        self.predicted_objects_buffer = deque(maxlen=20)

    def goal_callback(self, msg: PoseStamped):
        orientation = msg.pose.orientation
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.goal_center = (msg.pose.position.x, msg.pose.position.y, yaw)
        self.get_logger().info(f'Goal received: {self.goal_center}') 

    def initialpose_callback(self, msg: PoseWithCovarianceStamped): 
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation 
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w]) 

        self.start_center = (x, y, yaw)
        self.get_logger().info(f'Initialized: {self.start_center}') 

    def trajectory_callback(self, msg: Trajectory):
        now = self.get_clock().now().nanoseconds
        self.trajectory_buffer.append((now, msg))

    def odometry_callback(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds
        self.odometry_buffer.append((now, msg))

    def predicted_objects_callback(self, msg: PredictedObjects):
        now = self.get_clock().now().nanoseconds
        self.predicted_objects_buffer.append((now, msg))

    def timer_callback(self):
        self.sync_data()

        if self.start_center is not None and self.goal_center is not None:
            if self.predicted_objects_buffer is not None: 
                now, msg = self.predicted_objects_buffer[-1] 
                objects = msg.objects
                self.get_logger().info(f'Predicted objects: {len(objects)}') 
                obstacles = []
                for obj in objects:
                    # obj_id = obj.object_id
                    obj_class = obj.classification[0].label
                    obj_x = obj.kinematics.initial_pose_with_covariance.pose.position.x 
                    obj_y = obj.kinematics.initial_pose_with_covariance.pose.position.y
                    obj_yaw = self.quaternion_to_yaw(obj.kinematics.initial_pose_with_covariance.pose.orientation)
                    obj_center = (obj_x, obj_y, obj_yaw)
                    obstacles.append(obj_center)
                    self.get_logger().info(f'class: {obj_class}, center: {obj_center}')

            print('!'*50)
            # Get the map and trajectory
            # campus_map = CampusMap(self.map_path, self.trajectory_path)
            # campus_map.generate_simulator_scene(self.start_center, self.goal_center) 
            # simulate the environment 
            self.simulate(self.start_center, self.goal_center, obstacles=obstacles)
            print('!'*50)
            self.start_center, self.goal_center = None, None 

    def quaternion_to_yaw(self, orientation):
        """Convert ROS quaternion to yaw using tf."""
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    def sync_data(self):
        # if not self.trajectory_buffer or not self.odometry_buffer or not self.predicted_objects_buffer: 
        if not self.trajectory_buffer or not self.odometry_buffer: 
            # 3개의 데이터가 모두 존재할때만 sync 동작 수행
            return
        
        traj_time, traj_msg = self.trajectory_buffer[-1]
        odo_time, odo_msg = self.odometry_buffer[-1]
        pred_time, pred_msg = self.predicted_objects_buffer[-1]

        time_diff_sec = abs(traj_time - odo_time) / 1e9  # nanoseconds to seconds

        if time_diff_sec < 0.1:
            
            traj_point = traj_msg.points[0].pose.position 
            traj_orientation = traj_msg.points[0].pose.orientation
            traj_yaw = self.quaternion_to_yaw(traj_orientation)
            traj_center = (traj_point.x, traj_point.y, traj_yaw)
            odo_point = odo_msg.pose.pose.position
            odo_orientation = odo_msg.pose.pose.orientation
            odo_yaw = self.quaternion_to_yaw(odo_orientation)
            odo_center = (odo_point.x, odo_point.y, odo_yaw)

            self.get_logger().info(f'traj_center: {traj_center}, odo_center: {odo_center}') 

    def simulate(self, start_center, goal_center, obstacles=None): 
        scene_info = {
            'mode': 'simulate',
            'start': start_center,  
            'dest': goal_center, 
            'obstacles': obstacles, 
        }
        env = CampusEnvBase(render_mode="human", map_path=self.map_path, trajectory_path=self.trajectory_path)
        env.reset(case_id=None, scene_info=scene_info)
        step_count = 0
        while True:
            # action = env.action_space.sample()
            action = sample_straight_forward_action()

            obs, reward_info, status, info = env.step(action)

            # obs, reward_info, status, info = env.step()
            print("LIDAR:", obs['lidar'].shape)       # if use_lidar_observation is True
            print("Image:", obs['img'].shape)   # if use_img_observation is True
            print("Action:", action)
            print("Target repr:", obs['target'])
            print(f"Step: {step_count:3d}, Status: {status.name}, Reward: {reward_info}")
            print('--'*50)

            if status != Status.CONTINUE:
                print(f"Episode finished with status: {status.name}")
                break

            step_count += 1
            time.sleep(0.05)

        env.close()


def main(args=None):
    rclpy.init(args=args)
    node = PlanningSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()