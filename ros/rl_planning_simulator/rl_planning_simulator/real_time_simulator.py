import os
import math 
import pygame
import rclpy 
from rclpy.node import Node 
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped 
from tf_transformations import euler_from_quaternion 
from autoware_planning_msgs.msg import Trajectory
from autoware_perception_msgs.msg import PredictedObjects
from nav_msgs.msg import Odometry
from collections import deque
from shapely.affinity import affine_transform, translate, rotate  
from shapely.geometry import Polygon, Point 

from env.lanelet2_map_parser import LaneletMapParser
from env.vehicle import State, Vehicle
from env.map_base import Area 
from configs import (
    WIN_W, WIN_H, K, 
    NON_DRIVABLE_COLOR, BG_COLOR, OBSTACLE_COLOR, START_COLOR, DEST_COLOR,
    NUM_STEP, STEP_LENGTH,)

class RealTimeSimulator(Node): 
    def __init__(self): 
        super().__init__('real_time_simulator') 

        self.vehicle = Vehicle(n_step=NUM_STEP, step_len=STEP_LENGTH)
        self.start_center = None 
        self.goal_center = None 
        self.odom = None 

        self.create_timer(0.1, self.timer_callback)

        self.create_subscription(PoseStamped, '/planning/mission_planning/goal', self.goal_callback, 1) 
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 1) 

        self.map_parser = LaneletMapParser(osm_path='/media/k/part11/workspace/rl_planner/data/lanelet2_map/parking_lot_lanelet2_map_v1.osm')
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

    def _quaternion_to_yaw(self, orientation):
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    def sync_data(self):
        if not self.trajectory_buffer or not self.odometry_buffer:
            return None, None, None 
        
        traj_time, traj_msg = self.trajectory_buffer[-1]
        odo_time, odo_msg = self.odometry_buffer[-1]

        time_diff_sec = abs(traj_time - odo_time) / 1e9

        if time_diff_sec < 0.1:
            odom_pos = odo_msg.pose.pose.position
            odom_yaw = self._quaternion_to_yaw(odo_msg.pose.pose.orientation)
            odom_data = (odom_pos.x, odom_pos.y, odom_yaw) 
            
            traj_data = [] 
            for traj_point in traj_msg.points: 
                traj_pos = traj_point.pose.position 
                traj_yaw = self._quaternion_to_yaw(traj_point.pose.orientation)
                traj_data.append((traj_pos.x, traj_pos.y, traj_yaw)) 

            obs_data = [] 
            if self.predicted_objects_buffer:
                obs_time, obs_msg = self.predicted_objects_buffer[-1] 
                # TODO: obstacle 클래스 만들기 
                objects = obs_msg.objects
                self.get_logger().info(f'Predicted objects: {len(objects)}') 
                for obj in objects:
                    # obj_id = obj.object_id
                    obj_class = obj.classification[0].label
                    obj_x = obj.kinematics.initial_pose_with_covariance.pose.position.x 
                    obj_y = obj.kinematics.initial_pose_with_covariance.pose.position.y
                    obj_yaw = self._quaternion_to_yaw(obj.kinematics.initial_pose_with_covariance.pose.orientation)
                    obj_center = (obj_x, obj_y, obj_yaw)
                    obs_data.append(obj_center)
                    self.get_logger().info(f'class: {obj_class}, center: {obj_center}')

            return odom_data, traj_data, obs_data 
        return None, None, None
    
    def coord_transform_matrix(self) -> list:
        k = K
        ego_x = 0 
        ego_y = 0
        bx = WIN_W / 2 - k * ego_x
        by = (WIN_H * 2 / 3) + k * ego_y  
        self.k = k
        return [k, 0, 0, -k, bx, by] 
    
    def _coord_transform(self, object) -> list: 
        if hasattr(object, "shape"): 
            transformed = affine_transform(object.shape, self.matrix)
            return list(transformed.exterior.coords) 
        else: 
            transformed = affine_transform(object, self.matrix) 
            return list(transformed.coords)
        
    def _map_to_center(self, ego_point, target_point): 
        ego_x, ego_y, ego_yaw = ego_point
        target_x, target_y, target_yaw = target_point
        dyaw = target_yaw - ego_yaw 
        pt = translate(Point(target_x, target_y), xoff=-ego_x, yoff=-ego_y) 
        rotated_pt = rotate(pt, math.pi/2-ego_yaw, origin=(0, 0), use_radians=True) 
        return (rotated_pt.x, rotated_pt.y, dyaw+math.pi/2)   
    
    def _coord_transform_point(self, x: float, y: float) -> tuple:
        a, b, d, e, xoff, yoff = self.coord_transform_matrix()
        x_new = a * x + b * y + xoff
        y_new = d * x + e * y + yoff
        return int(x_new), int(y_new)
        
    def _render(self, surface:pygame.Surface, non_drivable_area, traj_data, obs_data): 
        surface.fill(BG_COLOR) 

        # non-drivable area 
        areas = []
        if non_drivable_area is not None:
            if isinstance(non_drivable_area, Polygon):
                areas.append(Area(shape=non_drivable_area, subtype="non_drivable", color=(220, 220, 220, 255)))
            else:  
                for poly in non_drivable_area.geoms:
                    areas.append(Area(shape=poly, subtype="non_drivable", color=(220, 220, 220, 255)))
        
        for area in areas:
            pygame.draw.polygon(surface, NON_DRIVABLE_COLOR, self._coord_transform(area))
        for obstacle in obs_data: 
            pygame.draw.polygon(surface, OBSTACLE_COLOR, self._coord_transform(obstacle.create_box(mode='obs_vehicle')))
        for traj_point in traj_data: 
            x, y, yaw = traj_point 
            pygame.draw.circle(surface, (0, 0, 255), self._coord_transform_point(x, y), 3)
            
        if self.start_center:
            start_center = self._map_to_center(self.odom, self.start_center)
            pygame.draw.polygon(surface, START_COLOR, self._coord_transform(State(start_center).create_box()), width=1) 
        if self.goal_center:
            goal_center = self._map_to_center(self.odom, self.goal_center)
            pygame.draw.polygon(surface, DEST_COLOR, self._coord_transform(State(goal_center).create_box()), width=1)
        pygame.draw.polygon(surface, self.vehicle.color, self._coord_transform(self.vehicle.get_ego_box()))

        # if RENDER_TRAJ and len(self.vehicle.trajectory) > 1:
        #     render_len = min(len(self.vehicle.trajectory), TRAJ_RENDER_LEN)
        #     for i in range(render_len):
        #         vehicle_box = self.vehicle.trajectory[-(render_len-i)].create_box()  
        #         points = self._coord_transform(vehicle_box)
        #         temp_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        #         pygame.draw.polygon(temp_surface, TRAJ_COLORS[-(render_len-i)], points)
        #         surface.blit(temp_surface, (0, 0))

    def timer_callback(self): 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                rclpy.shutdown()
                return

        odom_data, traj_data, obs_data = self.sync_data()
        if odom_data is not None: 
            self.odom = odom_data
            self.vehicle.reset(State(odom_data))
            self.matrix = self.coord_transform_matrix()
            
            processed_traj = [] 
            if traj_data is not None: 
                for traj_point in traj_data[::5]: 
                    point = self._map_to_center(odom_data, traj_point)
                    processed_traj.append(point)

            processed_obs = [] 
            if obs_data is not None: 
                for obs_point in obs_data: 
                    pose = self._map_to_center(odom_data, obs_point)
                    processed_obs.append(State(pose))

            drivable_area, non_drivable_area = self.map_parser.get_zoomed_area(odom_data, zoom_width=60, zoom_height=80) 
            self._render(self.screen, non_drivable_area, processed_traj, processed_obs)
        else: 
            pass 

        pygame.display.flip()
        self.clock.tick(10)  # FPS 제한

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