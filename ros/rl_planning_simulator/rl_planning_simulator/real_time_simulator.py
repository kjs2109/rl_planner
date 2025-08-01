import os
import math 
import time 
import rclpy 
import pygame
import threading
import numpy as np
from rclpy.node import Node 
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped 
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint
from autoware_perception_msgs.msg import PredictedObjects
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from std_srvs.srv import SetBool 
from tf_transformations import quaternion_from_euler, euler_from_quaternion 

from collections import deque
from shapely.affinity import affine_transform, translate, rotate  
from shapely.geometry import Polygon, Point, LinearRing 

from path_smoother import smooth_trajectory
from environment.utils import random_gaussian_num, random_uniform_num
from environment.map_base import Area
from environment.agent_simulator import AgentSimulator 
from environment.vehicle import State, Status
from environment.configs import (
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
        self.create_timer(0.1, self._publish_trajectory)

        # subscribers 
        self.create_subscription(PoseStamped, '/planning/mission_planning/goal', self.goal_callback, 1) 
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initialpose_callback, 1) 
        self.create_subscription(Trajectory, '/planning/scenario_planning/lane_driving/trajectory', self.trajectory_callback, 10)
        self.create_subscription(Odometry, '/localization/kinematic_state', self.odometry_callback, 10)
        self.create_subscription(PredictedObjects, '/perception/object_recognition/objects', self.predicted_objects_callback, 10)
        
        # publishers
        self.trajectory_pub_ = self.create_publisher(Trajectory, '/planning/scenario_planning/rl_planning/trajectory', 1)
        
        # services
        self.create_service(SetBool, '/rl_planner/set_rl_mode', self.set_rl_mode_callback)

        self.trajectory_buffer = deque(maxlen=20)
        self.odometry_buffer = deque(maxlen=20)
        self.predicted_objects_buffer = deque(maxlen=20)

        self.trajectory_lock = threading.Lock() 
        self.rl_stop_event = threading.Event()

        pygame.init()
        self.screen = pygame.display.set_mode((600, 800))
        pygame.display.set_caption("Trajectory Visualization")
        self.clock = pygame.time.Clock()

        root_path = '/home/k/rl_planner'
        self.simulator_1 = AgentSimulator(
            map_path=os.path.join(root_path, 'data/lanelet2_map_local.osm'),
            agent_path=os.path.join(root_path, 'data/SAC_opt_199999_s.pt'), 
            mode='short_term',
            tolerant_time=40
        )
        self.simulator_2 = AgentSimulator(
            map_path=os.path.join(root_path, 'data/lanelet2_map_local.osm'),
            agent_path=os.path.join(root_path, 'data/SAC_opt_103999_l.pt'), 
            mode='long_term', 
            tolerant_time=80
        )
        
        self.is_stop = True 
        self.rl_trajectory = [] 
        self.rl_mode = False
        self.vis_traj = False 
        self.stop_cnt = 0
        self.outtime_cnt_s = 0
        self.outtime_cnt_l = 0 
        self.dest_s = None 

    def yaw_to_quaternion(self, yaw):
        q = quaternion_from_euler(0, 0, yaw)
        quat = Quaternion()
        quat.x = q[0]
        quat.y = q[1]
        quat.z = q[2]
        quat.w = q[3]
        return quat

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
        self.is_stop = self._stop_filter(msg.twist.twist)
        self.odometry_buffer.append((now, odom_data))

    def _stop_filter(self, twist): 
        return twist.linear.x < 0.5 and twist.linear.y < 0.5 and twist.angular.z < 0.5

    def trajectory_callback(self, msg:Trajectory):
        now = self.get_clock().now().nanoseconds
        if len(msg.points) <= 3: 
            msg = self.prev_traj_msg 
        else: 
            self.prev_traj_msg = msg

        traj_data = [] 
        for traj_point in msg.points:
            traj_pos = traj_point.pose.position 
            traj_yaw = self._get_quaternion_yaw(traj_point.pose.orientation)
            traj_data.append((traj_pos.x, traj_pos.y, traj_yaw))
        
        # long term 
        long_term = traj_data[-1]
        # short term 
        _short_term_idx = int(len(traj_data)*(0.5 - self.outtime_cnt_s*0.05))
        short_term_idx = max(_short_term_idx, 25) if len(traj_data) > 25 else _short_term_idx
        short_term = traj_data[short_term_idx]  
        rel_distance = math.sqrt((short_term[0] - long_term[0])**2 + (short_term[1] - long_term[1])**2)
        if rel_distance < 10.0: 
            short_term = long_term 
        self.short_term_dest_center = (short_term[0], short_term[1], short_term[2])
        self.long_term_dest_center = (long_term[0], long_term[1], long_term[2])
        self.trajectory_buffer.append((now, traj_data))

    def predicted_objects_callback(self, msg:PredictedObjects):
        now = self.get_clock().now().nanoseconds
        obs_data = [] 
        for obj in msg.objects: 
            class_num = obj.classification[0].label
            obj_x = obj.kinematics.initial_pose_with_covariance.pose.position.x 
            obj_y = obj.kinematics.initial_pose_with_covariance.pose.position.y
            obj_yaw = self._get_quaternion_yaw(obj.kinematics.initial_pose_with_covariance.pose.orientation)
            obj = {'class': class_num, 'pose': (obj_x, obj_y, obj_yaw)}
            obs_data.append(obj)
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
                    self.rl_mode = not self.rl_mode

                    if self.rl_mode:
                        self.rl_stop_event.clear()
                        self.simulate_thread_1 = threading.Thread(target=self.simulate_loop_short, daemon=True)
                        self.simulate_thread_2 = threading.Thread(target=self.simulate_loop_long, daemon=True)
                        self.simulate_thread_1.start()
                        self.simulate_thread_2.start()
                        print("RL mode activated")
                    else:
                        self.rl_stop_event.set()
                        self.stop_cnt = 0
                        for name in ['simulate_thread_1', 'simulate_thread_2']:
                            thread = getattr(self, name, None)
                            if thread and thread.is_alive():
                                thread.join(timeout=2.0)
                                setattr(self, name, None)
                        print("RL mode deactivated")

                elif event.key == pygame.K_v:
                    self.vis_traj = not self.vis_traj

    def set_rl_mode_callback(self, request: SetBool, response: SetBool):
        if request.data:
            if not self.rl_mode:
                self.rl_mode = True
                self.rl_stop_event.clear()
                self.simulate_thread_1 = threading.Thread(target=self.simulate_loop_short, daemon=True)
                self.simulate_thread_2 = threading.Thread(target=self.simulate_loop_long, daemon=True)
                self.simulate_thread_1.start()
                self.simulate_thread_2.start()
                response.success = True
                response.message = "RL mode activated"
        else:
            if self.rl_mode:
                self.rl_mode = False
                self.rl_stop_event.set()
                for name in ['simulate_thread_1', 'simulate_thread_2']:
                    thread = getattr(self, name, None)
                    if thread and thread.is_alive():
                        thread.join(timeout=2.0)
                        setattr(self, name, None)
                response.success = True
                response.message = "RL mode deactivated"
        return response
    
    def timer_callback(self): 
        self._listen_events() 
        odom_data, traj_data, obs_data = self.sync_data()  # 맵 좌표 기준 데이터

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
                    for obs in obs_data: 
                        class_num, obs_point = obs['class'], obs['pose']
                        pose = self._map_coord_transform_base(self.curr_pose, obs_point)
                        if int(class_num) == 1 or int(class_num) == 2 or int(class_num) == 3: 
                            self.processed_obs.append(Area(shape=State(pose).create_box(mode='obs_vehicle'), subtype="obstacle"))
                        elif int(class_num) == 7: 
                            self.processed_obs.append(Area(shape=State(pose).create_box(mode='pedestrian'), subtype="obstacle"))

                drivable_area, non_drivable_area, lanelet_area = self.simulator_2.get_scene(self.curr_pose, zoom_width=60, zoom_height=80)
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
        if not self.vis_traj:
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
        
        for traj_point in self.rl_trajectory[3:]:  
            x, y, v, yaw = traj_point
            x, y, yaw = self._map_coord_transform_base(self.curr_pose, (x, y, yaw))
            if self.rl_mode:
                pygame.draw.circle(surface, (0, 255, 0), self._coord_transform_point(x, y), 3)
                if self.vis_traj:
                    pygame.draw.polygon(surface, (30, 144, 255), self._coord_transform_poly(State((x, y, yaw)).create_box()), width=1)

    def _publish_trajectory(self):
        traj_msg = Trajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.header.frame_id = 'map'

        if self.rl_mode:
            rl_trajectory = self.rl_trajectory 
        else:
            rl_trajectory = [] 

        for pt in rl_trajectory:
            x, y, v, yaw = pt

            traj_pt = TrajectoryPoint()
            traj_pt.pose.position.x = x
            traj_pt.pose.position.y = y
            traj_pt.pose.position.z = 0.0

            quat = self.yaw_to_quaternion(yaw)
            traj_pt.pose.orientation = quat
            traj_pt.longitudinal_velocity_mps = v 
            traj_msg.points.append(traj_pt)

        self.trajectory_pub_.publish(traj_msg)
    
    def simulate_loop_short(self):                 
        while rclpy.ok() and not self.rl_stop_event.is_set(): 
            tick = time.time() 

            try: 
                dest_center = self._map_coord_transform_base(self.curr_pose, self.short_term_dest_center)
            except:
                dest_center = self._map_coord_transform_base(self.curr_pose, self.goal_center)  

            if self.outtime_cnt_s != 0:  
                cnt = 0 
                while True:
                    dest_x, dest_y, dest_yaw = dest_center 
                    dest_x = random_uniform_num(dest_x-2, dest_x+2) 
                    dest_y = random_uniform_num(dest_y-2, dest_y+2)
                    car_rb, car_rf, car_lf, car_lb = list(State([dest_x, dest_y, dest_yaw, 0, 0]).create_box().coords)[:-1]
                    dest_box = LinearRing((car_rb, car_rf, car_lf, car_lb))
                    cnt += 1 
                    if any(dest_box.intersects(obs.shape) for obs in self.processed_obs):
                        continue
                    if any(dest_box.intersects(area.shape) for area in self.processed_areas):
                        continue
                    dest_center = (dest_x, dest_y, dest_yaw)
                    print(f'searching count: ', cnt, dest_center, self.outtime_cnt_s)
                    break 

            self.simulator_1.reset(self.screen, self.processed_areas, self.processed_obs, self.curr_pose, dest_center)
            rl_trajectory, status = self.simulator_1.run() 

            with self.trajectory_lock: 
                if len(self.rl_trajectory) <= len(rl_trajectory) or self.outtime_cnt_l > 1:
                    # rl_trajectory = smooth_trajectory(rl_trajectory)
                    self.rl_trajectory = rl_trajectory
                    time.sleep(1.5)
            
            if status == Status.OUTTIME or status == Status.COLLIDED:
                self.outtime_cnt_s += 1 
            elif status == Status.ARRIVED: 
                self.outtime_cnt_s = 0

            tock = time.time() - tick 
            print(f"[short: {status.name}] outtime: {self.outtime_cnt_s} | publish trajectory: {tock:.6f} sec | trajectory length: {len(rl_trajectory)}")
        
    def simulate_loop_long(self):             
        while rclpy.ok() and not self.rl_stop_event.is_set():
            tick = time.time() 

            try: 
                dest_center = self._map_coord_transform_base(self.curr_pose, self.long_term_dest_center)
            except:
                dest_center = self._map_coord_transform_base(self.curr_pose, self.goal_center)  

            self.simulator_2.reset(self.screen, self.processed_areas, self.processed_obs, self.curr_pose, dest_center)
            rl_trajectory, status = self.simulator_2.run()  

            if len(rl_trajectory) != 0:
                with self.trajectory_lock: 
                    rl_trajectory = smooth_trajectory(rl_trajectory)
                    if len(self.rl_trajectory) <10: 
                        time.sleep(1.5) 
                    self.rl_trajectory = rl_trajectory
            
            if status == Status.OUTTIME or status == Status.COLLIDED:
                self.outtime_cnt_l += 1 
            elif status == Status.ARRIVED: 
                self.outtime_cnt_l = 0
                
            tock = time.time() - tick
            print(f"[long: {status.name}] outtime: {self.outtime_cnt_l} | publish trajectory: {tock:.6f} sec | trajectory length: {len(rl_trajectory)}")

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