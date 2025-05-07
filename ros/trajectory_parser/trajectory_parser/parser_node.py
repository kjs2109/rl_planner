import rclpy
from rclpy.node import Node
from autoware_planning_msgs.msg import Trajectory
from nav_msgs.msg import Odometry
import math
from tf_transformations import euler_from_quaternion
import json
from collections import deque
import os

class TrajectoryOdometryParser(Node):
    def __init__(self):
        super().__init__('trajectory_parser')
        self.id = 0  
        self.save_interval = 10  # N개의 샘플마다 저장
        self.filename = 'synced_trajectory_odometry.json'

        # Timer: 0.5초마다 sync 시도
        self.create_timer(0.5, self.timer_callback)

        # Subscribe to topics
        self.create_subscription(Trajectory, '/planning/scenario_planning/trajectory', self.trajectory_callback, 10)
        self.create_subscription(Odometry, '/localization/kinematic_state', self.odometry_callback, 10)

        # Buffers
        self.trajectory_buffer = deque(maxlen=100)
        self.odometry_buffer = deque(maxlen=100)

        # Synced data
        self.synced_data = []

        self.get_logger().info('Trajectory and Odometry Parser Node has been started.')

    def trajectory_callback(self, msg: Trajectory):
        now = self.get_clock().now().nanoseconds
        self.trajectory_buffer.append((now, msg))

    def odometry_callback(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds
        self.odometry_buffer.append((now, msg))

    def timer_callback(self):
        # 주기적으로 sync 시도
        self.sync_data()

    def sync_data(self):
        if not self.trajectory_buffer or not self.odometry_buffer:
            return

        traj_time, traj_msg = self.trajectory_buffer[-1]
        odo_time, odo_msg = self.odometry_buffer[-1]

        time_diff_sec = abs(traj_time - odo_time) / 1e9  # nanoseconds to seconds

        if time_diff_sec < 0.1:
            self.get_logger().info(f'[{self.id}] Sync found with time diff: {time_diff_sec:.3f} sec')
            self.save_synced(self.id, traj_msg, odo_msg)
            self.id += 1

            if self.id % self.save_interval == 0:
                self.save_to_json()

    def save_synced(self, id, traj_msg: Trajectory, odo_msg: Odometry):
        traj_data = []
        for point in traj_msg.points:
            pos = point.pose.position
            orientation = point.pose.orientation
            yaw = self.quaternion_to_yaw(orientation)
            traj_data.append({
                "trajectory_point": {
                    "position": {"x": pos.x, "y": pos.y, "z": pos.z},
                    "yaw_deg": math.degrees(yaw),
                    "velocity_mps": point.longitudinal_velocity_mps
                }
            })

        odom_pos = odo_msg.pose.pose.position
        odom_orientation = odo_msg.pose.pose.orientation
        odom_yaw = self.quaternion_to_yaw(odom_orientation)
        odom_vel = odo_msg.twist.twist.linear

        odom_data = {
            "odometry": {
                "position": {"x": odom_pos.x, "y": odom_pos.y, "z": odom_pos.z},
                "yaw_deg": math.degrees(odom_yaw),
                "linear_velocity": {"x": odom_vel.x, "y": odom_vel.y, "z": odom_vel.z}
            }
        }

        self.synced_data.append({
            "synced_sample": {
                "id": id,
                "trajectory": traj_data,
                "odometry": odom_data
            }
        })

    def quaternion_to_yaw(self, orientation):
        """Convert ROS quaternion to yaw using tf."""
        q = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(q)
        return yaw

    def save_to_json(self):
        output_path = os.path.join(os.getcwd(), self.filename)
        with open(output_path, 'w') as f:
            json.dump(self.synced_data, f, indent=2)
        self.get_logger().info(f'Saved {len(self.synced_data)} samples to {self.filename}')

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryOdometryParser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_to_json()
    finally:
        node.destroy_node()
        rclpy.shutdown()
