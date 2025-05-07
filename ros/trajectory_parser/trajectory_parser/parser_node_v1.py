# import math 
# import rclpy
# from rclpy.node import Node
# from tf_transformations import euler_from_quaternion

# from autoware_planning_msgs.msg import Trajectory
# from nav_msgs.msg import Odometry

# class TrajectoryOdometryParser(Node):
#     def __init__(self):
#         super().__init__('trajectory_parser')

#         # Trajectory subscriber
#         self.create_subscription(
#             Trajectory,
#             '/planning/scenario_planning/trajectory',  
#             self.trajectory_callback,
#             2
#         )

#         # Odometry subscriber
#         self.create_subscription(
#             Odometry,
#             '/localization/kinematic_state',  
#             self.odometry_callback,
#             2
#         )

#         self.get_logger().info('Trajectory and Odometry Parser Node has been started.')

#     def trajectory_callback(self, msg: Trajectory):
#         self.get_logger().info(f'Received trajectory with {len(msg.points)} points.')
#         for idx, point in enumerate(msg.points):
#             position = point.pose.position
#             orientation = point.pose.orientation
#             yaw = self.quaternion_to_yaw(orientation)
#             velocity = point.longitudinal_velocity_mps
#             self.get_logger().info(
#                 f'Point {idx}: Position(x={position.x:.2f}, y={position.y:.2f}, z={position.z:.2f}), '
#                 f'Yaw={math.degrees(yaw):.2f} deg, '
#                 f'Velocity={velocity:.2f} m/s'
#             )

#     def odometry_callback(self, msg: Odometry):
#         pos = msg.pose.pose.position
#         orientation = msg.pose.pose.orientation
#         vel = msg.twist.twist.linear
#         yaw = self.quaternion_to_yaw(orientation)
#         self.get_logger().info(
#             f'Odometry: Position(x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}), '
#             f'Yaw={math.degrees(yaw):.2f} deg, '
#             f'Linear Velocity(x={vel.x:.2f}, y={vel.y:.2f}, z={vel.z:.2f})'
#         )
    
#     def quaternion_to_yaw(self, orientation):
#         q = [orientation.x, orientation.y, orientation.z, orientation.w]
#         _, _, yaw = euler_from_quaternion(q)
#         return yaw

# def main(args=None):
#     rclpy.init(args=args)
#     node = TrajectoryOdometryParser()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

import rclpy
from rclpy.node import Node
from autoware_planning_msgs.msg import Trajectory
from nav_msgs.msg import Odometry
import math
from tf_transformations import euler_from_quaternion
import json
from collections import deque

class TrajectoryOdometryParser(Node):
    def __init__(self):
        super().__init__('trajectory_parser')
        self.id = 0  
        self.create_timer(0.5, self.timer_callback)

        # Subscribe to topics
        self.create_subscription(Trajectory, '/planning/scenario_planning/trajectory', self.trajectory_callback, 2)
        self.create_subscription(Odometry, '/localization/kinematic_state', self.odometry_callback, 2)

        # Buffers to hold recent messages
        self.trajectory_buffer = deque(maxlen=100)  # recent 100
        self.odometry_buffer = deque(maxlen=100)

        # Synced data
        self.synced_data = []

        self.get_logger().info('Trajectory and Odometry Parser Node has been started.')

    def trajectory_callback(self, msg: Trajectory):
        now = self.get_clock().now().nanoseconds
        self.trajectory_buffer.append((now, msg))
        self.sync_data()

    def odometry_callback(self, msg: Odometry):
        now = self.get_clock().now().nanoseconds
        self.odometry_buffer.append((now, msg))
        self.sync_data()

    def sync_data(self):
        if not self.trajectory_buffer or not self.odometry_buffer:
            return

        traj_time, traj_msg = self.trajectory_buffer[-1]
        odo_time, odo_msg = self.odometry_buffer[-1]

        time_diff_sec = abs(traj_time - odo_time) / 1e9  # nanoseconds to seconds

        if time_diff_sec < 0.1:  # if within 0.1 sec, accept as synced
            self.get_logger().info(f'[{self.id}] Sync found with time diff: {time_diff_sec:.3f} sec')
            self.save_synced(self.id, traj_msg, odo_msg)
            self.id += 1 

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
                "id": self.id,
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
        with open('synced_trajectory_odometry.json', 'w') as f:
            json.dump(self.synced_data, f, indent=2)
        self.get_logger().info('Saved synced data to synced_trajectory_odometry.json')

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