import rclpy 
from rclpy.node import Node  
from std_srvs.srv import SetBool
from nav_msgs.msg import Odometry
import time


class RLModeTrigger(Node):
    def __init__(self):
        super().__init__('rl_mode_service')
        self.client = self.create_client(SetBool, '/rl_planner/set_rl_mode')
        self.create_subscription(Odometry, '/localization/kinematic_state', self.odometry_callback, 10)
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        self.req = SetBool.Request()
        self.stop_cnt = 0
        self.driving_cnt = 0 
        self.rl_mode_active = False

    def _stop_filter(self, twist):
        return twist.linear.x < 0.01 and twist.linear.y < 0.01 and twist.angular.z < 0.01

    def odometry_callback(self, msg: Odometry):
        if self._stop_filter(msg.twist.twist):
            self.driving_cnt = 0 
            self.stop_cnt += 1
        else:
            self.stop_cnt = 0  
            self.driving_cnt += 1  

    def set_rl_mode(self, mode: bool):
        if self.rl_mode_active == mode:
            return False  

        self.req.data = mode
        future = self.client.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            result = future.result()
            if result.success:
                self.get_logger().info(f'Service response: {result.message}')
                self.rl_mode_active = mode
            else:
                self.get_logger().error(f'Service failed: {result.message}')
            return result.success
        else:
            self.get_logger().error('Service call failed (no result)')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = RLModeTrigger()
    
    try:
        while rclpy.ok():
            if node.stop_cnt != 0: 
                print(f"Stop count: {node.stop_cnt}") 
            elif node.driving_cnt != 0:
                print(f"Driving count: {node.driving_cnt}") 

            if node.stop_cnt == 10:
                node.set_rl_mode(True)
            if node.driving_cnt == 25:
                node.set_rl_mode(False)

            rclpy.spin_once(node, timeout_sec=1.0)
            time.sleep(1.0) 

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
