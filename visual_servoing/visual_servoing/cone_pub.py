#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vs_msgs.msg import ConeLocation

class ConePublisher(Node):
    def __init__(self):
        super().__init__("cone_publisher")

        self.pub = self.create_publisher(ConeLocation, "/relative_cone", 10)
        self.create_timer(0.1, self.publish_cone)

        # Set the cone position here (metres, car frame: +x forward, +y left)
        self.cone_x = 2.0
        self.cone_y = 0.0

        self.get_logger().info(f"Publishing cone at x={self.cone_x}, y={self.cone_y}")

    def publish_cone(self):
        msg = ConeLocation()
        msg.x_pos = self.cone_x
        msg.y_pos = self.cone_y
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ConePublisher())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
