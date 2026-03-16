#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

from vs_msgs.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped


class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic")
        DRIVE_TOPIC = self.get_parameter("drive_topic").value  # set in launch file; different for simulator vs racecar

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.create_subscription(
            ConeLocation, "/relative_cone", self.relative_cone_callback, 1)

        self.parking_distance = 0.5  # meters; desired stop distance from cone
        self.relative_x = 0
        self.relative_y = 0

        self.wheelbase = 0.35        # meters
        self.max_speed = 1.0          # m/s
        self.max_steering = 0.34      # radians (~20 deg)
        self.parking_tolerance = 0.03  # meters; stop if within this of parking_distance

        self.kturn_active = False
        self.R_min = self.wheelbase / np.tan(self.max_steering)  # ~0.92 m
        self.buffer = 0.3

        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        dist = np.sqrt(self.relative_x**2 + self.relative_y**2)
        distance_error = dist - self.parking_distance

        # Pure pursuit steering toward the cone
        alpha = np.arctan2(self.relative_y, self.relative_x)
        steering = np.arctan(2.0 * self.wheelbase * np.sin(alpha) / dist)
        steering = np.clip(steering, -self.max_steering, self.max_steering)


        if dist < self.R_min + self.buffer and abs(alpha) > np.radians(50):
            self.kturn_active = True

        if self.kturn_active and abs(alpha) < np.radians(20):
            self.kturn_active = False

        cone_is_behind = self.relative_x < 0
        cone_is_to_side = abs(alpha) > np.radians(70)


        if self.kturn_active:
            # K-turn phase 1: reverse with max steering to swing nose toward cone.
            speed = -0.5
            steering = -np.sign(alpha) * self.max_steering
        elif cone_is_behind or cone_is_to_side:
            # Cone is behind or far to the side — back up with flipped pursuit steering.
            # -steering while reversing swings the nose toward the cone.
            speed = -0.5
            steering = -steering
        elif abs(distance_error) < self.parking_tolerance:
            # Within tolerance
            speed, steering = 0.0, 0.0
        else:
            # Normal forward pure pursuit
            speed = np.clip(distance_error, -self.max_speed, self.max_speed)
            if speed < 0:
                steering = -steering

        drive_cmd.drive.speed = speed
        drive_cmd.drive.steering_angle = steering

        self.drive_pub.publish(drive_cmd)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()

        d = np.sqrt(self.relative_x**2 + self.relative_y**2)
        error_msg.x_error = float(self.relative_x - self.parking_distance)
        error_msg.y_error = float(self.relative_y)
        error_msg.distance_error = float(d - self.parking_distance)

        self.error_pub.publish(error_msg)


def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
