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

        self.parking_distance = 0.75  # meters; desired stop distance from cone
        self.relative_x = 0
        self.relative_y = 0

        self.wheelbase = 0.325        # meters
        self.max_speed = 1.0          # m/s
        self.max_steering = 0.34      # radians (~20 deg)
        self.parking_tolerance = 0.03  # meters; stop if within this of parking_distance
        self.reversing = False

        self.get_logger().info("Parking Controller Initialized")

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        drive_cmd = AckermannDriveStamped()

        #################################
        # Pure Pursuit Controller
        #
        # The cone at (relative_x, relative_y) is used as the lookahead point.
        # Pure pursuit computes the steering angle to follow an arc through it:
        #
        #   steering = atan(2 * wheelbase * sin(alpha) / d)
        #
        #   alpha = heading angle to the lookahead point
        #   d     = Euclidean distance to the lookahead point
        #
        # Speed is proportional to the signed distance error (positive = drive
        # forward, negative = reverse), and clamped to max_speed.
        #################################

        d = np.sqrt(self.relative_x**2 + self.relative_y**2)
        distance_error = d - self.parking_distance

        # Pure pursuit steering toward the cone
        alpha = np.arctan2(self.relative_y, self.relative_x)
        steering = np.arctan(2.0 * self.wheelbase * np.sin(alpha) / d) if d > 0.01 else 0.0
        steering = np.clip(steering, -self.max_steering, self.max_steering)

        # Update reversing state: start reversing when parked but cone drifted behind,
        # stop reversing once cone is back in front
        if self.reversing and self.relative_x > 0:
            self.reversing = False
        elif abs(distance_error) < self.parking_tolerance and self.relative_x < self.parking_distance:
            self.reversing = True

        if d < 0.01:
            speed, steering = 0.0, 0.0
        elif self.reversing or self.relative_x < 0:
            # Cone is behind or we need to back up — reverse straight
            speed, steering = -self.max_speed, steering
        elif abs(distance_error) < self.parking_tolerance:
            # Within tolerance and aligned — stop
            speed, steering = 0.0, 0.0
        else:
            # Normal forward pure pursuit
            speed = np.clip(distance_error, -self.max_speed, self.max_speed)
            if speed < 0:
                steering = steering

        self.get_logger().info(f"dist_err: {distance_error:.2f}  speed: {speed:.2f}  reversing: {self.reversing}")
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
