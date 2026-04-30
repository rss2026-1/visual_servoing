#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path

from homography_transformer import PTS_IMAGE_PLANE, PTS_GROUND_PLANE

# import your color segmentation algorithm; call this function in ros_image_callback!
from visual_servoing.computer_vision.lane_color_segmentation import lane_segmentation

class LaneDetector(Node):
    """
    A class for applying lane detection algorithms to the real robot.
    """


    def __init__(self):
        super().__init__("lane_detector")
        # toggle line follower vs cone parker
        self.LineFollower = False

        self.debug_pub = self.create_publisher(Image, "/bev_debug_img", 10)
        self.path_pub  = self.create_publisher(Path, "/lane_center_path", 10)
        self.image_sub = self.create_subscription(Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)
        self.bridge = CvBridge()

        # BEV homography — computed once, reused every frame
        BEV_W, BEV_H = 1000, 500
        X_MIN, X_MAX =  0, 80
        Y_MIN, Y_MAX = -80, 80
        self.BEV_W, self.BEV_H = BEV_W, BEV_H
        self.X_MIN, self.X_MAX = X_MIN, X_MAX
        self.Y_MIN, self.Y_MAX = Y_MIN, Y_MAX
        self.car_bev_x = int((Y_MAX - 0) / (Y_MAX - Y_MIN) * BEV_W)
        self.car_bev_y = int((X_MAX - 0) / (X_MAX - X_MIN) * BEV_H)

        pts_ground_world = np.array(PTS_GROUND_PLANE, dtype=np.float32)
        pts_ground_bev = np.column_stack([
            (Y_MAX - pts_ground_world[:, 1]) / (Y_MAX - Y_MIN) * BEV_W,
            (X_MAX - pts_ground_world[:, 0]) / (X_MAX - X_MIN) * BEV_H,
        ])
        pts_image = np.array(PTS_IMAGE_PLANE, dtype=np.float32)
        self.H, _ = cv2.findHomography(pts_image, pts_ground_bev.astype(np.float32))

        self.get_logger().info("Lane Detector Initialized")

    def image_callback(self, image_msg):
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        bev = cv2.warpPerspective(img, self.H, (self.BEV_W, self.BEV_H))
        cv2.drawMarker(bev, (self.car_bev_x, self.car_bev_y), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 20, 2)

        mask, left_fit, right_fit, center_fit, bottom_x_left, bottom_x_right, bottom_x_center, debug = lane_segmentation(bev)

        # Sanity check: left and right lines should be ~1 m apart
        if left_fit is not None and right_fit is not None:
            y_left  = (self.Y_MAX - bottom_x_left  / self.BEV_W * (self.Y_MAX - self.Y_MIN)) * 0.0254
            y_right = (self.Y_MAX - bottom_x_right / self.BEV_W * (self.Y_MAX - self.Y_MIN)) * 0.0254
            lane_width_m = abs(y_right - y_left)
            if not (0.8 < lane_width_m < 1.2):
                self.get_logger().warn(
                    f"Lane width {lane_width_m:.2f} m out of range, discarding left/right fits")
                left_fit = right_fit = center_fit = None

        if center_fit is not None:
            self.path_pub.publish(self._fit_to_path(center_fit, image_msg.header))

        debug_msg = self.bridge.cv2_to_imgmsg(debug, "bgr8")
        self.debug_pub.publish(debug_msg)

    def _fit_to_path(self, center_fit, header):
        """
        Evaluate center_fit at 20 evenly-spaced BEV rows and convert each
        point from BEV pixels -> real-world car frame (meters, x=forward, y=left).
        """
        path = Path()
        path.header = header
        path.header.frame_id = "base_link"

        bev_ys = np.linspace(0, self.BEV_H - 1, 20)
        bev_xs = np.polyval(center_fit, bev_ys)

        for bev_x, bev_y in zip(bev_xs, bev_ys):
            # inverse of the BEV pixel transform
            x_world = (self.X_MAX - bev_y / self.BEV_H * (self.X_MAX - self.X_MIN)) * 0.0254  # inches → meters
            y_world = (self.Y_MAX - bev_x / self.BEV_W * (self.Y_MAX - self.Y_MIN)) * 0.0254

            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x_world
            pose.pose.position.y = y_world
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path


def main(args=None):
    rclpy.init(args=args)
    cone_detector = LaneDetector()
    rclpy.spin(cone_detector)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
