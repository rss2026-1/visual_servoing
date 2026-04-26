#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np

import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from visual_servoing.computer_vision.color_segmentation_final_challenge_b import cd_color_segmentation


class TrafficLightDetector(Node):
    """
    Detects a red traffic light using color segmentation.
    Publishes /red_light_detected (Bool) for the state machine.
    Does NOT publish /relative_cone_px — that topic is owned by the state machine
    (fed from YOLO detections) so both can share the homography pipeline cleanly.
    """

    MIN_DETECTION_AREA = 500  # pixels; below this we ignore as noise

    def __init__(self):
        super().__init__("traffic_light_detector")

        self.bridge = CvBridge()

        self.red_light_pub = self.create_publisher(Bool, "/red_light_detected", 10)
        self.debug_pub = self.create_publisher(Image, "/traffic_light_debug_img", 10)

        self.image_sub = self.create_subscription(
            Image, "/zed/zed_node/rgb/image_rect_color", self.image_callback, 5)

        self.get_logger().info("Traffic Light Detector Initialized")

    def image_callback(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        height, width, _ = image.shape

        # Crop to the region where a ~20 cm traffic light would appear.
        # percent=0.50 puts the band around the middle of the image; widen
        # the band (0.15) to be robust while we tune the exact height.
        filt = np.zeros_like(image)[:, :, 0]
        percent = 0.50
        top = int(round(height * (percent - 0.15)))
        bottom = int(round(height * percent))
        filt[top:bottom, 0:width] = 255
        focused = cv2.bitwise_and(image, image, mask=filt)

        detected = False
        try:
            ((xl, yl), (xh, yh)) = cd_color_segmentation(focused, focused)
            area = (xh - xl) * (yh - yl)
            if area >= self.MIN_DETECTION_AREA:
                detected = True
                boxed = cv2.rectangle(focused.copy(), (xl, yl), (xh, yh), (0, 0, 255), 2)
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(boxed, "bgr8"))
        except Exception:
            # No contours found — no red light visible
            pass

        msg = Bool()
        msg.data = detected
        self.red_light_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
