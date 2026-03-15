#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import torch

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dataclasses import dataclass
from rclpy.node import Node
from typing import List
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # Bounding box coordinates in the original image:
    x1: int
    y1: int
    x2: int
    y2: int


class YoloAnnotatorNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_annotator")

        # Declare and get ROS parameters
        self.model_name = (
            self.declare_parameter("model", "yolo11n.pt")
            .get_parameter_value()
            .string_value
        )
        self.conf_threshold = (
            self.declare_parameter("conf_threshold", 0.5)
            .get_parameter_value()
            .double_value
        )
        self.iou_threshold = (
            self.declare_parameter("iou_threshold", 0.7)
            .get_parameter_value()
            .double_value
        )

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.model_name)
        self.model.to(self.device)

        self.class_color_map = self.get_class_color_map()
        self.allowed_cls = [
            i for i, name in self.model.names.items()
            if name in self.class_color_map
        ]

        self.get_logger().info(f"Running {self.model_name} on device {self.device}")
        self.get_logger().info(f"Confidence threshold: {self.conf_threshold}")
        if self.allowed_cls:
            self.get_logger().info(f"You've chosen to keep these class IDs: {self.allowed_cls}")
        else:
            self.get_logger().warn("No allowed classes matched the model's class list.")

        # Create publisher and subscribers
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, "/zed/zed_node/rgb/image_rect_color", self.on_image, 10)
        self.pub = self.create_publisher(
            Image, "/yolo/annotated_image", 10)

    def get_class_color_map(self) -> dict[str, tuple[int, int, int]]:
        """
        Return a dictionary mapping a list of COCO class names you want to keep
        to the detection BGR colors in the annotated image. COCO class names include
        "chair", "couch", "tv", "laptop", "dining table", and many more. The list
        of available classes can be found in `self.model.names`.
        """
        return {
            "person": (0, 255, 0), # green
            "chair": (255, 0, 0), # blue
            "cone": (255, 165, 0), # orange
            "backpack": (0, 255, 255), # yellow
            "bottle": (255, 0, 255), # magenta
        }

    def on_image(self, msg: Image) -> None:
        # Convert ROS -> OpenCV (BGR)
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        # Run YOLO inference
        try:
            results = self.model(
                bgr,
                classes=self.allowed_cls,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        if not results:
            return

        # Convert results to Detection List
        dets = self.results_to_detections(results[0])

        # Draw detections on BGR image
        annotated = self.draw_detections(bgr, dets)

        # Publish annotated BGR image
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        self.pub.publish(out_msg)

    def results_to_detections(self, result) -> List[Detection]:
        """
        Convert an Ultralytics result into a Detection list.

        YOLOv11 outputs:
          result.boxes.xyxy: (N, 4) tensor
          result.boxes.conf: (N,) tensor
          result.boxes.cls:  (N,) tensor
        """
        detections = []

        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        # Convert Torch tensors -> CPU numpy
        xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
        conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
        cls_np = cls.detach().cpu().numpy() if hasattr(cls, "detach") else np.asarray(cls)

        for box, confidence, class_id in zip(xyxy_np, conf_np, cls_np):
            # box is [x1, y1, x2, y2] in pixel coordinates (float)
            # class_id is a float from the tensor; cast to int for indexing
            cid = int(class_id)
            detections.append(Detection(
                class_id=cid,
                class_name=self.model.names[cid],
                confidence=float(confidence),
                x1=int(box[0]),
                y1=int(box[1]),
                x2=int(box[2]),
                y2=int(box[3]),
            ))

        return detections

    def draw_detections(
        self,
        bgr_image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:

        out_image = bgr_image.copy()

        for det in detections:
            # Get the bounding box corners from the Detection dataclass
            pt1 = (det.x1, det.y1) # top-left
            pt2 = (det.x2, det.y2) # bottom-right

            # Look up the BGR color assigned to this class
            color = self.class_color_map[det.class_name]

            # Draw the bounding box around the detection on the output image
            cv2.rectangle(out_image, pt1, pt2, color, thickness=2)

            # Build the label string: "class_name conf%"
            label = f"{det.class_name} {det.confidence:.0%}"

            # Position the label just above the top-left corner of the box.
            # If the box is near the top of the image, shift the text inside
            text_y = det.y1 - 8 if det.y1 - 8 > 10 else det.y1 + 16

            # Draw a filled rectangle behind the text so it is readable
            # regardless of background color
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                out_image,
                (det.x1, text_y - text_h - 2),
                (det.x1 + text_w, text_y + 2),
                color,
                thickness=cv2.FILLED,
            )

            # Draw the label text in black over the colored background
            cv2.putText(
                out_image,
                label,
                (det.x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0), # black text
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return out_image


def main() -> None:
    rclpy.init()
    node = YoloAnnotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
