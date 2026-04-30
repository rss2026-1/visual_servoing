import argparse
import cv2
import numpy as np
from homography_transformer import PTS_IMAGE_PLANE, PTS_GROUND_PLANE
from visual_servoing.computer_vision.lane_color_segmentation import lane_segmentation

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=int, default=23, help="Image number (e.g. 24 for image24.png)")
parser.add_argument("--no-contour", action="store_true", help="Disable contour-based lane segmentation")
args = parser.parse_args()

use_contour = not args.no_contour
img_path = f"/home/racecar/racecar_ws/src/visual_servoing/visual_servoing/visual_servoing/computer_vision/racetrack_images/lane_1/image{args.image}.png"

# load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")


# BEV size
BEV_W, BEV_H = 1000, 500

# Real-world view range (inches): x=forward, y=left
X_MIN, X_MAX =  0, 90
Y_MIN, Y_MAX = -80, 80


pts_ground_world = np.array(PTS_GROUND_PLANE, dtype=np.float32)
pts_ground_bev = np.column_stack([
    (Y_MAX - pts_ground_world[:, 1]) / (Y_MAX - Y_MIN) * BEV_W,
    (X_MAX - pts_ground_world[:, 0]) / (X_MAX - X_MIN) * BEV_H,
])

PTS_IMAGE_PLANE = np.array(PTS_IMAGE_PLANE, dtype=np.float32)

H, _ = cv2.findHomography(PTS_IMAGE_PLANE, pts_ground_bev.astype(np.float32))

bev = cv2.warpPerspective(img, H, (BEV_W, BEV_H))

# Draw car position marker at bottom-center (x=0, y=0 in world coords)
car_bev_x = int((Y_MAX - 0) / (Y_MAX - Y_MIN) * BEV_W)
car_bev_y = int((X_MAX - 0) / (X_MAX - X_MIN) * BEV_H)
cv2.drawMarker(bev, (car_bev_x, car_bev_y), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 20, 2)

mask, left_fit, right_fit, center_fit, bottom_x_left, bottom_x_right, bottom_x_center, debug = lane_segmentation(bev, bev_w=BEV_W, y_min=Y_MIN, y_max=Y_MAX, use_contour=use_contour)

print(f"left_fit: {left_fit}")
print(f"right_fit: {right_fit}")
print(f"center_fit: {center_fit}")
print(f"bottom_x_left: {bottom_x_left}")
print(f"bottom_x_right: {bottom_x_right}")
print(f"bottom_x_center: {bottom_x_center}")

cv2.imshow("Original", img)
cv2.imshow("BEV", bev)
cv2.imshow("debug", debug)

while True:
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q') or key == 27:  # q or Escape
        break
    if cv2.getWindowProperty("Original", cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
