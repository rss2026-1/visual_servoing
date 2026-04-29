import cv2
import numpy as np
from homography_transformer import PTS_IMAGE_PLANE, PTS_GROUND_PLANE
from visual_servoing.computer_vision.lane_color_segmentation import lane_segmentation

# load image
img_path = "/home/racecar/racecar_ws/src/visual_servoing/visual_servoing/visual_servoing/computer_vision/racetrack_images/lane_3/image4.png"

# load image
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at {img_path}")


# BEV size
BEV_W, BEV_H = 1000, 500

# Real-world view range (inches): x=forward, y=left
X_MIN, X_MAX =  0, 80
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

mask, left_fit, right_fit, center_fit, bottom_x_left, bottom_x_right, bottom_x_center, debug = lane_segmentation(bev, bev_w=BEV_W, y_min=Y_MIN, y_max=Y_MAX)

print(f"left_fit: {left_fit}")
print(f"right_fit: {right_fit}")
print(f"center_fit: {center_fit}")
print(f"bottom_x_left: {bottom_x_left}")
print(f"bottom_x_right: {bottom_x_right}")
print(f"bottom_x_center: {bottom_x_center}")

mask, _, _, _, _, _, _, _ = lane_segmentation(bev, bev_w=BEV_W, y_min=Y_MIN, y_max=Y_MAX)  # get mask for histogram
h, w = mask.shape
midpoint = w // 2
histogram = np.sum(mask, axis=0)
left_x  = int(np.argmax(histogram[:midpoint]))
right_x = int(np.argmax(histogram[midpoint:])) + midpoint
print(f"left base x={left_x}, right base x={right_x}")

hist_img = np.zeros((200, w, 3), dtype=np.uint8)
norm = (histogram / (histogram.max() + 1e-6) * 190).astype(int)
for x, val in enumerate(norm):
    cv2.line(hist_img, (x, 199), (x, 199 - val), (255, 255, 255), 1)
cv2.line(hist_img, (midpoint, 0), (midpoint, 199), (0, 255, 0), 1)
cv2.line(hist_img, (left_x,   0), (left_x,   199), (255, 0, 0), 2)
cv2.line(hist_img, (right_x,  0), (right_x,  199), (0, 0, 255), 2)

cv2.imshow("Original", img)
cv2.imshow("BEV", bev)
cv2.imshow("debug", debug)
cv2.imshow("histogram", hist_img)

while True:
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q') or key == 27:  # q or Escape
        break
    if cv2.getWindowProperty("Original", cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
