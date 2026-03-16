import cv2
import numpy as np
import random
import json

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################


PARAMS_FILE = "best_params.json"
FROZEN = {4, 5, 10, 11}

def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cd_color_segmentation(img, template):
    """
    Implement the cone detection using color segmentation algorithm
    Input:
        img: np.3darray; the input image with a cone to be detected. BGR.
        template: Not required, but can optionally be used to automate setting hue filter values.
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
            (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
    """

    # height, width, _ = img.shape

    # kernel = np.ones((5, 5), np.uint8)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # filt = cv2.inRange(image, np.array([7, 150, 150]), np.array([35, 255, 255]))
    # filt = cv2.erode(filt, kernel, iterations=1)
    # # return cv2.bitwise_and(img, img, mask=filt)
    # contours, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
    # xpad = round(w/2)
    # ypad = round(h/2)

    # bb_mask = np.zeros_like(filt)
    # bb_mask[max(0, y-ypad):min(y+h+ypad, height-1), max(0, x-xpad):min(x+w+xpad, width-1)] = 255
    # # return cv2.bitwise_and(img, img, mask=bb_mask)
    # image = cv2.bitwise_and(image, image, mask=bb_mask)
    # filt = cv2.inRange(image, np.array([5, 150, 100]), np.array([35, 255, 255]))
    # # return cv2.bitwise_and(img, img, mask=filt)
    # contours, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
    # # return cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # return ((x, y), (x+w, y+h))

    params = np.array([0,59,92,26,255,255,0,88,109,18,255,255])
    try:
        return color_segmentation(
            img, template, params[0:3], params[3:6], params[6:9], params[9:]
        )
    except Exception as e:
        return ((0, 0), (0, 0))

def color_segmentation(img, template, lower1, upper1, lower2, upper2):
    height, width, _ = img.shape
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    filt = cv2.inRange(image, lower1, upper1)
    filt = cv2.erode(filt, kernel, iterations=1)

    contours, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
    xpad, ypad = round(w / 2), round(h / 2)

    bb_mask = np.zeros_like(filt)
    bb_mask[max(0, y-ypad):min(y+h+ypad, height-1),
            max(0, x-xpad):min(x+w+xpad, width-1)] = 255

    image = cv2.bitwise_and(image, image, mask=bb_mask)
    filt = cv2.inRange(image, lower2, upper2)

    contours, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])

    return ((x, y), (x + w, y + h))
