import cv2
import numpy as np

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
    ########## YOUR CODE STARTS HERE ##########

    height, width, _ = img.shape

    kernel = np.ones((5, 5), np.uint8)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    filt = cv2.inRange(image, np.array([7, 150, 150]), np.array([35, 255, 255]))
    filt = cv2.erode(filt, kernel, iterations=1)
    # return cv2.bitwise_and(img, img, mask=filt)
    contours, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
    xpad = round(w/2)
    ypad = round(h/2)

    bb_mask = np.zeros_like(filt)
    bb_mask[max(0, y-ypad):min(y+h+ypad, height-1), max(0, x-xpad):min(x+w+xpad, width-1)] = 255
    # return cv2.bitwise_and(img, img, mask=bb_mask)
    image = cv2.bitwise_and(image, image, mask=bb_mask)
    filt = cv2.inRange(image, np.array([5, 150, 100]), np.array([35, 255, 255]))
    # return cv2.bitwise_and(img, img, mask=filt)
    contours, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
    # return cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    ########### YOUR CODE ENDS HERE ###########

    # Return bounding box
    return ((x, y), (x+w, y+h))

if __name__ == "__main__":
    og = cv2.imread('./test_images_cone/test2.jpg')
    bb = cd_color_segmentation(og, og)
    image_print(bb)

    # image_print(cv2.rectangle(og, *bb, (0, 255, 0), 2))