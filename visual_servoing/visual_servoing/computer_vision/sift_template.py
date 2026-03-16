import cv2
import imutils
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
    Helper function to print out images, for debugging.
    Press any key to continue.
    """
    winname = "Image"
    cv2.namedWindow(winname)         # Create a named window
    cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cd_sift_ransac(img, template):
    """
    Implement the cone detection using SIFT + RANSAC algorithm.
    Input:
        img: np.3darray; the input image with a cone to be detected
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box in image coordinates (Y increasing downwards),
            where (x1, y1) is the top-left pixel of the box
            and (x2, y2) is the bottom-right pixel of the box.
    """
    # Minimum number of matching features
    MIN_MATCH = 10  # Adjust this value as needed
    # Create SIFT
    sift = cv2.SIFT_create()

    # Compute SIFT on template and test image
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(img, None)

    # Find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Find and store good matches
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # If enough good matches, find bounding box
    if len(good) > MIN_MATCH:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Create mask
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        ########## YOUR CODE STARTS HERE ##########

        # Project the template corners into the test image using the homography.
        # cv2.perspectiveTransform applies M to each point in pts, mapping
        # template pixel coordinates -> corresponding locations in the test image.
        dst = cv2.perspectiveTransform(pts, M)
        # dst shape: (4, 1, 2) — four (x, y) points in the test image.

        # Flatten to (4, 2) so we can easily extract x and y coordinates.
        dst = dst.reshape(-1, 2)

        # Compute the axis-aligned bounding box that encloses all four
        # projected corners.  np.min/max over the four points gives us the
        # tightest rectangle in the convention ((xmin, ymin), (xmax, ymax)).
        x_min = int(np.min(dst[:, 0]))
        y_min = int(np.min(dst[:, 1]))
        x_max = int(np.max(dst[:, 0]))
        y_max = int(np.max(dst[:, 1]))

        ########### YOUR CODE ENDS HERE ###########

        # Return bounding box
        return ((x_min, y_min), (x_max, y_max))
    else:
        print(f"[SIFT] not enough matches; matches: {len(good)}")

        # Return bounding box of area 0 if no match found
        return ((0, 0), (0, 0))


def cd_template_matching(img, template):
    """
    Implement the cone detection using template matching algorithm.
    Input:
        img: np.3darray; the input image with a cone to be detected
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box in px (Y increases downward),
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    """
    template_canny = cv2.Canny(template, 50, 200)

    # Perform Canny Edge detection on test image
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(grey_img, 50, 200)

    # Get dimensions of template
    (img_height, img_width) = img_canny.shape[:2]

    # Keep track of best-fit match
    best_match = None

    # Loop over different scales of image
    for scale in np.linspace(1.5, .5, 50):
        # Resize the image
        resized_template = imutils.resize(
            template_canny, width=int(template_canny.shape[1] * scale))
        (h, w) = resized_template.shape[:2]
        # Check to see if test image is now smaller than template image
        if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
            continue

        ########## YOUR CODE STARTS HERE ##########
        # Use OpenCV template matching functions to find the best match
        # across template scales.

        # Apply normalized cross-correlation between the Canny edge image
        # and the resized Canny template. TM_CCOEFF_NORMED scores range
        # from -1 to 1; higher means a better match. This is the
        # "Normalized Correlation" method described in lecture.
        result = cv2.matchTemplate(img_canny, resized_template, cv2.TM_CCOEFF_NORMED)

        # Find the location of the best score in the result map
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Keep track of the best match seen so far across all scales.
        # best_match stores (score, top-left location, template h, template w)
        if best_match is None or max_val > best_match[0]:
            best_match = (max_val, max_loc, h, w)

    # Unpack the best match found across all scales and build the bounding box.
    # max_loc is the top-left corner (x1, y1); add the template dimensions
    # at that scale to get the bottom-right corner (x2, y2).
    _, (x1, y1), best_h, best_w = best_match
    bounding_box = ((x1, y1), (x1 + best_w, y1 + best_h))

        ########### YOUR CODE ENDS HERE ###########

    return bounding_box
