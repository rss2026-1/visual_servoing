import cv2
import numpy as np
import os
import glob

def _get_mask(bev_img):
    # color: white pixels
    img_hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(img_hsv, np.array([0, 0, 200]), np.array([180, 20, 255]))

    # direction: keep only vertically-structured features
    # Blur the Sobel responses over a neighbourhood so interior pixels of a
    # vertical line also register as "vertical", then keep where sobelx > sobely.
    gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    sx = cv2.GaussianBlur(sx, (15, 15), 0)
    sy = cv2.GaussianBlur(sy, (15, 15), 0)

    # can change coeff value of sy
    vertical_mask = np.uint8(sx > 1.2 * sy) * 255

    combined = cv2.bitwise_and(color_mask, vertical_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN,   kernel, iterations=1)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_DILATE, kernel, iterations=2)

    steps = [
        ("color",    color_mask),
        ("sobel",    vertical_mask),
        ("combined", combined),
        ("morph",    mask),
    ]
    return mask, None, steps


def _get_mask_contour(bev_img):
    gray_uint8 = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)

    # color mask. white pixels
    img_hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(img_hsv, np.array([0, 0, 200]), np.array([180, 25, 255]))

    # Sobel filter to only keep vertical features.
    # Blur the Sobel responses over a neighbourhood so interior pixels of
    # vertical line also register as vertical, then keep where sobelx > sobely.
    gray_f = gray_uint8.astype(np.float32)
    sx = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3))
    sy = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3))
    sx = cv2.GaussianBlur(sx, (15, 15), 0)
    sy = cv2.GaussianBlur(sy, (15, 15), 0)
    vertical_mask = np.uint8(sx > 1.2 * sy) * 255

    # Combine color + direction before contour analysis so horizontal
    # crossing lines are already removed and won't merge contours with vertical
    # lanes

    pre_mask = cv2.bitwise_and(color_mask, vertical_mask)

    # contour filter on the already vertical image to reject lane numbers.
    # Small fragments (from angled/broken lines) pass unconditionally —
    # only large blobs need the aspect ratio test.
    MIN_AREA = 200
    contours, _ = cv2.findContours(pre_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # if area < MIN_AREA:
        #     filtered.append(contour)
        #     continue
        _, (cw, ch), _ = cv2.minAreaRect(contour)
        if ch == 0:
            continue
        aspect_ratio = float(cw) / ch
        if aspect_ratio < 1/8 or aspect_ratio > 8:
            filtered.append(contour)

    contour_mask = np.zeros_like(gray_uint8)
    cv2.drawContours(contour_mask, filtered, -1, 255, thickness=cv2.FILLED)
    post_contour = cv2.bitwise_and(pre_mask, contour_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(post_contour, cv2.MORPH_OPEN,   kernel, iterations=1)
    mask = cv2.morphologyEx(mask,         cv2.MORPH_DILATE, kernel, iterations=2)

    steps = [
        ("color",    color_mask),
        ("sobel",    vertical_mask),
        ("combined", pre_mask),
        ("contour",  post_contour),
        ("morph",    mask),
    ]
    return mask, filtered, steps


def _find_histogram_peaks(histogram, min_separation):
    """
    Find the two strongest peaks in a histogram that are at least
    min_separation pixels apart.  Returns (left_x, right_x) sorted by
    position, or (peak_x, None) if only one meaningful peak exists.
    """
    hist = histogram.astype(float)

    peak1_x = int(np.argmax(hist))
    if hist[peak1_x] == 0:
        return None, None

    suppressed = hist.copy()
    lo = max(0, peak1_x - min_separation)
    hi = min(len(hist), peak1_x + min_separation)
    suppressed[lo:hi] = 0

    peak2_x   = int(np.argmax(suppressed))
    peak2_val = suppressed[peak2_x]

    if peak2_val < 0.20 * hist[peak1_x]:
        return peak1_x, None

    left_x  = min(peak1_x, peak2_x)
    right_x = max(peak1_x, peak2_x)
    return left_x, right_x


def _find_blob_peaks(mask, min_separation):
    """
    Find the x-centroids of the two largest connected blobs in *mask* that are
    at least min_separation pixels apart.  Returns (left_x, right_x) sorted by
    x, or (peak_x, None) if only one usable blob exists.

    Scoring by blob area rather than histogram column-sums handles angled lane
    lines correctly: an angled line spreads pixels across many columns (lower
    per-column histogram peak) but its total area still beats noise fragments.
    """
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None, None

    # Sort by area descending, skipping background (label 0)
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1

    best_xs = []
    for idx in order:
        cx = int(centroids[idx][0])
        if all(abs(cx - bx) >= min_separation for bx in best_xs):
            best_xs.append(cx)
        if len(best_xs) == 2:
            break

    if not best_xs:
        return None, None
    if len(best_xs) == 1:
        return best_xs[0], None

    return min(best_xs), max(best_xs)


def _sliding_window_fit(mask, base_x):
    N_WINDOWS    = 9
    WINDOW_WIDTH = 80
    MIN_PIXELS   = 50

    h, w = mask.shape
    window_h = h // N_WINDOWS

    current_x = base_x
    all_xs = []
    all_ys = []

    for i in range(N_WINDOWS):
        y_low  = h - (i + 1) * window_h
        y_high = h - i * window_h
        x_low  = max(0, current_x - WINDOW_WIDTH)
        x_high = min(w, current_x + WINDOW_WIDTH)

        ys, xs = np.where(mask[y_low:y_high, x_low:x_high] > 0)
        xs = xs + x_low
        ys = ys + y_low

        all_xs.extend(xs)
        all_ys.extend(ys)
        if len(xs) >= MIN_PIXELS:
            current_x = int(np.mean(xs))

    if len(all_xs) < MIN_PIXELS * 2:
        return None, [], []

    coeffs = np.polyfit(all_ys, all_xs, 2)
    return coeffs, all_xs, all_ys


def lane_segmentation(bev_img, bev_w=None, y_min=-80, y_max=80, use_contour=True):
    """
    bev_w  - pixel width of the BEV image (defaults to img width)
    y_min/y_max - real-world Y range in inches that spans bev_w pixels
                  (y_min is rightmost edge, y_max is leftmost edge)
    """
    LANE_WIDTH_M = 0.9         # expected track lane width in metres
    INCHES_PER_M = 1.0 / 0.0254

    mask, contours, steps = _get_mask(bev_img)
    if use_contour:
        mask, contours, steps = _get_mask_contour(bev_img)

    h, w = mask.shape
    if bev_w is None:
        bev_w = w

    #half lane width in pixels
    half_lane_in = LANE_WIDTH_M * INCHES_PER_M / 2.0        # inches
    half_lane_px = int(half_lane_in / (y_max - y_min) * bev_w)

    # base detection: find the two largest blobs in the bottom 3/4 of the mask.
    # Blob area is orientation-independent — an angled lane line has the same
    # total pixel count whether it's vertical or diagonal, unlike a histogram
    # column-sum which penalises non-vertical lines.
    bottom_mask = mask[h // 4:, :]
    peak_left_x, peak_right_x = _find_blob_peaks(bottom_mask, min_separation=half_lane_px)
    histogram = np.sum(bottom_mask, axis=0)   # kept for debug visualisation only

    if peak_left_x is None:
        # No signal
        left_x = right_x = w // 2
    elif peak_right_x is None:
        # Only one peak — try both sides and use pixel support to decide which lane it is
        full_lane_px = 2 * half_lane_px
        _, xs_a, _ = _sliding_window_fit(mask, peak_left_x - full_lane_px)
        _, xs_b, _ = _sliding_window_fit(mask, peak_left_x + full_lane_px)
        if len(xs_a) >= len(xs_b):
            left_x  = peak_left_x - full_lane_px
            right_x = peak_left_x
        else:
            left_x  = peak_left_x
            right_x = peak_left_x + full_lane_px
    else:
        left_x  = peak_left_x
        right_x = peak_right_x

    # Clamp to valid image range
    left_x  = int(np.clip(left_x,  0, w - 1))
    right_x = int(np.clip(right_x, 0, w - 1))

    left_fit,  left_xs,  left_ys  = _sliding_window_fit(mask, left_x)
    right_fit, right_xs, right_ys = _sliding_window_fit(mask, right_x)

    # --- CCW curvature check ---
    # all curves should bend left -> a < 0.
    # ignore crazy bends -> most likely errorneous
    # A positive 'a' means the fit captured noise/wrong structure.
    MAX_CURVATURE = 5e-4
    discarded_fit = None
    if left_fit is not None and (abs(left_fit[0]) > MAX_CURVATURE or left_fit[0] > 8e-5):
        discarded_fit = left_fit
        left_fit, left_xs, left_ys = None, [], []
    if right_fit is not None and (abs(right_fit[0]) > MAX_CURVATURE or right_fit[0] > 8e-5):
        if discarded_fit is None:
            discarded_fit = right_fit
        right_fit, right_xs, right_ys = None, [], []

    # curvature mismatch: keep the fit with more supporting pixels
    if left_fit is not None and right_fit is not None:
        if abs(left_fit[0] - right_fit[0]) > 0.001:
            if len(left_xs) >= len(right_xs):
                discarded_fit = right_fit
                right_fit, right_xs, right_ys = None, [], []
            else:
                discarded_fit = left_fit
                left_fit, left_xs, left_ys = None, [], []

    # lane-width sanity check
    # Both fits present but the detected lines are too close or too far apart.
    if left_fit is not None and right_fit is not None:
        y_bottom_tmp = h - 1
        bx_l = int(np.polyval(left_fit,  y_bottom_tmp))
        bx_r = int(np.polyval(right_fit, y_bottom_tmp))
        y_left_m  = (y_max - bx_l / bev_w * (y_max - y_min)) * 0.0254
        y_right_m = (y_max - bx_r / bev_w * (y_max - y_min)) * 0.0254
        lane_width_m = abs(y_right_m - y_left_m)
        if not (0.6 < lane_width_m < 1.2):
            if len(left_xs) >= len(right_xs):
                discarded_fit = right_fit
                right_fit, right_xs, right_ys = None, [], []
            else:
                discarded_fit = left_fit
                left_fit, left_xs, left_ys = None, [], []

    #  compute center
    y_bottom = h - 1
    if left_fit is not None and right_fit is not None:
        bottom_x_left   = int(np.polyval(left_fit,  y_bottom))
        bottom_x_right  = int(np.polyval(right_fit, y_bottom))
        bottom_x_center = (bottom_x_left + bottom_x_right) // 2
        center_fit = (left_fit + right_fit) / 2
    elif left_fit is not None:
        # left line found -> center is half a lane to the right
        bottom_x_left   = int(np.polyval(left_fit, y_bottom))
        bottom_x_right  = None
        bottom_x_center = bottom_x_left + half_lane_px
        center_fit = left_fit.copy()
        center_fit[2] += half_lane_px
    elif right_fit is not None:
        # right line found -> center is half a lane to the left
        bottom_x_left   = None
        bottom_x_right  = int(np.polyval(right_fit, y_bottom))
        bottom_x_center = bottom_x_right - half_lane_px
        center_fit = right_fit.copy()
        center_fit[2] -= half_lane_px
    else:
        bottom_x_left = bottom_x_right = bottom_x_center = None
        center_fit = None

    debug = _draw_debug(bev_img, mask, left_fit, right_fit, center_fit,
                        left_xs, left_ys, right_xs, right_ys, bottom_x_center, discarded_fit,
                        histogram, peak_left_x, peak_right_x, steps, contours)

    return mask, left_fit, right_fit, center_fit, bottom_x_left, bottom_x_right, bottom_x_center, debug


def _draw_debug(bev_img, mask, left_fit, right_fit, center_fit,
                left_xs, left_ys, right_xs, right_ys, center_x, discarded_fit=None,
                histogram=None, peak_left_x=None, peak_right_x=None, steps=None, contours=None):
    vis = bev_img.copy()
    h, w = vis.shape[:2]

    overlay = np.zeros_like(vis)
    overlay[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    if contours:
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0, 255, 255), 1)

    for x, y in zip(left_xs, left_ys):
        cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 0), -1)
    for x, y in zip(right_xs, right_ys):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

    plot_ys = np.linspace(0, h - 1, h)
    if left_fit is not None:
        plot_xs = np.polyval(left_fit, plot_ys)
        pts = np.column_stack([plot_xs, plot_ys]).astype(np.int32)
        cv2.polylines(vis, [pts], False, (255, 100, 0), 2)
    if right_fit is not None:
        plot_xs = np.polyval(right_fit, plot_ys)
        pts = np.column_stack([plot_xs, plot_ys]).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 100, 255), 2)
    if discarded_fit is not None:
        plot_xs = np.polyval(discarded_fit, plot_ys)
        pts = np.column_stack([plot_xs, plot_ys]).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 0), 2)
    if center_fit is not None:
        plot_xs = np.polyval(center_fit, plot_ys)
        pts = np.column_stack([plot_xs, plot_ys]).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 255, 0), 2)

    if center_x is not None:
        cv2.line(vis, (center_x, 0), (center_x, h), (0, 255, 255), 2)

    if histogram is not None:
        HIST_H = 120
        hist_img = np.zeros((HIST_H, w, 3), dtype=np.uint8)
        norm = (histogram / (histogram.max() + 1e-6) * (HIST_H - 10)).astype(int)
        for x, val in enumerate(norm):
            cv2.line(hist_img, (x, HIST_H - 1), (x, HIST_H - 1 - val), (255, 255, 255), 1)
        if peak_left_x is not None:
            cv2.line(hist_img, (peak_left_x,  0), (peak_left_x,  HIST_H - 1), (255, 0, 0), 2)
        if peak_right_x is not None:
            cv2.line(hist_img, (peak_right_x, 0), (peak_right_x, HIST_H - 1), (0, 0, 255), 2)
        vis = np.vstack([vis, hist_img])

    if steps:
        STEP_H = 120
        step_imgs = []
        for label, m in steps:
            sh, sw = m.shape[:2]
            step_w = int(sw * STEP_H / sh)
            resized = cv2.resize(m, (step_w, STEP_H))
            bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR) if resized.ndim == 2 else resized
            cv2.putText(bgr, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            step_imgs.append(bgr)
        steps_row = np.hstack(step_imgs)
        # scale the whole row to exactly match vis width
        steps_row = cv2.resize(steps_row, (vis.shape[1], STEP_H))
        vis = np.vstack([vis, steps_row])

    return vis

if __name__ == "__main__":
   source_folder = '../../../../racetrack_images/lane_3'
   destination_folder = 'racetrack_images/lane_3_out'
   os.makedirs(destination_folder, exist_ok=True)

   for image_path in glob.glob(os.path.join(source_folder, '*.png')):
      print(image_path)
      original = cv2.imread(image_path,cv2.IMREAD_COLOR)
      processed_image, *rest = lane_segmentation(original)

      filename = os.path.basename(image_path)
      cv2.imwrite(os.path.join(destination_folder, filename), processed_image)
