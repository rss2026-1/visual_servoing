import cv2
import numpy as np

LANE_WIDTH_M = 1.1
INCHES_PER_M = 1.0 / 0.0254


def _get_mask(bev_img):
    img_hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(img_hsv, np.array([0, 0, 180]), np.array([180, 60, 255]))

    gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    sy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    sx = cv2.GaussianBlur(sx, (15, 15), 0)
    sy = cv2.GaussianBlur(sy, (15, 15), 0)
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

    img_hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(img_hsv, np.array([0, 0, 170]), np.array([180, 60, 255]))

    gray_f = gray_uint8.astype(np.float32)
    sx = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3))
    sy = np.abs(cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3))
    sx = cv2.GaussianBlur(sx, (15, 15), 0)
    sy = cv2.GaussianBlur(sy, (15, 15), 0)
    vertical_mask = np.uint8(sx > 1.2 * sy) * 255

    pre_mask = cv2.bitwise_and(color_mask, vertical_mask)

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
    hist = histogram.astype(float)
    peak1_x = int(np.argmax(hist))
    if hist[peak1_x] == 0:
        return None, None

    suppressed = hist.copy()
    suppressed[max(0, peak1_x - min_separation):min(len(hist), peak1_x + min_separation)] = 0

    peak2_x   = int(np.argmax(suppressed))
    peak2_val = suppressed[peak2_x]

    if peak2_val < 0.20 * hist[peak1_x]:
        return peak1_x, None

    return min(peak1_x, peak2_x), max(peak1_x, peak2_x)


def _hough_fit(mask, x_center, search_width):
    h, w = mask.shape
    x_lo = max(0, x_center - search_width)
    x_hi = min(w, x_center + search_width)

    strip = np.zeros_like(mask)
    strip[:, x_lo:x_hi] = mask[:, x_lo:x_hi]

    # Collect HoughLinesP segment endpoints only for debug visualization.
    lines = cv2.HoughLinesP(
        strip,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=40,
        maxLineGap=15,
    )
    dot_xs, dot_ys = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) > abs(x2 - x1):
                dot_xs.extend([x1, x2])
                dot_ys.extend([y1, y2])

    # Fit a line to ALL white pixels in the strip — far more robust than
    # fitting only the sparse HoughLinesP endpoints.
    ys, xs = np.nonzero(strip)
    if len(xs) < 20:
        return None, dot_xs, dot_ys

    coeffs = np.polyfit(ys, xs, 1)
    coeffs = np.concatenate([[0.0], coeffs])
    return coeffs, dot_xs, dot_ys


def _draw_debug(bev_img, mask, left_fit, right_fit, center_fit,
                left_xs, left_ys, right_xs, right_ys, center_x, discarded_fit=None,
                histogram=None, peak_left_x=None, peak_right_x=None, steps=None):
    vis = bev_img.copy()
    h, w = vis.shape[:2]

    overlay = np.zeros_like(vis)
    overlay[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

    for x, y in zip(left_xs, left_ys):
        cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 0), -1)
    for x, y in zip(right_xs, right_ys):
        cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)

    plot_ys = np.linspace(0, h - 1, h)
    if left_fit is not None:
        pts = np.column_stack([np.polyval(left_fit, plot_ys), plot_ys]).astype(np.int32)
        cv2.polylines(vis, [pts], False, (255, 100, 0), 2)
    if right_fit is not None:
        pts = np.column_stack([np.polyval(right_fit, plot_ys), plot_ys]).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 100, 255), 2)
    if discarded_fit is not None:
        pts = np.column_stack([np.polyval(discarded_fit, plot_ys), plot_ys]).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 0), 2)
    if center_fit is not None:
        pts = np.column_stack([np.polyval(center_fit, plot_ys), plot_ys]).astype(np.int32)
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
        steps_row = cv2.resize(steps_row, (vis.shape[1], STEP_H))
        vis = np.vstack([vis, steps_row])

    return vis


def lane_segmentation_hough(bev_img, bev_w=None, y_min=-80, y_max=80, use_contour=True):
    mask, _, steps = _get_mask(bev_img)
    if use_contour:
        mask, _, steps = _get_mask_contour(bev_img)

    h, w = mask.shape
    if bev_w is None:
        bev_w = w

    half_lane_in = LANE_WIDTH_M * INCHES_PER_M / 2.0
    half_lane_px = int(half_lane_in / (y_max - y_min) * bev_w)

    histogram = np.sum(mask[h // 4:, :], axis=0)
    peak_left_x, peak_right_x = _find_histogram_peaks(histogram, min_separation=half_lane_px)

    print(f"Before peak_left_x={peak_left_x}, peak_right_x={peak_right_x}")
    if peak_left_x is None:
        left_x = right_x = w // 2
    elif peak_right_x is None:
        full_lane_px = 2 * half_lane_px
        _, xs_a, _ = _hough_fit(mask, peak_left_x - full_lane_px, half_lane_px)
        _, xs_b, _ = _hough_fit(mask, peak_left_x + full_lane_px, half_lane_px)
        if len(xs_a) >= len(xs_b):
            print(f"[SINGLE PEAK] peak={peak_left_x} is RIGHT lane, guessing left at {peak_left_x - full_lane_px} ({len(xs_a)} vs {len(xs_b)} px)")
            left_x  = peak_left_x - full_lane_px
            right_x = peak_left_x
        else:
            print(f"[SINGLE PEAK] peak={peak_left_x} is LEFT lane, guessing right at {peak_left_x + full_lane_px} ({len(xs_b)} vs {len(xs_a)} px)")
            left_x  = peak_left_x
            right_x = peak_left_x + full_lane_px
    else:
        left_x  = peak_left_x
        right_x = peak_right_x
    print(f"After peak_left_x={peak_left_x}, peak_right_x={peak_right_x}")

    left_x  = int(np.clip(left_x,  0, w - 1))
    right_x = int(np.clip(right_x, 0, w - 1))
    print(f"left base x={left_x}, right base x={right_x}")

    # Cap search width to half the peak separation so windows never overlap.
    if peak_left_x is not None and peak_right_x is not None:
        SEARCH_WIDTH = min(half_lane_px, (right_x - left_x) // 2 - 5)
    else:
        SEARCH_WIDTH = half_lane_px
    SEARCH_WIDTH = max(SEARCH_WIDTH, 20)

    left_fit,  left_xs,  left_ys  = _hough_fit(mask, left_x,  SEARCH_WIDTH)
    right_fit, right_xs, right_ys = _hough_fit(mask, right_x, SEARCH_WIDTH)

    # CCW curvature check
    MAX_CURVATURE = 5e-4
    discarded_fit = None
    if left_fit is not None and (abs(left_fit[0]) > MAX_CURVATURE or left_fit[0] > 5e-5):
        print(f"[DISCARD] left_fit: a={left_fit[0]:.2e}")
        discarded_fit = left_fit
        left_fit, left_xs, left_ys = None, [], []
    if right_fit is not None and (abs(right_fit[0]) > MAX_CURVATURE or right_fit[0] > 5e-5):
        print(f"[DISCARD] right_fit: a={right_fit[0]:.2e}")
        if discarded_fit is None:
            discarded_fit = right_fit
        right_fit, right_xs, right_ys = None, [], []

    # curvature mismatch
    if left_fit is not None and right_fit is not None:
        if abs(left_fit[0] - right_fit[0]) > 0.001:
            if len(left_xs) >= len(right_xs):
                discarded_fit = right_fit
                right_fit, right_xs, right_ys = None, [], []
            else:
                discarded_fit = left_fit
                left_fit, left_xs, left_ys = None, [], []

    # lane width sanity check
    if left_fit is not None and right_fit is not None:
        y_bottom_tmp = h - 1
        bx_l = int(np.polyval(left_fit,  y_bottom_tmp))
        bx_r = int(np.polyval(right_fit, y_bottom_tmp))
        y_left_m  = (y_max - bx_l / bev_w * (y_max - y_min)) * 0.0254
        y_right_m = (y_max - bx_r / bev_w * (y_max - y_min)) * 0.0254
        lane_width_m = abs(y_right_m - y_left_m)
        if not (0.7 < lane_width_m < 1.5):
            if len(left_xs) >= len(right_xs):
                discarded_fit = right_fit
                right_fit, right_xs, right_ys = None, [], []
            else:
                discarded_fit = left_fit
                left_fit, left_xs, left_ys = None, [], []
        else:
            print(f"[OK] lane width {lane_width_m:.2f} m")

    y_bottom = h - 1
    if left_fit is not None and right_fit is not None:
        bottom_x_left   = int(np.polyval(left_fit,  y_bottom))
        bottom_x_right  = int(np.polyval(right_fit, y_bottom))
        bottom_x_center = (bottom_x_left + bottom_x_right) // 2
        center_fit = (left_fit + right_fit) / 2
    elif left_fit is not None:
        bottom_x_left   = int(np.polyval(left_fit, y_bottom))
        bottom_x_right  = None
        bottom_x_center = bottom_x_left + half_lane_px
        center_fit = left_fit.copy()
        center_fit[2] += half_lane_px
    elif right_fit is not None:
        bottom_x_left   = None
        bottom_x_right  = int(np.polyval(right_fit, y_bottom))
        bottom_x_center = bottom_x_right - half_lane_px
        center_fit = right_fit.copy()
        center_fit[2] -= half_lane_px
    else:
        bottom_x_left = bottom_x_right = bottom_x_center = None
        center_fit = None

    debug = _draw_debug(bev_img, mask, left_fit, right_fit, center_fit,
                        left_xs, left_ys, right_xs, right_ys, bottom_x_center,
                        discarded_fit, histogram, peak_left_x, peak_right_x, steps)

    return mask, left_fit, right_fit, center_fit, bottom_x_left, bottom_x_right, bottom_x_center, debug
