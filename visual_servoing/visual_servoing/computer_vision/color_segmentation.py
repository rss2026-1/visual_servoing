import cv2
import numpy as np
import cv_test
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


PARAMS_FILE = "line_params.json"
FROZEN = set()

def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
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

    params = np.array([7,150,147,40,255,255,8,178,75,37,255,254])
    return color_segmentation(
        img, template, params[0:3], params[3:6], params[6:9], params[9:]
    )

def color_segmentation(img, template, lower1, upper1, lower2, upper2):
    height, width, _ = img.shape
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    filt = cv2.inRange(image, lower1, upper1)
    filt = cv2.dilate(filt, kernel, iterations=1)
    # return cv2.bitwise_and(img, img, mask=filt)

    contours, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
    xpad, ypad = round(w / 2), round(h / 2)

    bb_mask = np.zeros_like(filt)
    bb_mask[max(0, y-ypad):min(y+h+ypad, height-1),
            max(0, x-xpad):min(x+w+xpad, width-1)] = 255

    image = cv2.bitwise_and(image, image, mask=bb_mask)
    # return cv2.bitwise_and(img, img, mask=bb_mask)
    filt = cv2.inRange(image, lower2, upper2)
    filt = cv2.dilate(filt, kernel, iterations=1)
    # return cv2.bitwise_and(img, img, mask=filt)

    contours, _ = cv2.findContours(filt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = max([cv2.boundingRect(c) for c in contours], key=lambda r: r[2] * r[3])
    # return cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    return ((x, y), (x + w, y + h))

def perturb(params, step=3, all=False):
    mutable = [i for i in range(len(params)) if i not in FROZEN]

    if all:
        indices = mutable
    else:
        indices = [random.choice(mutable)]

    candidate = list(params)
    for idx in indices:
        delta = random.randint(-step, step)
        candidate[idx] = int(np.clip(candidate[idx] + delta, 0, 255))
    return candidate


def evaluate(params, samples=10):
    lower1 = np.array(params[0:3])
    upper1 = np.array(params[3:6])
    lower2 = np.array(params[6:9])
    upper2 = np.array(params[9:12])

    def segmentation_fn(img, template):
        return color_segmentation(img, template, lower1, upper1, lower2, upper2)

    try:
        scores = cv_test.test_algorithm(
            segmentation_fn,
            "./line_images/test_line.csv",
            './test_images_cone/cone_template.png'
        )
        punish = 0
        for val in scores.values():
            if val == 0:
                punish += 1/samples
        return scores, sum(scores.values())/samples - punish
    except Exception as e:
        return {}, -float('inf')

def value_finder(initial_params, step=3, reset_every=20, samples=10, all=False):
    current = list(initial_params)
    best = list(initial_params)
    scores, best_score = evaluate(initial_params, samples=samples)
    print(best_score)

    i = 0

    while True:
        if (i + 1) % reset_every == 0:
            i = 0
            current = list(best)
            print(f"[reset] back to best")

        candidate = perturb(current, step, all=all)
        scores, avg = evaluate(candidate, samples=samples)

        if avg > best_score:
            best, best_score = candidate, avg
            with open(PARAMS_FILE, "r+") as f:
                history = json.load(f)
                history.append({"params": best, "scores": scores, "score": best_score})
                history.sort(key=lambda x: x["score"], reverse=True)
                f.seek(0); f.truncate(); json.dump(history, f, indent=2)
            print(f"[{i+1:03d}] NEW BEST: score={best_score:.4f}  params={best}")
        elif avg < 0.7:
            i = 0
            current = list(best)
            print(f"[reset] back to best")
            continue
        else:
            print(f"[{i+1:03d}] score={avg:.4f}")
        current = candidate
        i += 1


if __name__ == "__main__":
    og = cv2.imread('./test_images_cone/test7.jpg')
    # image_print(og)
    bb = cd_color_segmentation(og, og)
    image_print(bb)

    # image_print(cv2.rectangle(og, *bb, (0, 255, 0), 2))
    # print(bb)
    # scores = cv_test.test_algorithm(
    #     cd_color_segmentation,
    #     "./new_images/test_new_cone.csv",
    #     './test_images_cone/cone_template.png'
    #     )
    
    # print(scores)
    # print(min(scores.values()))
    # print(sum(scores.values())/10)

    # best = [
    #   0,
    #   118,
    #   37,
    #   34,
    #   231,
    #   238,
    #   6,
    #   94,
    #   124,
    #   62,
    #   238,
    #   240
    # ]
    # value_finder(best, step=10, reset_every=25, samples=25, all=True)