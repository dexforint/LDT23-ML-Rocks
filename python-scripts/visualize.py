import cv2
import numpy as np
from glob import glob
from random import choice
import yaml


def resize_with_pad(image, dsize=(640, 640)):
    dw, dh = dsize
    dratio = dw / dh

    h, w, _ = image.shape
    ratio = w / h

    canvas = np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)

    if ratio >= dratio:
        coef = w / dw
        new_w = dw
        new_h = round(h / coef)
        image = cv2.resize(image, (new_w, new_h))
        y_shift = (dh - new_h) // 2
        x_shift = 0
    else:
        coef = h / dh
        new_h = dh
        new_w = round(w / coef)
        image = cv2.resize(image, (new_w, new_h))
        x_shift = (dw - new_w) // 2
        y_shift = 0

    canvas[y_shift : y_shift + new_h, x_shift : x_shift + new_w] = image

    convertion_info = {
        "h": h,
        "w": w,
        "new_h": new_h,
        "new_w": new_w,
        "dh": dh,
        "dw": dw,
        "y_shift": y_shift,
        "x_shift": x_shift,
        "coef": coef,
    }

    return canvas, convertion_info


img_paths = glob("train/images/*")
img_paths = [path.replace("\\", "/") for path in img_paths]
img_paths.sort()

print(len(img_paths))

img_path = choice(img_paths)
ann_path = ".".join(img_path.replace("images", "labels").split(".")[:-1]) + ".txt"


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]


def draw_boxes(img, boxes, title="Title"):
    img = img.copy()
    H, W, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        color = colors[box[0]]
        cls = classes[box[0]]

        if len(box) == 5:
            _, x, y, w, h = box
            x_left = int(x - w / 2)
            x_right = int(x + w / 2)
            y_top = int(y - h / 2)
            y_bottom = int(y + h / 2)

            cv2.rectangle(
                img, (x_left, y_top), (x_right, y_bottom), (0, 0, 255), H // 300
            )
            cv2.putText(
                img,
                cls,
                (x_left, y_top),
                font,
                1,
                color,
                H // 300,
                cv2.LINE_AA,
            )
        else:
            contour = np.array(box[1:])
            contour = contour.reshape(-1, 1, 2)
            x_left = np.min(contour[:, 0, 0])
            y_top = np.min(contour[:, 0, 1])

            cv2.putText(
                img,
                cls,
                (x_left, y_top),
                font,
                1,
                color,
                H // 300,
                cv2.LINE_AA,
            )

            cv2.drawContours(img, [contour], -1, color, 3)

    img, _ = resize_with_pad(img)

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_boxes(ann_path):
    with open(ann_path) as file:
        text = file.read()

    lines = text.split("\n")

    boxes = []

    for line in lines:
        if len(line) == 0:
            continue
        line = line.split()
        line = [int(line[0])] + [float(el) for el in line[1:]]

        boxes.append(line)

    return boxes


def normalize_boxes(boxes, shape):
    H, W, _ = shape

    norm_boxes = []
    for box in boxes:
        if len(box) == 5:
            cls, x, y, w, h = box
            x = int(x * W)
            w = int(w * W)
            y = int(y * H)
            h = int(h * H)

            norm_boxes.append((cls, x, y, w, h))

        for i, el in enumerate(box):
            if i == 0:
                continue
            elif i % 2 == 1:
                box[i] = int(el * W)
            else:
                box[i] = int(el * H)

            norm_boxes.append(box)

    return norm_boxes


with open("data.yaml", "r") as stream:
    try:
        classes = yaml.safe_load(stream)["names"]
    except yaml.YAMLError as exc:
        print(exc)


img = cv2.imread(img_path)
H, W, _ = img.shape
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

boxes = read_boxes(ann_path)
boxes = normalize_boxes(boxes, img.shape)
img_name = img_path.split("/")[-1]
print(img_name)
print(H, W)
draw_boxes(img, boxes, title=img_name)
