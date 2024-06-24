import os
from glob import glob
from tqdm import tqdm
import shutil
import cv2
from tools import *
from draw_boxes import read_boxes, normalize_boxes
import numpy as np
from math import sqrt
import scipy.stats
import random
import json

if os.path.exists("combined_dataset/train"):
    shutil.rmtree("combined_dataset/train")

HEIGHT = 640
WIDTH = 640


def denormalize_boxes(boxes):
    for box in boxes:
        for i, el in enumerate(box):
            if i == 0:
                continue
            elif i % 2 == 1:
                box[i] = min(1, max(0, el / WIDTH))
            else:
                box[i] = min(1, max(0, el / HEIGHT))

    return boxes


def save_boxes(path, boxes):
    text = []
    for obox in boxes:
        obox = [str(el) for el in obox]
        obox = " ".join(obox)
        obox = obox.replace("-", "")
        text.append(obox)

    text = "\n".join(text)

    with open(path, "w") as file:
        file.write(text)


def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / boxBArea
    return iou


def filter_boxes(left_top_coords, right_bottom_coords, boxes):
    (x_left, y_top) = left_top_coords
    (x_right, y_bottom) = right_bottom_coords
    filtered_boxes = []

    for box in boxes:
        cls, x, y, w, h = box
        x1 = x - w / 2
        x2 = x + w / 2
        y1 = y - h / 2
        y2 = y + h / 2

        iou = get_iou((x_left, y_top, x_right, y_bottom), (x1, y1, x2, y2))

        # if ((x - w / 2) >= x_left) and ((x + w / 2) <= x_right) and ((y - h / 2) >= y_top) and ((y + h / 2) <= y_bottom):
        if iou > 0.5:
            if x1 >= x_left:
                assert x1 < x_right
                x2 = min(x2, x_right)
            else:
                assert x2 > x_left
                x1 = max(x1, x_left)

            if y1 >= y_top:
                assert y1 < y_bottom
                y2 = min(y2, y_bottom)
            else:
                assert y2 > y_top
                y1 = max(y1, y_top)

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            filtered_boxes.append((cls, x, y, w, h))

    return filtered_boxes


def cut(img, boxes):
    H, W, _ = img.shape

    box = random.choice(boxes)
    cls, x, y, w, h = box

    x_left = x - w / 2
    x_right = x + w / 2
    y_top = y - h / 2
    y_bottom = y + h / 2

    cut_left = round(x_left + w / 2 - WIDTH / 2)
    cut_right = cut_left + WIDTH
    cut_top = round(y_top + h / 2 - HEIGHT / 2)
    cut_bottom = cut_top + HEIGHT

    cut_left = max(0, cut_left)
    cut_right = min(W, cut_right)
    cut_top = max(0, cut_top)
    cut_bottom = min(H, cut_bottom)

    cutted_img = img[cut_top:cut_bottom, cut_left:cut_right]

    boxes = filter_boxes((cut_left, cut_top), (cut_right, cut_bottom), boxes)

    for i, box in enumerate(boxes):
        cls, x, y, w, h = box

        x = x - cut_left
        y = y - cut_top

        boxes[i] = [cls, x, y, w, h]

    return cutted_img, boxes


drop_counter = 0


def reshape(img, boxes, img_path):
    global drop_counter

    H, W, _ = img.shape

    mean_area = 0.0
    for box in boxes:
        _, x, y, w, h = box

        area = w * h
        mean_area += area

    mean_area = mean_area / len(boxes)

    lower = 1000
    upper = 30000  # 41706
    mu = 3073
    sigma = 5335

    target_area = scipy.stats.truncnorm.rvs(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=1
    )[0]

    if target_area < lower:
        target_area = lower

    scale_factor = sqrt(target_area / mean_area)
    if scale_factor > 2:
        scale_factor = 2

    new_H = round(scale_factor * H)
    new_W = round(scale_factor * W)
    # print((H, W), (new_H, new_W))

    img = cv2.resize(img, (new_W, new_H))

    for i, box in enumerate(boxes):
        cls, x, y, w, h = box

        x = x * scale_factor
        y = y * scale_factor
        w = w * scale_factor
        h = h * scale_factor

        boxes[i] = [cls, x, y, w, h]

    is_cutted = False
    if new_H > HEIGHT or new_W > WIDTH:
        img, boxes = cut(img, boxes)
        new_H, new_W, _ = img.shape
        is_cutted = True

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    shift_y = random.randint(0, HEIGHT - new_H)
    shift_x = random.randint(0, WIDTH - new_W)

    canvas[shift_y : shift_y + new_H, shift_x : shift_x + new_W] = img

    for i, box in enumerate(boxes):
        cls, x, y, w, h = box

        x = shift_x + x
        y = shift_y + y

        boxes[i] = [cls, x, y, w, h]

    info = {
        "mean_area": mean_area,
        "target_area": target_area,
        "scale_factor": scale_factor,
        "new_H": new_H,
        "new_W": new_W,
        "is_cutted": is_cutted,
    }

    return canvas, boxes, info


folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]
folder_paths = [
    folder_path
    for folder_path in folder_paths
    if folder_path.split("/")[-1].startswith("!")
]
folder_paths.sort()

infos = {}

for folder_path in folder_paths:
    print(folder_path)
    if os.path.exists(f"{folder_path}/transformed_train_reshaped"):
        shutil.rmtree(f"{folder_path}/transformed_train_reshaped")

    os.makedirs(f"{folder_path}/transformed_train_reshaped")
    os.makedirs(f"{folder_path}/transformed_train_reshaped/images")
    os.makedirs(f"{folder_path}/transformed_train_reshaped/labels")

    img_paths = glob(f"{folder_path}/transformed_train/images/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_paths.sort()

    for img_path in tqdm(img_paths):
        name, img_name, ann_name, img_path, ann_path, folder = get_names(
            img_path=img_path
        )

        # img_path = img_path.split("/")
        # img_path[-3] = "train_"
        # img_path = "/".join(img_path)

        img = cv2.imread(img_path)
        H, W, _ = img.shape
        boxes = read_boxes(ann_path)
        boxes = normalize_boxes(boxes, img.shape)

        new_img, new_boxes, info = reshape(img, boxes, img_path)
        infos[img_path] = info

        if new_img is None:
            continue
        new_boxes = denormalize_boxes(new_boxes)

        cv2.imwrite(
            f"{folder_path}/transformed_train_reshaped/images/{img_name}", new_img
        )
        save_boxes(
            f"{folder_path}/transformed_train_reshaped/labels/{ann_name}",
            new_boxes,
        )

with open("infos.json", "w") as file:
    json.dump(infos, file)

print("drop_counter", drop_counter)
