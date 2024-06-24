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
        # obox = obox.replace("-", "")
        text.append(obox)

    text = "\n".join(text)

    with open(path, "w") as file:
        file.write(text)


def get_iou(boxA, boxB, H, W):
    boxA = np.around(boxA).reshape(-1, 1, 2).astype(np.int32)
    boxB = np.around(boxB).reshape(-1, 1, 2).astype(np.int32)

    canvasA = np.zeros((H, W), dtype=np.uint8)
    canvasB = np.zeros((H, W), dtype=np.uint8)

    cv2.drawContours(canvasA, [boxA], -1, 1, -1)
    cv2.drawContours(canvasB, [boxB], -1, 1, -1)

    boxAArea = canvasA.sum()
    boxBArea = canvasB.sum()

    intersection = cv2.bitwise_and(canvasA, canvasB)

    interArea = intersection.sum()

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def filter_boxes(left_top_coords, right_bottom_coords, boxes, H, W):
    (x_left, y_top) = left_top_coords
    (x_right, y_bottom) = right_bottom_coords
    filtered_boxes = []

    for box in boxes:
        iou = get_iou(
            [x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom],
            box[1:],
            H,
            W,
        )
        if iou > 0.5:
            filtered_boxes.append(box)

    return filtered_boxes


def get_end_points(box):
    box = np.around(box).reshape(-1, 2)
    x_left = box[:, 0].min()
    x_right = box[:, 0].max()
    y_top = box[:, 1].min()
    y_bottom = box[:, 1].max()

    return round(x_left), round(y_top), round(x_right), round(y_bottom)


def cut(img, boxes):
    H, W, _ = img.shape

    box = random.choice(boxes)
    x_left, y_top, x_right, y_bottom = get_end_points(box[1:])
    w = x_right - x_left
    h = y_bottom - y_top

    cut_left = round(x_left + w / 2 - WIDTH / 2)
    cut_right = cut_left + WIDTH
    cut_top = round(y_top + h / 2 - HEIGHT / 2)
    cut_bottom = cut_top + HEIGHT

    cut_left = max(0, cut_left)
    cut_right = min(W, cut_right)
    cut_top = max(0, cut_top)
    cut_bottom = min(H, cut_bottom)

    cutted_img = img[cut_top:cut_bottom, cut_left:cut_right]

    boxes = filter_boxes((cut_left, cut_top), (cut_right, cut_bottom), boxes, H, W)

    for i, box in enumerate(boxes):
        cls, x1, y1, x2, y2, x3, y3, x4, y4 = box

        x1 = x1 - cut_left
        x2 = x2 - cut_left
        x3 = x3 - cut_left
        x4 = x4 - cut_left

        y1 = y1 - cut_top
        y2 = y2 - cut_top
        y3 = y3 - cut_top
        y4 = y4 - cut_top

        boxes[i] = [cls, x1, y1, x2, y2, x3, y3, x4, y4]

    return cutted_img, boxes


drop_counter = 0


def get_dist(x1, y1, x2, y2):
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def reshape(img, boxes):
    global drop_counter

    H, W, _ = img.shape

    mean_area = 0.0
    for box in boxes:
        cls, x1, y1, x2, y2, x3, y3, x4, y4 = box

        area = get_dist(x1, y1, x2, x2) * get_dist(x1, y1, x4, x4)
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
        cls, x1, y1, x2, y2, x3, y3, x4, y4 = box

        x1 = x1 * scale_factor
        x2 = x2 * scale_factor
        x3 = x3 * scale_factor
        x4 = x4 * scale_factor
        y1 = y1 * scale_factor
        y2 = y2 * scale_factor
        y3 = y3 * scale_factor
        y4 = y4 * scale_factor

        boxes[i] = [cls, x1, y1, x2, y2, x3, y3, x4, y4]

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
        cls, x1, y1, x2, y2, x3, y3, x4, y4 = box

        x1 = x1 + shift_x
        x2 = x2 + shift_x
        x3 = x3 + shift_x
        x4 = x4 + shift_x

        y1 = y1 + shift_y
        y2 = y2 + shift_y
        y3 = y3 + shift_y
        y4 = y4 + shift_y

        boxes[i] = [cls, x1, y1, x2, y2, x3, y3, x4, y4]

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

        # !!!!!!!!!!
        img_path = img_path.split("/")
        img_path[-3] = "train_"
        img_path = "/".join(img_path)

        ann_path = ann_path.split("/")
        ann_path[-2] = "olabels"
        ann_path[-3] = "transformed_train"
        ann_path = "/".join(ann_path)
        # !!!!!!!!!!

        img = cv2.imread(img_path)
        H, W, _ = img.shape
        boxes = read_boxes(ann_path)
        boxes = normalize_boxes(boxes, img.shape)

        new_img, new_boxes, info = reshape(img, boxes)
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
