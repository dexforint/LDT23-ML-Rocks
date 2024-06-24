import cv2
import numpy as np
from glob import glob
from random import choice
import yaml
import os
from tqdm import tqdm
from tools import *
import shutil

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]

colors = colors * 10

SUBFOLDER = "train"  # "transformed_train"
LABELFOLDER = "labels"


def draw_boxes(img, boxes, classes):
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

            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), color, H // 300)
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

    return img


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
            continue

        for i, el in enumerate(box):
            if i == 0:
                continue
            elif i % 2 == 1:
                box[i] = int(el * W)
            else:
                box[i] = int(el * H)

        norm_boxes.append(box)

    return norm_boxes


if __name__ == "__main__":
    folder_paths = ["combined_dataset/datasets"]
    folder_paths = [path.replace("\\", "/") for path in folder_paths]

    for folder_path in folder_paths:
        print(folder_path)
        if os.path.exists(f"{folder_path}/train_"):
            shutil.rmtree(f"{folder_path}/train_")

        os.makedirs(f"combined_dataset/train_")
        os.makedirs(f"combined_dataset/train_/images")
        os.makedirs(f"combined_dataset/train_/labels")

        img_paths = glob(f"{folder_path}/{SUBFOLDER}/images/*")
        img_paths = [path.replace("\\", "/") for path in img_paths]
        img_paths.sort()

        with open(f"{folder_path}/data.yaml", "r") as stream:
            classes = yaml.safe_load(stream)["names"]

        for img_path in tqdm(img_paths):
            name, img_name, ann_name, img_path, ann_path, folder = get_names(
                img_path=img_path
            )

            ann_path = ann_path.split("/")
            ann_path[-2] = LABELFOLDER
            ann_path = "/".join(ann_path)

            img = cv2.imread(img_path)
            H, W, _ = img.shape
            boxes = read_boxes(ann_path)
            boxes = normalize_boxes(boxes, img.shape)

            img = draw_boxes(img, boxes, classes)
            cv2.imwrite(f"combined_dataset/train_/images/{img_name}", img)
            shutil.copyfile(ann_path, f"combined_dataset/train_/labels/{ann_name}")
