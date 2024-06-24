import cv2
import numpy as np
from glob import glob
from random import choice
import yaml
import os
from tqdm import tqdm
from tools import get_boxes, get_names
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

IMG_FOLDER = "transformed_images"
# IMG_FOLDER = "images"
LABEL_FOLDER = "transformed_labels"
# LABEL_FOLDER = "labels"
classes = ["copter", "airplane", "helicopter", "bird", "plaindrone"]


def draw_boxes(img, boxes, classes):
    img = img.copy()
    H, W, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        color = colors[box["label"]]
        cls = classes[box["label"]]

        x_left = box["x_left"]
        y_top = box["y_top"]
        x_right = box["x_right"]
        y_bottom = box["y_bottom"]

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
        # else:
        #     contour = np.array(box[1:])
        #     contour = contour.reshape(-1, 1, 2)
        #     x_left = np.min(contour[:, 0, 0])
        #     y_top = np.min(contour[:, 0, 1])

        #     cv2.putText(
        #         img,
        #         cls,
        #         (x_left, y_top),
        #         font,
        #         1,
        #         color,
        #         H // 300,
        #         cv2.LINE_AA,
        #     )

        #     cv2.drawContours(img, [contour], -1, color, 3)

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
        line = [int(float(line[0]))] + [float(el) for el in line[1:]]

        boxes.append(line)

    return boxes


if __name__ == "__main__":
    folder_paths = glob("datasets/*")
    folder_paths = [path.replace("\\", "/") for path in folder_paths]

    folder_paths.sort()

    # folder_paths = ["datasets/datasetDrone.v29i.yolov8"]  # !!!

    for folder_path in folder_paths:
        print(folder_path)

        img_paths = glob(f"{folder_path}/train/{IMG_FOLDER}/*")
        img_paths = [path.replace("\\", "/") for path in img_paths]
        img_paths.sort()

        if os.path.exists(f"{folder_path}/train/drawn_images"):
            # drawn_img_paths = glob(f"{folder_path}/train/drawn_images/*")

            # if len(drawn_img_paths) == len(img_paths):
            #     continue

            shutil.rmtree(f"{folder_path}/train/drawn_images")

        # os.makedirs(f"{folder_path}/train_")
        os.makedirs(f"{folder_path}/train/drawn_images")
        # os.makedirs(f"{folder_path}/train_/labels")

        # with open(f"{folder_path}/data.yaml", "r") as stream:
        #     classes = yaml.safe_load(stream)["names"]

        for img_path in tqdm(img_paths):
            name, img_name, ann_name, img_path, ann_path, folder, format = get_names(
                img_path=img_path, ann_folder=LABEL_FOLDER
            )

            img = cv2.imread(img_path)
            H, W, _ = img.shape
            boxes = get_boxes(ann_path, H, W)

            img = draw_boxes(img, boxes, classes)
            cv2.imwrite(f"{folder_path}/train/drawn_images/{img_name}", img)
            # shutil.copyfile(ann_path, f"{folder_path}/train_/labels/{ann_name}")
