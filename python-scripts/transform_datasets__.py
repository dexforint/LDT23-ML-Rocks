import cv2
import numpy as np
from glob import glob
from random import choice
import yaml
import os
from tqdm import tqdm
from tools import *
import shutil
from draw_boxes import read_boxes, normalize_boxes


def save_boxes(path, oboxes):
    text = []
    for obox in oboxes:
        obox = [str(el) for el in obox]
        obox = " ".join(obox)
        obox = obox.replace("-", "")
        text.append(obox)

    text = "\n".join(text)

    with open(path, "w") as file:
        file.write(text)


folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]
folder_paths = [
    folder_path
    for folder_path in folder_paths
    if folder_path.split("/")[-1].startswith("!")
]
folder_paths.sort()

for folder_path in folder_paths:
    print(folder_path)

    if os.path.exists(f"{folder_path}/transformed_train"):
        shutil.rmtree(f"{folder_path}/transformed_train")

    os.makedirs(f"{folder_path}/transformed_train")
    os.makedirs(f"{folder_path}/transformed_train/images")
    os.makedirs(f"{folder_path}/transformed_train/labels")
    os.makedirs(f"{folder_path}/transformed_train/olabels")

    img_paths = glob(f"{folder_path}/train/images/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_paths.sort()

    with open(f"{folder_path}/data.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        classes = data["names"]
        target_classes = data["target_classes"]

    target_labels = set()
    for i, cls in enumerate(classes):
        if cls in target_classes:
            target_labels.add(i)

    for img_path in tqdm(img_paths):
        name, img_name, ann_name, img_path, ann_path, folder = get_names(
            img_path=img_path
        )

        img = cv2.imread(img_path)
        H, W, _ = img.shape
        boxes = read_boxes(ann_path)
        boxes = normalize_boxes(boxes, img.shape)

        boxes = [box for box in boxes if box[0] in target_labels]

        if len(boxes) == 0:
            continue

        oboxes = []
        nboxes = []
        for box in boxes:
            line = box[1:]
            if len(line) == 4:
                x, y, w, h = line

                x1 = x - w // 2
                y1 = y - h // 2

                x2 = x + w // 2
                y2 = y - h // 2

                x3 = x + w // 2
                y3 = y + h // 2

                x4 = x - w // 2
                y4 = y + h // 2

            else:
                contour = np.array(line)
                contour = contour.reshape(-1, 1, 2)
                rotatedRect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rotatedRect)
                box = np.around(box)
                box = box.reshape(-1)
                x1, y1, x2, y2, x3, y3, x4, y4 = box

                x_left, y_top, w, h = cv2.boundingRect(contour)
                x = x_left + w / 2
                y = y_top + h / 2

            x = x / W
            y = y / H
            w = w / W
            h = h / H

            x1 = x1 / W
            y1 = y1 / H

            x2 = x2 / W
            y2 = y2 / H

            x3 = x3 / W
            y3 = y3 / H

            x4 = x4 / W
            y4 = y4 / H

            oboxes.append([0, x1, y1, x2, y2, x3, y3, x4, y4])
            nboxes.append([0, x, y, w, h])

        cv2.imwrite(f"{folder_path}/transformed_train/images/{img_name}", img)
        save_boxes(
            f"{folder_path}/transformed_train/olabels/{ann_name}",
            oboxes,
        )
        save_boxes(
            f"{folder_path}/transformed_train/labels/{ann_name}",
            nboxes,
        )
