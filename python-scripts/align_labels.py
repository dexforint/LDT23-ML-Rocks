import cv2
import numpy as np
from glob import glob
from random import choice
import yaml
import os
from tqdm import tqdm
from tools import get_names
import shutil
from draw_boxes import read_boxes

TRAIN_FOLDER = "train"
IMG_FOLDER = "images"
LABEL_FOLDER = "labels"

TARGET_LABEL_FOLDER = "alabels"


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

    if os.path.exists(f"{folder_path}/{TRAIN_FOLDER}"):
        shutil.rmtree(f"{folder_path}/{TRAIN_FOLDER}")

    os.makedirs(f"{folder_path}/{TRAIN_FOLDER}/{TARGET_LABEL_FOLDER}")

    img_paths = glob(f"{folder_path}/{TRAIN_FOLDER}/{IMG_FOLDER}/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_paths.sort()

    with open(f"{folder_path}/data.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        classes = data["names"]
        if "target_classes" in data:
            target_classes = data["target_classes"]
        else:
            target_classes = classes

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

        boxes = [box for box in boxes if box[0] in target_labels]

        if len(boxes) == 0:
            continue

        cv2.imwrite(f"{folder_path}/{TRAIN_FOLDER}/{IMG_FOLDER}/{img_name}", img)
        save_boxes(
            f"{folder_path}/{TRAIN_FOLDER}/{TARGET_LABEL_FOLDER}/{ann_name}",
            boxes,
        )
