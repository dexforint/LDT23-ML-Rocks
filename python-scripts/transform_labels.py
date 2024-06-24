import cv2
import numpy as np
from glob import glob
from random import choice
import yaml
import os
from tqdm import tqdm
from tools import *
import shutil

cls2target_label = {
    "copter": 0,
    "airplane": 1,
    "helicopter": 2,
    "bird": 3,
    "plaindrone": 4,
}

folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]

folder_paths.sort()

all_target_classes = set(
    ["copter", "airplane", "helicopter", "bird", "plaindrone", "none"]
)

for folder_path in folder_paths:
    print(folder_path)
    with open(f"{folder_path}/data.yaml", "r") as stream:
        data = yaml.safe_load(stream)
        classes = data["names"]
        target_classes = data["target_names"]

        assert len(classes) == len(target_classes)

    for target_class in target_classes:
        assert target_class in all_target_classes

    label2target_cls = target_classes

    if os.path.exists(f"{folder_path}/train/transformed_labels"):
        shutil.rmtree(f"{folder_path}/train/transformed_labels")

    os.makedirs(f"{folder_path}/train/transformed_labels")

    ann_paths = glob(f"{folder_path}/train/labels/*")
    ann_paths = [path.replace("\\", "/") for path in ann_paths]

    for ann_path in tqdm(ann_paths):
        with open(ann_path) as file:
            text = file.read()

        lines = text.split("\n")

        boxes = []

        for line in lines:
            if len(line) == 0:
                continue
            line = line.split()
            label = int(float(line[0]))

            assert label in range(len(classes))

            target_cls = label2target_cls[label]

            if target_cls == "none":
                continue

            target_label = cls2target_label[target_cls]

            target_label = str(target_label)

            line[0] = target_label

            line = " ".join(line)

            boxes.append(line)

        ann_text = "\n".join(boxes)

        ann_name = os.path.basename(ann_path)

        with open(f"{folder_path}/train/transformed_labels/{ann_name}", "w") as file:
            file.write(ann_text)
