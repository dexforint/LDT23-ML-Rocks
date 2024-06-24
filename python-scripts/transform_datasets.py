import cv2
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from tools import (
    get_boxes,
    save_boxes,
    correct_filter_boxes,
    find_min_box,
    get_crop_box,
    get_names,
    crop_image,
)
import shutil

IGNORE_OK = False

TARGET_IMG_WIDTH = 640
TARGET_IMG_HEIGHT = 640

TRAIN_FOLDER = "train"
IMG_FOLDER = "images"
LABEL_FOLDER = "labels"

TARGET_IMAGE_FOLDER = "transformed_images"
TARGET_LABEL_FOLDER = "transformed_labels"

folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]
folder_paths.sort()

# folder_paths = ["datasets/Bird_Only.v1i.yolov8"]  # !!!!

for folder_path in folder_paths:
    print(folder_path)

    img_paths = glob(f"{folder_path}/{TRAIN_FOLDER}/{IMG_FOLDER}/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_paths.sort()

    TARGET_LABEL_PATH = f"{folder_path}/{TRAIN_FOLDER}/{TARGET_LABEL_FOLDER}"
    TARGET_IMAGE_PATH = f"{folder_path}/{TRAIN_FOLDER}/{TARGET_IMAGE_FOLDER}"

    ok_path = f"{folder_path}/{TRAIN_FOLDER}/ok.txt"
    if not IGNORE_OK:
        if os.path.exists(ok_path):
            continue

    if os.path.exists(TARGET_LABEL_PATH):
        shutil.rmtree(TARGET_LABEL_PATH)
    os.makedirs(TARGET_LABEL_PATH)

    if os.path.exists(TARGET_IMAGE_PATH):
        shutil.rmtree(TARGET_IMAGE_PATH)
    os.makedirs(TARGET_IMAGE_PATH)

    for img_path in tqdm(img_paths):
        name, img_name, ann_name, img_path, ann_path, folder, format = get_names(
            img_path=img_path, ann_folder=LABEL_FOLDER
        )

        img = cv2.imread(img_path)
        H, W, _ = img.shape
        boxes = get_boxes(ann_path, H, W)

        if H < TARGET_IMG_HEIGHT or W < TARGET_IMG_WIDTH:
            canvas_height = max(TARGET_IMG_HEIGHT, H)
            canvas_width = max(TARGET_IMG_WIDTH, W)
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            canvas[:H, :W] = img
            img = canvas
            boxes = correct_filter_boxes(
                {
                    "x_left": 0,
                    "y_top": 0,
                    "x_right": canvas_width,
                    "y_bottom": canvas_height,
                    "width": canvas_width,
                    "height": canvas_height,
                },
                boxes,
                iou_threshold=0.33,
            )

            H, W, _ = img.shape

        if len(boxes) == 0:
            continue

        min_box = find_min_box(boxes)

        crop_box = get_crop_box(min_box, W, H, TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT)

        boxes = correct_filter_boxes(crop_box, boxes, iou_threshold=0.33)

        img = crop_image(img, crop_box)
        cv2.imwrite(f"{TARGET_IMAGE_PATH}/{name}.{format}", img)
        save_boxes(boxes, f"{TARGET_LABEL_PATH}/{name}.txt")

    with open(ok_path, "w") as file:
        file.write("")
