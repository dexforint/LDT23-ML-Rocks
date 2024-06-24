from glob import glob
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import os

DRAWN_IMAGES_FOLDER = "drawn_images"
TARGET_IMAGES_FOLDER = "images"
TARGET_LABELS_FOLDER = "labels"

folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]

folder_paths.sort()

to_remove = True

# folder_paths = ["datasets/dataset"]

for folder_path in folder_paths:
    print(folder_path)

    img_paths = glob(f"{folder_path}/train/{TARGET_IMAGES_FOLDER}/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]

    drawn_img_paths = glob(f"{folder_path}/train/{DRAWN_IMAGES_FOLDER}/*")
    drawn_img_paths = [path.replace("\\", "/") for path in drawn_img_paths]

    label_paths = glob(f"{folder_path}/train/{TARGET_LABELS_FOLDER}/*")
    label_paths = [path.replace("\\", "/") for path in label_paths]

    img_names = []
    for img_path in img_paths:
        img_name = img_path.split("/")[-1]
        img_name = ".".join(img_name.split(".")[:-1])
        img_names.append(img_name)
    img_names = set(img_names)

    drawn_img_names = []
    for img_path in drawn_img_paths:
        img_name = img_path.split("/")[-1]
        img_name = ".".join(img_name.split(".")[:-1])
        drawn_img_names.append(img_name)
    drawn_img_names = set(drawn_img_names)

    label_names = []
    for label_path in label_paths:
        label_name = label_path.split("/")[-1]
        label_name = ".".join(label_name.split(".")[:-1])
        label_names.append(label_name)

    label_names = set(label_names)

    intersection = img_names.intersection(label_names).intersection(drawn_img_names)

    for drawn_img_path in tqdm(drawn_img_paths):
        img_name = drawn_img_path.split("/")[-1]
        img_name = ".".join(img_name.split(".")[:-1])

        if not (img_name in intersection):
            assert os.path.exists(drawn_img_path)
            if to_remove:
                os.remove(drawn_img_path)

    img_remove_counter = 0
    for img_path in tqdm(img_paths):
        img_name = img_path.split("/")[-1]
        img_name = ".".join(img_name.split(".")[:-1])

        if not (img_name in intersection):
            assert os.path.exists(img_path)
            img_remove_counter += 1
            if to_remove:
                os.remove(img_path)

    print("img_remove_counter:", img_remove_counter)

    label_remove_counter = 0
    for label_path in tqdm(label_paths):
        label_name = label_path.split("/")[-1]
        label_name = ".".join(label_name.split(".")[:-1])

        if not (label_name in intersection):
            label_remove_counter += 1
            assert os.path.exists(label_path)
            if to_remove:
                os.remove(label_path)
    print("label_remove_counter:", label_remove_counter)
    print(
        len(glob(f"{folder_path}/train/{TARGET_IMAGES_FOLDER}/*")),
        len(glob(f"{folder_path}/train/{DRAWN_IMAGES_FOLDER}/*")),
        len(glob(f"{folder_path}/train/{TARGET_LABELS_FOLDER}/*")),
    )

print("Total images:", len(glob(f"datasets/*/train/{TARGET_IMAGES_FOLDER}/*")))
