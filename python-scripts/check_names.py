from glob import glob
import cv2
from tqdm import tqdm
import pickle
import numpy as np
import os


img_paths = glob(f"ldt-23-dataset/train/images/*")
img_paths = [path.replace("\\", "/") for path in img_paths]

label_paths = glob(f"ldt-23-dataset/train/labels/*")
label_paths = [path.replace("\\", "/") for path in label_paths]

img_names = []
for img_path in img_paths:
    img_name = img_path.split("/")[-1]
    img_name = ".".join(img_name.split(".")[:-1])
    img_names.append(img_name)
img_names = set(img_names)


label_names = []
for label_path in label_paths:
    label_name = label_path.split("/")[-1]
    label_name = ".".join(label_name.split(".")[:-1])
    label_names.append(label_name)

label_names = set(label_names)

intersection = img_names.intersection(label_names)

print(len(intersection), len(img_names), len(label_names))


img_paths = glob(f"ldt-23-dataset/val/images/*")
img_paths = [path.replace("\\", "/") for path in img_paths]

label_paths = glob(f"ldt-23-dataset/val/labels/*")
label_paths = [path.replace("\\", "/") for path in label_paths]

img_names = []
for img_path in img_paths:
    img_name = img_path.split("/")[-1]
    img_name = ".".join(img_name.split(".")[:-1])
    img_names.append(img_name)
img_names = set(img_names)


label_names = []
for label_path in label_paths:
    label_name = label_path.split("/")[-1]
    label_name = ".".join(label_name.split(".")[:-1])
    label_names.append(label_name)

label_names = set(label_names)

intersection = img_names.intersection(label_names)

print(len(intersection), len(img_names), len(label_names))
