from glob import glob
from tqdm import tqdm
import hashlib
import json
import shutil
import os

os.makedirs("datasets")
os.makedirs("datasets/train")
os.makedirs("datasets/val")
os.makedirs("datasets/train/images")
os.makedirs("datasets/train/labels")
os.makedirs("datasets/val/images")
os.makedirs("datasets/val/labels")

img_paths = glob("./!*/crops/images/*")
img_paths = [path.replace("\\", "/") for path in img_paths]
img_paths.sort()

for img_path in tqdm(img_paths):
    img_name = img_path.split("/")[-1]

    ann_path = img_path.split("/")
    ann_path[-2] = "labels"
    ann_path[-1] = ".".join(ann_path[-1].split(".")[:-1]) + ".txt"
    ann_path = "/".join(ann_path)

    ann_name = ann_path.split("/")[-1]

    shutil.copyfile(img_path, f"datasets/train/images/{img_name}")
    shutil.copyfile(ann_path, f"datasets/train/labels/{ann_name}")

print(len(img_paths))
