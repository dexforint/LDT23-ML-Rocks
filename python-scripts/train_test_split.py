from glob import glob
from tqdm import tqdm
import random
import shutil
import os

FOLDER = "dataset"

os.makedirs(f"{FOLDER}/val")
os.makedirs(f"{FOLDER}/val/images")
os.makedirs(f"{FOLDER}/val/labels")

img_paths = glob(f"{FOLDER}/train/images/*")
img_paths = [path.replace("\\", "/") for path in img_paths]

img_paths.sort()
print(len(img_paths))


random.seed(42)
val_img_paths = random.sample(img_paths, len(img_paths) // 20)

for val_img_path in tqdm(val_img_paths):
    new_val_img_path = val_img_path.split("/")
    new_val_img_path[-3] = "val"
    new_val_img_path = "/".join(new_val_img_path)

    assert os.path.exists(val_img_path), val_img_path
    shutil.move(val_img_path, new_val_img_path)

    img_name = val_img_path.split("/")[-1]
    ann_name = ".".join(img_name.split(".")[:-1])

    val_ann_path = f"{FOLDER}/train/labels/{ann_name}.txt"
    new_val_ann_path = val_ann_path.split("/")
    new_val_ann_path[-3] = "val"
    new_val_ann_path = "/".join(new_val_ann_path)

    assert os.path.exists(val_ann_path), val_ann_path
    shutil.move(val_ann_path, new_val_ann_path)
