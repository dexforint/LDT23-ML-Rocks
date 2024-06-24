from glob import glob
import shutil
from tqdm.auto import tqdm
import os

os.makedirs("dataset")
os.makedirs("dataset/train")
os.makedirs("dataset/train/images")
os.makedirs("dataset/train/labels")

TARGET_IMG_FOLDER = "transformed_images"
TARGET_LABEL_FOLDER = "transformed_labels"

img_paths = glob(f"datasets/*/train/{TARGET_IMG_FOLDER}/*")
img_paths = [path.replace("\\", "/") for path in img_paths]

label_paths = glob(f"datasets/*/train/{TARGET_LABEL_FOLDER}/*")
label_paths = [path.replace("\\", "/") for path in label_paths]

print(len(img_paths), len(label_paths))

for img_path in tqdm(img_paths):
    dataset_name = img_path.split("/")[-4]
    img_name = img_path.split("/")[-1]
    shutil.copyfile(img_path, f"dataset/train/images/{dataset_name}_{img_name}")

for label_path in tqdm(label_paths):
    dataset_name = label_path.split("/")[-4]
    label_name = label_path.split("/")[-1]
    shutil.copyfile(label_path, f"dataset/train/labels/{dataset_name}_{label_name}")
