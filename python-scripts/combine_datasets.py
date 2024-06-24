import os
from glob import glob
from tqdm import tqdm
import shutil

if os.path.exists("combined_dataset/datasets/train"):
    shutil.rmtree("combined_dataset/datasets/train")

os.makedirs("combined_dataset/datasets", exist_ok=True)
os.makedirs("combined_dataset/datasets/train", exist_ok=True)
os.makedirs("combined_dataset/datasets/train/images", exist_ok=True)
os.makedirs("combined_dataset/datasets/train/labels", exist_ok=True)

folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]
folder_paths = [
    folder_path
    for folder_path in folder_paths
    if folder_path.split("/")[-1].startswith("!")
]
folder_paths.sort()

for folder_path in tqdm(folder_paths):
    shutil.copytree(
        f"{folder_path}/transformed_train_reshaped",
        "combined_dataset/datasets/train",
        dirs_exist_ok=True,
    )
