from glob import glob
from tqdm import tqdm
import os
import shutil

os.makedirs("bad", exist_ok=True)

folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]

counter = 0
for folder_path in folder_paths:
    img_paths = glob(f"{folder_path}/train/images/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_names = [path.split("/")[-1] for path in img_paths]
    names = [".".join(img_name.split(".")[:-1]) for img_name in img_names]
    set_names = set(names)

    ann_paths = glob(f"{folder_path}/train/labels/*")
    ann_paths = [path.replace("\\", "/") for path in ann_paths]

    for ann_path in ann_paths:
        ann_name = ann_path.split("/")[-1]
        name = ".".join(ann_name.split(".")[:-1])

        if name in set_names:
            continue

        counter += 1
        os.remove(ann_path)

    img_paths = glob(f"{folder_path}/train/images/*")
    ann_paths = glob(f"{folder_path}/train/labels/*")
    print(len(img_paths), len(ann_paths), folder_path)

print(counter)
