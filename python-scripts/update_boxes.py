from glob import glob
from tqdm import tqdm
import os
import shutil

os.makedirs("bad", exist_ok=True)

folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]
folder_paths = [path for path in folder_paths if path.split("/")[-1].startswith("!")]

counter = 0
for folder_path in tqdm(folder_paths):
    os.makedirs(f"bad/{folder_path}", exist_ok=True)
    os.makedirs(f"bad/{folder_path}/train", exist_ok=True)
    os.makedirs(f"bad/{folder_path}/train/images", exist_ok=True)
    os.makedirs(f"bad/{folder_path}/train/labels", exist_ok=True)

    img_paths = glob(f"{folder_path}/train_/images/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]

    img_names = [path.split("/")[-1] for path in img_paths]
    img_names = set(img_names)

    img_paths = glob(f"{folder_path}/train/images/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]

    orig_img_names = [path.split("/")[-1] for path in img_paths]
    orig_img_names = set(orig_img_names)

    names_to_remove = orig_img_names - img_names

    for img_name in names_to_remove:
        name = ".".join(img_name.split(".")[:-1])
        ann_name = f"{name}.txt"

        shutil.move(
            f"{folder_path}/train/images/{img_name}",
            f"bad/{folder_path}/train/images/{img_name}",
        )

        shutil.move(
            f"{folder_path}/train/labels/{ann_name}",
            f"bad/{folder_path}/train/labels/{ann_name}",
        )

        counter += 1

print(counter)
