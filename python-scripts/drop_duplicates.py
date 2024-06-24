from tqdm import tqdm
import os
import shutil
import hashlib
from tools import *

os.makedirs("duplicates", exist_ok=True)

to_count = True


def get_file_hash(file_path):
    BUF_SIZE = 65536

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


img_paths = get_img_paths("datasets/*/train/images/*")
folders = set([img_path.split("/")[-4] for img_path in img_paths])

hashes = {}

img_wo_objects = []

for img_path in tqdm(img_paths):
    name, img_name, ann_name, img_path, ann_path, folder = get_names(img_path=img_path)

    n_boxes = read_boxes_num(ann_path)
    if n_boxes == 0:
        img_wo_objects.append((name, img_name, ann_name, img_path, ann_path, folder))
    else:
        img_hash = get_file_hash(img_path)

        if not (img_hash in hashes):
            hashes[img_hash] = []
        hashes[img_hash].append((name, img_name, ann_name, img_path, ann_path, folder))

img_wo_objects_counter_ratio = round(100 * len(img_wo_objects) / len(img_paths), 1)
print(
    f"Images without objects: {len(img_wo_objects)} ({img_wo_objects_counter_ratio}%)"
)


# !Move code
folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]
folder_paths.sort()


for folder_path in folder_paths:
    folder = folder_path.split("/")[-1]
    os.makedirs(f"duplicates/{folder}", exist_ok=True)
    os.makedirs(f"duplicates/{folder}/train", exist_ok=True)
    os.makedirs(f"duplicates/{folder}/train/images", exist_ok=True)
    os.makedirs(f"duplicates/{folder}/train/labels", exist_ok=True)

if not to_count:
    for name, img_name, ann_name, img_path, ann_path, folder in tqdm(
        img_wo_objects, total=len(img_wo_objects)
    ):
        shutil.move(img_path, f"duplicates/{folder}/train/images/{img_name}")
        shutil.move(ann_path, f"duplicates/{folder}/train/labels/{ann_name}")


duplicate_counter = 0

flag = True

for img_hash, duplicate_paths in tqdm(hashes.items(), total=len(hashes)):
    duplicate_paths = duplicate_paths[1:]
    duplicate_counter += len(duplicate_paths)

    for name, img_name, ann_name, img_path, ann_path, folder in duplicate_paths:
        folder = img_path.split("/")[-4]
        if not to_count:
            shutil.move(img_path, f"duplicates/{folder}/train/images/{img_name}")
            shutil.move(ann_path, f"duplicates/{folder}/train/labels/{ann_name}")


duplicate_counter_ratio = round(100 * duplicate_counter / len(img_paths), 1)
print(f"Duplicates: {duplicate_counter} ({duplicate_counter_ratio}%)")
