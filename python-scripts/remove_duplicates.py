from glob import glob
from tqdm import tqdm
import hashlib
import json
import shutil
import os


def read_boxes(ann_path):
    with open(ann_path) as file:
        text = file.read()

    lines = text.split("\n")
    return len(lines)


def get_hash(path):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(path, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


folders = glob("./!*")
folders = [path.replace("\\", "/") for path in folders]
folders.sort()
print(folders)

hashes = set()
duplicate_paths = []

duplicate_counter = 0
file_counter = 0
for folder in folders:
    os.makedirs(f"duplicates/{folder}", exist_ok=True)
    os.makedirs(f"duplicates/{folder}/train_", exist_ok=True)
    os.makedirs(f"duplicates/{folder}/train_/images", exist_ok=True)
    os.makedirs(f"duplicates/{folder}/train_/labels", exist_ok=True)

    img_paths = glob(f"{folder}/train_/images/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_paths.sort()
    for img_path in tqdm(img_paths):
        img_name = img_path.split("/")[-1]
        file_counter += 1

        ann_path = img_path.split("/")
        ann_path[-1] = ".".join(ann_path[-1].split(".")[:-1]) + ".txt"
        ann_path[-2] = "labels"
        ann_path = "/".join(ann_path)
        ann_name = ann_path.split("/")[-1]

        img_hash = get_hash(img_path)
        ann_hash = get_hash(ann_path)
        n = read_boxes(ann_path)
        if img_hash in hashes or ann_hash in hashes or n > 10:
            duplicate_counter += 1
            duplicate_paths.append(img_path)
            shutil.move(img_path, f"duplicates/{folder}/train_/images/{img_name}")
            shutil.move(ann_path, f"duplicates/{folder}/train_/labels/{ann_name}")

        hashes.add(img_hash)
        hashes.add(ann_hash)

print(duplicate_counter / file_counter, duplicate_counter, file_counter)
with open("img_duplicate_paths.json", "w") as file:
    json.dump(duplicate_paths, file)
