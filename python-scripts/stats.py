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
print(len(folders))

counter = 0

for folder in folders:
    img_paths = glob(f"{folder}/train_/images/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_paths.sort()

    ann_paths = glob(f"{folder}/train_/labels/*")
    ann_paths = [path.replace("\\", "/") for path in ann_paths]
    ann_paths.sort()
    print(len(img_paths), len(ann_paths))

    counter += len(img_paths)

    for img_path, ann_path in zip(img_paths, ann_paths):
        img_name = ".".join(img_path.split("/")[-1].split(".")[:-1])
        ann_name = ".".join(ann_path.split("/")[-1].split(".")[:-1])

        assert img_name == ann_name

print(counter)
