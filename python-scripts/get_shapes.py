from glob import glob
from tqdm import tqdm
import cv2
import numpy as np

folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]
folder_paths = [
    folder_path
    for folder_path in folder_paths
    if folder_path.split("/")[-1].startswith("!")
]
folder_paths.sort()


for folder_path in folder_paths:
    img_paths = glob(f"{folder_path}/transformed_train/images/*")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_paths = [path for path in img_paths if path.split("/")[1].startswith("!")]
    img_paths.sort()

    hs = []
    ws = []
    areas = []

    shapes = set()
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        H, W, _ = img.shape

        shapes.add((H, W))

        hs.append(H)
        ws.append(W)
        areas.append(H * W)

    print(folder_path.split("/")[-1])
    print(shapes)

    ws = np.array(ws)
    hs = np.array(hs)
    areas = np.array(areas)

    # print(ws.mean(), ws.std())
    # print(hs.mean(), hs.std())
    # print(areas.mean(), areas.std())
