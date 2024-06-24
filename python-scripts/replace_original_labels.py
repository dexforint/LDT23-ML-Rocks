import cv2
import numpy as np
from glob import glob
from random import choice
import yaml
import os
from tqdm import tqdm
from tools import *
import shutil

folder_paths = glob("datasets/*")
folder_paths = [path.replace("\\", "/") for path in folder_paths]

folder_paths.sort()

for folder_path in tqdm(folder_paths):
    os.rename(f"{folder_path}/train/labels", f"{folder_path}/train/original_labels")
    os.rename(f"{folder_path}/train/transformed_labels", f"{folder_path}/train/labels")
