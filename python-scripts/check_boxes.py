from tqdm import tqdm
from glob import glob

ann_paths = glob("datasets/*/train/labels/*")

num_counter = {}
counter = 0

for ann_path in tqdm(ann_paths):
    with open(ann_path) as file:
        text = file.read()

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue

        nums = line.split()
        num = len(nums)

        num_counter[num] = num_counter.get(num, 0) + 1
        counter += 1

for key, value in num_counter.items():
    value /= counter
    value = round(value, 3)
    print(key, ":", value)
