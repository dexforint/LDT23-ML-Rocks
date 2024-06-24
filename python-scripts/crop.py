import pickle 
from glob import glob
from tqdm import tqdm
import cv2
from random import randint, random
import json
import os

os.makedirs("crops")
os.makedirs("crops/images")
os.makedirs("crops/labels")

DRAW_BOXES_FLAG = False
RANDOM_ERRORS = []
IMAGE_FOLDER = "train_/images/*"
target_shape = (640, 640)
start_index = 0
CROP_FOLDER = "crops"


def read_boxes(ann_path, shape):
    H, W = shape
    with open(ann_path) as file:
        text = file.read()

    lines = text.split("\n")

    boxes = []

    for line in lines:
        if len(line) == 0:
            continue
        line = line.split()
        
        assert len(line) == 5, ann_path
        cls, x, y, w, h = line
        assert cls == "0", ann_path
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        x1 = x - w / 2
        x2 = x + w / 2
        y1 = y - h / 2
        y2 = y + h / 2

        if x1 < 0:
            x1 = 0
        if x2 > 1:
            x2 = 1
        if y1 < 0:
            y1 = 0
        if y2 > 1:
            y2 = 1

        x1 = int(x1 * W)
        x2 = int(x2 * W)
        y1 = int(y1 * H)
        y2 = int(y2 * H)

        x = (x1 + x2) // 2
        w = x2 - x1
        y = (y1 + y2) // 2
        h = y2 - y1

        boxes.append((x, y, w, h, x1, y1, x2, y2))
    return boxes


def write_annotation(folder, img_name, index, boxes): 
    text = []
    
    for box in boxes:
        x, y, w, h = box
        text.append(f"0 {x:.9f} {y:.9f} {w:.9f} {h:.9f}")
        
    text = "\n".join(text)
    
    path = f"{folder}/labels/{img_name}_{index:02}.txt"
    with open(path, 'w') as file:
        file.write(text)

def draw_boxes(img, boxes):
    H, W, _ = img.shape
    for box in boxes:
        x, y, w, h = box
        x1 = int((x - w / 2) * W)
        x2 = int((x + w / 2) * W)
        y1 = int((y - h / 2) * H)
        y2 = int((y + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        

def check_intersections(left_top_coords, right_bottom_coords, boxes):
    (x_left, y_top) = left_top_coords
    (x_right, y_bottom) = right_bottom_coords
    for box in boxes:
        x, y, w, h, x1, y1, x2, y2 = box
        
        if (x_left > x1 and x_left < x2):
            return False
        
        if (x_right > x1 and x_right < x2):
            return False
        
        if (y_top > y1 and y_top < y2):
            return False
        
        if (y_bottom > y1 and y_bottom < y2):
            return False
        
    return True

def get_iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / boxBArea
	return iou

def filter_boxes(left_top_coords, right_bottom_coords, boxes):
    (x_left, y_top) = left_top_coords
    (x_right, y_bottom) = right_bottom_coords
    filtered_boxes = []
    
    for box in boxes:
        x, y, w, h, x1, y1, x2, y2 = box

        iou = get_iou((x_left, y_top, x_right, y_bottom), (x1, y1, x2, y2))
        if iou > 0.4:
            if x1 >= x_left:
                assert x1 < x_right
                x2 = min(x2, x_right)
            else:
                assert x2 > x_left
                x1 = max(x1, x_left)
            
            if y1 >= y_top:
                assert y1 < y_bottom
                y2 = min(y2, y_bottom)
            else:
                assert y2 > y_top
                y1 = max(y1, y_top)

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            filtered_boxes.append((iou > 0.8, box, (x, y, w, h, x1, y1, x2, y2)))
        
    return filtered_boxes

def correct_boxes(left_top_coords, target_shape, boxes):
    x_left, y_top = left_top_coords
    corrected_boxes = []
    
    for box in boxes:
        x, y, w, h, x1, y1, x2, y2 = box
        
        x = x - x_left
        x = x / target_shape[0]
        
        # print(x)
        assert 0 <= x and x <= 1
        
        y = y - y_top
        y = y / target_shape[1]
        
        assert 0 <= y and y <= 1
        
        w = w / target_shape[0]
        h = h / target_shape[1]
        
        assert 0 <= w and w <= 1
        assert 0 <= h and h <= 1
        
        corrected_boxes.append((x, y, w, h))
        
    return corrected_boxes


def write_random_crop(img, boxes, folder, target_shape=(640, 640), img_name=None):
    H, W, _ = img.shape
    
    boxes_set = set(boxes)
    while len(boxes_set) != 0:
        box = boxes_set.pop()
        x, y, w, h, x1, y1, x2, y2 = box
        index = len(glob(f"{folder}/images/{img_name}_*"))
        if h > target_shape[1] or w > target_shape[0]:
            continue

        for i in range(10):
            try:
                x_left = randint(max(int(x1) - target_shape[0] + w, 0), min(int(x1), W - target_shape[0]))
            except:
                print("x_left", img_name)
                print("W:", W, "x1:", x1, "w:", w)
                print(int(x1) - target_shape[0] + w, int(x1),  W - target_shape[0])
                assert False

            try:
                y_top = randint(max(int(y1) - target_shape[1] + h, 0), min(int(y1), H - target_shape[1]))
            except:
                print("y_top", img_name)
                print("H:", H, "y1:", y1, "h:", h)
                print(int(y1) - target_shape[1] + h, int(y1),  H - target_shape[1])
                assert False

            if check_intersections((x_left, y_top), (x_left + target_shape[0], y_top + target_shape[1]), boxes):
                break

            if i == 9:
                RANDOM_ERRORS.append((img_name, index, box))

        filtered_boxes = filter_boxes((x_left, y_top), (x_left + target_shape[0], y_top + target_shape[1]), boxes)
        filtered_boxes_ = []
        for flag, orig_box, box in filtered_boxes:
            if flag:
                if orig_box in boxes_set:
                    boxes_set.remove(orig_box)

            filtered_boxes_.append(box)

        corrected_boxes = correct_boxes((x_left, y_top), target_shape, filtered_boxes_)

        sub_img = img[y_top:y_top + target_shape[1], x_left:x_left + target_shape[0]]
        sub_img = sub_img.copy()

        if DRAW_BOXES_FLAG:
            draw_boxes(sub_img, corrected_boxes)
        
        cv2.imwrite(f"{folder}/images/{img_name}_{index:02}.jpg", sub_img)
        write_annotation(folder, img_name, index, corrected_boxes)


img_paths = glob(IMAGE_FOLDER)
img_paths = [path.replace("\\", "/") for path in img_paths]
img_paths.sort()
# img_paths = ["train_/images/aitdata_frame_000032_jpg.rf.4aea4925ac9dd31e0f54f20a7d23d9ac.jpg"]
less_counter = 0

for img_path in tqdm(img_paths[start_index:], initial=start_index, total=len(img_paths)):
    name = ".".join(img_path.split("/")[-1].split(".")[:-1])
    ann_path = img_path.split("/")
    ann_path[-1] = f"{name}.txt"
    ann_path[-2] = "labels"
    ann_path = "/".join(ann_path)

    img = cv2.imread(img_path)
    H, W, _ = img.shape
    if H < target_shape[1] or W < target_shape[0]:
        less_counter += 1
        continue
    boxes = read_boxes(ann_path, (H, W))
    write_random_crop(img, boxes, CROP_FOLDER, target_shape=target_shape, img_name=name)

print("less_counter:", less_counter)
# with open("RANDOM_ERRORS.json", "w") as file:
#     json.dump(RANDOM_ERRORS, file)