from glob import glob
import numpy as np


def read_boxes_num(ann_path):
    with open(ann_path) as file:
        text = file.read().strip()

    lines = text.split("\n")

    n_boxes = 0
    for line in lines:
        if len(line) == 0:
            continue
        n_boxes += 1

    return n_boxes


def read_boxes(ann_path):
    with open(ann_path) as file:
        text = file.read().strip()

    lines = text.split("\n")

    boxes = []
    for line in lines:
        if len(line) == 0:
            continue
        line = line.split()
        line = [int(line[0])] + [float(el) for el in line[1:]]

        boxes.append(line)

    return boxes


def get_boxes(ann_path, H, W):
    with open(ann_path) as file:
        text = file.read()

    lines = text.split("\n")

    boxes = []

    for line in lines:
        line = line.strip()

        if len(line) == 0:
            continue
        line = line.split()
        line = [int(float(line[0]))] + [float(el) for el in line[1:]]

        if len(line) == 5:
            _, x_norm, y_norm, width_norm, height_norm = line
            x_left_norm = x_norm - width_norm / 2
            x_right_norm = x_norm + width_norm / 2
            y_top_norm = y_norm - height_norm / 2
            y_bottom_norm = y_norm + height_norm / 2

        else:
            contour = np.array(line[1:])
            contour = contour.reshape(-1, 1, 2)

            x_left_norm = np.min(contour[:, 0, 0])
            y_top_norm = np.min(contour[:, 0, 1])
            x_right_norm = np.max(contour[:, 0, 0])
            y_bottom_norm = np.max(contour[:, 0, 1])

            x_norm = (x_right_norm + x_left_norm) / 2
            y_norm = (y_top_norm + y_bottom_norm) / 2
            width_norm = x_right_norm - x_left_norm
            height_norm = y_bottom_norm - y_top_norm

        x_left = int(W * x_left_norm)
        y_top = int(H * y_top_norm)
        x_right = int(W * x_right_norm)
        y_bottom = int(H * y_bottom_norm)

        x = int(W * x_norm)
        y = int(H * y_norm)

        width = x_right - x_left
        height = y_bottom - y_top

        if width < 8 or height < 8:
            continue

        boxes.append(
            {
                "label": line[0],
                ####
                "width_norm": width_norm,
                "height_norm": height_norm,
                "width": width,
                "height": height,
                ####
                "x": x,
                "y": y,
                ####
                "x_left": x_left,
                "y_top": y_top,
                "x_right": x_right,
                "y_bottom": y_bottom,
                ####
                "x_norm": x_norm,
                "y_norm": y_norm,
                ####
                "x_left_norm": x_left_norm,
                "y_top_norm": y_top_norm,
                "x_right_norm": x_right_norm,
                "y_bottom_norm": y_bottom_norm,
                ####
                "img_width": W,
                "img_height": H,
                "ann_path": ann_path,
            }
        )

    return boxes


def save_boxes(boxes, ann_path):
    lines = []

    for box in boxes:
        line = f"{box['label']} {box['x_norm']} {box['y_norm']} {box['width_norm']} {box['height_norm']}"
        lines.append(line)

    with open(ann_path, "w") as file:
        file.write("\n".join(lines))


def get_img_paths(template):
    img_paths = glob(template)
    img_paths = [img_path.replace("\\", "/") for img_path in img_paths]
    img_paths.sort()

    return img_paths


def get_names(img_path=None, ann_path=None, img_folder="images", ann_folder="labels"):
    if not (img_path is None):
        img_name = img_path.split("/")[-1]
        format = img_name.split(".")[-1]
        name = ".".join(img_name.split(".")[:-1])
        ann_name = f"{name}.txt"
        ann_path = "/".join(img_path.split("/")[:-2]) + f"/{ann_folder}/{ann_name}"
        folder = img_path.split("/")[-4]
        return name, img_name, ann_name, img_path, ann_path, folder, format
    elif not (ann_path is None):
        ann_name = ann_path.split("/")[-1]
        format = ann_name.split(".")[-1]
        name = ".".join(ann_name.split(".")[:-1])
        img_name = f"{name}.{format}"
        img_path = "/".join(ann_path.split("/")[:-2]) + f"/{img_folder}/{img_name}"
        folder = ann_path.split("/")[-4]
        return name, img_name, ann_name, img_path, ann_path, folder, format

    assert False


def get_iou(boxA, boxB):
    xA = max(boxA["x_left"], boxB["x_left"])
    yA = max(boxA["y_top"], boxB["y_top"])
    xB = min(boxA["x_right"], boxB["x_right"])
    yB = min(boxA["y_bottom"], boxB["y_bottom"])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = boxA["width"] * boxA["height"]
    boxBArea = boxB["width"] * boxB["height"]

    assert boxBArea > 0, (boxB["ann_path"], boxB["width_norm"], boxB["height_norm"])

    iou = interArea / (boxAArea + boxBArea - interArea)
    iouA = interArea / boxAArea
    iouB = interArea / boxBArea
    return iou, iouA, iouB


def correct_filter_boxes(crop_box, boxes, iou_threshold=0.4):
    x1 = crop_box["x_left"]
    y1 = crop_box["y_top"]
    x2 = crop_box["x_right"]
    y2 = crop_box["y_bottom"]

    target_width = x2 - x1
    target_height = y2 - y1

    corrected_boxes = []

    for box in boxes:
        iou, iouA, iouB = get_iou(crop_box, box)

        if iouA < 1 and iouB < iou_threshold:
            continue

        x_left = max(x1, box["x_left"]) - x1
        x_right = min(x2, box["x_right"]) - x1
        x = (x_right + x_left) // 2

        y_top = max(y1, box["y_top"]) - y1
        y_bottom = min(y2, box["y_bottom"]) - y1
        y = (y_bottom + y_top) // 2

        width = x_right - x_left
        height = y_bottom - y_top

        corrected_boxes.append(
            {
                "label": box["label"],
                ####
                "width_norm": width / target_width,
                "height_norm": height / target_height,
                "width": width,
                "height": height,
                ####
                "x": x,
                "y": y,
                ####
                "x_left": x_left,
                "y_top": y_top,
                "x_right": x_right,
                "y_bottom": y_bottom,
                ####
                "x_norm": x / target_width,
                "y_norm": y / target_height,
                ####
                "x_left_norm": x_left / target_width,
                "y_top_norm": y_top / target_height,
                "x_right_norm": x_right / target_width,
                "y_bottom_norm": y_bottom / target_height,
                ####
                "img_width": target_width,
                "img_height": target_height,
                "ann_path": box["ann_path"],
            }
        )

    # if len(boxes):
    #     assert len(corrected_boxes) > 0, (box["ann_path"], crop_box, box, iouB)

    return corrected_boxes


def find_min_box(boxes):
    min_box = boxes[0]
    min_area = boxes[0]["width"] * boxes[0]["height"]
    for box in boxes[1:]:
        area = box["width"] * box["height"]
        if area < min_area:
            min_box = box
            min_area = area
    return min_box


def get_crop_box(min_box, IMG_WIDTH, IMG_HEIGHT, CROP_WIDTH, CROP_HEIGHT):
    x_left = min_box["x_left"]
    y_top = min_box["y_top"]
    width = min_box["width"]
    height = min_box["height"]

    x_margin = (CROP_WIDTH - width) // 2
    y_margin = (CROP_HEIGHT - height) // 2

    x1 = max(0, x_left - x_margin)
    x1 = min(x1, IMG_WIDTH - CROP_WIDTH)

    y1 = max(0, y_top - y_margin)
    y1 = min(y1, IMG_HEIGHT - CROP_HEIGHT)

    return {
        "x_left": x1,
        "y_top": y1,
        "x_right": x1 + CROP_WIDTH,
        "y_bottom": y1 + CROP_HEIGHT,
        "width": CROP_WIDTH,
        "height": CROP_HEIGHT,
    }


def crop_image(img, crop_box):
    x1 = crop_box["x_left"]
    y1 = crop_box["y_top"]
    x2 = crop_box["x_right"]
    y2 = crop_box["y_bottom"]
    return img[y1:y2, x1:x2]
