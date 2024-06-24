def check_intersections(crop_box, boxes):
    x_left = crop_box["x_left"]
    y_top = crop_box["y_top"]
    x_right = crop_box["x_right"]
    y_bottom = crop_box["y_bottom"]

    for box in boxes:
        x1 = box["x_left"]
        y1 = box["y_top"]
        x2 = box["x_right"]
        y2 = box["y_bottom"]

        if x_left > x1 and x_left < x2:
            return False

        if x_right > x1 and x_right < x2:
            return False

        if y_top > y1 and y_top < y2:
            return False

        if y_bottom > y1 and y_bottom < y2:
            return False

    return True


def filter_boxes(crop_box, boxes):
    x_left = crop_box["x_left"]
    y_top = crop_box["y_top"]
    x_right = crop_box["x_right"]
    y_bottom = crop_box["y_bottom"]

    filtered_boxes = []
    for box in boxes:
        x1 = box["x_left"]
        y1 = box["y_top"]
        x2 = box["x_right"]
        y2 = box["y_bottom"]
        x = box["x"]
        y = box["y"]
        w = box["width"]
        h = box["height"]

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
