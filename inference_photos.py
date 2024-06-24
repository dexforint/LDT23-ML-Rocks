import argparse
import os
from glob import glob
from tqdm.auto import tqdm
import gradio
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="./models/yolov8s.pt",
    confidence_threshold=0.2,
    device="cuda:0",  # or 'cuda:0'
)


def predict_image(img):
    # result = get_prediction(img, detection_model)
    result = get_sliced_prediction(
        img,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=False,
    )
    H, W, _ = img.shape
    preds = []
    for pred in result.object_prediction_list:
        x1 = pred.bbox.minx
        x2 = pred.bbox.maxx
        y1 = pred.bbox.miny
        y2 = pred.bbox.maxy

        x1 = x1 / W
        x2 = x2 / W
        y1 = y1 / H
        y2 = y2 / H

        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        preds.append((pred.category.id, x, y, w, h))

    return preds


def inference(image_path, label_folder):
    if label_folder.endswith("/"):
        label_folder = label_folder[:-1]

    image_name = image_path.split("/")[-1]
    name = ".".join(image_name.split(".")[:-1])

    img = read_image(image_path)

    preds = predict_image(img)

    lines = []
    for pred in preds:
        lines.append(f"{pred[0]};{pred[1]};{pred[2]};{pred[3]};{pred[4]}")

    text = "\n".join(lines)

    with open(f"{label_folder}/{name}.txt", "w") as file:
        file.write(text)


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Обработка изображений и видео")

    parser.add_argument("--image_folder", type=str, help="Путь к папке с изображениями")

    parser.add_argument(
        "--output_label_folder", type=str, help="Путь к папке для выходных файлов"
    )

    args = parser.parse_args()
    return args


def main(args):
    IMG_FOLDER = args.image_folder
    LABEL_FOLDER = args.output_label_folder

    if LABEL_FOLDER.endswith("/"):
        LABEL_FOLDER = LABEL_FOLDER[:-1]

    os.makedirs(LABEL_FOLDER, exist_ok=True)

    img_paths = glob(f"{IMG_FOLDER}/*")

    allowed_formats = set(["jpg", "png", "jpeg"])
    img_paths = [
        path for path in img_paths if path.split(".")[-1].lower() in allowed_formats
    ]

    for img_path in tqdm(img_paths):
        inference(img_path, LABEL_FOLDER)


if __name__ == "__main__":
    args = parse_args()

    if args.output_label_folder and not os.path.exists(args.output_label_folder):
        os.makedirs(args.output_label_folder)

    main(args)
