import argparse
import os
from glob import glob
from tqdm.auto import tqdm
import gradio
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.utils.cv import read_image
from sahi.utils.cv import read_video, visualize_object_predictions
from sahi.utils.file_handlers import video_writer


detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="./models/yolov8s.pt",
    confidence_threshold=0.3,
    device="cuda:0",  # or 'cuda:0'
)


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description="Обработка изображений и видео")

    parser.add_argument("--video_path", type=str, help="Путь к видео файлу")
    parser.add_argument("--output_video_path", type=str, help="Путь выходного видео")

    args = parser.parse_args()
    return args


def main(args):
    video_path = args.video_path
    output_video_path = args.output_video_path

    video = read_video(video_path)

    # Создаем объект для записи видео
    writer = video_writer(output_video_path, fps=video["fps"])

    # Обрабатываем каждый кадр видео
    for frame in video["frames"]:
        # Детекция объектов на кадре
        result = detection_model(frame)

        # Визуализация результатов детекции на кадре
        result_frame = visualize_object_predictions(
            frame=frame,
            object_prediction_list=result.object_prediction_list,
        )

        # Запись обработанного кадра в выходное видео
        writer.write(result_frame)

    # Закрываем writer после обработки всех кадров
    writer.release()


if __name__ == "__main__":
    args = parse_args()

    # Доступ к аргументам
    video_path = args.video_path
    output_video_path = args.output_video_path
