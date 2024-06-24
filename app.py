import gradio as gr
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.postprocess.backend import PostprocessBackend
from sahi.postprocess.objects import BoundingBox
from sahi.utils.file import read_image
from sahi.utils.cv2 import draw_bbox
from sahi.utils.torch import load_model_from_checkpoint

# Загрузка модели
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="models/yolov8s.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)


# Функция для обработки изображения
def predict_image(image):
    img = read_image(image)
    # prediction = get_prediction(
    #     image=img,
    #     model=model,
    #     postprocess_backend=PostprocessBackend.ALGORITHM,
    #     return_postprocessed=True,
    # )

    prediction = get_sliced_prediction(
        img,
        model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=False,
    )

    # Создание списка с результатами для каждого обнаруженного объекта
    results = []
    for detection in prediction.object_prediction_list:
        bbox = detection.bbox
        results.append(
            f"{detection.category_id};{bbox.center_x.value};{bbox.center_y.value};{bbox.width.value};{bbox.height.value}"
        )

    # Создание текстовых файлов с результатами для каждого изображения
    file_name = image.name.split(".")[0]
    with open(f"{file_name}.txt", "w") as f:
        for result in results:
            f.write(f"{result}\n")

    return [f"{file_name}.txt"]


# Функция для обработки видео
def predict_video(video):
    cap = cv2.VideoCapture(video.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Обработка кадра
        img = read_image(frame)
        prediction = get_prediction(
            image=img,
            model=model,
            postprocess_backend=PostprocessBackend.ALGORITHM,
            return_postprocessed=True,
        )

        # Отрисовка боксов объектов
        for detection in prediction.object_prediction_list:
            bbox = detection.bbox
            frame = draw_bbox(
                image=frame,
                bbox=BoundingBox(
                    box_points=bbox.to_xyxy(),
                    category_id=detection.category_id,
                    score=detection.score.value,
                ),
                thickness=2,
                color=(0, 255, 0),
                text_color=(0, 0, 0),
                text_size=0.5,
            )

        # Вывод видео
        yield frame

    cap.release()


# Создание веб-интерфейса Gradio
demo = gr.Interface(
    fn=[predict_image, predict_video],
    inputs=[gr.Image(label="Изображение"), gr.Video(label="Видео")],
    outputs=[gr.File(label="Результаты"), gr.Video(label="Видео с боксами")],
    title="YOLOv8s Object Detection",
    theme="default",
)

# Запуск веб-интерфейса
demo.launch()
