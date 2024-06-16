# Лидеры Цифровой Трансформации

Нейросеть для мониторинга воздушного пространства вокруг аэропортов

Команда: ML Rocks \
Telegram: @dl_hello

# Прототип

1. http://147.45.237.203/ldt23
2. https://colab.research.google.com/drive/1XXraPtQiMlCkWApklfuI1__iSOrwo3i1#scrollTo=XQHIXrqY6B6T
3. ./inference_photos.py
4. ./inference_video.py

# Запуск:

# 1

```bash
docker-compose up -d --build
```

После запуска перейдите по ссылке http://localhost:7860

# 2

```bash
python inference_photos.py --image_folder path/to/image/folder --output_label_folder path/to/output/label/folder
```

# 3

```bash
python inference_video.py --video_path path/to/video --output_video_path path/to/output/video
```
