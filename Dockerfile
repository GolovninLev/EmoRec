FROM python:3.9-slim

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Обновление пакетного менеджера и установка необходимых инструментов
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt-utils \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Установка libgl1-mesa-glx и libglib2.0-0 для cv2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копирование моделей в контейнер
COPY ./models/haarcascade_frontalface_default.xml /models/haarcascade_frontalface_default.xml
COPY ./models/emo_rec_model.pth /models/emo_rec_model.pth
COPY ./models/YOLO_face_model.pkl /models/YOLO_face_model.pkl

COPY ./emo_imgs /emo_imgs

# Копирование кода в контейнер
COPY ./src/emo_rec.py /src/emo_rec.py
COPY ./src/my_bot.py /src/my_bot.py
COPY ./src/graphs.py /src/graphs.py
COPY ./src/run.py /src/run.py

RUN mkdir /output
RUN mkdir /.secrets

COPY ./.secrets/client_secrets.json /.secrets/client_secrets.json

RUN apt-get update && apt-get install -y git  # Установка Git для ultralytics
RUN git clone https://github.com/ultralytics/ultralytics
WORKDIR /ultralytics
RUN pip install -e .

CMD ["python3", "/src/run.py"]
