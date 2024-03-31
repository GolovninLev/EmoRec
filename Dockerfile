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
COPY ./models/k03.28_13-28-55_e19.pth /models/k03.28_13-28-55_e19.pth

COPY ./emo_imgs /emo_imgs

# Копирование кода в контейнер
COPY ./src/emo_rec.py /src/emo_rec.py
COPY ./src/my_bot.py /src/my_bot.py
COPY ./src/vgg_face_dag.py /src/vgg_face_dag.py
COPY ./src/run.py /src/run.py

RUN mkdir /output
RUN mkdir /.secrets

CMD ["python3", "/src/run.py"]
