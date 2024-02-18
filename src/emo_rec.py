import os
from pathlib import Path
import pickle

import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, models

from collections import deque
from collections import Counter


import vgg_face_dag

from io import BytesIO
import tempfile


def get_most_common_elem(arr):
    counter = Counter(arr)
    return counter.most_common(1)[0][0] if counter else None


class EmoRec:
    def __init__(self):
        root_dir = Path(__file__).resolve().parent.parent
        # ##################################### Свойства
        model_name = 'resnet50' # resnet50 vgg19
        path_to_pr_model = str(root_dir / 'models' / 'k01.23 13-28-01.pth') # k01.23 08-47-18.pth
        
        self.hist_len = 2
        self.total_eval_ever_n_frames = 8
        
        self.image_transforms =  transforms.Compose([
                transforms.ToPILImage(),            # Преобразование в PIL Image
                transforms.Resize((224, 224)),      # Измените размер изображения на 224x224
                transforms.ToTensor(),              # Преобразуйте изображение в тензор
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],     # Нормализация по средним значениям и стандартным отклонениям ImageNet
                    std =[0.229, 0.224, 0.225]
            )
        ])

        # self.emotion_labels = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
        self.emotion_labels = {0: "disgust", 1: "contempt", 2: "anger", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
        
        self.output_file_path = Path(r'./output/')
        self.output_file_name = Path(r'output.mp4')
        # ##################################### Свойства



        # Инициализация модели
        num_classes = len(self.emotion_labels)

        if model_name == 'resnet50':
            self.model = models.resnet50()
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if model_name == 'vgg19':
            self.model = models.vgg19()
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, num_classes)

        if model_name == 'vgg_face_dag':
            self.model = vgg_face_dag()
            self.model.fc8 = nn.Linear(self.model.fc8.in_features, num_classes)
            # with open(path_to_pr_model, 'rb') as file:
            #     model = pickle.load(file)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.model.load_state_dict(torch.load(path_to_pr_model)) 
        self.model.to(self.device)
        
        path_to_face_finder_model = str(root_dir / 'models' / 'haarcascade_frontalface_default.xml')
        self.face_finder = cv2.CascadeClassifier(path_to_face_finder_model) 
        print('EmoRec init successful')


# ################################################################################################################

    def predict(self, clipped_face_frame):
        
        clipped_face_tensor = self.image_transforms(clipped_face_frame).unsqueeze(0).clone().detach()
        clipped_face_tensor = clipped_face_tensor.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.forward(clipped_face_tensor)
        
        return prediction.cpu()


# ################################################################################################################

    def make_image(self, file_content):
        
        # сохраняем файл в оперативной памяти
        numpy_arr = np.asarray(np.frombuffer(file_content, np.uint8))
        
        # Загрузка изображения с диска
        image = cv2.imdecode(numpy_arr, cv2.IMREAD_COLOR)
        if image is None:
            print("Не удалось загрузить изображение.")
            return "Не удалось загрузить изображение."
        
        # Поиск лиц на фото
        faces_locations = self.face_finder.detectMultiScale(image, minNeighbors=14, minSize=(48, 48), scaleFactor=1.1) 
        

        # Рисование прямоугольников вокруг обнаруженных лиц
        for (x, y, w, h) in faces_locations:
            
            # Дорисовывание прямоугольника вокруг лица
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # Вырезание лица для передачи модели
            clipped_face_frame = image[y:y + h, x:x + w]
            clipped_face_frame = cv2.resize(clipped_face_frame, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Если детектор нашёл лица
            if clipped_face_frame.size != 0: 
                prediction = self.predict(clipped_face_frame)
                result_emotion_label = self.emotion_labels[int(np.argmax(prediction))]
                
                # Прописывание значка эмоции
                cv2.putText(image, result_emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        
        
        _, img_encoded = cv2.imencode('.jpg', image)
        image_bytes = img_encoded.tobytes()
        return image_bytes




# ################################################################################################################

# ##########################################################

    def make_video(self, file_content, file_id,  is_compress_result=False, compress_video_fps=20):
        
        try:
            
            # # Создание именного временного файла для cv2.VideoCapture и запись в него данных
            with tempfile.NamedTemporaryFile(delete=True, suffix='.mp4', mode='wb') as temp_file:
                print(temp_file.name)
                temp_file.write(file_content)
                temp_file_name = temp_file.name
                # Создаём объект исходного видео
                input_video = cv2.VideoCapture(temp_file_name) 
            
            if not input_video.isOpened():
                print("Ошибка: Не удалось открыть видеофайл")
                return "Ошибка: Не удалось открыть видеофайл"
            
            # Получение характеристик видео для создания нового с дорисовками
            width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            all_frames_num = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))


            output_file_path = self.output_file_path / self.output_file_name
        
            # VideoWriter - сохранятор нового видео
            if is_compress_result:
                codec = cv2.VideoWriter_fourcc(*'XVID')
                output_video = cv2.VideoWriter(str(output_file_path), codec, compress_video_fps, (width, height))
            else:
                fps = input_video.get(cv2.CAP_PROP_FPS)
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                output_video = cv2.VideoWriter(str(output_file_path), codec, fps, (width, height))
            if not output_video.isOpened():
                print("Ошибка: Не удалось создать объект VideoWriter")
                return "Ошибка: Не удалось создать объект VideoWriter"


            # Инициализация параметров хранения недавней истории расположения лиц
            # faces_loc_hist = deque(maxlen=self.hist_len)
            # es = 10
            # el = 30
            
            # Инициализация параметров хранения недавней истории эмоций на лицах
            faces_emo_hist = deque(maxlen=self.hist_len + 1)
            
            frames_counter = 0


            # Чтение и обработка кадров
            while True:
                
                # Захват очередного кадра
                video_is_still_processing, frame = input_video.read() 
                frames_counter += 1
                
                # Проверка, достигнут ли конец видео
                if not video_is_still_processing:
                    print('100%')
                    break
                
                # Поиск лиц на кадре
                faces_locations = self.face_finder.detectMultiScale(frame, minNeighbors=12, minSize=(48, 48), scaleFactor=1.1) 
                # faces_loc_hist.append(faces_locations) # и запись их расположений в недавную историю
                
                
                
                # Рисование прямоугольников вокруг обнаруженных лиц
                for (x, y, w, h) in faces_locations:
                    
                    # Проверка есть ли найденное лицо в недавней истории
                    # if frames_counter > self.hist_len and \
                    #         not ( \
                    #         abs(x - faces_loc_hist[0]) < el and abs(x - faces_loc_hist[1]) < el and abs(x - faces_loc_hist[2]) < el and \
                    #         abs(y - faces_loc_hist[0]) < el and abs(y - faces_loc_hist[1]) < el and abs(y - faces_loc_hist[2]) < el and \
                    #         abs(w - faces_loc_hist[0]) < es and abs(w - faces_loc_hist[1]) < es and abs(w - faces_loc_hist[2]) < es and \
                    #         abs(h - faces_loc_hist[0]) < es and abs(h - faces_loc_hist[1]) < es and abs(h - faces_loc_hist[2]) < es):
                        
                    
                    
                    # Дорисовывание прямоугольника вокруг лица
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    
                    # Вырезание лица для передачи модели
                    clipped_face_frame = frame[y:y + h, x:x + w]
                    clipped_face_frame = cv2.resize(clipped_face_frame, (224, 224), interpolation=cv2.INTER_AREA)


                    # Если детектор нашёл лица И ОНИ БЫЛИ НА 3 ПРЕД. КАДРАХ
                    if clipped_face_frame.size != 0: 
                        
                        if frames_counter % self.total_eval_ever_n_frames == 0 or frames_counter == 1: # можно не каждый кадр модели отдавать модели, чтоб урать мелькание нарисованных эмоций и снизить нагрузку
                            prediction = self.predict(clipped_face_frame)
                            # prediction = 1

                            faces_emo_hist.append(int(np.argmax(prediction)))
                            most_common_pred = get_most_common_elem(faces_emo_hist)
                            
                            result_emotion_label = self.emotion_labels[most_common_pred]
                            
                            print(f'{(frames_counter / (all_frames_num * 1.1) * 100.0):.2f}%')
                            
                            
                        elif frames_counter % self.total_eval_ever_n_frames in range(8 - self.hist_len, 8): 
                            prediction = self.predict(clipped_face_frame)
                            
                            faces_emo_hist.append(int(np.argmax(prediction)))
                        
                        
                        # Прописывание значка эмоции
                        cv2.putText(frame, result_emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                
                
                # Запись обработанного кадра в выходной файл
                output_video.write(frame) 
                
        finally:
            # Закрытие input и output файлов
            input_video.release()
            output_video.release()
            
            
            with open(str(output_file_path), 'rb') as f:
                result = f.read()

            return result, str(output_file_path)

