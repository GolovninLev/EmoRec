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


from io import BytesIO
import tempfile

import traceback
import csv

from graphs import generate_emotion_map_html
from graphs import generate_emotion_map_png

import time



def get_most_common_elem(arr):
    counter = Counter(arr)
    return counter.most_common(1)[0][0] if counter else None


class EmoRec:
    def __init__(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        # ##################################### Свойства
        model_name_photo = 'vgg19' # resnet50 vgg19
        model_name_video = 'vgg19' # resnet50 vgg19

        self.face_finder_name = 'yolo'
        
        path_to_pr_model_photo = str(self.root_dir / 'models' / 'emo_rec_model.pth')
        path_to_pr_model_video = str(self.root_dir / 'models' / 'emo_rec_model.pth')
        
        self.offset_photo = 15
        self.photo_min_threshold = 0.15
        self.video_min_threshold = 0.01
        
        self.image_transforms =  transforms.Compose([
                transforms.ToPILImage(),            # Преобразование в PIL Image
                transforms.Resize((224, 224)),      # Измените размер изображения на 224x224
                transforms.ToTensor(),              # Преобразуйте изображение в тензор
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],     # Нормализация по средним значениям и стандартным отклонениям ImageNet
                    std =[0.229, 0.224, 0.225]
            )
        ])

        self.emotion_labels = {0: "anger", 1: "contempt", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
        self.labels_ru = ["Злость", "Презрение", "Отвращение", "Страх", 
                         "Счастье", "Нейтральность", "Печаль", "Удивление"]
        self.emo_buttons = [
            f'\U0001F621 {self.labels_ru[0]}',
            f'\U0001F60F {self.labels_ru[1]}',
            f'\U0001F922 {self.labels_ru[2]}',
            f'\U0001F628 {self.labels_ru[3]}',
            f'\U0001F604 {self.labels_ru[4]}',
            f'\U0001F610 {self.labels_ru[5]}',
            f'\U0001F61E {self.labels_ru[6]}',
            f'\U0001F631 {self.labels_ru[7]}']
        
        self.output_file_path = Path(str(self.root_dir / 'output'))
        self.output_file_name = Path(r'output.mp4')
        # ##################################### Свойства 



        self.model_photo = self.init_model(model_name_photo, path_to_pr_model_photo)
        self.model_video = self.init_model(model_name_video, path_to_pr_model_video)

        with open(self.root_dir / 'models' / 'YOLO_face_model.pkl', 'rb') as f:
            self.model_YOLO_face = pickle.load(f)
        path_to_face_finder_model = str(self.root_dir / 'models' / 'haarcascade_frontalface_default.xml')
        self.model_Haar_cascades = cv2.CascadeClassifier(path_to_face_finder_model) 
        print('EmoRec init successful')
        
        
        self.face_sensitivity_photo = 14
        self.face_sensitivity_video = 12
        self.smile_size = 120
        self.alpha = 0.5
        
        self.init_smiles()


    def init_smiles(self):
        self.smiles = dict()
        self.alphas = dict()
        self.smile_w = self.smile_size
        self.smile_h = self.smile_size
        
        for smile in self.emotion_labels.values():
            
            smile_path = str(self.root_dir / 'emo_imgs' / f'{smile}.png')
            smiley = cv2.imread(smile_path, cv2.IMREAD_UNCHANGED)
            
            self.smiles.update({smile: cv2.resize(smiley, (self.smile_w, self.smile_h))})



# ################################################################################################################

    def init_model(self, model_name, path_to_pr_model):
        
        num_classes = len(self.emotion_labels)
        
        if model_name == 'resnet50':
            model = models.resnet50()
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        if model_name == 'vgg19':
            model = models.vgg19()
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

        if model_name == 'vgg_face_dag':
            model = vgg_face_dag()
            model.fc8 = nn.Linear(model.fc8.in_features, num_classes)
            # with open(path_to_pr_model, 'rb') as file:
            #     model = pickle.load(file)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path_to_pr_model)) 
            model.to(self.device)
        else:
            model.to(self.device)
            model.load_state_dict(torch.load(path_to_pr_model, map_location=torch.device('cpu'))) 
        
        return model



    def predict(self, clipped_face_frame, mode='photo'):
        
        clipped_face_tensor = self.image_transforms(clipped_face_frame).unsqueeze(0).clone().detach()
        clipped_face_tensor = clipped_face_tensor.to(self.device)
        
        if mode == 'photo':
            model = self.model_photo
        elif mode == 'video':
            model = self.model_video
        
        model.eval()
        with torch.no_grad():
            prediction = model.forward(clipped_face_tensor)
        
        return prediction.cpu()



    def detect_faces_with_yolo(self, image):
        output = self.model_YOLO_face(image, verbose=False)
        
        faces_locations = []

        for pred in output[0].boxes.xyxy:  # output[0].boxes.xyxy - массив координат [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, pred)  # Преобразование в целые числа
            faces_locations.append((x1, y1, x2 - x1, y2 - y1))  # Преобразование в (x, y, w, h)

        return faces_locations



# ################################################################################################################

    def make_image(self, file_content, face_sensitivity, smile_size, alpha):
        
        if self.smile_size != smile_size:
            self.smile_size = smile_size
            self.init_smiles()
        self.face_sensitivity_photo = face_sensitivity
        self.alpha = alpha
        
        # сохраняем файл в оперативной памяти
        numpy_arr = np.asarray(np.frombuffer(file_content, np.uint8))
        
        # Загрузка изображения с диска
        image = cv2.imdecode(numpy_arr, cv2.IMREAD_COLOR)
        if image is None:
            print("Не удалось загрузить изображение.")
            return "Не удалось загрузить изображение."
        
        # Поиск лиц на фото
        if self.face_finder_name == 'yolo':
            faces_locations = self.detect_faces_with_yolo(image)
        else:
            faces_locations = self.model_Haar_cascades.detectMultiScale(
                image, minNeighbors=self.face_sensitivity_photo, 
                minSize=(48, 48), scaleFactor=1.1) 
        
        
        texts_return = []
        faces_return = []


        # Рисование прямоугольников вокруг обнаруженных лиц
        for (x, y, w, h) in faces_locations:
            
            # Дорисовывание прямоугольника вокруг лица
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # Вырезание лица для передачи модели
            image_height, image_width = image.shape[:2]
            
            x1 = max(0, x - self.offset_photo)
            y1 = max(0, y - self.offset_photo)
            x2 = min(image_width, x + w + self.offset_photo)
            y2 = min(image_height, y + h + self.offset_photo)
            
            clipped_face_frame = image[y1:y2, x1:x2]
            
            clipped_face_frame = cv2.resize(clipped_face_frame, (224, 224), interpolation=cv2.INTER_AREA)
            _, img_encoded = cv2.imencode('.jpg', clipped_face_frame)
            faces_return.append(img_encoded.tobytes())
            
            # Если детектор нашёл лица
            if clipped_face_frame.size != 0: 
                prediction = self.predict(clipped_face_frame, 'photo')
                p = torch.softmax(torch.tensor(np.array(prediction)[0]), dim=0).numpy()
                
                sorted_emo = sorted(zip(self.emo_buttons, p), key=lambda x: x[1], reverse=True)
                text_return = f'Распределение эмоций на фото ниже:\n'
                text_return += '\n'.join([f'{label} - {(100 * prob):.0f}%' for label, prob in sorted_emo if prob > self.photo_min_threshold])
                
                texts_return.append(text_return)
                result_emotion_label = self.emotion_labels[int(np.argmax(prediction))]
                
                
                
                x_smile = x + w // 2 - self.smile_w // 2
                y_smile = y - self.smile_h // 2
                
                if y_smile < 0:
                    y_smile = 0
                if x_smile < 0:
                    x_smile = 0
                if y_smile + self.smile_h > image.shape[0]:
                    y_smile = image.shape[0] - self.smile_h
                if x_smile + self.smile_w > image.shape[1]:
                    x_smile = image.shape[1] - self.smile_w
                    
                
                for c in range(3):
                    # Применяем альфа-канал смайлика
                    image[y_smile:y_smile+self.smile_h, 
                          x_smile:x_smile+self.smile_w, c] = \
                                self.smiles[result_emotion_label][:, :, c] * \
                               (self.smiles[result_emotion_label][:, :, 3] / (255.0 / self.alpha)) + \
                                    image[y_smile:y_smile+self.smile_h, 
                                          x_smile:x_smile+self.smile_w, c] * \
                                    (1.0 - self.smiles[result_emotion_label][:, :, 3] / (255.0 / self.alpha))
                
                # Прописывание текста эмоции
                # cv2.putText(image, result_emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        
        
        _, img_encoded = cv2.imencode('.jpg', image)
        image_bytes = img_encoded.tobytes()
        return image_bytes, texts_return, faces_return



# ################################################################################################################

# ##########################################################

    def make_video(self, file_content, face_sensitivity, smile_size, alpha, is_compress_result=False, compress_video_fps=20, area_plot_dote_by_sec=0.05):
        
        if self.smile_size != smile_size:
            self.smile_size = smile_size
            self.init_smiles()
        self.face_sensitivity_video = face_sensitivity
        self.alpha = alpha
        
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
            input_video_fps = input_video.get(cv2.CAP_PROP_FPS)
            if is_compress_result:
                codec = cv2.VideoWriter_fourcc(*'XVID')
                output_video = cv2.VideoWriter(str(output_file_path), codec, compress_video_fps, (width, height))
            else:
                codec = cv2.VideoWriter_fourcc(*'mp4v')
                output_video = cv2.VideoWriter(str(output_file_path), codec, input_video_fps, (width, height))
            if not output_video.isOpened():
                print("Ошибка: Не удалось создать объект VideoWriter")
                return "Ошибка: Не удалось создать объект VideoWriter"


            # Инициализация параметров хранения недавней истории расположения лиц
            # faces_loc_hist = deque(maxlen=self.hist_len)
            # es = 10
            # el = 30
            
            
            frames_counter = 0
            
            video_emotions_stats = [np.zeros(len(self.emotion_labels)) 
                                    for _ in range((all_frames_num 
                                                    # // (int(input_video_fps * area_plot_dote_by_sec) + 1)
                                                    + 1))]
            total_prediction = np.zeros(len(self.emotion_labels))
            csv_data = [
                ['Вермя (час:мин:сек.мс)', 'Кадр', *self.labels_ru]
            ]
            
            start_time = time.time()

            # Чтение и обработка кадров
            while True:
                
                # Захват очередного кадра
                video_is_still_processing, frame = input_video.read() 
                frames_counter += 1
                frame_prediction_sum = np.zeros(len(self.emotion_labels))
                
                # Проверка, достигнут ли конец видео
                if not video_is_still_processing:
                    print('100%')
                    break
                
                # Поиск лиц на кадре
                if self.face_finder_name == 'yolo':
                    faces_locations = self.detect_faces_with_yolo(frame)
                else:
                    faces_locations = self.model_Haar_cascades.detectMultiScale(
                        frame, minNeighbors=self.face_sensitivity_photo, 
                        minSize=(48, 48), scaleFactor=1.1) 
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
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    
                    # Вырезание лица для передачи модели
                    clipped_face_frame = frame[y:y + h, x:x + w]
                    clipped_face_frame = cv2.resize(clipped_face_frame, (224, 224), interpolation=cv2.INTER_AREA)


                    # Если детектор нашёл лица И ОНИ БЫЛИ НА 3 ПРЕД. КАДРАХ
                    if clipped_face_frame.size != 0: 
                        
                        prediction = self.predict(clipped_face_frame, 'photo')

                        t = int(input_video_fps * area_plot_dote_by_sec)
                        if t == 0 or frames_counter % t == 0:
                            pred_soft = torch.softmax(torch.tensor(np.array(prediction[0])), dim=0).numpy()
                            frame_prediction_sum += pred_soft
                            
                            print(f'{frames_counter} / {all_frames_num}: \tone_face_pred: \t{pred_soft}')
                        
                        
                        result_emotion_label = self.emotion_labels[int(np.argmax(prediction))]
                            
                        
                        
                        
                        
                        
                        # Прописывание значка эмоции
                        # cv2.putText(frame, result_emotion_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
                        
                        x_smile = x + w // 2 - self.smile_w // 2
                        y_smile = y - self.smile_h // 2
                        
                        if y_smile < 0:
                            y_smile = 0
                        if x_smile < 0:
                            x_smile = 0
                        if y_smile + self.smile_h > frame.shape[0]:
                            y_smile = frame.shape[0] - self.smile_h
                        if x_smile + self.smile_w > frame.shape[1]:
                            x_smile = frame.shape[1] - self.smile_w
                            
                        
                        for c in range(3):
                            # Применяем альфа-канал смайлика
                            frame[y_smile:y_smile+self.smile_h, 
                                x_smile:x_smile+self.smile_w, c] = \
                                        self.smiles[result_emotion_label][:, :, c] * \
                                    (self.smiles[result_emotion_label][:, :, 3] / (255.0 / self.alpha)) + \
                                            frame[y_smile:y_smile+self.smile_h, 
                                                x_smile:x_smile+self.smile_w, c] * \
                                            (1.0 - self.smiles[result_emotion_label][:, :, 3] / (255.0 / self.alpha))
                
                
                # Запись обработанного кадра в выходной файл
                output_video.write(frame) 
                
                if len(faces_locations) > 0: 
                    frame_prediction_norm = frame_prediction_sum / len(faces_locations)
                else:
                    frame_prediction_norm = np.zeros(len(self.emotion_labels))
                video_emotions_stats[frames_counter] += frame_prediction_norm
                total_prediction += frame_prediction_norm
                print(f'{frames_counter} / {all_frames_num} : frame_prediction: \t\t{frame_prediction_norm}')
                
                total_seconds = frames_counter / input_video_fps
                h = int(total_seconds // 3600)
                m = int((total_seconds % 3600) // 60)
                s = int(total_seconds % 60)
                ms = int((total_seconds - int(total_seconds)) * 1000)
                csv_data.append([f"{h:02.0f}:{m:02.0f}:{s:02.0f}.{ms:03.0f}", frames_counter, *frame_prediction_norm])
                
                if int(frames_counter % input_video_fps) == 0:
                    # print(f'{frames_counter} кадров обработано')
                    print(f'{(frames_counter / (all_frames_num * 1.1) * 100.0):.2f}% кадров обработано')
        
        except Exception as e:
            traceback.print_exc()
                
        finally:
            end_time = time.time()
            time_log = f"Время обработки видео из {all_frames_num} кадров: {(end_time - start_time):.2f} секунд." +\
                    f" ({(end_time - start_time) / all_frames_num:.2f} секунд на кадр)"
            print(time_log)
            
            # Закрытие input и output файлов
            input_video.release()
            output_video.release()
            
            
            with open(str(output_file_path), 'rb') as f:
                result = f.read()

            # video_emotions_stats = [arr for arr in video_emotions_stats if np.any(arr)]
            
            html_graph = generate_emotion_map_html(
                video_emotions_stats, 
                self.emo_buttons, 
                input_video_fps)
            
            html_png = generate_emotion_map_png(
                video_emotions_stats, 
                self.emo_buttons, 
                input_video_fps)
            
            
            
            text_return = f'Распределение наиболее частых эмоций в видео:\n'
            
            p = np.array(total_prediction) / np.sum(total_prediction)
            sorted_emo = sorted(zip(self.emo_buttons, p), key=lambda x: x[1], reverse=True)
            text_return += '\n'.join([f'{label} - {(100 * prob):.0f}%:' for label, prob in sorted_emo if prob >= self.video_min_threshold])
            
            csvfile = BytesIO()
            csvfile.write(b'')  # Очистить файл
            writer = csv.writer(csvfile)
            for row in csv_data:
                csvfile.write(b','.join([str(x).encode('utf-8') for x in row]) + b'\n')
            csvfile.seek(0)


            return result, str(output_file_path), html_graph, html_png, text_return, csvfile
