import os
import time
import json
import threading
from pathlib import Path

import io
from io import BytesIO
from io import StringIO
import traceback

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import telebot
from telebot.types import ReplyKeyboardMarkup
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from telebot.types import KeyboardButton
from telebot.types import InputMediaPhoto

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseUpload

from emo_rec import EmoRec



class MyBot:
    
    def __init__(self):
        
        self.bot = telebot.TeleBot('6829160910:AAEmmlh0aB567vnpfSsFeTA7CV1Z_vGl3XA', skip_pending=True)
        
        self.emo_rec = EmoRec()
        
        self.shipping_method = 'telegram' # telegram google_drive
        
        self.is_compress_result = True
        self.compress_video_fps = 20
        
        self.update_login_google = True
        
        # self.buffer_to_send_photo = []
        # self.img_num_from_user = 1
        
        
        self.t_auth_response_part = "https://t.me/EemoRecBot"
        self.t_auth_response_full = ""
        self.authorized_user_file = './.secrets/token.json'
        
        self.send_url_resp_event = threading.Event()
        
        
        self.previous_command = ''
        self.face_sensitivity_photo = 14
        self.face_sensitivity_video = 12
        self.smile_size = 120
        self.alpha = 0.5
        self.new_icon_sent = False
        self.root_dir = Path(__file__).resolve().parent.parent
        
        self.t_telegram = "Telegram"
        self.t_google_drive = "Google drive"
        
        
        
        self.t_settings = "\U00002699 Настройки"
        self.t_help = "\U00002753 Помощь"
        
        self.t_set_icons = "\U00002699 \U0001F60A Иконки смайлика"
        self.t_set_smile_size = "\U00002699 \U0001F60A Размер смайлика"
        self.t_set_alpha = "\U00002699 \U0001F60A Прозрачность смайлика"
        self.t_set_face_sensitivity_photo = "\U00002699 \U0001F50D Сила поиска лиц на фото"
        self.t_set_face_sensitivity_video = "\U00002699 \U0001F50D Силу поиска лиц на видео"
        
        self.t_location_receive_result = "\U000026F3 Способ получения результата"
        self.t_reset_google_drive = "\U0001F512 Войти/сменить аккаунт google drive"
        
        self.t_back = "\U000025C0 Назад"
        
        
        self.keyboard_base = ReplyKeyboardMarkup(resize_keyboard = True)
        self.keyboard_base.add(KeyboardButton(text=self.t_settings), 
                               KeyboardButton(text=self.t_help))
        
        
        self.keyboard_settings = ReplyKeyboardMarkup(resize_keyboard = True)
        
        self.keyboard_settings.add(KeyboardButton(text=self.t_set_icons))
        self.keyboard_settings.add(KeyboardButton(text=self.t_set_smile_size))
        self.keyboard_settings.add(KeyboardButton(text=self.t_set_alpha))
        self.keyboard_settings.add(KeyboardButton(text=self.t_set_face_sensitivity_photo))
        self.keyboard_settings.add(KeyboardButton(text=self.t_set_face_sensitivity_video))
        
        self.keyboard_settings.add(KeyboardButton(text=self.t_location_receive_result))
        self.keyboard_settings.add(KeyboardButton(text=self.t_reset_google_drive))
        
        self.keyboard_settings.add(KeyboardButton(text=self.t_back))
        
        
        self.keyboard_shipping_method = ReplyKeyboardMarkup(resize_keyboard = True)
        self.keyboard_shipping_method.add(KeyboardButton(text=self.t_telegram), 
                                          KeyboardButton(text=self.t_google_drive))
        
        
        self.emotion_labels    = {0: "Гнев",  1: "Презрение", 2: "Отвращение", 3: "Страх", 4: "Счастье", 5: "Нейтральность", 6: "Грусть", 7: "Удивление"}
        self.emotion_labels_en = {0: "anger", 1: "contempt",  2: "disgust",    3: "fear",  4: "happy",   5: "neutral",       6: "sad",    7: "surprise"}


        self.keyboard_emos = ReplyKeyboardMarkup(resize_keyboard = True)
        
        self.emo_buttons = [f'\U0001F616 {self.emotion_labels[0]}',
            f'\U0001F60F {self.emotion_labels[1]}',
            f'\U0001F621 {self.emotion_labels[2]}',
            f'\U0001F628 {self.emotion_labels[3]}',
            f'\U0001F604 {self.emotion_labels[4]}',
            f'\U0001F610 {self.emotion_labels[5]}',
            f'\U0001F622 {self.emotion_labels[6]}',
            f'\U0001F62E {self.emotion_labels[7]}']
        
        self.keyboard_emos.add(KeyboardButton(text=self.emo_buttons[0]))
        self.keyboard_emos.add(KeyboardButton(text=self.emo_buttons[1]))
        self.keyboard_emos.add(KeyboardButton(text=self.emo_buttons[2]))
        self.keyboard_emos.add(KeyboardButton(text=self.emo_buttons[3]))
        self.keyboard_emos.add(KeyboardButton(text=self.emo_buttons[4]))
        self.keyboard_emos.add(KeyboardButton(text=self.emo_buttons[5]))
        self.keyboard_emos.add(KeyboardButton(text=self.emo_buttons[6]))
        self.keyboard_emos.add(KeyboardButton(text=self.emo_buttons[7]))
        self.keyboard_emos.add(KeyboardButton(text=self.t_back))

        




    def handle_photo_doc(self, message): 
        if self.update_login_google == True and self.shipping_method == 'google_drive':
            self.bot.send_message(message.chat.id, 
                        f"Сначала авторизируйтесь в аккаунте google drive...", 
                        reply_markup=self.keyboard_settings)
            return


        try:
            if message.content_type == 'document':
                file = self.bot.get_file(message.document.file_id)

            if message.content_type == 'photo':
                # -1 = присланное пользователем фото в самом хорошем качестве (меньшие индексы = хуже)
                file = self.bot.get_file(message.photo[-1].file_id) 
    
            file_content = self.bot.download_file(file.file_path)
            
            if self.new_icon_sent == True:
                image = Image.open(io.BytesIO(file_content))
                path_to_emo = str(self.root_dir / 'emo_imgs' / f'{self.emotion_labels_en[self.new_icon_num]}.png')
                
                # Добавление альфа-канала, если его нет
                if image.mode != 'RGBA':
                    image = image.convert('RGBA')
                    
                image.save(path_to_emo, format='PNG')
                self.bot.send_message(message.chat.id, 
                        f'Новая иконка для эмоции "{self.emotion_labels[self.new_icon_num]}" установлена', 
                        reply_markup=self.keyboard_settings)
                
                self.new_icon_sent = False
                self.smile_size += 1
                return
            
            image_res = self.emo_rec.make_image(file_content, self.face_sensitivity_photo, self.smile_size, self.alpha)
            
            # self.bot.send_photo(message.chat.id, result)
            
            # self.buffer_to_send_photo.append(InputMediaPhoto(result))
            # if len(self.buffer_to_send_photo) == self.img_num_from_user:
            #     self.bot.send_media_group(chat_id=message.chat.id, media=self.buffer_to_send_photo)
            #     self.buffer_to_send_photo = []
            # print(self.buffer_to_send_photo)

        except Exception as e:
            self.bot.reply_to(message, str(e))
            traceback.print_exc()
        
        
         # Отправка фото
        try:
            if self.shipping_method == 'telegram':
                    self.bot.send_photo(message.chat.id, image_res)
                    
            if self.shipping_method == 'google_drive':
                    self.upload_file_to_google_drive(image_res, mode='photo')
                    self.bot.send_message(message.chat.id, 
                        f"Фото отправлено на google drive...", 
                        reply_markup=self.keyboard_base)

        except Exception as e:
            self.bot.reply_to(message, str(e))
            traceback.print_exc()



    def handle_video_animation(self, message):
        if self.update_login_google == True and self.shipping_method == 'google_drive':
            self.bot.send_message(message.chat.id, 
                        f"Сначала авторизируйтесь в аккаунте google drive...", 
                        reply_markup=self.keyboard_settings)
            return

        try:
            if message.content_type == 'video':
                file = self.bot.get_file(message.video.file_id)
                file_content = self.bot.download_file(file.file_path)
                self.bot.send_message(message.chat.id, 
                    f"Начинаем обрабатывать ваше видео...", 
                    reply_markup=self.keyboard_base)
                video_res, path_to_res_video = self.emo_rec.make_video(file_content, message.video.file_id, self.face_sensitivity_video, self.smile_size, self.alpha, is_compress_result=self.is_compress_result, compress_video_fps = self.compress_video_fps)
            if message.content_type == 'animation': # .animation. ~= .document. != .gif.
                file = self.bot.get_file(message.animation.file_id) # .animation. ~= .document. != .gif.
                file_content = self.bot.download_file(file.file_path)
                self.bot.send_message(message.chat.id, 
                    f"Начинаем обрабатывать ваше видео...", 
                    reply_markup=self.keyboard_base)
                video_res, path_to_res_video = self.emo_rec.make_video(file_content, message.animation.file_id, self.face_sensitivity_video, self.smile_size, self.alpha, is_compress_result=self.is_compress_result, compress_video_fps = self.compress_video_fps) # .animation. ~= .document. != .gif.
                
        except Exception as e:
            self.bot.reply_to(message, str(e))
            traceback.print_exc()

        # Отправка видео
        if self.shipping_method == 'telegram':
            try:
                self.bot.send_video(message.chat.id, video_res)
                
            except Exception as e:
                print({str(e)}) # A request to the Telegram API was unsuccessful. Error code: 413. Description: Request Entity Too Large 
                self.bot.send_message(message.chat.id, 
                    f"\nВидео оказалось слишком большимм для текущего уровня Telegram API, попробуйте выбрать в вариантах отправки google drive", 
                    reply_markup=self.keyboard_base)
            finally:
                os.remove(str(path_to_res_video))
        
        if self.shipping_method == 'google_drive':
            try: 
                self.bot.send_message(message.chat.id, 
                    f"Отправляем видео на google drive...", 
                    reply_markup=self.keyboard_base)
                self.upload_file_to_google_drive(path_to_res_video, mode='video')
                self.bot.send_message(message.chat.id, 
                    f"Видео отправлено на google drive", 
                    reply_markup=self.keyboard_base)
            except Exception as e:
                self.bot.reply_to(message, str(e))
                traceback.print_exc()
            finally:
                os.remove(str(path_to_res_video))




    def check_creds_to_google_drive(self, message):
        # Подключение к Google Drive API
        API_application_area = ['https://www.googleapis.com/auth/drive.file']
        self.credentials = None


        # Если уже зарегестрировались, то берём из файла token.json
        if os.path.exists(self.authorized_user_file) \
                    and not self.update_login_google: # Для смены выбранного ранее аккаунт гугла
            
            with open(self.authorized_user_file, 'r') as token:
                self.credentials = Credentials.from_authorized_user_info(info=json.load(token))


        # Если credentials не валидные или их ещё нет
        if not self.credentials or not self.credentials.valid \
                    or self.update_login_google: # Для смены выбранного ранее аккаунт гугла
            
            
            # Если истёк, обновить
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())
                # Запись в token.json 
                with open(self.authorized_user_file, 'w') as token:
                    token.write(self.credentials.to_json())
            # Иначе создать из client_secrets.json
            else:
                flow = Flow.from_client_secrets_file('./.secrets/client_secrets.json', 
                                                     scopes=API_application_area, 
                                                     redirect_uri=self.t_auth_response_part)
                authorization_url, state = flow.authorization_url(prompt='consent')
                
                # Отправка сообщения с кнопкой авторизыции
                auth_keyboard_msg = InlineKeyboardMarkup()
                auth_keyboard_msg.add(InlineKeyboardButton("Авторизоваться", url=authorization_url))
                self.bot.send_message(message.chat.id, 
                                    "Авторизируйтесь в браузере в том аккаунте, на который надо отправить обработанный файл", 
                                    reply_markup=auth_keyboard_msg)
                self.bot.send_message(message.chat.id, 
                    f"Отправьте в чат URL на который вас перенаправят после авторизации: ", 
                    reply_markup=self.keyboard_settings)
                
                # Продолжение в обработке ответа пользователя
                # return
                self.send_url_resp_event.wait()
                
                flow.fetch_token(authorization_response=self.t_auth_response_full)
                self.credentials = flow.credentials
            
            
            # Запись в token.json 
            with open(self.authorized_user_file, 'w') as token:
                token.write(self.credentials.to_json())
        
        
        
        # Проверка корректности
        if self.credentials and self.credentials.valid:
            self.bot.send_message(message.chat.id, 
                                        "Авторизация прошла успешно", 
                                        reply_markup=self.keyboard_settings)
            self.update_login_google = False
        else:
            self.bot.send_message(message.chat.id, 
                                        "Авторизация не удалась", 
                                        reply_markup=self.keyboard_settings)




    def upload_file_to_google_drive(self, file_path, mode): 
        # Подготовка файла
        if mode == 'video':
            media = MediaFileUpload(file_path, resumable=True)
            file_metadata = {'name': 'emo_rec_output.mp4'}
        if mode == 'photo':
            file_metadata = {'name': 'emo_rec_output.jpg'}
            media = MediaIoBaseUpload(io.BytesIO(file_path), mimetype="image/jpeg", resumable=True)


        # Загрузка файла на Google Диск
        service = build('drive', 'v3', credentials=self.credentials)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        
        print('Файл успешно загружен. ID файла: %s' % file.get('id'))



    def handle_start(self, message):
        """Обработчик комнады "start"; функция для создания стартового меню
        Args:
            message (Message): Объект сообщения
        """
        # Создание меню с кнопками
        
        self.bot.send_message(message.chat.id, 
                        "Отпрвьте фото или видео и получите их обработанную версию назад \n\nЕсли пределы API телеграмма будут превышены, то вам будет предложено сохранить видео на google drive", 
                        reply_markup=self.keyboard_base)



    def start_polling(self):
        """Функция для запуска бота"""
        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            self.handle_start(message)
            
        @self.bot.message_handler(content_types=['photo', 'document'])
        def handle_photo_doc(message): 
            self.handle_photo_doc(message)
        
        @self.bot.message_handler(content_types=['video', 'animation']) # 'gif' = 'animation'
        def handle_video_animation(message): 
            self.handle_video_animation(message)
            
        @self.bot.message_handler(content_types=['text'])
        def handle_text(message): 
            self.handle_text(message)
            
        
        # Запуск бота
        self.bot.polling(none_stop=True, interval=0)

    def is_float(self, s):
        try:
            float(s.replace(',', '.'))
            return True
        except ValueError:
            return False
        

    def get_key_by_value(self, dictionary, value):
        for key, val in dictionary.items():
            if val == value:
                return key
        # If the value is not found, return None or handle it according to your needs
        return None


    def handle_text(self, message):
        
        if message.text == self.t_help:
            self.handle_start(message)
            
            
        if message.text == self.t_settings:
            self.bot.send_message(message.chat.id, 
                f"Выберите то, что хотите настроить", 
                reply_markup=self.keyboard_settings)
        
        if message.text == self.t_location_receive_result:
            self.bot.send_message(message.chat.id, 
                f"Выберите место получения результата", 
                reply_markup=self.keyboard_shipping_method)


        if message.text == self.t_reset_google_drive:
            self.update_login_google = True
            # self.check_creds_to_google_drive(message,)
            threading.Thread(target=self.check_creds_to_google_drive, args=(message,)).start()
        
        if message.text == self.t_telegram:
            self.shipping_method = 'telegram'
            self.bot.send_message(message.chat.id, 
                f"Вы будете получать результаты обработки в этот чат", 
                reply_markup=self.keyboard_settings)
            
        if message.text == self.t_google_drive:
            self.shipping_method = 'google_drive'
            self.bot.send_message(message.chat.id, 
                f"Вы будете получать результаты обработки на ваш google drive (после авторизации)", 
                reply_markup=self.keyboard_settings)


        if message.text == self.t_back:
            self.bot.send_message(message.chat.id, 
                f"Пришлите фото или видео файл", 
                reply_markup=self.keyboard_base)
            
        if self.t_auth_response_part in message.text:
            self.t_auth_response_full = message.text
            self.send_url_resp_event.set()


        if message.text == self.t_set_smile_size:
            self.previous_command = self.t_set_smile_size
            self.bot.send_message(message.chat.id, 
                f"Напишите целое число в пиксалях, которое будет использоваться для установки ширины и высоты смайлика", 
                reply_markup=self.keyboard_settings)

        if message.text == self.t_set_face_sensitivity_photo:
            self.previous_command = self.t_set_face_sensitivity_photo
            self.bot.send_message(message.chat.id, 
                f"Напишите целое число от 1 до 30, устанавливающее чувствительность поиска лиц для фото (где 1 - много ложных срабатываний; 30 - недостаток детектирования; по умолчанию: для фото - 14", 
                reply_markup=self.keyboard_settings)
            
        if message.text == self.t_set_face_sensitivity_video:
            self.previous_command = self.t_set_face_sensitivity_video
            self.bot.send_message(message.chat.id, 
                f"Напишите целое число от 1 до 30, устанавливающее чувствительность поиска лиц для видео (где 1 - много ложных срабатываний; 30 - недостаток детектирования; по умолчанию: для видео - 12", 
                reply_markup=self.keyboard_settings)

        if message.text == self.t_set_alpha:
            self.previous_command = self.t_set_alpha
            self.bot.send_message(message.chat.id, 
                f"Напишите не целое число от 0.0 до 1.0, устанавливающее прозрачность смайликов (где 1.0 - непрозрачное, 0.0 - прозрачное)", 
                reply_markup=self.keyboard_settings)

        if message.text.isdigit() or self.is_float(message.text.replace(',', '.')):
            if self.previous_command == self.t_set_smile_size:
                self.smile_size = int(message.text)
                self.bot.send_message(message.chat.id, 
                    f"Установлен размер смайлика в {self.smile_size} пикселей", 
                    reply_markup=self.keyboard_settings)
                
            if self.previous_command == self.t_set_face_sensitivity_photo:
                self.face_sensitivity_photo = int(message.text)
                self.bot.send_message(message.chat.id, 
                    f"Установлена чувствительность поиска лиц для фото = {self.face_sensitivity_photo}", 
                    reply_markup=self.keyboard_settings)
                
            if self.previous_command == self.t_set_face_sensitivity_video:
                self.face_sensitivity_video = int(message.text)
                self.bot.send_message(message.chat.id, 
                    f"Установлена чувствительность поиска лиц для видео = {self.face_sensitivity_video}", 
                    reply_markup=self.keyboard_settings)
                
            if self.previous_command == self.t_set_alpha:
                self.alpha = float(message.text.replace(',', '.'))
                self.bot.send_message(message.chat.id, 
                    f"Установлен прозрачность смайликов = {self.alpha}", 
                    reply_markup=self.keyboard_settings)
            
        if message.text == self.t_set_icons:
            self.bot.send_message(message.chat.id, 
                f"Выбеите эмоцию, иконку которой хотите замменить", 
                reply_markup=self.keyboard_emos)
        
        if message.text in self.emo_buttons:
            self.new_icon_sent = True
            self.new_icon_num = self.emo_buttons.index(message.text) 
            self.bot.send_message(message.chat.id, 
                f"Пришлите как файл (не сжатую) новую иконку эмоции", 
                reply_markup=self.keyboard_emos)
