import os
import io
from io import BytesIO
from io import StringIO
import traceback

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import telebot
from telebot.types import ReplyKeyboardMarkup
from telebot.types import KeyboardButton
from telebot.types import InputMediaPhoto

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

from emo_rec import EmoRec



class MyBot:
    
    
    def __init__(self):
        
        self.SECRETS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'client_secrets.json')
        gauth = GoogleAuth()
        gauth.client_config_file = self.SECRETS_FILE

        # self.upload_file_to_google_drive(r'src\vgg_face_dag.py', 'vgg_face_dag.py')
        
        
        self.bot = telebot.TeleBot(os.getenv('TOKEN'), skip_pending=True)
        
        self.emo_rec = EmoRec()
        
        
        self.is_compress_result = True
        self.compress_video_fps = 20
        
        # self.buffer_to_send_photo = []
        # self.img_num_from_user = 1
        
        self.t_settings = "Настройки"
        self.t_help = "Помощь"
        self.keyboard = ReplyKeyboardMarkup(resize_keyboard = True)
        self.keyboard.add(KeyboardButton(text=self.t_settings), KeyboardButton(text=self.t_help))
        




    def start_polling(self):
        """Функция для запуска бота"""
        @self.bot.message_handler(content_types=['photo', 'document'])
        def handle_photo_doc(message): 
            self.handle_photo_doc(message)
        
        @self.bot.message_handler(content_types=['video', 'animation']) # 'gif' = 'animation'
        def handle_video_animation(message): 
            self.handle_video_animation(message)
            
        @self.bot.message_handler(content_types=['text'])
        def handle_text(message): 
            self.handle_text(message)
            
        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            self.handle_start(message)
        
        # Запуск бота
        self.bot.polling(none_stop=True, interval=0)



    def handle_video_animation(self, message):
        try:
            if message.content_type == 'video':
                file = self.bot.get_file(message.video.file_id)
                file_content = self.bot.download_file(file.file_path)
                video_res, path_to_res_video = self.emo_rec.make_video(file_content, message.video.file_id, is_compress_result=self.is_compress_result, compress_video_fps = self.compress_video_fps)
            if message.content_type == 'animation': # .animation. ~= .document. != .gif.
                file = self.bot.get_file(message.animation.file_id) # .animation. ~= .document. != .gif.
                file_content = self.bot.download_file(file.file_path)
                video_res, path_to_res_video = self.emo_rec.make_video(file_content, message.animation.file_id, is_compress_result=self.is_compress_result, compress_video_fps = self.compress_video_fps) # .animation. ~= .document. != .gif.
                
        except Exception as e:
            self.bot.reply_to(message, str(e))
            traceback.print_exc()
            self.bot.send_video(message.chat.id, video_res)


        try:
            self.bot.send_video(message.chat.id, video_res)
            
        except Exception as e:
            print({str(e)}) # A request to the Telegram API was unsuccessful. Error code: 413. Description: Request Entity Too Large 
            self.send_to_google_drive(path_to_res_video, message, message_to_user_text = f"\nВидео оказалось слишком большимм для текущего уровня Telegram API, но мы попытаемся отправить его на ваш гугл диск")
                
        finally:
            os.remove(str(path_to_res_video))



    def send_to_google_drive(self, path_to_res_video, message, message_to_user_text):
        download_link = self.upload_file_to_google_drive(path_to_res_video, os.path.basename(path_to_res_video))
        self.bot.send_message(message.chat.id, 
                message_to_user_text, 
                reply_markup=self.keyboard)
        
        if download_link:
            self.bot.send_message(message.chat.id, 
                f"Ссылка для скачивания файла: {download_link}", 
                reply_markup=self.keyboard)
        else:
            self.bot.send_message(message.chat.id, 
                f"Что-то пошло не так",
                reply_markup=self.keyboard)


    
    def handle_start(self, message):
        """Обработчик комнады "start"; функция для создания стартового меню
        Args:
            message (Message): Объект сообщения
        """
        # Создание меню с кнопками
        
        self.bot.send_message(message.chat.id, 
                        "Отпрвьте фото или видео и получите их обработанную версию назад \n\nЕсли пределы API телеграмма будут превышены, то вам будет предложено сохранить видео на гугл диск", 
                        reply_markup=self.keyboard)



    def handle_photo_doc(self, message): 
        try:
            if message.content_type == 'document':
                file = self.bot.get_file(message.document.file_id)

            if message.content_type == 'photo':
                # -1 = присланное пользователем фото в самом хорошем качестве (меньшие индексы = хуже)
                file = self.bot.get_file(message.photo[-1].file_id) 
    
            file_content = self.bot.download_file(file.file_path)
            result = self.emo_rec.make_image(file_content)
            self.bot.send_photo(message.chat.id, result)
            
            # self.buffer_to_send_photo.append(InputMediaPhoto(result))
            # if len(self.buffer_to_send_photo) == self.img_num_from_user:
            #     self.bot.send_media_group(chat_id=message.chat.id, media=self.buffer_to_send_photo)
            #     self.buffer_to_send_photo = []
            # print(self.buffer_to_send_photo)

        except Exception as e:
            self.bot.reply_to(message, str(e))
            traceback.print_exc()



    def upload_file_to_google_drive(self, file_path, file_name):
        
        drive = GoogleDrive(GoogleAuth().LocalWebserverAuth()) # аутентификация через браузер

        try:
            # Создание папки
            folder_metadata = {'title': 'emo_rec', 'mimeType': 'application/vnd.google-apps.folder'}
            folder = drive.CreateFile(folder_metadata)
            folder.Upload()

            # Загрузка файла в созданную папку
            file_metadata = {'title': file_name, 'parents': [{'id': folder['id']}]}
            file = drive.CreateFile(file_metadata)
            file.SetContentFile(file_path)
            file.Upload()

            # Создание ссылки для скачивания
            file.InsertPermission({
                'type': 'anyone',
                'value': 'anyone',
                'role': 'reader'
            })
            download_link = file['alternateLink']
            
            return download_link
        
        except Exception as e:
            print("Произошла ошибка во время отправки файла на гугл-диск:", e)
            return None



    def handle_text(self, message):
        
        if message.text == self.t_help:
            self.handle_start(message)
            
        if message.text == self.t_settings:
            pass

