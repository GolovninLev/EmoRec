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

from emo_rec import EmoRec


class MyBot:
    
    def __init__(self):
        # os.getenv('TOKEN')
        self.bot = telebot.TeleBot('6829160910:AAEmmlh0aB567vnpfSsFeTA7CV1Z_vGl3XA', skip_pending=True)
        
        self.emo_rec = EmoRec()
        
        self.is_compress_result = True
        
        # self.buffer_to_send_photo = []
        # self.img_num_from_user = 1
        
        self.t_settings = "Настройки"


    def start_polling(self):
        """Функция для запуска бота"""
        @self.bot.message_handler(content_types=['photo', 'document'])
        def handle_photo_doc(message): 
            self.handle_photo_doc(message)
        
        @self.bot.message_handler(content_types=['video'])
        def handle_video(message): 
            self.handle_video(message)
            
        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            self.handle_start(message)
        
        # Запуск бота
        self.bot.polling(none_stop=True, interval=0)


    def handle_video(self, message):
        try:
            file = self.bot.get_file(message.video.file_id)
            file_content = self.bot.download_file(file.file_path)
            result = self.emo_rec.make_video(file_content, message.video.file_id, is_compress_result=self.is_compress_result)
            self.bot.send_video(message.chat.id, result)
            
        except Exception as e:
            self.bot.reply_to(message, str(e))
            traceback.print_exc()

    
    def handle_start(self, message):
        """Обработчик комнады "start"; функция для создания стартового меню
        Args:
            message (Message): Объект сообщения
        """
        # Создание меню с кнопками
        keyboard = ReplyKeyboardMarkup(resize_keyboard = True)
        
        keyboard.add(KeyboardButton(text=self.t_settings))
        
        self.bot.send_message(message.chat.id, 
                        "Отпрвьте фото или видео и получите их обработанную версию назад", 
                        reply_markup=keyboard)


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



my_bot = MyBot()
my_bot.start_polling()



