from my_bot import MyBot

import requests


session = requests.Session()
session.timeout = 300

my_bot = MyBot()

my_bot.start_polling()
