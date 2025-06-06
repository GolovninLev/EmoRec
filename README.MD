<img src="https://i.ibb.co/dt22f23/2024-11-30-22-35-39.png" alt="EmoRec-Demo" width="280" height="695" align="right"/>
<img src="https://i.ibb.co/bdLk5Df/2024-11-30-22-37-54.png" alt="EmoRec-Demo" width="280" height="570" align="right"/>

## Общее описание проекта

Бот в Телеграмме, которому можно отправить фото, видео или gif-анимацию. Бот распознаёт выражения лиц людей в кадре и возвращает:

*   Текстовый отчёт о выраженности эмоций на фото и видео
*   График в формате png  и в формате html с возможностью масштабирования и фильтрации, отражающий преобладающие эмоции на видео в зависимости от времени
*   Csv файл, в котором отражаются вероятности эмоций для каждого кадра, на котором были распознаны лица
*   Медиафайл (фото/видео) с добавленными аннотациями эмоций

Приложение запускается посредством контейнера Docker, что позволяет ему быть независимым от операционной системы и её окружения. 



## Настройки приложения

**В приложении предусмотрены следующие настройки:**

*   Установка пользовательских изображений для отображения эмоций. 
*   Изменение размера изображений для отображения эмоций. 
*   Изменение прозрачности изображений для отображения эмоций.
*   Выбор места получения обработанных фото и видео: обратно в чат или на облачное хранилище Google Диск после авторизации. 
*   Смена привязанного облачного хранилища Google Диск для получения обработанных фото и видео. 

**В приложении предусмотрено два вида получения результата:**

*   В виде ответных сообщений телеграм-бота. 
*   В виде загрузки файла на Google Диск после авторизации. 



## Описание содержимого репозитория

- В папке models хранится модель для обнаружения лиц на изображении.
  - В эту папку следует поместить обученные веса модели VGG19.

- В папке emo_imgs хранятся изображения эмоций по умолчанию.

- В папке src хранятся исходники бота и модуля распознавания эмоций.

- В папке model_learner хранится скрипт для обучения модели распознавания эмоций.

- В папке papers хранятся статьи, связанные с проведёнными экспериментами по обучению моделей, использующихся в данном проекте. 



## Сборка и запуск в Docker-контейнере
1. Клонируйте данный репозиторий

2. Положите в католог `models` веса модели используемой для распознования эмоций в файле с именем `emo_rec_model.pth` (не помещаются в GitHub) 

3. Если вы планируете использвоать функцию отправки данных на гугл диск (любой), предоставте файлы `token.json` и `client_secrets.json`, которые можно получить из своего аккаунта Google Cloud Console (Google API)

4. Для сборки контейнера выполните следующую команду в папке с Docker-файлом (где `emo_rec` - имя создаваемого образа, а `0` - тег версии образа):

    ```sh
    cd EmoRec
    ```

    ```sh
    docker build -t emo_rec:0 .
    ```

5. Создайте токен и название бота, например, согласно [следующей инструкции](https://web7.pro/kak-poluchit-token-bota-telegram-api/)

6. Запуск контейнера с помощью команды (где флаг `-d` указывает на запуск контейнера в фоновом режиме) (не забудьте вставить своё значение токена телеграмм бота):

    ```sh
    docker run --rm -it -d -e TOKEN=<значение токена телеграмм бота> emo_rec:0
    ```

    или 

    ```sh
    docker run --rm -it -d --env-file /путь/к/файлу/.env emo_rec:0
    ```
    

6. После сборки и запуска проекта, становится доступен бот по ссылке: 
    
    ```link
    https://t.me/<название бота, которое вы ему дали при генерации своего токена>
    ```


