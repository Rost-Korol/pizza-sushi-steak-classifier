# pizza-sushi-steak-classifier
Model that can predict what kind of food (in 3 categories) in the image. Also there are a simple GUI and telegram-bot API to interact with model 

### Модель

Реализовано два варианта модели:
1. кастомная VGG модель с двумя свёрточными блоками. В каждом 2 сверточных слоя, между ними ReLU активация, затем MaxPool слой
2. Переученная на свои данные efficientnet

модель использует уже ранее обученные веса, находящиеся в дирректории models/

### Данные

Данные взяты из датасета Food101. Для того, чтобы модель быстрее обучалась, использовалось только три класса самой популярной еды: пицца, суши и стейки. Использовался TrivialAugmentWide аугментатор для изображений

### GUI

Пользовательский интерфейс реализован в фреймворке TKinter. Представляет из себя окно в котором предлагается выбрать изображение из вашего локального хранилища 

![gui_main](https://github.com/Rost-Korol/pizza-sushi-steak-classifier/assets/91683515/1e05e6a7-e1c5-4892-a584-75fa15d460c3)

Затем после нажатия кнопки predict 

![with_image](https://github.com/Rost-Korol/pizza-sushi-steak-classifier/assets/91683515/388846ef-fcdb-4a0d-b37c-b8c7ab16f965)

модель выдаёт предсказание, котоорое исчезает через пару секунд

![prediction](https://github.com/Rost-Korol/pizza-sushi-steak-classifier/assets/91683515/3c0bd830-aca9-4316-808d-57564d33bc5f)

### Telegram bot

https://t.me/pizza_sushi_steak_bot

Телеграм бот реализован на базе фреймворка python-telegram-bot он принимает изображения* и выдаёт ответ в виде сообщения.

![telegram-bot](https://github.com/Rost-Korol/pizza-sushi-steak-classifier/assets/91683515/ad73ccbb-7251-4528-a285-32be19212401)

Если вы хотите реализовать свой бот на базе этого кода, вам нужно получить свой token через bot_father в телеграм.

* В данной версии не работает через Desktop версию telegram. Пользуйтесь мобильной версией!
