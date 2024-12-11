from ultralytics import YOLO
from bson import decode_all
import random
import os
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
import time

# Инициализируем две модели
model_last = YOLO('last.pt')
model_badya = YOLO('badya.pt')

# model.train(
#     data="data/data.yaml",  # path to dataset YAML
#     epochs=150,  # number of training epochs
#     imgsz=500,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )

# При несовпадении ответов отправлять на модерацию в отдельную папку

script_dir = os.path.dirname(os.path.abspath(__file__))
moderation_dir = os.path.join(script_dir, 'moderation')
os.makedirs(moderation_dir, exist_ok=True)

bson_file_path = os.path.join(script_dir, 'users.bson')

with open(bson_file_path, 'rb') as f:
    users_data = f.read()

users = decode_all(users_data)
random.seed(3)

def save_to_moderation(img, filename):
    filepath = os.path.join(moderation_dir, filename)
    img.save(filepath)
    print(f"Изображение сохранено на модерацию: {filepath}")

# Функция для обработки одного изображения и вывода результатов по модели
def process_image(model, img, model_number):
    results = model(img)
    counter_f = 0
    counter_m = 0
    # Считаем количество обнаруженных объектов
    for result in results:
        for data in result.boxes.data:
            if model_number == 1:
                if data[5] == 0.0:
                    counter_f += 1
                else:
                    counter_m += 1
            else:
                if data[5] == 0.0:
                    counter_m += 1
                else:
                    counter_f += 1
    if counter_f + counter_m != 0:
        result.show()
        print(f"Кол-во мужчин: {counter_m}\nКол-во женщин: {counter_f}")
        if counter_m + counter_f == 1:
            print("Одиночное фото")
        elif counter_m + counter_f > 2:
            print("Групповое фото")
        elif counter_m == 2 or counter_f == 2:
            print("Дружеское фото")
        elif counter_m == 1 and counter_f == 1:
            print("Романтическое фото")
        else:
            print("Класс не распознан")
    else:
        print("Объекты не распознаны на изображении.")
    
    # Всегда возвращаем результаты
    return counter_m, counter_f

while True:
    random_user = random.choice(users)
    random_user_photo_url = random_user["photo_max_orig"]
    print(random_user_photo_url)

    if random_user_photo_url == "https://sun6-22.userapi.com/impf/DW4IDqvukChyc-WPXmzIot46En40R00idiUAXw/l5w5aIHioYc.jpg?quality=96&as=32x32,48x48,72x72,108x108,160x160,240x240,360x360&sign=10ad7d7953daabb7b0e707fdfb7ebefd&u=I6EtahnrCRLlyd0MhT2raQt6ydhuyxX4s72EHGuUSoM&cs=240x240":
        continue
    print(random_user_photo_url)

    # Загрузка изображения по URL
    response = requests.get(random_user_photo_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    if img.size[0] < 240 or img.size[1] < 240:
        continue

    print("\n=== Результаты для модели 'last.pt' ===")
    count_m1, count_f1 = process_image(model_last, img, 1)  # Обработка изображения на первой модели

    print("\n=== Результаты для модели 'badya.pt' ===")
    count_m2, count_f2 = process_image(model_badya, img, 2)  # Обработка изображения на второй модели

    if (count_m1, count_f1) != (count_m2, count_f2):
        print("Результаты моделей не совпали!")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"moderation_{timestamp}.jpg"
        save_to_moderation(img, filename)
    else:
        print("Результаты моделей совпали.")

    choice = input("Продолжить? Y/N\n")
    if choice == "N":
        break

    
