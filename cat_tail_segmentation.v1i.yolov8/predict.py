from ultralytics import YOLO

# Абсолютный путь к модели
model_path = r'D:\Dev\intehros\intehros_test\runs\segment\cat_tail_final\experiment\weights\best.pt'
model = YOLO(model_path)

# Проверка на тестовом фото
test_image = (
    r'D:\Dev\intehros\intehros_test\cats_photo\my_cat.jpg'  #  свой путь
)
results = model(test_image)
results[0].show()  # confidence score
results[0].save('result.jpg')
