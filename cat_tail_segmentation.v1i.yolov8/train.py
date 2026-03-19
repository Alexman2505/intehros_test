"""
Обучение YOLOv8-seg для сегментации кошачьих хвостов
"""

import os
import torch
from pathlib import Path
from ultralytics import YOLO

# ============================================
# 1. КОНФИГУРАЦИЯ - ОПТИМИЗИРОВАНО ПОД 39 ЭПОХ
# ============================================
if __name__ == '__main__':
    DATA_YAML = "data.yaml"  # файл с описанием датасета
    EPOCHS = 39  # количество эпох
    BATCH_SIZE = 8  # размер батча
    IMG_SIZE = 640  # размер изображений
    WORKERS = 4  # потоки загрузки
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    PROJECT_NAME = "cat_tail_final"  # папка для результатов

    # ============================================
    # 2. ПРОВЕРКА ОКРУЖЕНИЯ
    # ============================================

    print("=" * 50)
    print("ПРОВЕРКА КОНФИГУРАЦИИ")
    print("=" * 50)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    print(f"Устройство для обучения: {DEVICE}")
    print(f"Количество эпох: {EPOCHS}")
    print(f"Размер батча: {BATCH_SIZE}")

    # ============================================
    # 3. ЗАГРУЗКА МОДЕЛИ
    # ============================================

    print("\n" + "=" * 50)
    print("ЗАГРУЗКА МОДЕЛИ")
    print("=" * 50)

    # Загружаем предобученную модель для сегментации
    model = YOLO('yolov8n-seg.pt')
    print(" Модель загружена")

    # ============================================
    # 4. ОБУЧЕНИЕ
    # ============================================

    print("\n" + "=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)
    print(f"Результаты будут сохранены в: {PROJECT_NAME}/")

    # Запускаем обучение
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT_NAME,
        name='experiment',
        # Только проверенные аугментации
        augment=True,
        copy_paste=0.2,  # как в первом запуске
        mosaic=1.0,
        fliplr=0.5,  # горизонтальный флип (безопасно)
        save=True,
        save_period=5,
        weight_decay=0.0005,  # увеличили
        plots=True,
        val=True,
    )

    print("\n ОБУЧЕНИЕ ЗАВЕРШЕНО!")

    # ============================================
    # 5. ВАЛИДАЦИЯ ЛУЧШЕЙ МОДЕЛИ
    # ============================================

    print("\n" + "=" * 50)
    print("ВАЛИДАЦИЯ ЛУЧШЕЙ МОДЕЛИ")
    print("=" * 50)

    # Путь к лучшей модели
    project_root = Path(__file__).parent  # папка где лежит train.py
    best_model_path = Path(
        r'D:\Dev\intehros\intehros_test\runs\segment\cat_tail_final\experiment\weights\best.pt'
    )
    print(best_model_path)
    best_model = YOLO(str(best_model_path))

    # Валидация
    metrics = best_model.val(data=DATA_YAML)

    print("\n ИТОГОВЫЕ МЕТРИКИ ДЕТЕКЦИИ:")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")

    print("\n МЕТРИКИ СЕГМЕНТАЦИИ:")
    print(f"  - Mask Precision: {metrics.seg.mp:.4f}")
    print(f"  - Mask Recall: {metrics.seg.mr:.4f}")
    print(f"  - Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"  - Mask mAP50-95: {metrics.seg.map:.4f}")
    # F1-score
    if metrics.box.mp + metrics.box.mr > 0:
        f1 = (
            2
            * (metrics.box.mp * metrics.box.mr)
            / (metrics.box.mp + metrics.box.mr)
        )
        print(f"\n  - F1-score: {f1:.4f}")

    print("\n МЕТРИКИ СЕГМЕНТАЦИИ:")
    print(f"\n  - Mask Precision: {metrics.seg.mp:.4f}")
    print(f"\n  - Mask Recall: {metrics.seg.mr:.4f}")
    print(f"\n  - Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"\n  - Mask mAP50-95: {metrics.seg.map:.4f}")

    # ============================================
    # 6. КОНВЕРТАЦИЯ В ONNX
    # ============================================

    print("\n" + "=" * 50)
    print("КОНВЕРТАЦИЯ В ONNX")
    print("=" * 50)

    # Экспортируем в ONNX
    onnx_path = best_model.export(
        format='onnx',
        imgsz=IMG_SIZE,
        half=False,
        simplify=True,
        opset=12,
        device='cpu',
    )

    # ============================================
    # 7. ИТОГ
    # ============================================

    print("\n" + "=" * 50)
    print("ГОТОВО!")
    print("=" * 50)
    print(f"\n Результаты обучения: {PROJECT_NAME}/experiment/")
    print(
        f"\n Лучшая модель PyTorch: {PROJECT_NAME}/experiment/weights/best.pt"
    )
    print(f"\n ONNX модель: best.onnx")
    print(f"\n Графики обучения: {PROJECT_NAME}/experiment/results.png")
