from ultralytics import YOLO

def main():
    # ===============================
    # MODEL ve DATA AYARLARI
    # ===============================
    MODEL_PATH = "yolov8s-seg.pt"
    DATA_YAML = "data.yaml"
    PROJECT_DIR = "runs/segment"

    # ===============================
    # MODELİ YÜKLE
    # ===============================
    model = YOLO(MODEL_PATH)

    # ===============================
    # EĞİTİM
    # ===============================
    model.train(
        task="segment",
        data=DATA_YAML,
        epochs=80,
        imgsz=512,
        batch=8,
        name="seg_trainV",
        project=PROJECT_DIR,
        device=0,        # GPU
        workers=0,       # 🔴 Windows için şart
        patience=10,
        verbose=True
    )

    print("✔ YOLOv8 Segmentasyon Eğitimi Başarıyla Tamamlandı")

# ===============================
# WINDOWS GÜVENLİK BLOĞU
# ===============================
if __name__ == "__main__":
    main()
