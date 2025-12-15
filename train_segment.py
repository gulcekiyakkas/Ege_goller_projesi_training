from ultralytics import YOLO

def main():
    # YOLOv8 segmentation modeli
    model = YOLO("yolov8s-seg.pt")

    # Modeli eğit
    model.train(
        data="data.yaml",      # dataset yolu
        epochs=50,             # eğitim sayısı
        imgsz=512,             # görüntü boyutu
        batch=8,               # batch size
        name="seg_train",      # kayıt klasörü adı
        workers=2,             # windows için önerilen
        device=0               # GPU kullan
    )

if __name__ == "__main__":
    main()
