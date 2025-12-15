import cv2
import numpy as np
from pathlib import Path
from PIL import Image

DATA_DIR = Path("training_images")

OUT_DIR = Path("dataset")
OUT_IMG = OUT_DIR / "images/train"
OUT_LBL = OUT_DIR / "labels/train"
OUT_MASK = OUT_DIR / "masks/train"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)

def compute_ndwi(img):
    green = img[:, :, 1].astype(float)
    blue = img[:, :, 0].astype(float)
    return (green - blue) / (green + blue + 1e-6)

images = list(DATA_DIR.glob("*.jpg"))
print(f"{len(images)} görüntü bulundu. Polygon maskeler üretiliyor...")

for img_path in images:
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w, _ = img.shape

    ndwi = compute_ndwi(img)
    mask = (ndwi > 0.05).astype(np.uint8) * 255

    # Maske kaydet
    cv2.imwrite(str(OUT_MASK / f"{img_path.stem}.png"), mask)

    # Polygon bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_path = OUT_LBL / f"{img_path.stem}.txt"
    with open(label_path, "w") as f:
        if len(contours) == 0:
            print(f"⚠ {img_path.name}: su bulunamadı → boş mask kaydedildi.")
            continue

        cnt = max(contours, key=cv2.contourArea)

        points = []
        for p in cnt:
            x, y = p[0]
            points.append(x / w)
            points.append(y / h)

        # YOLO SEG formatı:
        # class x1 y1 x2 y2 x3 y3 ...
        f.write("0 " + " ".join(map(str, points)))

    # Orijinal görüntüyü de kopyala
    cv2.imwrite(str(OUT_IMG / img_path.name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print("✔ YOLOv8-SEGMANTASYON dataset hazır!")
