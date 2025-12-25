import random
import shutil
from pathlib import Path

# ===============================
# AYARLAR
# ===============================
BASE_DIR = Path("dataset")

IMG_TRAIN = BASE_DIR / "images/train"
LBL_TRAIN = BASE_DIR / "labels/train"
MSK_TRAIN = BASE_DIR / "masks/train"

IMG_VAL = BASE_DIR / "images/val"
LBL_VAL = BASE_DIR / "labels/val"
MSK_VAL = BASE_DIR / "masks/val"

IMG_TEST = BASE_DIR / "images/test"
LBL_TEST = BASE_DIR / "labels/test"
MSK_TEST = BASE_DIR / "masks/test"

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.1,
    "test": 0.2
}

# ===============================
# KLASÖRLERİ OLUŞTUR
# ===============================
for p in [IMG_VAL, LBL_VAL, MSK_VAL, IMG_TEST, LBL_TEST, MSK_TEST]:
    p.mkdir(parents=True, exist_ok=True)

# ===============================
# GÖRÜNTÜ LİSTESİ
# ===============================
images = list(IMG_TRAIN.glob("*.jpg"))
random.shuffle(images)

n_total = len(images)
n_train = int(n_total * SPLIT_RATIO["train"])
n_val = int(n_total * SPLIT_RATIO["val"])

train_imgs = images[:n_train]
val_imgs = images[n_train:n_train + n_val]
test_imgs = images[n_train + n_val:]

# ===============================
# TAŞIMA FONKSİYONU
# ===============================
def move_files(img_list, img_dst, lbl_dst, msk_dst):
    for img_path in img_list:
        name = img_path.stem

        shutil.move(str(img_path), img_dst / img_path.name)
        shutil.move(str(LBL_TRAIN / f"{name}.txt"), lbl_dst / f"{name}.txt")
        shutil.move(str(MSK_TRAIN / f"{name}.png"), msk_dst / f"{name}.png")

# ===============================
# SPLIT UYGULA
# ===============================
move_files(val_imgs, IMG_VAL, LBL_VAL, MSK_VAL)
move_files(test_imgs, IMG_TEST, LBL_TEST, MSK_TEST)

print("✔ Dataset başarıyla train / val / test olarak ayrıldı.")
print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
