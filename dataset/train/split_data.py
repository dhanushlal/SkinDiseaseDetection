import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = "D:/skin_disease_project/dataset/train"
val_dir = "D:/skin_disease_project/dataset/val"
os.makedirs(val_dir, exist_ok=True)

# Loop through each class folderpred
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    
    if len(images) < 2:
        print(f"⚠️ Skipping '{class_name}' because it has less than 2 images.")
        continue

    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in val_imgs:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_dir, img)
        shutil.move(src, dst)

print("✅ Validation set created with 20% of each class (if at least 2 images).")
