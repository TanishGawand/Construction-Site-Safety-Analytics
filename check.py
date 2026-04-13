import os

# 🔁 CHANGE THIS PATH
DATASET_PATH = r"C:\Users\Tanish\OneDrive\Desktop\ConstructionSite_Project\dataset3"

label_dirs = [
    os.path.join(DATASET_PATH, "train/labels"),
    os.path.join(DATASET_PATH, "valid/labels"),
    os.path.join(DATASET_PATH, "test/labels")
]

removed_files = 0

for label_dir in label_dirs:
    image_dir = label_dir.replace("labels", "images")
    
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            label_path = os.path.join(label_dir, file)
            
            if os.path.getsize(label_path) == 0:
                os.remove(label_path)
                
                img_file = file.replace(".txt", ".jpg")  # adjust if png
                img_path = os.path.join(image_dir, img_file)
                
                if os.path.exists(img_path):
                    os.remove(img_path)
                    removed_files += 1

print(f"Removed {removed_files} empty images + labels")