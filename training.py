from ultralytics import YOLO

def train_model():
    # ✅ Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # ✅ Train (fine-tune)
    model.train(
        data="dataset3/data.yaml",   # correct path
        epochs=30,                  # more epochs for better learning
        imgsz=640,                  # standard resolution
        batch=8,                    # adjust if GPU allows
        lr0=0.001,                  # lower learning rate (important)
        optimizer="AdamW",          # better optimizer
        patience=20,                # early stopping
        augment=True,               # built-in augmentation
        degrees=10,                 # rotation
        fliplr=0.5,                 # horizontal flip
        hsv_h=0.015,                # color augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,                 # YOLO augmentation
        mixup=0.1,                  # blending images
        workers=2,                  # avoid Windows crash
        device=0,                   # GPU (use "cpu" if no GPU)
        name="helmet_bigdata",    # output folder
        exist_ok=True               # overwrite if exists
    )



if __name__ == "__main__":
    train_model()