from ultralytics import YOLO

def run_comparison():
    # Base model
    model_base = YOLO("yolov8n.pt")
    metrics_base = model_base.val(data="data.yaml")

    # Fine-tuned model
    model_custom = YOLO("runs/detect/helmet_bigdata/weights/best.pt")
    metrics_custom = model_custom.val(data="data.yaml")

    print("\n===== BASE YOLO =====")
    print(f"mAP50: {metrics_base.box.map50}")
    print(f"mAP50-95: {metrics_base.box.map}")
    print(f"Precision: {metrics_base.box.mp}")
    print(f"Recall: {metrics_base.box.mr}")

    print("\n===== FINE-TUNED =====")
    print(f"mAP50: {metrics_custom.box.map50}")
    print(f"mAP50-95: {metrics_custom.box.map}")
    print(f"Precision: {metrics_custom.box.mp}")
    print(f"Recall: {metrics_custom.box.mr}")


# 🔥 IMPORTANT (Windows fix)
if __name__ == "__main__":
    run_comparison()