import cv2
from ultralytics import YOLO
from datetime import datetime

# ✅ Load model
model = YOLO("runs/detect/helmet_bigdata/weights/best.pt")

cap = cv2.VideoCapture("video.mp4")

width, height = 960, 540

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_final.mp4", fourcc, fps, (width, height))

# 🔥 Counters
total_violations = 0
violation_ids = set()

# 🔥 Tracking history
history = {}
history_size = 8

# 🔥 Frame skipping
frame_count = 0
skip_frames = 2

def get_id(x1, y1, x2, y2):
    return ((x1 + x2)//2 // 80, (y1 + y2)//2 // 80)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    frame = cv2.resize(frame, (width, height))
    frame_violations = 0

    results = model(frame, conf=0.5)

    current_ids = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            obj_id = get_id(x1, y1, x2, y2)
            current_ids.append(obj_id)

            if obj_id not in history:
                history[obj_id] = []

            # ✅ FIXED LABEL LOGIC
            if label == "helmet":
                val = 1
            elif label == "head":
                val = 0
            else:
                continue

            history[obj_id].append(val)

            if len(history[obj_id]) > history_size:
                history[obj_id].pop(0)

            avg = sum(history[obj_id]) / len(history[obj_id])

            # ================= DECISION =================
            if avg > 0.6:
                text = "SAFE"
                color = (0, 255, 0)
            else:
                text = "NO HELMET"
                color = (0, 0, 255)

                # ✅ Count only once per person
                if obj_id not in violation_ids:
                    violation_ids.add(obj_id)
                    total_violations += 1

                frame_violations += 1

            # ================= DRAW =================
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame,
                          (x1, y1 - th - 10),
                          (x1 + tw + 10, y1),
                          color,
                          -1)

            # Label text
            cv2.putText(frame, text,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

    # cleanup old IDs
    history = {k: v for k, v in history.items() if k in current_ids}

    # ================= UI =================
    cv2.rectangle(frame, (0, 0), (width, 40), (30, 30, 30), -1)
    cv2.putText(frame, "Construction Safety Monitoring System",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(frame, (10, 50), (300, 140), (40, 40, 40), -1)

    cv2.putText(frame, "SYSTEM STATUS",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, f"Current: {frame_violations}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"Total: {total_violations}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 🚨 Alert banner
    if frame_violations > 0:
        cv2.putText(frame, "SAFETY VIOLATION DETECTED",
                    (width // 2 - 200, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(frame, timestamp,
                (width - 250, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # =====================================

    out.write(frame)
    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()