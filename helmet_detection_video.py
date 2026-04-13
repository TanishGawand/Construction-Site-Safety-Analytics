import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("video.mp4")

width = 960
height = 540

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output1.mp4", fourcc, fps, (width, height))

total_violations = 0

cooldown_frames = 20
cooldown_counter = 0

helmet_histories = {}
history_size = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    frame_violations = 0

    results = model(frame)

    current_people = []

    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                person_id = (x1 // 50, y1 // 50)
                current_people.append(person_id)

                if person_id not in helmet_histories:
                    helmet_histories[person_id] = []

                # Head region
                head_y2 = y1 + int((y2 - y1) * 0.15)
                x_margin = int((x2 - x1) * 0.2)

                hx1 = x1 + x_margin
                hx2 = x2 - x_margin

                head = frame[y1:head_y2, hx1:hx2]
                if head.size == 0:
                    continue

                hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)

                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([35, 255, 255])

                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                h, w = mask.shape
                mask[h//2:, :] = 0

                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

                yellow_pixels = cv2.countNonZero(mask)
                total_pixels = head.shape[0] * head.shape[1]

                if total_pixels == 0:
                    continue

                yellow_ratio = yellow_pixels / total_pixels
                current_detection = (yellow_ratio > 0.12) and (yellow_pixels > 500)

                helmet_histories[person_id].append(current_detection)

                if len(helmet_histories[person_id]) > history_size:
                    helmet_histories[person_id].pop(0)

                history = helmet_histories[person_id]
                helmet_detected = history.count(True) >= int(0.8 * len(history))

                # 🔥 FINAL STATUS
                if helmet_detected:
                    text = "SAFE"
                    color = (0,255,0)
                else:
                    text = "NO HELMET"
                    color = (0,0,255)

                    if cooldown_counter == 0:
                        frame_violations += 1
                        total_violations += 1
                        cooldown_counter = cooldown_frames

                # 🔥 DRAW BOX
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Cleanup old IDs
    helmet_histories = {k: v for k, v in helmet_histories.items() if k in current_people}

    if cooldown_counter > 0:
        cooldown_counter -= 1

    # ================= UI PART =================

    # 🔷 TOP BAR
    cv2.rectangle(frame, (0, 0), (width, 40), (30, 30, 30), -1)
    cv2.putText(frame, "Construction Safety Monitoring System",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # 🔷 STATUS PANEL
    cv2.rectangle(frame, (10, 50), (300, 140), (40, 40, 40), -1)

    cv2.putText(frame, "SYSTEM STATUS",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    cv2.putText(frame, f"Current: {frame_violations}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)

    cv2.putText(frame, f"Total: {total_violations}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)

    # 🔷 TIMESTAMP
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(frame, timestamp,
                (width - 250, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    # ==================================================

    out.write(frame)
    cv2.imshow("CCTV Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()