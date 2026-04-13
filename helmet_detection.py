import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load image
image = cv2.imread("image.jpg")

# Run detection
results = model(image)

for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        # Only process persons
        if label == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw person bounding box (GREEN)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)

            # Define head region (top 25%)
            head_y2 = y1 + int((y2 - y1) * 0.25)

            # Crop head region
            head = image[y1:head_y2, x1:x2]

            # Avoid empty crop
            if head.size == 0:
                continue

            # Convert to HSV
            hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)

            # Yellow helmet detection range
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([35, 255, 255])

            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # Calculate ratio of yellow pixels
            yellow_ratio = cv2.countNonZero(mask) / (head.shape[0] * head.shape[1])

            # Decision threshold
            helmet_detected = yellow_ratio > 0.05

            # Display result
            if helmet_detected:
                text = "Helmet OK"
                color = (0,255,0)
            else:
                text = "NO HELMET!"
                color = (0,0,255)

            # Put label
            cv2.putText(image, text, (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Show output
cv2.imshow("Helmet Detection Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()