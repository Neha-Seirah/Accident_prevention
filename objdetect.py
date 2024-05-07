from ultralytics import YOLO
import cv2
import math
import pygame

# Initialize Pygame mixer
pygame.mixer.init()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Model
model = YOLO("yolo-Weights/yolov8n.pt")

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Function to calculate distance from camera
def calculate_distance(rectangle_width_pixels):
    focal_length = 450  # Focal length of the camera in pixels
    object_width_cm = 4  # Width of the object in centimeters
    distance = (object_width_cm * focal_length) / rectangle_width_pixels
    return distance

# Alarm sound
alarm_sound_path = "alert.wav"

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    alarm_triggered = False

    # Coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculate distance
            object_width_pixels = x2 - x1
            distance = calculate_distance(object_width_pixels)

            # If an object is too close, trigger the alarm
            if distance < 10 and not alarm_triggered:
                alarm_sound = pygame.mixer.Sound(alarm_sound_path)
                alarm_sound.play()
                alarm_triggered = True

            # Display object name and distance on the frame
            cls = int(box.cls[0])
            class_name = classNames[cls]
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 0, 0)
            thickness = 2
            text = f"{class_name}: {distance:.2f} cm"
            cv2.putText(img, text, org, font, font_scale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
