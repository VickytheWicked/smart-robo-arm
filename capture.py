import cv2
import os
import time

os.makedirs("eraser_dataset", exist_ok=True)

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam")
        break

    cv2.imshow("Eraser Dataset - SPACE to capture, Q to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = f"eraser_dataset/eraser_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()