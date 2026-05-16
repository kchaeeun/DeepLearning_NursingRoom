import torch
import cv2
import time
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='xlarge_best.onnx', force_reload=True)
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 corresponds to the first webcam, you may need to change this value if you have multiple webcams

# Set frame rate to 30 frames per second
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    start = time.time()
    
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to 640x640
    frame = cv2.resize(frame, (640, 640))

    # Perform object detection on the frame
    results = model(frame)

    # Draw bounding boxes and labels on the frame
    for obj in results.xyxy[0]:
        xyxy = obj.numpy()
        class_index = int(xyxy[5])
        class_label = results.names[class_index]
        print("Detected Class:", class_label)
        
        # 바운딩 박스의 좌측 상단 좌표
        x_min, y_min = int(xyxy[0]), int(xyxy[1])
        
        # 클래스 레이블 출력 위치 조정
        text_offset_x = 10
        text_offset_y = 25
        
        # 바운딩 박스에 클래스 레이블 출력
        cv2.putText(frame, class_label, (x_min + text_offset_x, y_min + text_offset_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                    color=(0, 255, 0), thickness=2)

    # Calculate inference time
    dt = (time.time() - start) * 1000
    cv2.putText(frame, f'Inference : {dt:.1f}ms', (0, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 0, 255), thickness=2)

    # Show the frame
    cv2.imshow('Webcam', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()