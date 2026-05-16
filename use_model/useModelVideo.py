import torch
import cv2
import time
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# 모델 로딩
model = torch.hub.load('ultralytics/yolov5', 'custom', path='xlarge_best.onnx', force_reload=True)
model.eval()

input_file = 'video4.mp4'
output_file = 'video4_yolov5x_best_onnx.mp4'

cap = cv2.VideoCapture(input_file)

# Set frame rate to 30 frames per second
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
resolution = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter(output_file, fourcc, 24.0, resolution)

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 화면에 표시
    results = model(cv2.resize(frame, (640, 640)))

    for volume in results.xyxy[0]:
        xyxy = volume.numpy()
        xyxy[0] = xyxy[0] / 640 * resolution[0]
        xyxy[2] = xyxy[2] / 640 * resolution[0]
        xyxy[1] = xyxy[1] / 640 * resolution[1]
        xyxy[3] = xyxy[3] / 640 * resolution[1]
        class_index = int(xyxy[5])
        class_label = results.names[class_index]
        print("Detected Class:", class_label)

        # 클래스에 따라 다른 색상 선택
        if class_label == 'adult':
            color = (148, 0, 211)  # 보라색
        elif class_label == 'baby':
            color = (0, 0, 255)     # 파란색
        elif class_label == 'kids':
            color = (0, 255, 0)     # 초록색
        elif class_label == 'stroller':
            color = (0, 255, 255)   # 노란색
        else:
            color = (0, 255, 0)     # 기본적으로 초록색 사용

        # 바운딩 박스 좌표 추출
        x_min, y_min, x_max, y_max = map(int, xyxy[:4])

        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                        color=color, thickness=2)
        
        # 클래스 레이블 출력 위치 및 색상 조정
        text_offset_x = 10
        text_offset_y = 25
        cv2.putText(frame, class_label, (x_min + text_offset_x, y_min + text_offset_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)


    dt = (time.time() - start) * 1000
    cv2.putText(frame, f'Inference : {dt:.1f}ms', (0, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 0, 255), thickness=2)
    # 결과를 출력 동영상 파일에 저장합니다.
    out.write(frame)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(f'{dt}')

cap.release()
out.release()
cv2.destroyAllWindows()
