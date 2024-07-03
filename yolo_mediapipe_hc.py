import os
import torch
import cv2
import time
import numpy as np
import math
import mediapipe as mp
import onnxruntime as ort
import serial

# 노트북 B에서 HC-06 모듈이 연결된 포트를 적절히 변경
bluetooth_serial_port_led = 'COM13'  # 윈도우에서는 'COMX' 형식으로 설정
bluetooth_serial_port_room = 'COM14'  # 윈도우에서는 'COMX' 형식으로 설정

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/4-1/capstone/xlarge_best.onnx', force_reload=True)
model.iou = 0.5

camera_angle_deg = 0

def connect_bluetooth(bluetooth_serial_port):
    try:
        ser = serial.Serial(bluetooth_serial_port, 9600)
        # print(f"{bluetooth_serial_port}에 연결되었습니다")
        return ser
    except serial.SerialException as e:
        print(f"{bluetooth_serial_port}에 연결 실패: {e}")
        return None

def send_command(ser, command):
    if ser:
        ser.write(command.encode())
        time.sleep(1)  # 아두이노가 응답할 시간을 줌
        while ser.in_waiting:
            response = ser.readline().decode('utf-8').strip()
            print(f"응답: {response}")

def open():
    ser = connect_bluetooth(bluetooth_serial_port_room)
    if ser:
        try:
            # 모터 회전을 위한 't' 명령어 전송
            send_command(ser, 't')
        except KeyboardInterrupt:
            print("사용자에 의해 프로그램 중단")
        finally:
            ser.close()
            # print("블루투스 연결 종료")
            
def led():
    ser = connect_bluetooth(bluetooth_serial_port_led)
    if ser:
        try:
            # 모터 회전을 위한 't' 명령어 전송
            send_command(ser, 'f')
        except KeyboardInterrupt:
            print("사용자에 의해 프로그램 중단")
        finally:
            ser.close()
            print("블루투스 연결 종료")          

def calculate_distance(object_width_mm, focal_length, object_width_in_frame):
    # Calculate distance using the formula
    # object_width_mm => 객체의 실제 너비, focal_length =>  카메라의 초점 거리, object_width_in_frame => 객체의 이미지에서의 너비
    distance = ((object_width_mm * focal_length) / (object_width_in_frame * math.cos(math.radians(camera_angle_deg)))) / 10
    return distance

def calculate_height(object_height_in_frame, camera_height_mm, image_height_px):
    actual_object_height_mm = (object_height_in_frame * camera_height_mm) / (image_height_px * math.cos(math.radians(camera_angle_deg)))
    return actual_object_height_mm

# 선분 각도를 계산하는 함수
def calculate_line_angle(point2, point3, point4):
    # 두 선분의 기울기 계산
    slope1 = math.atan2(point3[1] - point2[1], point3[0] - point2[0])
    slope2 = math.atan2(point4[1] - point3[1], point4[0] - point3[0])

    # 두 선분의 기울기 차이 계산
    angle_rad = abs(slope2 - slope1)

    # 라디안을 도(degree)로 변환하여 반환
    angle_deg = math.degrees(angle_rad)
    return 180 - angle_deg

# 웹캠 연결
cap = cv2.VideoCapture(1)
adult_detected = False

# Recommended values for measurements (you need to measure these values)
object_width_mm = 300  # Actual width of the object in millimeters
object_height_mm = 200
focal_length = 600  # Focal length of the camera in pixels

# 미디어 파이프로 포즈 추정
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        start = time.time()
        
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 크기 조정 (640x640)
        frame_resized = cv2.resize(frame, (640, 640))

        # YOLO 모델에 이미지 입력
        results = model(frame_resized)
        # results = model(frame)
        
        color_mapping = {
            'adult': (0, 0, 255),    # Red
            'kids': (0, 255, 0),     # Green
            'baby': (0, 255, 255),   # Yellow
            'stroller': (255, 0, 255)  # Pink
        }   

        # 이미지의 세로 해상도 가져오기
        image_height_px = frame.shape[0] 

        adult_detected = False
        # Process detections
        for obj in results.xyxy[0]:
            xyxy = obj.cpu().numpy()
            x_min, y_min, x_max, y_max = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            class_index = int(xyxy[5])
            class_label = results.names[class_index]
            
            object_height_in_frame = y_max - y_min

            # Calculate actual object height in millimeters
            actual_object_height_mm = calculate_height(object_height_in_frame, 200, image_height_px)


            # Calculate object width in the frame
            object_width_in_frame = x_max - x_min
            object_height_in_frame = y_max - y_min
            
            # Calculate distance using the function
            distance = calculate_distance(object_width_mm, focal_length, object_width_in_frame)

            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x_min, y_min - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=color_mapping[class_label], thickness=2)

            # Draw bounding box if distance is less than 200cm
            if distance < 200:      # 2m 이상인 경우 측정 x
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_mapping[class_label], 2)

                # Draw class label
                cv2.putText(frame, class_label, (x_min, y_min - 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                            color=color_mapping[class_label], thickness=2)

                # Show message if distance is less than 100cm
                if distance < 100:
                    if class_label == 'baby' or class_label == 'stroller':
                        print('아기가 인식되어 수유실이 열렸습니다.')
                        open()
                    else:
                        cv2.putText(frame, "Please go behind the line for height measurement.", (x_min, y_min - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        
                # else:
                    if class_label == 'adult' or actual_object_height_mm >= 140:
                        class_label = 'adult'
                        adult_detected = True
                        # print('dect 관절')
                        cv2.putText(frame, f"Tall: {actual_object_height_mm:.2f} cm", (x_min, y_min - 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color_mapping[class_label], thickness=2)
                        
        if adult_detected:
            image_height, image_width, _ = frame.shape
        
            # 오른팔과 왼팔의 각도를 계산하고 조건에 따라 행동을 취함
            # 미디어 파이프로 포즈 추정
            results = pose.process(frame)
            frame.flags.writeable = True
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

            # 포즈가 감지되지 않으면 다음 프레임으로 넘어감
            if not results.pose_landmarks:
                continue

            # 포즈 각 관절의 좌표를 추출하고 출력
            points = []
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * image_width)
                y = int(landmark.y * image_height)
                points.append((x, y))

            # 오른팔과 왼팔의 각도를 계산하고 조건에 따라 행동을 취함
            if None not in [points[12], points[14], points[16], points[11], points[13], points[15]]:
                right_line_angle = calculate_line_angle(points[12], points[14], points[16])
                left_line_angle = calculate_line_angle(points[11], points[13], points[15])

                if 50 < right_line_angle < 90 and 50 < left_line_angle < 90:
                    print("---------------------------------------------------------------------")
                    print("오른팔의 각도:", right_line_angle, ", 왼팔의 각도:", left_line_angle)
                    print("업는 자세가 확인되어 수유실 출입문이 열립니다.")
                    open()
                else:
                    print("---------------------------------------------------------------------")
                    print("아이를 안거나 업는 자세가 감지되지 않았습니다.")
                    led()

                    
            if None not in [points[12], points[14], points[16]]:
                if points[15] is not None:                                                                   # 왼쪽 손목의 관출 추출 시 안은 자세로 추정 => 후에 학습 진행
                    # 2번과 3번을 연결한 선과 3번과 4번을 연결한 선 사이의 각도 계산
                    hug_line_angle = calculate_line_angle(points[12], points[14], points[16])
                    if hug_line_angle < 75:
                        print("---------------------------------------------------------------------")
                        print("12번과 14번을 연결한 선과 14번과 16번을 연결한 선 사이의 각도:", hug_line_angle)
                        print("안는 자세가 확인되어 수유실 출입문이 열립니다.")
                        open()
                else:
                    print("---------------------------------------------------------------------")
                    print("아이를 안거나 업는 자세가 감지되지 않았습니다.")
                    led()
        else:
            # adult가 감지되지 않으면 MediaPipe 종료
            # print("Adult가 감지되지 않아 MediaPipe가 종료됩니다.")
            adult_detected = False
            pass

        # Calculate inference time
        dt = (time.time() - start) * 1000

        # Show the frame
        cv2.imshow('Webcam', frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
