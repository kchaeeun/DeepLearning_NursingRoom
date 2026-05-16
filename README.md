# 영상분석 기반 임산부 인증 수유실 자동 출입 관리 시스템
YOLOv5 모델을 사용한 아기, 키즈, 성인, 유아차 객체 인식 및 MediaPipe를 사용한 자세 검출 후 임베디드 시스템을 이용한 자동 출입 시스템 구현

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv5](https://img.shields.io/badge/YOLOv5-x-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)
![ONNX](https://img.shields.io/badge/ONNX-Acceleration-black)

## 📌 프로젝트 소개

**Birth**는 임산부와 아기 동반 보호자가 보다 안전하고 편리하게 수유실을 이용할 수 있도록 설계한
딥러닝 기반 수유실 자동 출입 관리 시스템입니다.

기존 수유실은 무단 출입 문제와 역무원의 수동 출입 관리로 인해 불편함이 존재했습니다.
본 프로젝트는 이를 해결하기 위해 **YOLOv5 객체 인식**, **MediaPipe 자세 검출**, **거리 기반 필터링**을 결합하여
수유 대상자를 자동으로 인식하고 출입문을 제어하는 시스템을 구현했습니다.

- 논문 게재: 한국정보통신학회논문지 Vol.28 No.11 (2024)
- 논문명: **영상분석 기반 임산부 인증 수유실 자동 출입 관리**
- DOI: 10.6109/jkiice.2024.28.11.1379

---

## 🎯 프로젝트 목표

- 임산부 및 아기 동반 보호자의 수유실 접근성 향상
- 외부인 무단 출입 방지를 통한 보안 강화
- 역무원의 수동 출입 관리 부담 감소
- 실시간 객체 인식 기반 자동 출입 시스템 구현
- Raspberry Pi 환경에서도 동작 가능한 경량 AI 시스템 구축

---

## 🧠 주요 기술 스택

| 분야 | 기술 |
|---|---|
| AI / Vision | YOLOv5-x, MediaPipe Pose |
| Computer Vision | OpenCV |
| Edge Device | Raspberry Pi 4 |
| Hardware | Arduino UNO, Step Motor, Webcam(Logitech C270) |
| Model Optimization | ONNX |
| Dataset | Roboflow |
| Development | Python, Google Colab, VSCode |

---

## 🏗 시스템 구조

```text
웹캠 입력
   ↓
YOLOv5 객체 인식
   ├── baby / stroller 검출 → 자동문 개방
   └── adult 검출
            ↓
     MediaPipe 자세 분석
            ├── 아기 안기 / 업기 자세 검출 → 자동문 개방
            └── 일반 사용자 → 출입 제한
```

### 시스템 구성 요소

- Raspberry Pi 4
- Arduino UNO
- Logitech C270 Webcam
- Step Motor 기반 자동문
- LED 전광판

---

## 📂 데이터셋 구성

### 클래스 구성

| Class | Count |
|---|---:|
| adult | 1,154 |
| kids | 988 |
| baby | 943 |
| stroller | 469 |

- 총 데이터 수: 3,896장
- train: 3,675
- valid: 114
- test: 107

### 데이터셋 구축 과정

1. Roboflow 기반 이미지 수집
2. Bounding Box 라벨링
3. 클래스 불균형 해결을 위한 웹 크롤링 데이터 추가
4. 데이터 증강 진행

### 적용한 데이터 증강

- 이미지 크기: `640x640`
- 회전: `±12°`
- 노이즈 픽셀 추가
- 그레이 스케일 처리

---

## 🔍 YOLOv5 모델 선택 이유

| Version | 특징 |
|---|---|
| YOLOv5 | 경량화 모델 지원 및 빠른 추론 속도 |
| YOLOv6 | 포즈 추정 기능 강화 |
| YOLOv7 | YOLOv4 기반 성능 확장 |
| YOLOv8 | Detection/Tracking 고도화 |
| YOLOv9 | PGI, GELAN 도입 |

본 프로젝트는 **실시간 처리 속도**와 **라즈베리파이 환경에서의 동작 가능성**이 중요했기 때문에,
성능과 속도의 균형이 우수한 **YOLOv5**를 선택했습니다.

---

## ⚙ 모델 학습

### 학습 환경

- GPU: NVIDIA RTX 3060
- CUDA: 11.8
- 학습 환경: Google Colab

### 초기 하이퍼파라미터

| img-size | batch | epoch | lr0 | lrf |
|---|---:|---:|---:|---:|
| 640x640 | 16 | 30 | 0.01 | 0.1 |

### 학습 명령어

```bash
python train.py \
  --batch 16 \
  --epochs 30 \
  --data dataset/data.yaml \
  --hyp data/hyps/hyp_evolve.yaml \
  --name birth_test_xl_5_3
```

---

## 📊 모델 성능 비교

### YOLOv5 모델별 성능

| Model | Precision | Recall | mAP@0.5 |
|---|---:|---:|---:|
| YOLOv5-s | 66.6 | 67.2 | 69.6 |
| YOLOv5-m | 85.7 | 79.8 | 87.0 |
| YOLOv5-l | 86.6 | 79.6 | 87.2 |
| YOLOv5-x | 87.8 | 85.6 | 90.4 |

### 최종 학습 결과

| Num | lr0 | lrf | Precision | Recall | mAP@0.5 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.01 | 0.1 | 87.8 | 85.6 | 90.4 |
| 2 | 0.1 | 0.1 | 68.6 | 79.6 | 87.2 |
| 3 | 0.001 | 0.01 | 85.9 | 87.5 | **91.9** |

최종적으로 Precision과 Recall의 균형이 가장 우수하고,
mAP 성능이 가장 높은 **3번 모델(YOLOv5-x)** 을 최종 모델로 선정했습니다.

---

## 🧍 거리 기반 객체 필터링

지하철 역사 환경 특성상 지나가는 보행자를 잘못 인식하는 문제를 줄이기 위해
객체와 카메라 사이의 거리를 계산하는 로직을 추가했습니다.

### 적용 목적

- 원거리 성인을 아기로 잘못 인식하는 문제 해결
- 수유실 앞 사용자만 분석 대상으로 제한
- 불필요한 연산 감소

### 결과

- 거리 측정 오차: ±2.33cm
- 키 추정 오차: ±2.35cm

이를 통해 2m 이상 거리의 객체를 필터링하고,
140cm 이상의 객체를 성인으로 분류하도록 개선했습니다.

---

## 🤱 MediaPipe 자세 검출

YOLO 객체 인식만으로는 아기 검출 정확도에 한계가 존재했기 때문에,
MediaPipe Pose를 활용해 다음 두 가지 자세를 추가적으로 분석했습니다.

### 자세 인식 종류

- 아기 안기 자세
- 아기 업기 자세

### 인식 정확도

| 자세 | Detection Rate |
|---|---:|
| 안기 자세 | 92.9% |
| 업기 자세 | 98.4% |

이를 통해 객체 인식만 사용했을 때보다 더 안정적인 출입 인증이 가능해졌습니다.

---

## 🚀 모델 경량화 및 최적화

### 적용 기술

#### TensorFlow Lite (.tflite)
- 모바일 및 Edge 환경 최적화
- 경량화 목적

#### ONNX (.onnx)
- 추론 속도 향상
- Raspberry Pi 환경 최적화

### 결과

- 경량화보다 ONNX 기반 가속화 효과가 더 우수
- 실시간 추론 성능 향상 확인

---

## 💡 프로젝트 성과

### 논문 게재

- 한국정보통신학회논문지 게재
- Vol.28 No.11, 2024

### 사용자 반응

덕성여대 재학생 70명을 대상으로 진행한 설문 결과:

> “실제로 수유실 자동 출입 시스템이 도입된다면 도움이 될 것인가?”

- 매우 그렇다: 48명
- 그렇다: 18명
- 긍정 응답 비율: **94.28%**

---

## 📈 기대 효과

- 임산부 및 보호자의 수유실 이용 편의 향상
- 공공시설 보안 강화
- 역무원 업무 효율 증가
- 사회적 약자를 위한 AI 접근성 확대
- 공항·쇼핑몰·공공기관 등으로 확장 가능

---

## 📄 논문 정보

### 논문 제목
영상분석 기반 임산부 인증 수유실 자동 출입 관리

### DOI
https://doi.org/10.6109/jkiice.2024.28.11.1379

---

## 🔗 Reference

- YOLOv5 Official Repository
  - https://github.com/ultralytics/yolov5

- YOLOv5 Official Docs
  - https://docs.ultralytics.com/ko/models/yolov5/

- MediaPipe
  - https://ai.google.dev/edge/mediapipe/solutions/guide


![image](https://github.com/kchaeeun/DeepLearning_NursingRoom/assets/102590823/095933e0-7bf7-4cfa-9bc8-272c5f6c3f6c)
