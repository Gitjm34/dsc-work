import torch
import cv2
import time

# YOLOv5 모델 로드 (사전 학습된 모델 사용)
print("Loading YOLOv5 model for passenger detection...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded successfully.")

# 기본 타이머 및 시간 설정
base_timer = 5          # 문 기본 열림 시간 (초)
standing_time = 10      # 승객이 서 있을 때 추가 타이머 (초)
movement_time = 15      # 승객이 이동 중일 때 추가 타이머 (초)
exit_time = 5           # 승객이 완전히 내릴 때까지 기다리는 시간 (초)

# 승객 상태 변수
passenger_detected = False
passenger_moving = False
passenger_standing = False
passenger_seated = False

# 카메라 입력 처리
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# 타이머 동작 함수
def apply_timer(seconds):
    print(f"Door will remain open for {seconds} seconds.")
    time.sleep(seconds)
    print("Closing the bus door now.")

# 객체 인식 및 승객 상태에 따른 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break
    
    # 객체 감지
    results = model(frame)
    results.render()

    # 감지된 객체 목록 확인
    detected_classes = [result['name'] for result in results.xyxy[0]]

    # 승객 감지 여부 확인
    passenger_detected = 'person' in detected_classes

    # 승객의 상태 확인 (서 있는지, 앉아 있는지, 이동 중인지)
    if passenger_detected:
        # 승객이 문 근처에서 서 있거나 하차를 기다리고 있는 경우
        if 'standing' in detected_classes:
            print("Passenger is standing near the door.")
            passenger_standing = True
            apply_timer(standing_time)
        # 승객이 이동 중일 경우 (앉은 자리에서 일어나거나 이동 중)
        elif 'moving' in detected_classes:
            print("Passenger is moving toward the door.")
            passenger_moving = True
            apply_timer(movement_time)
        # 승객이 앉아 있을 경우 (문 근처로 오기 전 상태)
        elif 'seated' in detected_classes:
            print("Passenger is seated.")
            passenger_seated = True
            apply_timer(base_timer)
        else:
            # 승객이 더 이상 감지되지 않으면 문을 닫음
            print("Passenger has exited the bus.")
            apply_timer(exit_time)
    else:
        print("No passenger detected near the door.")
        apply_timer(base_timer)

    # 실시간 영상 출력
    cv2.imshow('Passenger Detection and Timer System', results.imgs[0])

    # 'q' 키를 누르면 프로그램 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
