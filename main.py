import cv2
import time
from gesture_ai import MarshallerAI
from docking_ai import DockingAI

# --- 설정 ---
CAMERA_ID = 0  # Jetson 연결된 카메라 (CSI는 gstreamer 문자열 필요할 수 있음)
STATE = "MARSHAL" # 초기 상태: MARSHAL or DOCKING

def main():
    global STATE
    
    # 1. 카메라 열기
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Camera Open Failed!")
        return
    
    # 해상도 설정 (속도 향상을 위해 적절히 조절)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2. AI 모듈 초기화
    marshal_ai = MarshallerAI() # YOLO 로드 시간 소요됨
    docking_ai = DockingAI()

    print("=== System Started ===")
    print("Press 'm' for MARSHAL Mode")
    print("Press 'd' for DOCKING Mode")
    print("Press 'q' to Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- 상태 머신 (State Machine) ---
        if STATE == "MARSHAL":
            # [모드 1] 제스처 인식
            cmd, debug_frame = marshal_ai.detect_gesture(frame)
            
            # 여기서 cmd를 로봇 제어부(Serial/ROS)로 전송
            # ex) serial.write(f"{cmd}\n".encode())
            
            # 화면 표시
            cv2.putText(debug_frame, "[MODE: MARSHAL]", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow("TowCar AI View", debug_frame)

        elif STATE == "DOCKING":
            # [모드 2] 도킹 (AprilTag)
            data, debug_frame = docking_ai.process(frame)
            
            if data["found"]:
                # P-Control 예시
                # steering = Kp * data["error_x"]
                # throttle = BaseSpeed if data["area"] < TARGET_AREA else 0
                pass
            
            cv2.putText(debug_frame, "[MODE: DOCKING]", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("TowCar AI View", debug_frame)

        # --- 키 입력 처리 ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            STATE = "MARSHAL"
            print("Switched to MARSHAL Mode")
        elif key == ord('d'):
            STATE = "DOCKING"
            print("Switched to DOCKING Mode")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()