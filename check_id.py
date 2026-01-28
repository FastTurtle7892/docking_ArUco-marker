import cv2
import cv2.aruco as aruco
import numpy as np

def check_aruco_ids():
    # ---------------------------------------------------------
    # [설정] 마커 딕셔너리 변경 (8x8 격자 -> 6x6 데이터)
    # 테두리 포함 8칸이면 실제로는 6x6 마커입니다.
    # ---------------------------------------------------------
    try:
        # 6x6 딕셔너리 사용
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    except AttributeError:
        # 혹시 구버전이라 위 코드가 안되면 이걸로 시도
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    # 웹캠 실행
    cap = cv2.VideoCapture(0)
    
    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("📸 카메라가 켜졌습니다. (6x6 모드)")
    print("종료하려면 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 흑백 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 마커 검출
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 화면에 ID 출력
            id_list_str = f"IDs: {ids.flatten()}"
            
            # 텍스트 위치 계산 (첫 번째 마커 근처)
            cx = int(corners[0][0][:, 0].mean())
            cy = int(corners[0][0][:, 1].mean()) - 50
            
            # 마커 위에 큼지막하게 ID 띄우기
            cv2.putText(frame, id_list_str, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            print(f"감지됨! ID: {ids.flatten()}")

        cv2.imshow('ArUco ID Checker (6x6)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_aruco_ids()