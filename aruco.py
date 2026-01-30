import cv2
import numpy as np

# 6x6 딕셔너리 사용 (파일 이름 참고)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# 마커 ID 0번을 200x200 픽셀로 생성 예시
# 실제 인쇄 시 크기는 워드/한글에 넣고 위와 동일하게 조절하면 
# 원본 해상도가 깡패라 인식률이 훨씬 좋습니다.
img = cv2.aruco.generateImageMarker(aruco_dict, id=0, sidePixels=200)
cv2.imwrite("marker_id0.png", img)