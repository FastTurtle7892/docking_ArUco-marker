import cv2
import os

# 사진 저장할 폴더 생성
save_dir = 'calib_imgs'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)
# 해상도는 아루코 인식할 때와 똑같이 맞춰야 합니다!
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0
print("--- 카메라 캘리브레이션 촬영 ---")
print("'c' 키: 촬영 저장")
print("'q' 키: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Calibration Capture', frame)
    key = cv2.waitKey(1)

    if key == ord('c'):
        img_name = f"{save_dir}/img{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"📸 저장됨: {img_name}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()