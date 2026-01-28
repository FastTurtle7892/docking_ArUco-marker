import cv2
import numpy as np
import glob

# ==========================================
# [최종 설정] 사각형 9x6개 기준
# ------------------------------------------
# OpenCV는 교차점 수를 세므로 1씩 뺍니다.
CHECKERBOARD = (8, 5) 

# 한 칸의 실제 크기 (3cm = 30.0mm)
SQUARE_SIZE = 30.0 
# ==========================================

# 3D 점 좌표 생성
# (8 * 5)개의 점을 준비
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

objpoints = [] # 3D points
imgpoints = [] # 2D points

images = glob.glob('calib_imgs/*.jpg')

print(f"총 {len(images)}장의 이미지를 분석합니다... (설정: 8x5 교차점, 30mm)")

success_count = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        print(f"✅ 성공: {fname}")
        objpoints.append(objp)
        
        # 코너 정밀 보정
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        success_count += 1
    else:
        # 혹시 종이가 세로로 찍혔을 수도 있으니 (5, 8)로도 한번 시도해봄
        ret_rev, corners_rev = cv2.findChessboardCorners(gray, (CHECKERBOARD[1], CHECKERBOARD[0]), None)
        if ret_rev == True:
            print(f"✅ 성공 (회전됨): {fname}")
            # 회전된 패턴에 맞게 objp 재생성 필요하지만, 보통 가로세로 돌려가며 찍으므로 
            # 일단 주 패턴(8,5)만 통과시켜도 결과는 나옵니다.
            # 여기서는 엄격하게 (8,5)만 수집합니다.
        else:
            print(f"❌ 실패: {fname}")

if success_count > 0:
    print(f"\n🎉 {success_count}장의 사진으로 계산을 시작합니다!")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n\n======== [이걸 복사해서 DockingAI에 붙여넣으세요] ========")
    print("1. Camera Matrix (self.camera_matrix):")
    print("-" * 30)
    print(f"np.array([\n    [{mtx[0][0]:.5f}, {mtx[0][1]:.5f}, {mtx[0][2]:.5f}],\n    [{mtx[1][0]:.5f}, {mtx[1][1]:.5f}, {mtx[1][2]:.5f}],\n    [{mtx[2][0]:.5f}, {mtx[2][1]:.5f}, {mtx[2][2]:.5f}]\n], dtype=np.float32)")
    print("-" * 30)

    print("\n2. Distortion Coeffs (self.dist_coeffs):")
    print("-" * 30)
    print(f"np.array([\n    [{dist[0][0]:.5f}, {dist[0][1]:.5f}, {dist[0][2]:.5f}, {dist[0][3]:.5f}, {dist[0][4]:.5f}]\n], dtype=np.float32)")
    print("-" * 30)
    
    # 오차율 확인
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print(f"\n평균 오차(Error): {mean_error/len(objpoints):.5f}")

else:
    print("\n🚨 여전히 실패한다면 다음을 확인하세요:")
    print("1. 사진에 체스보드 테두리 여백(흰색 공간)이 충분히 있나요?")
    print("2. 체스보드가 너무 멀리 있거나 흐릿하지 않나요?")