import cv2
import numpy as np
import math

class DockingAI:
    def __init__(self):
        # ---------------------------------------------------------
        # [1] 사용자 설정
        self.MARKER_SIZE = 3.4  # 마커 크기 (cm)
        self.target_dict = cv2.aruco.DICT_6X6_250 
        # ---------------------------------------------------------

        # ArUco 설정
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.target_dict)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
        
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.width = 640
        self.height = 480
        
        # [2] 캘리브레이션 결과 (오차율 0.06 적용)
        self.camera_matrix = np.array([
            [872.23558, 0.00000, 315.00614],
            [0.00000, 873.47815, 240.01070],
            [0.00000, 0.00000, 1.00000]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([
            [0.14923, -1.11676, 0.00511, 0.00329, 7.40075]
        ], dtype=np.float32)

        # 3D 좌표 기준점
        ms = self.MARKER_SIZE / 2
        self.obj_points = np.array([
            [-ms, ms, 0],
            [ms, ms, 0],
            [ms, -ms, 0],
            [-ms, -ms, 0]
        ], dtype=np.float32)

        self.smooth_data = {}
        self.ALPHA = 0.3 

    def euler_from_quaternion(self, rvec):
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(rmat[2,1] , rmat[2,2])
            y = math.atan2(-rmat[2,0], sy)
            z = math.atan2(rmat[1,0], rmat[0,0])
        else:
            x = math.atan2(-rmat[1,2], rmat[1,1])
            y = math.atan2(-rmat[2,0], sy)
            z = 0

        pitch = x * 180.0 / math.pi
        yaw   = y * 180.0 / math.pi
        roll  = z * 180.0 / math.pi
        return roll, pitch, yaw

    def process(self, frame):
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        data = {
            "found": False, "id": -1, "dist_cm": 0.0,
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            "center": (0, 0)
        }

        min_distance = float('inf')
        best_marker_idx = -1
        best_rvec, best_tvec = None, None

        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]

                # [수정됨] 오직 ID 11번만 찾도록 고정
                if marker_id == 11: 
                    target_corners = corners[i][0]
                    
                    success, rvec, tvec = cv2.solvePnP(
                        self.obj_points, target_corners, 
                        self.camera_matrix, self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE 
                    )
                    
                    if success:
                        if marker_id in self.smooth_data:
                            prev = self.smooth_data[marker_id]
                            rvec = self.ALPHA * rvec + (1 - self.ALPHA) * prev['rvec']
                            tvec = self.ALPHA * tvec + (1 - self.ALPHA) * prev['tvec']
                        self.smooth_data[marker_id] = {'rvec': rvec, 'tvec': tvec}

                        dist = math.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)

                        # 혹시 11번이 여러 개일 경우 가장 가까운 것
                        if dist < min_distance:
                            min_distance = dist
                            best_marker_idx = i
                            best_rvec = rvec
                            best_tvec = tvec
                            data["id"] = int(marker_id)

            if best_marker_idx != -1:
                data["found"] = True
                data["dist_cm"] = min_distance
                
                roll, pitch, yaw = self.euler_from_quaternion(best_rvec)
                data["roll"] = roll
                data["pitch"] = pitch
                data["yaw"] = yaw 

                target_corners = corners[best_marker_idx][0]
                cx = int(target_corners[:, 0].mean())
                cy = int(target_corners[:, 1].mean())
                data["center"] = (cx, cy)

                # --- 시각화 ---
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, best_rvec, best_tvec, self.MARKER_SIZE)

                # --- UI 정보 박스 ---
                target_dist = 12.0
                remain_dist = min_distance - target_dist 

                box_width, box_height = 280, 160
                box_x = w - box_width - 20
                box_y = h - box_height - 20

                overlay = frame.copy()
                cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                dist_color = (0, 255, 0) if abs(remain_dist) < 2.0 else (0, 255, 255)
                yaw_color = (0, 255, 0) if abs(yaw) < 5.0 else (0, 255, 255)

                cv2.putText(frame, f"TARGET ID : {data['id']}", 
                           (box_x + 10, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.putText(frame, f"Dist   : {min_distance:.1f} cm (To: {remain_dist:.1f})", 
                           (box_x + 10, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)
                
                cv2.putText(frame, f"Yaw(Y) : {yaw:.1f} deg", 
                           (box_x + 10, box_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yaw_color, 2)
                cv2.putText(frame, f"Pit(X) : {pitch:.1f} deg", 
                           (box_x + 10, box_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, f"Rol(Z) : {roll:.1f} deg", 
                           (box_x + 10, box_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return data, frame