import cv2
import numpy as np
from ultralytics import YOLO

class MarshallerAI:
    def __init__(self):
        self.model = YOLO('yolov8n-pose.pt') 
        self.status = "IDLE"

    def calculate_angle(self, a, b, c):
        """세 점 사이의 각도 계산"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle > 180.0: angle = 360-angle
        return angle

    def draw_custom_skeleton(self, frame, kpts):
        """상반신 커스텀 시각화"""
        connections = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
        line_color = (0, 255, 0)
        joint_color = (0, 0, 255)
        
        for start_idx, end_idx in connections:
            if kpts[start_idx][2] > 0.5 and kpts[end_idx][2] > 0.5:
                x1, y1 = int(kpts[start_idx][0]), int(kpts[start_idx][1])
                x2, y2 = int(kpts[end_idx][0]), int(kpts[end_idx][1])
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 3)

        for idx in [0, 5, 6, 7, 8, 9, 10]:
             if kpts[idx][2] > 0.5:
                cx, cy = int(kpts[idx][0]), int(kpts[idx][1])
                cv2.circle(frame, (cx, cy), 6, joint_color, -1)

    def detect_gesture(self, frame):
        h, w, _ = frame.shape
        results = self.model(frame, verbose=False, conf=0.5)
        
        current_action = "IDLE"
        info_text = ""
        
        has_person = False
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            kpts_raw = results[0].keypoints.data[0].cpu().numpy()
            if kpts_raw[5][2] > 0.5 and kpts_raw[6][2] > 0.5:
                has_person = True

        if not has_person:
            self.status = "IDLE"
            box_w, box_h = 280, 80
            x1, y1 = w - box_w, h - box_h
            cv2.rectangle(frame, (x1, y1), (w, h), (100, 100, 100), -1)
            cv2.putText(frame, "IDLE", (x1+10, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2, cv2.LINE_AA)
            return "IDLE", frame

        # -------------------------------------------------------------
        # 로직 수행
        # -------------------------------------------------------------
        kpts_raw = results[0].keypoints.data[0].cpu().numpy()
        def get_norm(idx): return [kpts_raw[idx][0]/w, kpts_raw[idx][1]/h]

        nose = get_norm(0) if kpts_raw[0][2] > 0.5 else None

        l_sh, r_sh = get_norm(5), get_norm(6)   # 어깨
        l_el, r_el = get_norm(7), get_norm(8)   # 팔꿈치
        l_wr, r_wr = get_norm(9), get_norm(10)  # 손목

        angle_l = self.calculate_angle(l_sh, l_el, l_wr)
        angle_r = self.calculate_angle(r_sh, r_el, r_wr)
        
        wrist_dist_x = abs(l_wr[0] - r_wr[0])
        shoulder_width = abs(l_sh[0] - r_sh[0])

        current_action = "READY"

        # [1] STOP: X자 교차 OR 고공 초밀착 (최우선)
        is_crossed = r_wr[0] > l_wr[0] 
        is_high_touching = (wrist_dist_x < 0.05) and (l_wr[1] < l_sh[1] - 0.1)

        if (is_crossed or is_high_touching) and (l_wr[1] < l_sh[1] + 0.4):
            current_action = "STOP"

        # =============================================================
        # [2] ENGINE_CUT (목 긋기) - 가로 범위(Box) 대폭 확장
        # =============================================================
        else:
            # 1. 세로 범위 (Y축): 코 ~ 겨드랑이 위 (여유 있게)
            neck_top = nose[1] if nose else (l_sh[1] - 0.2)
            neck_bottom = l_sh[1] + 0.2

            # 2. 가로 범위 (X축): 어깨 바깥쪽까지 길게(Longer) 확장
            # 어깨 너비의 절반만큼 양쪽으로 더 늘림 (총 2배 넓이)
            margin = shoulder_width * 0.5 
            
            # r_sh[0]는 화면상 왼쪽(작은값), l_sh[0]는 화면상 오른쪽(큰값)
            box_left_limit = r_sh[0] - margin # 화면상 왼쪽 끝 한계
            box_right_limit = l_sh[0] + margin # 화면상 오른쪽 끝 한계

            # 3. 왼손 체크: 높이는 목, 좌우는 확장된 박스 안
            l_in_throat = (neck_top < l_wr[1] < neck_bottom) and \
                          (box_left_limit < l_wr[0] < box_right_limit)

            # 4. 오른손 체크
            r_in_throat = (neck_top < r_wr[1] < neck_bottom) and \
                          (box_left_limit < r_wr[0] < box_right_limit)

            # 5. 반대 손은 내려가 있어야 함 (Set Brakes 방지)
            r_is_down = r_wr[1] > r_sh[1] + 0.2
            l_is_down = l_wr[1] > l_sh[1] + 0.2

            if (l_in_throat and r_is_down) or (r_in_throat and l_is_down):
                 current_action = "ENGINE_CUT"
            
            # [3] SET_BRAKES
            elif (l_wr[1] < l_sh[1] and r_wr[1] > r_sh[1] + 0.2) or \
                 (r_wr[1] < r_sh[1] and l_wr[1] > l_sh[1] + 0.2):
                current_action = "SET_BRAKES"

            # [4] FORWARD
            elif abs(l_el[1] - l_sh[1]) < 0.2 and abs(r_el[1] - r_sh[1]) < 0.2 and \
                 l_wr[1] < l_el[1] and r_wr[1] < r_el[1] and \
                 angle_l < 125 and angle_r < 125:
                current_action = "FORWARD"

            # [5] APPROACHING
            elif angle_l > 130 and angle_r > 130:
                if l_wr[1] > l_sh[1] + 0.25 and r_wr[1] > r_sh[1] + 0.25:
                    if wrist_dist_x > shoulder_width * 1.5:
                        current_action = "APPROACHING"
                        info_text = "SPEED: FAST"
                    else:
                        current_action = "READY"
                elif l_wr[1] < l_sh[1] + 0.25 and r_wr[1] < r_sh[1] + 0.25:
                    if wrist_dist_x < shoulder_width * 1.0:
                         current_action = "APPROACHING"
                         info_text = "SPEED: VERY SLOW"
                    elif wrist_dist_x < shoulder_width * 1.6:
                         current_action = "APPROACHING"
                         info_text = "SPEED: SLOW"
                    else:
                        current_action = "APPROACHING"
                        info_text = "SPEED: NORMAL"

        self.draw_custom_skeleton(frame, kpts_raw)
        self.status = current_action
        
        box_w, box_h = 280, 80
        x1, y1 = w - box_w, h - box_h
        cv2.rectangle(frame, (x1, y1), (w, h), (245, 117, 16), -1)
        cv2.putText(frame, current_action, (x1+10, y1+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        if info_text:
            cv2.putText(frame, info_text, (x1+10, y1+65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

        return current_action, frame