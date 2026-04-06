"""
OpenVLA 파인튜닝용 시연 데이터 수집 스크립트
- D455 카메라 이미지 + Indy7 관절/TCP 데이터를 10Hz로 동시 수집
- Direct teaching으로 로봇을 움직이며 녹화
- 에피소드(시연 1회) 단위로 저장

사용법:
1. Indy7을 direct teaching 모드로 전환 (Conty 앱에서)
2. 이 스크립트 실행
3. 's' 키로 녹화 시작 → 로봇을 손으로 움직여 시연 → 's' 키로 녹화 종료
4. 반복 (30~50회 시연)
5. 'q' 키로 프로그램 종료
"""

import os
import time
import json
import cv2
import numpy as np
import pyrealsense2 as rs
from neuromeka import IndyDCP3

# =========================
# 설정
# =========================
ROBOT_IP = "192.168.1.2"        # Indy7 IP (본인 환경에 맞게 수정)
SAVE_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_dataset"  # 윈도우 드라이브에 저장
TASK_NAME = "pick_up_cup"       # 작업 이름 (작업마다 변경)
LANGUAGE_INSTRUCTION = "pick up the cup"  # VLA에 줄 언어 명령
RECORD_HZ = 10                  # 수집 주기 (10Hz)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# =========================
# 초기화
# =========================
print("로봇 연결 중...")
indy = IndyDCP3(ROBOT_IP, index=0)
print(f"로봇 연결 완료: {ROBOT_IP}")

print("카메라 초기화 중...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, 30)
pipeline.start(config)
print("카메라 초기화 완료")

# 저장 경로 생성
task_dir = os.path.join(SAVE_DIR, TASK_NAME)
os.makedirs(task_dir, exist_ok=True)

# 기존 에피소드 수 확인
existing = [d for d in os.listdir(task_dir) if d.startswith("episode_")]
episode_count = len(existing)
print(f"기존 에피소드 수: {episode_count}")

# =========================
# 수집 루프
# =========================
recording = False
episode_data = []

print("\n========================================")
print("  VLA 데이터 수집기")
print("========================================")
print(f"  작업: {TASK_NAME}")
print(f"  언어 명령: {LANGUAGE_INSTRUCTION}")
print(f"  수집 주기: {RECORD_HZ}Hz")
print("----------------------------------------")
print("  's' : 녹화 시작 / 종료")
print("  'q' : 프로그램 종료")
print("========================================\n")

cv2.namedWindow("VLA Data Collector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("VLA Data Collector", 960, 720)

try:
    while True:
        # 카메라 프레임 읽기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())

        # 화면 표시
        vis = image.copy()
        status = "RECORDING" if recording else "STANDBY"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(vis, f"[{status}] Episode: {episode_count}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(vis, f"Task: {LANGUAGE_INSTRUCTION}", (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if recording:
            cv2.putText(vis, f"Steps: {len(episode_data)}", (12, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # 녹화 중 빨간 원 표시
            cv2.circle(vis, (CAMERA_WIDTH - 30, 30), 15, (0, 0, 255), -1)

        cv2.putText(vis, "'s': start/stop | 'q': quit", (12, CAMERA_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("VLA Data Collector", vis)

        # 녹화 중이면 데이터 수집
        if recording:
            # 로봇 데이터 읽기
            robot_data = indy.get_robot_data()
            joint_positions = robot_data['q']      # 관절 각도 6개
            tcp_position = robot_data['p']         # TCP 위치 6개 [x,y,z,rx,ry,rz]

            # 타임스텝 데이터 저장
            step = {
                'timestamp': time.time(),
                'image': image.copy(),               # 640x480x3 numpy
                'joint_positions': list(joint_positions),  # [j1,j2,j3,j4,j5,j6]
                'tcp_position': list(tcp_position),        # [x,y,z,rx,ry,rz]
                'gripper_state': 0.0,                # 그리퍼 없으면 0 (나중에 수정)
            }
            episode_data.append(step)

        # 키 입력 처리
        key = cv2.waitKey(int(1000 / RECORD_HZ)) & 0xFF

        if key == ord('s'):
            if not recording:
                # 녹화 시작
                recording = True
                episode_data = []
                print(f"\n녹화 시작! (Episode {episode_count})")
            else:
                # 녹화 종료 → 저장
                recording = False
                if len(episode_data) < 5:
                    print(f"데이터가 너무 적음 ({len(episode_data)} steps). 저장 안 함.")
                    episode_data = []
                    continue

                # 에피소드 저장
                ep_dir = os.path.join(task_dir, f"episode_{episode_count:04d}")
                os.makedirs(ep_dir, exist_ok=True)

                # 이미지 저장
                for i, step in enumerate(episode_data):
                    img_path = os.path.join(ep_dir, f"image_{i:04d}.jpg")
                    cv2.imwrite(img_path, step['image'])

                # action 계산 (TCP 변화량)
                actions = []
                for i in range(len(episode_data) - 1):
                    curr_tcp = np.array(episode_data[i]['tcp_position'])
                    next_tcp = np.array(episode_data[i + 1]['tcp_position'])
                    delta_tcp = (next_tcp - curr_tcp).tolist()  # 6개 변화량
                    gripper = episode_data[i + 1]['gripper_state']
                    actions.append(delta_tcp + [gripper])  # 7차원
                # 마지막 스텝의 action은 0으로
                actions.append([0.0] * 7)

                # 수치 데이터 저장
                metadata = {
                    'language_instruction': LANGUAGE_INSTRUCTION,
                    'num_steps': len(episode_data),
                    'record_hz': RECORD_HZ,
                    'timestamps': [s['timestamp'] for s in episode_data],
                    'joint_positions': [s['joint_positions'] for s in episode_data],
                    'tcp_positions': [s['tcp_position'] for s in episode_data],
                    'gripper_states': [s['gripper_state'] for s in episode_data],
                    'actions': actions,
                }

                meta_path = os.path.join(ep_dir, "metadata.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                print(f"Episode {episode_count} 저장 완료! ({len(episode_data)} steps → {ep_dir})")
                episode_count += 1
                episode_data = []

        elif key == ord('q'):
            print("\n프로그램 종료.")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print(f"\n총 {episode_count}개 에피소드 수집 완료.")
print(f"저장 위치: {task_dir}")
