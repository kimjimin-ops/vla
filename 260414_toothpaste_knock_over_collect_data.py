import os
import time
import json
import cv2
import numpy as np
import pyrealsense2 as rs

from neuromeka import IndyDCP3

# =========================================================
# 설정
# =========================================================
ROBOT_IP = "192.168.1.6"
SAVE_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_dataset"
TASK_NAME = "toothpaste_knock_over"
LANGUAGE_INSTRUCTION = "knock over the standing toothpaste"

TARGET_EPISODES = 50
RECORD_HZ = 10

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# gripper는 이번 task에서 사용하지 않음
# action 7차원을 유지하기 위해 마지막 차원만 고정값 사용
FIXED_GRIPPER_VALUE = 0.0

MIN_STEPS_PER_EPISODE = 10
JPEG_QUALITY = 95


# =========================================================
# 유틸
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_robot_state(indy):
    """
    joint_positions (6개), tcp_position (6개) 반환.
    SDK 버전에 따라 get_control_data / get_robot_data 자동 선택.
    """
    if hasattr(indy, "get_control_data"):
        data = indy.get_control_data()
    elif hasattr(indy, "get_robot_data"):
        data = indy.get_robot_data()
    else:
        raise RuntimeError("IndyDCP3에 get_control_data / get_robot_data 없음")

    if not isinstance(data, dict):
        raise RuntimeError(f"로봇 상태 반환 형식 이상: {type(data)}")
    if "q" not in data or "p" not in data:
        raise RuntimeError(f"dict에 q/p 없음. keys={list(data.keys())}")

    q = list(data["q"])
    p = list(data["p"])

    if len(q) < 6 or len(p) < 6:
        raise RuntimeError(f"관절/TCP 길이 비정상: q={len(q)}, p={len(p)}")

    return q[:6], p[:6]


def compute_actions_fixed_gripper(tcp_positions, fixed_gripper_value=0.0):
    """
    action[i] = [Δx, Δy, Δz, Δrx, Δry, Δrz, fixed_gripper_value]
    마지막 step은 [0,0,0,0,0,0,fixed_gripper_value]
    """
    actions = []

    for i in range(len(tcp_positions) - 1):
        curr = np.array(tcp_positions[i], dtype=np.float32)
        nxt = np.array(tcp_positions[i + 1], dtype=np.float32)
        delta = (nxt - curr).tolist()
        actions.append(delta + [float(fixed_gripper_value)])

    actions.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(fixed_gripper_value)])
    return actions


def save_episode(
    ep_dir,
    episode_steps,
    language_instruction,
    record_hz,
    fixed_gripper_value=0.0,
):
    ensure_dir(ep_dir)

    image_files = []
    timestamps = []
    joint_positions = []
    tcp_positions = []

    for i, step in enumerate(episode_steps):
        img_name = f"image_{i:04d}.jpg"
        img_path = os.path.join(ep_dir, img_name)

        ok = cv2.imwrite(
            img_path,
            step["image"],
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError(f"이미지 저장 실패: {img_path}")

        image_files.append(img_name)
        timestamps.append(step["timestamp"])
        joint_positions.append(step["joint_positions"])
        tcp_positions.append(step["tcp_position"])

    gripper_states = [float(fixed_gripper_value)] * len(episode_steps)
    actions = compute_actions_fixed_gripper(tcp_positions, fixed_gripper_value)

    metadata = {
        "language_instruction": language_instruction,
        "num_steps": len(episode_steps),
        "record_hz": record_hz,
        "image_files": image_files,
        "timestamps": timestamps,
        "joint_positions": joint_positions,
        "tcp_positions": tcp_positions,
        "gripper_states": gripper_states,
        "actions": actions,
        "fixed_gripper_value": float(fixed_gripper_value),
        "task_type": "tcp_only_no_gripper",
    }

    with open(os.path.join(ep_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# =========================================================
# 메인
# =========================================================
def main():
    print("=" * 72)
    print("OpenVLA Data Collector - Toothpaste Knock Over")
    print("=" * 72)
    print(f"Robot IP             : {ROBOT_IP}")
    print(f"Save Dir             : {SAVE_DIR}")
    print(f"Task Name            : {TASK_NAME}")
    print(f"Instruction          : {LANGUAGE_INSTRUCTION}")
    print(f"Target Episodes      : {TARGET_EPISODES}")
    print(f"Record Hz            : {RECORD_HZ}")
    print(f"Fixed Gripper Value  : {FIXED_GRIPPER_VALUE}")
    print("=" * 72)

    task_dir = os.path.join(SAVE_DIR, TASK_NAME)
    ensure_dir(task_dir)

    existing = sorted([d for d in os.listdir(task_dir) if d.startswith("episode_")])
    episode_count = len(existing)
    print(f"\n[INFO] 기존 에피소드 수: {episode_count}")

    print("\n[1] 로봇 연결 중...")
    indy = IndyDCP3(ROBOT_IP, index=0)
    print("[OK] 로봇 연결 완료")

    print("\n[2] D455 초기화 중...")
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(
        rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT,
        rs.format.bgr8, CAMERA_FPS
    )
    pipeline.start(cfg)

    for _ in range(15):
        pipeline.wait_for_frames()

    print("[OK] D455 초기화 완료")

    recording = False
    episode_steps = []
    last_sample_ts = 0.0
    sample_period = 1.0 / RECORD_HZ

    window_name = "OpenVLA Toothpaste Knock Over Collector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 800)

    print("\n조작 키:")
    print("  s : 녹화 시작 / 종료")
    print("  q : 종료")
    print("\n권장 절차:")
    print("  1) direct teaching 모드로 로봇을 손으로 움직인다")
    print("  2) s로 녹화 시작")
    print("  3) 서 있는 치약 쪽으로 접근해서 넘어뜨리는 시연")
    print("  4) s로 녹화 종료 및 저장")
    print("  5) 총 50개 episode 수집")
    print()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            vis = image.copy()

            now = time.time()
            if recording and (now - last_sample_ts) >= sample_period:
                last_sample_ts = now
                try:
                    jpos, tcp = get_robot_state(indy)
                    episode_steps.append({
                        "timestamp": now,
                        "image": image.copy(),
                        "joint_positions": jpos,
                        "tcp_position": tcp,
                    })
                except Exception as e:
                    print(f"[WARN] 로봇 상태 읽기 실패: {e}")

            status_text = "RECORDING" if recording else "STANDBY"
            status_color = (0, 0, 255) if recording else (0, 255, 0)

            cv2.putText(
                vis, f"[{status_text}]",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2
            )
            cv2.putText(
                vis, f"Episode: {episode_count} / {TARGET_EPISODES}",
                (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
            cv2.putText(
                vis, f"Task: {LANGUAGE_INSTRUCTION}",
                (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            cv2.putText(
                vis, f"Gripper fixed value: {FIXED_GRIPPER_VALUE}",
                (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
            )
            cv2.putText(
                vis, "s:record  q:quit",
                (15, CAMERA_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            if recording:
                cv2.circle(vis, (CAMERA_WIDTH - 30, 30), 15, (0, 0, 255), -1)
                cv2.putText(
                    vis, f"Steps: {len(episode_steps)}",
                    (15, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

            cv2.imshow(window_name, vis)

            key = cv2.waitKey(30) & 0xFF

            if key == ord("s"):
                if not recording:
                    recording = True
                    episode_steps = []
                    last_sample_ts = 0.0
                    print(f"\n[INFO] 녹화 시작 — episode_{episode_count:04d}")
                else:
                    recording = False

                    if len(episode_steps) < MIN_STEPS_PER_EPISODE:
                        print(f"[WARN] step 수 부족 ({len(episode_steps)}개). 저장 안 함.")
                        episode_steps = []
                        continue

                    ep_dir = os.path.join(task_dir, f"episode_{episode_count:04d}")
                    try:
                        save_episode(
                            ep_dir,
                            episode_steps,
                            LANGUAGE_INSTRUCTION,
                            RECORD_HZ,
                            fixed_gripper_value=FIXED_GRIPPER_VALUE,
                        )
                        print(
                            f"[OK] episode_{episode_count:04d} 저장 완료 "
                            f"({len(episode_steps)} steps)"
                        )
                        episode_count += 1
                    except Exception as e:
                        print(f"[ERROR] 저장 실패: {e}")

                    episode_steps = []

                    if episode_count >= TARGET_EPISODES:
                        print(f"\n[INFO] 목표 {TARGET_EPISODES}개 수집 완료!")
                        break

            elif key == ord("q"):
                print("\n[INFO] 종료")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"\n[SUMMARY] 저장 위치     : {task_dir}")
    print(f"[SUMMARY] 완료 에피소드 : {episode_count}개")


if __name__ == "__main__":
    main()