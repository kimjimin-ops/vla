import os
import cv2
import torch
import numpy as np
import pyrealsense2 as rs

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# =========================
# 사용자 설정
# =========================
OPENVLA_PATH = "openvla/openvla-7b"  # 로컬 모델 폴더를 직접 지정
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 예시 명령
INSTRUCTION = "move toward the red circle"

# 너무 자주 추론하지 않도록 수동 트리거만 사용
WINDOW_NAME = "D455 + OpenVLA Preview"

# =========================
# dtype 설정
# =========================
if torch.cuda.is_available():
    # 최신 GPU에서는 bfloat16이 될 수도 있지만,
    # Windows 환경에서는 float16이 더 무난한 경우가 많음
    TORCH_DTYPE = torch.float16
else:
    TORCH_DTYPE = torch.float32

# =========================
# OpenVLA 로드
# =========================
print(f"[INFO] Loading OpenVLA from: {OPENVLA_PATH}")
print(f"[INFO] Device: {DEVICE}, dtype: {TORCH_DTYPE}")

processor = AutoProcessor.from_pretrained(
    OPENVLA_PATH,
    trust_remote_code=True,
)

vla = AutoModelForVision2Seq.from_pretrained(
    OPENVLA_PATH,
    torch_dtype=TORCH_DTYPE if DEVICE.startswith("cuda") else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

if DEVICE.startswith("cuda"):
    vla = vla.to(DEVICE)

vla.eval()

# =========================
# RealSense 초기화
# =========================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

print("[INFO] D455 started.")
print("[INFO] Controls:")
print("  t : run OpenVLA inference on current frame")
print("  q : quit")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

def run_openvla_inference(bgr_frame: np.ndarray, instruction: str):
    """
    현재 프레임 1장을 OpenVLA에 넣고 action을 출력만 함.
    로봇 제어는 하지 않음.
    """
    # OpenCV BGR -> RGB -> PIL
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    with torch.no_grad():
        inputs = processor(prompt, pil_img)

        if DEVICE.startswith("cuda"):
            inputs = {
                k: (v.to(DEVICE, dtype=TORCH_DTYPE) if torch.is_floating_point(v) else v.to(DEVICE))
                for k, v in inputs.items()
            }

        # bridge_orig는 BridgeData V2 기준 action 역정규화 키
        action = vla.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False,
        )

    return action

try:
    # 카메라 warm-up
    for _ in range(10):
        pipeline.wait_for_frames()

    while True:
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        display = color.copy()

        # 화면 안내
        cv2.putText(
            display,
            f"Instruction: {INSTRUCTION}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            display,
            "Press 't' to infer, 'q' to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("t"):
            print("\n[INFO] Running OpenVLA inference...")
            try:
                action = run_openvla_inference(color, INSTRUCTION)
                print(f"[RESULT] instruction = {INSTRUCTION}")
                print(f"[RESULT] predicted action = {action}")
            except Exception as e:
                print(f"[ERROR] OpenVLA inference failed: {e}")

        elif key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] Closed.")