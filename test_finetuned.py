"""
파인튜닝된 OpenVLA 모델 추론 테스트
- 베이스 모델 + LoRA 어댑터 로드
- 수집한 데이터셋의 첫 이미지로 추론 실행
- 정답 액션과 비교

사용법:
    conda activate openvla
    cd /media/kimjimin/02B092A4B0929E2B/openvla
    python test_finetuned.py
"""

import os
import json
import glob
import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import PeftModel

# =========================
# 설정
# =========================
DATASET_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_dataset/pick_up_cup"
CHECKPOINT_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_checkpoints/final_model"
MODEL_ID = "openvla/openvla-7b"

print("=" * 50)
print("  파인튜닝된 OpenVLA 추론 테스트")
print("=" * 50)

# =========================
# 1. 모델 로드 (베이스 + LoRA)
# =========================
print("\n프로세서 로딩 중...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

print("베이스 모델 로딩 중 (4-bit 양자화)...")
qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=qlora_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
)

print(f"LoRA 어댑터 로딩 중: {CHECKPOINT_DIR}")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.eval()
print("모델 로드 완료!\n")

# =========================
# 2. 테스트 데이터 로드
# =========================
episode_dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "episode_*")))
print(f"테스트할 에피소드: {len(episode_dirs)}개")

# 첫 번째 에피소드의 첫 이미지로 테스트
test_episode = episode_dirs[0]
meta_path = os.path.join(test_episode, "metadata.json")
with open(meta_path, 'r') as f:
    metadata = json.load(f)

language = metadata['language_instruction']
print(f"언어 명령: {language}\n")

# =========================
# 3. 여러 타임스텝에서 추론 비교
# =========================
print("타임스텝별 추론 결과:")
print("-" * 80)

test_steps = [0, 10, 20, 30]  # 여러 시점에서 테스트

for step_idx in test_steps:
    if step_idx >= metadata['num_steps']:
        continue

    # 이미지 로드
    img_path = os.path.join(test_episode, f"image_{step_idx:04d}.jpg")
    image = Image.open(img_path).convert("RGB")

    # 정답 액션
    gt_action = np.array(metadata['actions'][step_idx])

    # 추론
    prompt = f"In: What action should the robot take to {language}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        predicted_action = model.predict_action(
            **inputs,
            unnorm_key='bridge_orig',
            do_sample=False
        )

    # 결과 출력
    print(f"\n[Step {step_idx}]")
    print(f"  정답 액션: {gt_action}")
    print(f"  예측 액션: {predicted_action}")

    # 차이 계산
    diff = np.abs(predicted_action - gt_action)
    print(f"  오차 (절대값): {diff}")
    print(f"  평균 오차: {diff.mean():.6f}")

print("\n" + "=" * 50)
print("  추론 테스트 완료")
print("=" * 50)
print("\n해석:")
print("- 예측 액션이 정답과 비슷한 패턴(부호, 크기)을 보이면 학습 성공")
print("- 모두 0이거나 완전히 다른 값이면 학습 실패")
print("- joint1만 움직였으니 첫 번째 값이 가장 많이 변해야 함")
