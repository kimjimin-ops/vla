"""
파인튜닝된 OpenVLA 모델 추론 테스트 v2
- v2 학습 스크립트로 학습한 모델 테스트
- 액션 토큰을 다시 숫자로 변환

사용법:
    conda activate openvla
    cd /media/kimjimin/02B092A4B0929E2B/openvla
    python test_finetuned_v2.py
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
CHECKPOINT_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_checkpoints_v2/final_model"
STATS_PATH = "/media/kimjimin/02B092A4B0929E2B/vla_checkpoints_v2/action_stats.json"
MODEL_ID = "openvla/openvla-7b"
ACTION_DIM = 7

print("=" * 60)
print("  파인튜닝된 OpenVLA 추론 테스트 v2")
print("=" * 60)

# 액션 통계 로드
with open(STATS_PATH, 'r') as f:
    stats = json.load(f)
action_low = np.array(stats['action_low'])
action_high = np.array(stats['action_high'])
num_bins = stats['num_bins']

print(f"\n액션 통계 로드:")
print(f"  Low: {action_low}")
print(f"  High: {action_high}")
print(f"  Bins: {num_bins}")

# 모델 로드
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

vocab_size = processor.tokenizer.vocab_size
action_token_start = vocab_size - num_bins


def tokens_to_action(tokens):
    """토큰 ID를 액션 숫자로 역변환"""
    tokens = np.array(tokens)
    # 토큰 ID -> bin 인덱스 [0, num_bins-1]
    bins = tokens - action_token_start
    bins = np.clip(bins, 0, num_bins - 1)
    # bin -> [-1, 1]
    normalized = (bins / num_bins) * 2 - 1
    # [-1, 1] -> [low, high]
    action = (normalized + 1) / 2 * (action_high - action_low) + action_low
    return action


# =========================
# 추론 테스트
# =========================
episode_dirs = sorted(glob.glob(os.path.join(DATASET_DIR, "episode_*")))
test_episode = episode_dirs[0]
meta_path = os.path.join(test_episode, "metadata.json")
with open(meta_path, 'r') as f:
    metadata = json.load(f)

language = metadata['language_instruction']
print(f"테스트 에피소드: {test_episode}")
print(f"언어 명령: {language}\n")

print("타임스텝별 추론 결과:")
print("-" * 80)

test_steps = [0, 10, 20, 30]

for step_idx in test_steps:
    if step_idx >= metadata['num_steps']:
        continue

    img_path = os.path.join(test_episode, f"image_{step_idx:04d}.jpg")
    image = Image.open(img_path).convert("RGB")
    gt_action = np.array(metadata['actions'][step_idx])

    # 추론: 7개 토큰을 생성하도록 강제
    prompt = f"In: What action should the robot take to {language}?\nOut:"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

    with torch.no_grad():
        # 액션 토큰 7개 생성
        generated = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=torch.ones_like(inputs['input_ids']),
            pixel_values=inputs['pixel_values'],
            max_new_tokens=ACTION_DIM,
            do_sample=False,
        )

    # 생성된 토큰 중 마지막 7개가 액션 토큰
    new_tokens = generated[0, -ACTION_DIM:].cpu().numpy()
    predicted_action = tokens_to_action(new_tokens)

    print(f"\n[Step {step_idx}]")
    print(f"  정답 액션: {gt_action}")
    print(f"  생성된 토큰: {new_tokens}")
    print(f"  예측 액션: {predicted_action}")

    diff = np.abs(predicted_action - gt_action)
    print(f"  평균 오차: {diff.mean():.6f}")

print("\n" + "=" * 60)
print("  추론 테스트 완료")
print("=" * 60)
print("\n해석:")
print("- 생성된 토큰이 [31488 ~ 32000] 범위 안에 있어야 함 (액션 토큰 영역)")
print("- 예측 액션이 정답과 비슷한 부호와 크기면 학습 성공")
print("- 모든 step의 예측이 똑같으면 여전히 학습 실패")
