"""
OpenVLA 정식 파인튜닝 스크립트 (action tokenization 포함)
- 4-bit 양자화 + LoRA + gradient checkpointing
- 액션을 256개 구간으로 양자화하여 토큰으로 변환
- 변환된 토큰을 정답 라벨로 사용하여 학습

이전 버전과의 차이점:
- 이전: labels=input_ids (모델이 자기 입력 그대로 따라하도록 학습 → 잘못됨)
- 현재: 액션을 토큰으로 변환 후 라벨로 사용 (OpenVLA의 정식 방식)

사용법:
    conda activate openvla
    cd /media/kimjimin/02B092A4B0929E2B/openvla
    python finetune_indy7_v2.py
"""

import os
import json
import glob
import time
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# =========================
# 설정
# =========================
DATASET_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_dataset/pick_up_cup"
OUTPUT_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_checkpoints_v2"
MODEL_ID = "openvla/openvla-7b"

LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
LORA_RANK = 16
SAVE_EVERY_EPOCH = 2
GRAD_ACCUMULATION = 4

# OpenVLA 액션 토큰 설정
NUM_ACTION_BINS = 256  # 액션을 256개 구간으로 양자화
ACTION_DIM = 7         # 7차원 액션 (6 DoF + gripper)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 액션 정규화 통계 계산
# =========================
def compute_action_stats(dataset_dir):
    """전체 데이터셋의 액션 통계를 계산하여 정규화에 사용"""
    all_actions = []
    episode_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode_*")))

    for ep_dir in episode_dirs:
        meta_path = os.path.join(ep_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        for action in metadata['actions']:
            all_actions.append(action)

    all_actions = np.array(all_actions, dtype=np.float32)
    # 1, 99 percentile로 outlier 제거 (안정적인 정규화)
    q01 = np.percentile(all_actions, 1, axis=0)
    q99 = np.percentile(all_actions, 99, axis=0)

    print(f"\n액션 통계:")
    print(f"  전체 액션 수: {len(all_actions)}")
    print(f"  1% percentile: {q01}")
    print(f"  99% percentile: {q99}")
    print(f"  Min: {all_actions.min(axis=0)}")
    print(f"  Max: {all_actions.max(axis=0)}")

    return q01, q99


# =========================
# 데이터셋
# =========================
class Indy7VLADataset(Dataset):
    def __init__(self, dataset_dir, action_low, action_high, processor, num_bins=256):
        self.dataset_dir = dataset_dir
        self.action_low = action_low
        self.action_high = action_high
        self.processor = processor
        self.num_bins = num_bins

        episode_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode_*")))
        self.samples = []
        self.episode_metadata = {}

        for ep_dir in episode_dirs:
            meta_path = os.path.join(ep_dir, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            self.episode_metadata[ep_dir] = metadata
            for step_idx in range(metadata['num_steps']):
                self.samples.append((ep_dir, step_idx))

        print(f"데이터셋: {len(episode_dirs)}개 에피소드, {len(self.samples)}개 타임스텝")

    def __len__(self):
        return len(self.samples)

    def normalize_action(self, action):
        """액션을 [-1, 1] 범위로 정규화"""
        action = np.clip(action, self.action_low, self.action_high)
        # [low, high] -> [-1, 1]
        normalized = 2 * (action - self.action_low) / (self.action_high - self.action_low + 1e-8) - 1
        return normalized

    def action_to_tokens(self, action):
        """정규화된 액션을 토큰 ID로 변환
        OpenVLA는 마지막 256개 토큰을 액션 토큰으로 사용한다.
        """
        normalized = self.normalize_action(action)
        # [-1, 1] -> [0, num_bins-1]
        bins = np.clip(((normalized + 1) / 2 * self.num_bins).astype(np.int64),
                       0, self.num_bins - 1)
        # OpenVLA tokenizer의 vocab_size에서 마지막 256개를 액션 토큰으로 사용
        vocab_size = self.processor.tokenizer.vocab_size
        action_token_ids = vocab_size - self.num_bins + bins
        return action_token_ids  # shape: (7,)

    def __getitem__(self, idx):
        ep_dir, step_idx = self.samples[idx]
        metadata = self.episode_metadata[ep_dir]

        img_path = os.path.join(ep_dir, f"image_{step_idx:04d}.jpg")
        image = Image.open(img_path).convert("RGB")

        action = np.array(metadata['actions'][step_idx], dtype=np.float32)
        action_tokens = self.action_to_tokens(action)

        return {
            'image': image,
            'action': action,
            'action_tokens': action_tokens,
            'language_instruction': metadata['language_instruction'],
        }


# =========================
# 메인
# =========================
def main():
    print("=" * 60)
    print("  OpenVLA 정식 파인튜닝 v2 (Indy7, action tokenization)")
    print("=" * 60)

    # 액션 통계 계산
    action_low, action_high = compute_action_stats(DATASET_DIR)

    # 통계 저장 (추론 시 다시 사용)
    stats_path = os.path.join(OUTPUT_DIR, "action_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'action_low': action_low.tolist(),
            'action_high': action_high.tolist(),
            'num_bins': NUM_ACTION_BINS,
        }, f, indent=2)
    print(f"\n액션 통계 저장: {stats_path}")

    # 프로세서 로드
    print("\n프로세서 로딩 중...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 데이터셋 생성
    dataset = Indy7VLADataset(
        DATASET_DIR, action_low, action_high, processor,
        num_bins=NUM_ACTION_BINS
    )

    # 첫 샘플 확인
    sample = dataset[0]
    print(f"\n첫 샘플 예시:")
    print(f"  원본 action: {sample['action']}")
    print(f"  action tokens: {sample['action_tokens']}")
    print(f"  토큰 범위: [{sample['action_tokens'].min()}, {sample['action_tokens'].max()}]")

    # 4-bit 양자화 모델 로드
    print("\n모델 로딩 중 (4-bit 양자화)...")
    qlora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        quantization_config=qlora_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA 적용
    print(f"LoRA 적용 중 (rank={LORA_RANK})...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # 옵티마이저
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nVRAM 사용: {vram_used:.1f} / {vram_total:.1f} GB")

    # =========================
    # 학습 루프
    # =========================
    print(f"\n학습 시작: {NUM_EPOCHS} 에폭, {len(dataset)} 스텝/에폭")
    print("-" * 60)

    model.train()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        indices = np.random.permutation(len(dataset))
        start_time = time.time()
        optimizer.zero_grad()

        for i, idx in enumerate(indices):
            sample = dataset[int(idx)]

            # 프롬프트 구성
            prompt = f"In: What action should the robot take to {sample['language_instruction']}?\nOut:"

            # 입력 토큰화 (이미지 + 텍스트)
            inputs = processor(prompt, sample['image']).to("cuda:0", dtype=torch.bfloat16)
            input_ids = inputs['input_ids']

            # 정답 생성: 입력 뒤에 액션 토큰 + EOS 추가
            action_token_tensor = torch.tensor(
                sample['action_tokens'], dtype=torch.long, device="cuda:0"
            ).unsqueeze(0)  # shape: (1, 7)

            # 전체 시퀀스 = 입력 + 액션 토큰
            full_input_ids = torch.cat([input_ids, action_token_tensor], dim=1)

            # labels: 입력 부분은 -100(무시), 액션 토큰만 학습
            labels = full_input_ids.clone()
            labels[:, :input_ids.shape[1]] = -100

            # attention_mask 확장
            attention_mask = torch.ones_like(full_input_ids)

            # pixel_values는 이미지 입력
            pixel_values = inputs['pixel_values']

            try:
                outputs = model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss / GRAD_ACCUMULATION
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nVRAM 부족! step {i}에서 중단.")
                    torch.cuda.empty_cache()
                    return
                raise

            epoch_loss += loss.item() * GRAD_ACCUMULATION

            if (i + 1) % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (i + 1) % 100 == 0:
                avg_loss = epoch_loss / (i + 1)
                vram = torch.cuda.memory_allocated() / 1024**3
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Step {i+1}/{len(dataset)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"VRAM: {vram:.1f}GB | "
                      f"시간: {elapsed:.0f}s")

        avg_loss = epoch_loss / len(dataset)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} 완료 | Avg Loss: {avg_loss:.4f} | 시간: {elapsed:.0f}s")

        if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(ckpt_dir)
            print(f"  체크포인트 저장: {ckpt_dir}")

    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\n학습 완료! 최종 모델: {final_dir}")


if __name__ == "__main__":
    main()
