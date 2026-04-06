"""
OpenVLA QLoRA 파인튜닝 스크립트 (15.5GB VRAM용)
- 4-bit 양자화 + LoRA + gradient checkpointing
- batch size 1로 최소 VRAM 사용
- 수집된 Indy7 데이터로 파인튜닝

사용법:
    conda activate openvla
    cd /media/kimjimin/02B092A4B0929E2B/openvla
    python finetune_indy7.py
"""

import os
import sys
import json
import glob
import time
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# =========================
# 설정
# =========================
DATASET_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_dataset/pick_up_cup"
OUTPUT_DIR = "/media/kimjimin/02B092A4B0929E2B/vla_checkpoints"
MODEL_ID = "openvla/openvla-7b"

LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
BATCH_SIZE = 1
LORA_RANK = 16          # 32보다 작게 해서 VRAM 절약
SAVE_EVERY_EPOCH = 2     # 2 에폭마다 체크포인트 저장
GRAD_ACCUMULATION = 4    # 실효 배치사이즈 = 1 * 4 = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 데이터셋
# =========================
class Indy7VLADataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
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

    def __getitem__(self, idx):
        ep_dir, step_idx = self.samples[idx]
        metadata = self.episode_metadata[ep_dir]

        img_path = os.path.join(ep_dir, f"image_{step_idx:04d}.jpg")
        image = Image.open(img_path).convert("RGB")

        action = metadata['actions'][step_idx]
        language = metadata['language_instruction']

        return {
            'image': image,
            'action': np.array(action, dtype=np.float32),
            'language_instruction': language,
        }


# =========================
# 메인
# =========================
def main():
    print("=" * 50)
    print("  OpenVLA QLoRA 파인튜닝 (Indy7)")
    print("=" * 50)

    # 데이터셋 로드
    dataset = Indy7VLADataset(DATASET_DIR)
    if len(dataset) == 0:
        print("에러: 데이터가 없습니다!")
        return

    # 프로세서 로드
    print("\n프로세서 로딩 중...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 4-bit 양자화 설정
    print("모델 로딩 중 (4-bit 양자화)...")
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
    )

    # Gradient checkpointing 활성화 (VRAM 절약)
    model.gradient_checkpointing_enable()

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

    # VRAM 상태 출력
    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nVRAM 사용: {vram_used:.1f} / {vram_total:.1f} GB")
    print(f"남은 VRAM: {vram_total - vram_used:.1f} GB")

    # =========================
    # 학습 루프
    # =========================
    print(f"\n학습 시작: {NUM_EPOCHS} 에폭, {len(dataset)} 스텝/에폭")
    print(f"실효 배치사이즈: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print("-" * 50)

    model.train()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        indices = np.random.permutation(len(dataset))
        start_time = time.time()

        optimizer.zero_grad()

        for i, idx in enumerate(indices):
            sample = dataset[int(idx)]

            # 입력 준비
            prompt = f"In: What action should the robot take to {sample['language_instruction']}?\nOut:"
            inputs = processor(prompt, sample['image']).to("cuda:0", dtype=torch.bfloat16)

            # 정답 action을 토큰으로 변환
            action = sample['action']

            # Forward pass
            try:
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss / GRAD_ACCUMULATION
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nVRAM 부족! step {i}에서 중단.")
                    print("해결 방법: LORA_RANK를 8로 줄이거나 외부 서버를 사용하세요.")
                    torch.cuda.empty_cache()
                    return
                raise

            epoch_loss += loss.item() * GRAD_ACCUMULATION

            # Gradient accumulation
            if (i + 1) % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # 진행 상황 출력 (100스텝마다)
            if (i + 1) % 100 == 0:
                avg_loss = epoch_loss / (i + 1)
                vram = torch.cuda.memory_allocated() / 1024**3
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Step {i+1}/{len(dataset)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"VRAM: {vram:.1f}GB | "
                      f"시간: {elapsed:.0f}s")

        # 에폭 완료
        avg_loss = epoch_loss / len(dataset)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} 완료 | Avg Loss: {avg_loss:.4f} | 시간: {elapsed:.0f}s")

        # 체크포인트 저장
        if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(ckpt_dir)
            print(f"  체크포인트 저장: {ckpt_dir}")

    # 최종 모델 저장
    final_dir = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\n학습 완료! 최종 모델: {final_dir}")


if __name__ == "__main__":
    main()
