"""
OpenVLA 파인튜닝용 PyTorch Dataset
- collect_vla_data.py로 수집한 데이터를 로드
- RLDS 변환 없이 바로 파인튜닝에 사용 가능

이 파일을 OpenVLA의 vla-scripts/finetune.py에서 import하여 사용합니다.
"""

import os
import json
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Indy7VLADataset(Dataset):
    """
    수집된 Indy7 시연 데이터를 OpenVLA 파인튜닝에 사용할 수 있도록
    변환하는 PyTorch Dataset 클래스.

    저장 구조 예시:
        vla_dataset/
        └── pick_up_cup/
            ├── episode_0000/
            │   ├── image_0000.jpg
            │   ├── image_0001.jpg
            │   ├── ...
            │   └── metadata.json
            ├── episode_0001/
            │   └── ...
    """

    def __init__(self, dataset_dir, image_size=(224, 224)):
        """
        Args:
            dataset_dir: 데이터셋 최상위 경로 (예: /media/.../vla_dataset/pick_up_cup)
            image_size: 이미지 리사이즈 크기 (OpenVLA는 224x224 사용)
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size

        # 모든 에피소드 폴더 찾기
        episode_dirs = sorted(glob.glob(os.path.join(dataset_dir, "episode_*")))

        # 모든 타임스텝을 (에피소드 경로, 스텝 인덱스)로 펼치기
        self.samples = []
        self.episode_metadata = {}

        for ep_dir in episode_dirs:
            meta_path = os.path.join(ep_dir, "metadata.json")
            if not os.path.exists(meta_path):
                continue

            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            num_steps = metadata['num_steps']
            self.episode_metadata[ep_dir] = metadata

            for step_idx in range(num_steps):
                self.samples.append((ep_dir, step_idx))

        print(f"데이터셋 로드 완료: {len(episode_dirs)}개 에피소드, {len(self.samples)}개 타임스텝")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_dir, step_idx = self.samples[idx]
        metadata = self.episode_metadata[ep_dir]

        # 이미지 로드 및 리사이즈
        img_path = os.path.join(ep_dir, f"image_{step_idx:04d}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)
        image = np.array(image)  # (224, 224, 3)

        # 상태: 관절 각도 6개 + 그리퍼 1개 = 7개
        joint_pos = metadata['joint_positions'][step_idx]
        gripper = metadata['gripper_states'][step_idx]
        state = joint_pos + [gripper]  # 7차원

        # 동작: TCP 변화량 6개 + 그리퍼 1개 = 7개
        action = metadata['actions'][step_idx]  # 7차원

        # 언어 명령
        language = metadata['language_instruction']

        return {
            'image': image,              # (224, 224, 3) numpy array
            'state': np.array(state, dtype=np.float32),    # (7,)
            'action': np.array(action, dtype=np.float32),  # (7,)
            'language_instruction': language,               # str
        }


def verify_dataset(dataset_dir):
    """데이터셋이 올바르게 수집되었는지 확인하는 함수"""
    dataset = Indy7VLADataset(dataset_dir)

    if len(dataset) == 0:
        print("에러: 데이터가 없습니다!")
        return False

    # 첫 번째 샘플 확인
    sample = dataset[0]
    print(f"\n=== 데이터셋 검증 ===")
    print(f"총 샘플 수: {len(dataset)}")
    print(f"이미지 shape: {sample['image'].shape}")
    print(f"상태 shape: {sample['state'].shape}")
    print(f"동작 shape: {sample['action'].shape}")
    print(f"언어 명령: {sample['language_instruction']}")
    print(f"상태 예시: {sample['state']}")
    print(f"동작 예시: {sample['action']}")
    print(f"=== 검증 완료 ===\n")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("사용법: python indy7_dataset.py <데이터셋 경로>")
        print("예시: python indy7_dataset.py /media/kimjimin/02B092A4B0929E2B/vla_dataset/pick_up_cup")
        sys.exit(1)

    verify_dataset(sys.argv[1])
