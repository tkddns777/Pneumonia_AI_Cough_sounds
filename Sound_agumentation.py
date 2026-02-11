import os
import random
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# =====================================================
# 경로 설정
# =====================================================
INPUT_ROOT = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database"
OUTPUT_ROOT = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database\Augmented_Wav_TimeShift"

CLASSES = ["Pneumonia", "Healthy"]

# =====================================================
# 오디오 설정
# =====================================================
SR = 22050
DURATION = 5.0                 # 모든 샘플을 5초로 통일 (원하면 None로 두고 원본 길이 유지 가능)
MAX_SHIFT_SEC = 1.5            # 최대 이동 범위 (예: ±1초)
N_AUG_PER_FILE = 100             # 원본 파일 당 몇 개 증강 생성
SHIFT_MODE = "roll"            # "roll" 또는 "zero_pad"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# =====================================================
# 유틸: 길이 표준화
# =====================================================
def fix_length(y, sr, duration):
    if duration is None:
        return y
    target_len = int(sr * duration)
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

# =====================================================
# 유틸: 시간 이동
# =====================================================
def time_shift(y, sr, max_shift_sec, mode="roll"):
    max_shift = int(sr * max_shift_sec)
    if max_shift <= 0:
        return y, 0

    shift = random.randint(-max_shift, max_shift)  # samples 단위
    if shift == 0:
        return y, 0

    if mode == "roll":
        y_shift = np.roll(y, shift)
    elif mode == "zero_pad":
        y_shift = np.zeros_like(y)
        if shift > 0:
            y_shift[shift:] = y[:-shift]
        else:
            y_shift[:shift] = y[-shift:]
    else:
        raise ValueError("mode must be 'roll' or 'zero_pad'")

    return y_shift, shift

# =====================================================
# 출력 폴더 생성
# =====================================================
for cls in CLASSES:
    os.makedirs(os.path.join(OUTPUT_ROOT, cls), exist_ok=True)

# =====================================================
# 증강 실행
# =====================================================
for cls in CLASSES:
    in_dir = os.path.join(INPUT_ROOT, cls)
    out_dir = os.path.join(OUTPUT_ROOT, cls)

    wav_files = list(Path(in_dir).glob("*.wav"))
    print(f"\n▶ {cls}: {len(wav_files)} files")

    for wav_path in tqdm(wav_files):
        # 원본 로드
        y, sr = librosa.load(str(wav_path), sr=SR, mono=True)

        # 길이 통일 (권장)
        y = fix_length(y, sr, DURATION)

        # 원본도 복사 저장(원하면 주석 처리)
        # sf.write(os.path.join(out_dir, wav_path.name), y, sr)

        # 증강 생성
        for k in range(N_AUG_PER_FILE):
            y_aug, shift = time_shift(y, sr, MAX_SHIFT_SEC, mode=SHIFT_MODE)

            shift_ms = int(1000 * shift / sr)
            save_name = f"{wav_path.stem}_shift{shift_ms:+d}ms_{k}.wav"
            save_path = os.path.join(out_dir, save_name)

            sf.write(save_path, y_aug, sr)

print("\n✅ Time-shift 증강 wav 저장 완료")
