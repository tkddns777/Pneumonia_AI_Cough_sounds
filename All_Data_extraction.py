import pandas as pd
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# =====================================================
# 경로 설정
# =====================================================

AUDIO_DIR = r"C:\Users\user\Downloads\archive\Respiratory_Sound_Database\Respiratory_Sound_Database\audio_and_txt_files"
DIAGNOSIS_FILE = r"C:\Users\user\Downloads\archive\Respiratory_Sound_Database\Respiratory_Sound_Database\patient_diagnosis.csv"

OUTPUT_DIR = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\disease_dataset"

# =====================================================
# diagnosis 읽기
# =====================================================

diag_df = pd.read_csv(
    DIAGNOSIS_FILE,
    header=None,
    names=["patient_id","disease"]
)

# pneumonia vs healthy만 사용
target_classes = ["Pneumonia","Healthy"]

diag_df = diag_df[diag_df["disease"].isin(target_classes)]

# =====================================================
# patient split
# =====================================================

patients = diag_df["patient_id"].unique()

train_patients, test_patients = train_test_split(
    patients,
    test_size=0.2,
    random_state=42
)

# =====================================================
# audio 파일 리스트
# =====================================================

wav_files = list(Path(AUDIO_DIR).glob("*.wav"))

for wav_path in wav_files:

    filename = wav_path.stem

    patient_id = int(filename.split("_")[0])

    if patient_id not in diag_df["patient_id"].values:
        continue

    disease = diag_df.loc[
        diag_df["patient_id"] == patient_id,
        "disease"
    ].values[0]

    if patient_id in train_patients:
        split = "train"
    else:
        split = "test"

    dst_dir = os.path.join(
        OUTPUT_DIR,
        split,
        disease
    )

    os.makedirs(dst_dir, exist_ok=True)

    shutil.copy(
        wav_path,
        os.path.join(dst_dir, wav_path.name)
    )