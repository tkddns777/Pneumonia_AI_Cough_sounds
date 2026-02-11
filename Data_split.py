import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# =====================================================
# 설정: 여기만 수정
# =====================================================
SRC_ROOT = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database\Spectrogram_Output")

# 폴더명
CLASS_DIRS = ["Pneumonia", "Healthy"]

# 분할된 데이터 저장 폴더
OUT_ROOT = SRC_ROOT.parent / "MelSpectrogram_Dataset_Split"

TRAIN_RATIO = 0.8
SEED = 42

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# =====================================================


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def extract_group_id(filepath: Path) -> str:
    """
    ✅ 누수 방지 핵심: 같은 환자/같은 원본에서 나온 이미지들이
    train/test에 동시에 들어가지 않도록 group id를 추출.

    기본 규칙:
      - 파일명에서 '_' 또는 '-' 기준으로 첫 토큰을 group으로 사용
        예) patient001_03.png -> patient001
            p12-aug1.png      -> p12

    ⚠️ 네 파일 규칙이 다르면 이 함수만 너 데이터 규칙에 맞게 바꾸면 됨.
    """
    stem = filepath.stem
    if "_" in stem:
        return stem.split("_")[0]
    if "-" in stem:
        return stem.split("-")[0]
    return stem


@dataclass
class Sample:
    path: Path
    cls: str
    group: str


def collect_samples(src_root: Path, class_dirs: List[str]) -> List[Sample]:
    samples: List[Sample] = []
    for cls in class_dirs:
        cls_dir = src_root / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_dir}")

        files = [p for p in cls_dir.rglob("*") if p.is_file() and is_image(p)]
        if len(files) == 0:
            raise RuntimeError(f"No image files found in: {cls_dir}")

        for p in files:
            g = extract_group_id(p)
            samples.append(Sample(path=p, cls=cls, group=g))
    return samples


def group_split(samples: List[Sample], train_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    rng = random.Random(seed)

    group_to_samples: Dict[str, List[Sample]] = {}
    for s in samples:
        group_to_samples.setdefault(s.group, []).append(s)

    groups = list(group_to_samples.keys())
    rng.shuffle(groups)

    n_train_groups = int(len(groups) * train_ratio)
    train_groups = set(groups[:n_train_groups])
    test_groups = set(groups[n_train_groups:])

    train_samples: List[Sample] = []
    test_samples: List[Sample] = []

    for g in train_groups:
        train_samples.extend(group_to_samples[g])
    for g in test_groups:
        test_samples.extend(group_to_samples[g])

    # 누수 검증
    assert train_groups.isdisjoint(test_groups), "Group leakage: train/test groups overlap!"
    train_paths = set(str(s.path) for s in train_samples)
    test_paths = set(str(s.path) for s in test_samples)
    assert train_paths.isdisjoint(test_paths), "Sample leakage: train/test samples overlap!"

    return train_samples, test_samples


def copy_split(train_samples: List[Sample], test_samples: List[Sample], out_root: Path, class_dirs: List[str]):
    train_root = out_root / "train"
    test_root = out_root / "test"

    # 기존 OUT_ROOT 제거 후 새로 생성
    if out_root.exists():
        print(f"[INFO] Removing existing split folder: {out_root}")
        shutil.rmtree(out_root)

    for split in ["train", "test"]:
        for cls in class_dirs:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    # 복사
    for s in train_samples:
        dst = train_root / s.cls / s.path.name
        shutil.copy2(s.path, dst)

    for s in test_samples:
        dst = test_root / s.cls / s.path.name
        shutil.copy2(s.path, dst)

    return train_root, test_root


def count_by_class(samples: List[Sample], class_dirs: List[str]):
    counts = {c: 0 for c in class_dirs}
    for s in samples:
        counts[s.cls] += 1
    return counts


def main():
    print("==== Collecting samples ====")
    samples = collect_samples(SRC_ROOT, CLASS_DIRS)

    groups_all = set(s.group for s in samples)
    print(f"Total samples: {len(samples)}")
    print(f"Total groups : {len(groups_all)}")

    print("\n==== Group-wise split (80/20) ====")
    train_samples, test_samples = group_split(samples, TRAIN_RATIO, SEED)

    train_groups = set(s.group for s in train_samples)
    test_groups = set(s.group for s in test_samples)

    print(f"Train samples: {len(train_samples)} | Train groups: {len(train_groups)}")
    print(f"Test  samples: {len(test_samples)} | Test  groups: {len(test_groups)}")
    print(f"Group leakage check: {len(train_groups & test_groups)} (must be 0)")
    print(f"Train class counts: {count_by_class(train_samples, CLASS_DIRS)}")
    print(f"Test  class counts: {count_by_class(test_samples, CLASS_DIRS)}")

    print("\n==== Copying to output folders ====")
    train_root, test_root = copy_split(train_samples, test_samples, OUT_ROOT, CLASS_DIRS)
    print("✅ Done!")
    print(f"Train folder: {train_root}")
    print(f"Test  folder: {test_root}")

    print("\n[NOTE] 만약 Group leakage check가 0이 아닌 경우,")
    print("       extract_group_id()를 파일명 규칙에 맞게 수정해야 합니다.")


if __name__ == "__main__":
    main()
