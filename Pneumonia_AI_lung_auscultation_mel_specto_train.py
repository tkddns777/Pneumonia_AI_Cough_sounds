import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

import random
import os


# =====================================================
# 설정: 여기만 수정
# =====================================================
DATA_ROOT = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database\Mel_Data")
TRAIN_DIR = DATA_ROOT / "train_augmented"
TEST_DIR  = DATA_ROOT / "test"

SAVE_DIR = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database\results_inception_v3")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 2.5e-4
WEIGHT_DECAY = 1e-4

SEED_LIST = list(range(0, 101, 10))   # 🔥 여기 마음대로 추가 가능 

IMG_SIZE = 224
NUM_WORKERS = 0  # Windows에서 멈추면 0으로
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_WEIGHTED_SAMPLER = False    # ✅ 추천: True
USE_LOSS_CLASS_WEIGHT = True  # 필요 시 True (Sampler와 동시 사용 시 과보정 가능)

USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING = 0.05  # 보통 0.05~0.2

# =====================================================


# =====================================================
# 유사도 재현을 위한 시드 설정
# =====================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # 재현성 ↑ (속도 ↓ 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)

        loss = criterion(out, y)
        preds = out.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model(x)
        loss = criterion(out, y)
        preds = out.argmax(dim=1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def build_model(num_classes: int):

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model


def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Train/Test 폴더가 없습니다.\n"
            "먼저 split_dataset_groupwise.py를 실행해서 MelSpectrogram_Dataset_Split을 만드세요.\n"
            f"Expected: {TRAIN_DIR} and {TEST_DIR}"
        )

    all_best_acc = []

    for seed in SEED_LIST:
        print("\n" + "=" * 70)
        print(f"🚀 Running with SEED = {seed}")
        print("=" * 70)

        # ✅ seed 고정
        set_seed(seed)

        # =====================================================
        # Transform (증강 없음)
        # =====================================================
        train_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        test_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
        test_ds  = datasets.ImageFolder(str(TEST_DIR),  transform=test_tf)

        # ✅ 반드시 여기서 먼저 클래스 정의
        class_names = train_ds.classes
        num_classes = len(class_names)

        print(f"DEVICE: {DEVICE}")
        print(f"Classes: {class_names}")
        print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")

        # =====================================================
        # ✅ Class imbalance handling
        # =====================================================
        targets = np.array(train_ds.targets)
        class_counts = np.bincount(targets, minlength=num_classes)
        print(f"Train class counts: {class_counts.tolist()}")

        class_weights = (class_counts.sum() / (class_counts + 1e-12)).astype(np.float32)
        class_weights = class_weights / class_weights.mean()
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

        sample_weights = class_weights[targets]
        sample_weights_t = torch.tensor(sample_weights, dtype=torch.double)

        sampler = None
        shuffle_flag = True

        if USE_WEIGHTED_SAMPLER:
            sampler = WeightedRandomSampler(
                weights=sample_weights_t,
                num_samples=len(sample_weights_t),
                replacement=True
            )
            shuffle_flag = False
            print("✅ Using WeightedRandomSampler for balanced training.")
        else:
            print("ℹ️ Using default random shuffle (no sampler).")

        # =====================================================
        # DataLoader
        # =====================================================
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle_flag,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == "cuda")
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == "cuda")
        )

        # =====================================================
        # Model / Loss / Optimizer
        # =====================================================
        model = build_model(num_classes).to(DEVICE)

        if USE_LOSS_CLASS_WEIGHT:
            criterion = nn.CrossEntropyLoss(
                weight=class_weights_t,
                label_smoothing=(LABEL_SMOOTHING if USE_LABEL_SMOOTHING else 0.0)
            )
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=(LABEL_SMOOTHING if USE_LABEL_SMOOTHING else 0.0)
            )

        print(f"✅ Label smoothing: {LABEL_SMOOTHING if USE_LABEL_SMOOTHING else 0.0}")


        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

        best_acc = 0.0
        best_state = None
        best_epoch = -1

        # =====================================================
        # Training Loop
        # =====================================================
        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            test_loss, test_acc = evaluate(model, test_loader, criterion)

            scheduler.step(test_acc)
            lr_now = optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            print(
                f"[Seed {seed}] Epoch {epoch:02d}/{NUM_EPOCHS} | lr={lr_now:.2e} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
                f"time={dt:.1f}s"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        all_best_acc.append(best_acc)

        # =====================================================
        # Save best checkpoint for this seed
        # =====================================================
        ckpt = {
            "seed": seed,
            "best_acc": float(best_acc),
            "best_epoch": int(best_epoch),
            "class_names": class_names,
            "model_state_dict": best_state,
            "img_size": IMG_SIZE,
            "arch": "ResNet18",
            "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
            "use_loss_class_weight": USE_LOSS_CLASS_WEIGHT,
        }
        save_path = SAVE_DIR / f"resnet18_seed{seed}_best{best_acc:.3f}_epoch{best_epoch:03d}.pth"
        torch.save(ckpt, save_path)
        print(f"✅ [Seed {seed}] Saved best checkpoint: {save_path}")
        print(f"🔥 [Seed {seed}] Best test acc = {best_acc:.4f} at epoch {best_epoch}")

    # =====================================================
    # Summary across seeds
    # =====================================================
    all_best_acc = np.array(all_best_acc, dtype=np.float32)
    print("\n" + "=" * 70)
    print("📊 Multi-seed summary")
    print("=" * 70)
    print(f"Seeds: {SEED_LIST}")
    print(f"Best acc per seed: {[float(x) for x in all_best_acc.tolist()]}")
    print(f"Mean best acc: {all_best_acc.mean():.4f}")
    print(f"Std  best acc: {all_best_acc.std(ddof=1):.4f}" if len(all_best_acc) > 1 else "Std  best acc: N/A (only 1 seed)")

    if DEVICE != "cuda":
        print("\n[WARN] CUDA를 못 잡았어. GPU 사용 원하면 PyTorch CUDA 설치/환경을 확인해줘.")
    print("[Tip] Windows에서 멈추면 NUM_WORKERS=0으로 바꿔서 실행.")


if __name__ == "__main__":
    main()
