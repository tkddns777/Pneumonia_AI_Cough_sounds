import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import inception_v3, Inception_V3_Weights

# =====================================================
# 설정: 여기만 수정
# =====================================================
DATA_ROOT = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database\MelSpectrogram_Dataset_Split")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR  = DATA_ROOT / "test"

SAVE_DIR = DATA_ROOT / "results_inception_v3"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4

IMG_SIZE = 299
NUM_WORKERS = 4  # Windows에서 멈추면 0으로
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =====================================================


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
        if isinstance(out, tuple):
            out = out[0]

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
        # Inception v3 train mode + aux_logits=True => (logits, aux_logits)
        if isinstance(out, tuple):
            logits, aux_logits = out
            loss_main = criterion(logits, y)
            loss_aux = criterion(aux_logits, y)
            loss = loss_main + 0.4 * loss_aux
            preds = logits.argmax(dim=1)
        else:
            loss = criterion(out, y)
            preds = out.argmax(dim=1)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


def build_model(num_classes: int):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, aux_logits=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if model.aux_logits and model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)

    return model


def main():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Train/Test 폴더가 없습니다.\n"
            "먼저 split_dataset_groupwise.py를 실행해서 MelSpectrogram_Dataset_Split을 만드세요.\n"
            f"Expected: {TRAIN_DIR} and {TEST_DIR}"
        )

    # Transform (MelSpectrogram 특성상 augmentation은 약하게)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomApply([transforms.RandomRotation(5)], p=0.3),
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

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )

    class_names = train_ds.classes
    num_classes = len(class_names)

    print(f"DEVICE: {DEVICE}")
    print(f"Classes: {class_names}")
    print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")

    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        scheduler.step(test_acc)
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
            f"time={dt:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

            ckpt = {
                "epoch": epoch,
                "best_acc": best_acc,
                "class_names": class_names,
                "model_state_dict": best_state,
                "img_size": IMG_SIZE,
                "arch": "inception_v3",
            }
            save_path = SAVE_DIR / f"inception_v3_best_acc{best_acc:.3f}_epoch{epoch:03d}.pth"
            torch.save(ckpt, save_path)
            print(f"  ✅ Saved best checkpoint: {save_path}")

    print(f"\n✅ Training finished. Best test acc = {best_acc:.4f}")
    print(f"Checkpoints folder: {SAVE_DIR}")

    if DEVICE != "cuda":
        print("\n[WARN] CUDA를 못 잡았어. GPU 사용 원하면 PyTorch CUDA 설치/환경을 확인해줘.")
    print("[Tip] Windows에서 멈추면 NUM_WORKERS=0으로 바꿔서 실행.")


if __name__ == "__main__":
    main()
