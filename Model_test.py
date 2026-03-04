import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay
)

# =====================================================
# 설정: 여기만 수정
# =====================================================
DATA_ROOT = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database\Mel_Data")
TEST_DIR  = DATA_ROOT / "test"

CKPT_PATH = Path(r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Respiratory_Sound_Database\Respiratory_Sound_Database\results_inception_v3\resnet18_best.pth")

BATCH_SIZE = 64
NUM_WORKERS = 0  # Windows에서 멈추면 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_CM_FIG = True
OUT_DIR = CKPT_PATH.parent / "eval_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# =====================================================


def build_model_for_eval(num_classes: int):

    model = resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model


def show_confusion_matrix(y_true, y_pred, class_names, save_path=None):

    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Confusion Matrix (Counts) ===")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=True,
        values_format="d"
    )

    plt.title("Confusion Matrix")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return cm


@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        out = model(x)


        prob = torch.softmax(out, dim=1)
        pred = prob.argmax(dim=1)

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
        y_prob.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)
    return y_true, y_pred, y_prob


def main():
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test 폴더가 없습니다: {TEST_DIR}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"CKPT 파일이 없습니다: {CKPT_PATH}")

    # ---- Load checkpoint ----
    # 본인 파일이면 보안경고는 무시 가능 (원하면 weights_only=False 명시 가능)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    class_names = ckpt.get("class_names", None)
    img_size = ckpt.get("img_size", 224)

    print("CKPT PATH:", CKPT_PATH)
    print("CKPT epoch:", ckpt.get("epoch"))
    print("CKPT best_acc:", ckpt.get("best_acc"))

    if class_names is None:
        raise ValueError("ckpt 안에 class_names가 없습니다.")

    num_classes = len(class_names)

    # ---- Data transform (test 전용) ----
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=test_tf)

    print(f"[CKPT class_names] : {class_names}")
    print(f"[TEST  class_names] : {test_ds.classes}")
    print(f"[TEST  class_to_idx] : {test_ds.class_to_idx}")

    if test_ds.classes != class_names:
        print("\n[WARN] test_ds.classes와 ckpt class_names 순서가 다릅니다! (지표 뒤틀릴 수 있음)\n")

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )

    # ---- Build & load model ----
    model = build_model_for_eval(num_classes).to(DEVICE)

    # 가능하면 strict=True로 정확히 맞추는 게 좋음
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print("[Info] load_state_dict: strict=True 성공")
    except RuntimeError as e:
        print("[WARN] strict=True 로드 실패 -> strict=False로 재시도")
        print("       원인:", str(e).split("\n")[0])

        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("[Info] load_state_dict(strict=False) 결과:")
        print("  missing keys   :", len(missing))
        print("  unexpected keys:", len(unexpected))

    # ---- Predict ----
    y_true, y_pred, y_prob = predict_all(model, test_loader)

    # ROC AUC 계산
    auc = roc_auc_score(y_true, y_prob[:,1])
    print("\nROC AUC:", auc)

    # ---- Metrics ----
    acc = accuracy_score(y_true, y_pred)

    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # ---- Confusion Matrix (table + figure) ----
    cm_save_path = str(OUT_DIR / "confusion_matrix_table_heatmap.png") if SAVE_CM_FIG else None
    cm = show_confusion_matrix(y_true, y_pred, class_names, save_path=cm_save_path)

    # ✅ 혼돈행렬로부터 accuracy 재계산(불일치 검증)
    acc_from_cm = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    print("\n=== Accuracy check ===")
    print(f"Accuracy(from sklearn) = {acc:.6f}")
    print(f"Accuracy(from CM)      = {acc_from_cm:.6f}")
    if abs(acc - acc_from_cm) > 1e-9:
        print("[WARN] sklearn accuracy와 CM accuracy가 다릅니다. (코드/데이터 매칭 버그 가능)")

    print("\n==================== Overall ====================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro    : Precision={p_macro:.4f} Recall={r_macro:.4f} F1={f1_macro:.4f}")
    print(f"Weighted : Precision={p_weighted:.4f} Recall={r_weighted:.4f} F1={f1_weighted:.4f}")

    print("\n==================== Per-class ==================")
    for i, name in enumerate(class_names):
        print(f"[{i}] {name:12s} | Precision={p_c[i]:.4f} Recall={r_c[i]:.4f} F1={f1_c[i]:.4f} Support={sup_c[i]}")

    print("\n==================== Classification Report ======")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    if SAVE_CM_FIG:
        print(f"\n✅ Confusion matrix figure saved: {cm_save_path}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

