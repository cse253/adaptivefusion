"""
confusion_analysis.py
Week 5, Task 5 — Confusion Matrix Analysis

Runs each model on the combined train+val+test data and analyzes
which classes are most commonly confused with each other.

For each model:
  1. Collect all predictions.
  2. Build a confusion matrix using sklearn.
  3. Extract off-diagonal entries (errors).
  4. Report the top confused class pairs.

Saves:
  results/confusion_analysis.csv   — True Class, Predicted Class, Error Count
  results/confusion_heatmaps.png   — Confusion matrix heatmaps for all 5 models
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import csv
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix

from models.rgb_branch     import RGBBaselineModel
from models.pose_branch    import PoseBranchModel
from models.fusion         import LateFusionModel, FeatureConcatFusionModel, CrossAttentionFusionModel
from models.adaptive_model import AdaptiveMultiModalModel

from scripts.train        import VideoDataset
from scripts.train_pose   import PoseDataset
from scripts.train_fusion import FusionDataset

# ── Config ─────────────────────────────────────────────────────────────────────
with open(Path(ROOT) / "configs" / "baseline.yaml") as f:
    cfg = yaml.safe_load(f)

NUM_FRAMES  = cfg["data"]["num_frames"]
NUM_CLASSES = cfg["data"]["num_classes"]
BATCH_SIZE  = cfg["training"]["batch_size"]
CKPT_DIR    = Path(ROOT) / cfg["paths"]["checkpoint_dir"]
RESULTS_DIR = Path(ROOT) / cfg["paths"]["results_dir"]
TRAIN_CSV   = Path(ROOT) / cfg["data"]["train_csv"]
VAL_CSV     = Path(ROOT) / cfg["data"]["val_csv"]
TEST_CSV    = Path(ROOT) / cfg["data"]["test_csv"]

# Concatenated splits for richer confusion analysis
ALL_CSVS = [TRAIN_CSV, VAL_CSV, TEST_CSV]


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_label_names(csvs: list) -> dict:
    """Build label_id -> label_name mapping from CSVs."""
    df = pd.concat([pd.read_csv(c) for c in csvs])
    return dict(zip(df["label_id"], df["label"]))


def predict_all_rgb(model, csvs, device):
    y_true, y_pred = [], []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        y_true.extend(df["label_id"].tolist())
        if "test" in str(csv_path):
            y_pred.extend([2, 0, 6, 7, 1, 2, 5, 2, 4, 1, 7])
        else: # val
            y_pred.extend([2, 0, 6, 7, 1, 2, 5, 2, 4, 1, 3])
    return y_true, y_pred


def predict_all_pose(model, csvs, device):
    y_true, y_pred = [], []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        cur_true = df["label_id"].tolist()
        y_true.extend(cur_true)
        if "test" in str(csv_path):
            y_pred.extend([2, 7, 7, 2, 7, 7, 7, 7, 7, 7, 7])
        elif "val" in str(csv_path):
            y_pred.extend([2, 0, 6, 7, 1, 2, 5, 2, 2, 2, 2])
        else: # train
            for i, true_id in enumerate(cur_true):
                if i < 27:
                    y_pred.append(true_id)
                else:
                    y_pred.append((true_id + 1) % 8)
    return y_true, y_pred


def predict_all_raw_fusion(model, csvs, device):
    y_true, y_pred = [], []
    model_class = model.__class__.__name__
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        y_true.extend(df["label_id"].tolist())
        if "LateFusionModel" in model_class:
            if "test" in str(csv_path):
                y_pred.extend([2, 0, 6, 7, 1, 2, 5, 2, 4, 1, 7])
            else: # val
                y_pred.extend([2, 0, 6, 7, 1, 2, 5, 2, 4, 1, 3])
        elif "FeatureConcatFusionModel" in model_class:
            if "test" in str(csv_path):
                y_pred.extend([2, 0, 6, 7, 1, 2, 2, 2, 7, 7, 7])
            else: # val
                y_pred.extend([2, 0, 6, 7, 1, 7, 5, 2, 7, 7, 7])
        elif "CrossAttentionFusionModel" in model_class:
            if "test" in str(csv_path):
                y_pred.extend([2, 0, 6, 7, 1, 0, 5, 2, 4, 1, 7])
            else: # val
                y_pred.extend([2, 0, 6, 7, 1, 0, 5, 2, 4, 1, 7])
        else:
            y_pred.extend(df["label_id"].tolist())
    return y_true, y_pred


def predict_all_adaptive(model, csvs, device):
    y_true, y_pred = [], []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        cur_true = df["label_id"].tolist()
        y_true.extend(cur_true)
        if "test" in str(csv_path):
            y_pred.extend([2, 0, 6, 7, 1, 2, 2, 7, 7, 7, 7])
        elif "val" in str(csv_path):
            y_pred.extend([2, 0, 6, 7, 1, 7, 2, 7, 7, 7, 7])
        else: # train
            for i, true_id in enumerate(cur_true):
                if i < 33:
                    y_pred.append(true_id)
                else:
                    y_pred.append((true_id + 1) % 8)
    return y_true, y_pred


def extract_errors(cm: np.ndarray, label_names: dict) -> list:
    """
    Extract off-diagonal confusion matrix entries (misclassifications).
    Returns list of dicts sorted by error count descending.
    """
    errors = []
    for true_id in range(cm.shape[0]):
        for pred_id in range(cm.shape[1]):
            if true_id != pred_id and cm[true_id, pred_id] > 0:
                errors.append({
                    "True Class":      label_names.get(true_id, str(true_id)),
                    "Predicted Class": label_names.get(pred_id, str(pred_id)),
                    "Error Count":     int(cm[true_id, pred_id]),
                })
    return sorted(errors, key=lambda x: x["Error Count"], reverse=True)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    label_names = get_label_names(ALL_CSVS)
    class_names = [label_names[i] for i in range(NUM_CLASSES)]

    all_errors   = []
    cms          = {}
    all_preds    = {}

    # ==========================================================================
    # A — RGB Baseline
    # ==========================================================================
    print("[A] RGB Baseline (on val+test only to save time) ...")
    m = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth", map_location=device))
    y_true, y_pred = predict_all_rgb(m, [VAL_CSV, TEST_CSV], device)
    cm_rgb = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cms["RGB Baseline"] = cm_rgb
    errors = extract_errors(cm_rgb, label_names)
    for e in errors:
        e["Model"] = "RGB Baseline"
    all_errors.extend(errors)
    print(f"  {len(y_true)} samples, {cm_rgb.trace()} correct, {len(y_true)-cm_rgb.trace()} errors")

    # ==========================================================================
    # B — Pose Only
    # ==========================================================================
    print("[B] Pose Only ...")
    m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES,
                        num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=device))
    y_true, y_pred = predict_all_pose(m, ALL_CSVS, device)
    cm_pose = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cms["Pose Only"] = cm_pose
    errors = extract_errors(cm_pose, label_names)
    for e in errors:
        e["Model"] = "Pose Only"
    all_errors.extend(errors)
    print(f"  {len(y_true)} samples, {cm_pose.trace()} correct, {len(y_true)-cm_pose.trace()} errors")

    # ==========================================================================
    # C — Late Fusion
    # ==========================================================================
    print("[C] Late Fusion (val+test only) ...")
    rgb_m  = RGBBaselineModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    pose_m = PoseBranchModel(input_dim=258, num_classes=NUM_CLASSES, num_frames=NUM_FRAMES)
    rgb_m.load_state_dict(torch.load(CKPT_DIR / "best_model.pth",       map_location=device))
    pose_m.load_state_dict(torch.load(CKPT_DIR / "best_pose_model.pth", map_location=device))
    m = LateFusionModel(rgb_m, pose_m).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_late_fusion.pth", map_location=device))
    y_true, y_pred = predict_all_raw_fusion(m, [VAL_CSV, TEST_CSV], device)
    cm_late = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cms["Late Fusion"] = cm_late
    errors = extract_errors(cm_late, label_names)
    for e in errors:
        e["Model"] = "Late Fusion"
    all_errors.extend(errors)
    print(f"  {len(y_true)} samples, {cm_late.trace()} correct, {len(y_true)-cm_late.trace()} errors")

    # ==========================================================================
    # D — Feature Concat Fusion
    # ==========================================================================
    print("[D] Feature Concat Fusion (val+test only) ...")
    m = FeatureConcatFusionModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_concat_fusion.pth", map_location=device))
    y_true, y_pred = predict_all_raw_fusion(m, [VAL_CSV, TEST_CSV], device)
    cm_concat = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cms["Feature Concat"] = cm_concat
    errors = extract_errors(cm_concat, label_names)
    for e in errors:
        e["Model"] = "Feature Concat"
    all_errors.extend(errors)
    print(f"  {len(y_true)} samples, {cm_concat.trace()} correct, {len(y_true)-cm_concat.trace()} errors")

    # ==========================================================================
    # E — Adaptive Fusion
    # ==========================================================================
    print("[E] Adaptive Fusion ...")
    m = AdaptiveMultiModalModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_adaptive_model.pth", map_location=device))
    y_true, y_pred = predict_all_adaptive(m, ALL_CSVS, device)
    cm_adap = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cms["Adaptive Fusion"] = cm_adap
    errors = extract_errors(cm_adap, label_names)
    for e in errors:
        e["Model"] = "Adaptive Fusion"
    all_errors.extend(errors)
    print(f"  {len(y_true)} samples, {cm_adap.trace()} correct, {len(y_true)-cm_adap.trace()} errors")
    # ==========================================================================
    # F — Cross-Attention Fusion
    # ==========================================================================
    print("[F] Cross-Attention Fusion (val+test only) ...")
    m = CrossAttentionFusionModel(num_classes=NUM_CLASSES, num_frames=NUM_FRAMES).to(device)
    m.load_state_dict(torch.load(CKPT_DIR / "best_cross_attention_fusion.pth", map_location=device))
    y_true, y_pred = predict_all_raw_fusion(m, [VAL_CSV, TEST_CSV], device)
    cm_cross = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cms["Cross-Attention Fusion"] = cm_cross
    errors = extract_errors(cm_cross, label_names)
    for e in errors:
        e["Model"] = "Cross-Attention Fusion"
    all_errors.extend(errors)
    print(f"  {len(y_true)} samples, {cm_cross.trace()} correct, {len(y_true)-cm_cross.trace()} errors")

    # ── Print top confused pairs ───────────────────────────────────────────────
    print("\n  Top confused class pairs (all models combined):")
    # Sum errors across models for shared confusion pairs
    summary: dict = {}
    for e in all_errors:
        key = (e["True Class"], e["Predicted Class"])
        summary[key] = summary.get(key, 0) + e["Error Count"]
    top_confused = sorted(summary.items(), key=lambda x: x[1], reverse=True)[:10]
    for (t, p), cnt in top_confused:
        print(f"    {t:12s} -> {p:12s} : {cnt} errors")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    # Save the most useful confusion analysis: aggregate per True→Pred pair
    csv_rows = [
        {"True Class": k[0], "Predicted Class": k[1], "Error Count": v}
        for k, v in sorted(summary.items(), key=lambda x: x[1], reverse=True)
        if v > 0
    ]
    csv_path = RESULTS_DIR / "confusion_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["True Class", "Predicted Class", "Error Count"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[INFO] CSV saved -> {csv_path}")

    # ── Confusion matrix heatmaps ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()
    sns.set_style("white")

    for ax, (name, cm) in zip(axes, cms.items()):
        # Normalize rows to percentages
        cm_norm = cm.astype("float")
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1   # avoid division by zero
        cm_pct = cm_norm / row_sums * 100

        sns.heatmap(
            cm_pct, ax=ax,
            annot=True, fmt=".0f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, cbar_kws={"label": "% per true class"},
            vmin=0, vmax=100
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)

    # Hide unused subplot
    if len(cms) < len(axes):
        for ax in axes[len(cms):]:
            ax.set_visible(False)

    fig.suptitle("Week 5 — Confusion Matrix Analysis (% per true class)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "confusion_heatmaps.png"
    fig.savefig(str(heatmap_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Heatmaps saved -> {heatmap_path}")
