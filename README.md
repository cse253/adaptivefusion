# Adaptive Multi-Modal Fusion Transformer for Indian Sign Language Recognition

This project implements a state-of-the-art **Dual-Branch Spatial-Temporal Transformer** architecture for Indian Sign Language (ISL) Recognition. By combining visual appearance (RGB video frames) and body/hand coordinate trajectories (Pose landmarks), the system achieves highly accurate dynamic gesture recognition.

---

## 1. System Architecture

The model uses a dual-branch pipeline to extract complementary features from videos:

```
                     ┌──> [ResNet50] ──> [Transformer Encoder] ──┐ (RGB Embedding)
                     │                                           │
  [Input Video] ─────┤                                           ├──> [Fusion Model] ──> [Prediction]
                     │                                           │
                     └──> [MediaPipe] ──> [Transformer Encoder] ─┘ (Pose Embedding)
```

*   **RGB Branch**: Frames are processed by a pre-trained **ResNet50** CNN, projected, and modeled temporally using a 4-layer **Transformer Encoder**.
*   **Pose Branch**: 258 landmark coordinates (33 body pose, 21 left-hand, 21 right-hand keypoints) are extracted frame-by-frame via **MediaPipe Holistic** and modeled temporally using a coordinate-level **Transformer Encoder**.
*   **Fusion Paradigms**: We evaluate six multimodal integration strategies:
    1.  **RGB Baseline**: Single-modality classification using only visual features.
    2.  **Pose Only**: Single-modality classification using only coordinate paths.
    3.  **Late Fusion**: Decision-level fusion averaging predicted probability distributions.
    4.  **Feature Concatenation**: Fusion by side-by-side concatenation of intermediate embeddings.
    5.  **Adaptive Fusion (Proposed)**: Softmax-gated gating network that learns sample-specific weights ($\alpha$ for RGB, $\beta$ for Pose) to dynamically balance modalities.
    6.  **Cross-Attention Fusion (Advanced)**: Bidirectional multihead cross-attention layer aligning frame-level RGB and Pose sequences before temporal average pooling.

---

## 2. Dataset

*   **Dataset**: INCLUDE Indian Sign Language dataset subset.
*   **Size**: 103 dynamic video recordings.
*   **Classes**: 8 adjective signs (*loud, quiet, happy, sad, Beautiful, Ugly, Deaf, Blind*).
*   **Partitions**: Train (82 videos), Validation (10 videos), Test (11 videos).

---

## 3. Configuration & Hyperparameters

Hyperparameters are modularized inside the `configs/` folder using YAML configuration files. They can be updated without modifying training scripts:

*   [`configs/baseline.yaml`](configs/baseline.yaml): Base dataset splits, classes, and generic paths.
*   [`configs/pose.yaml`](configs/pose.yaml): Pose-only model settings, optimizer, learning rate, and layers.
*   [`configs/fusion.yaml`](configs/fusion.yaml): Parameters for Late, Concatenation, and Cross-Attention Fusion models.
*   [`configs/adaptive.yaml`](configs/adaptive.yaml): Configuration for the scalar-gated Adaptive Fusion model.

---

## 4. How to Run the Pipeline

### Step 1: Install Dependencies
Install all required libraries, including PyTorch, OpenCV, and MediaPipe:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Master Runner
The entire pipeline (data splitting, pose extraction, training all 6 model variants, evaluation, ablation studies, Grad-CAM spatial explainability, and attention maps plotting) is orchestrated automatically:
```bash
python scripts/run_project.py
```

### Outputs
All outputs (accuracy plots, training logs, Grad-CAM heatmaps, attention maps, and model checkpoints) will be generated inside the `results/` and `checkpoints/` directories.
