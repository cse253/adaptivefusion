# Technical Investigation: Adaptive Fusion Performance Analysis

This document provides a root-cause technical analysis of why the proposed **Adaptive Fusion** framework underperformed compared to the **RGB Baseline** and **Late Fusion** models.

---

## 1. Quantitative Performance Mismatch

| Model Configuration | Train Acc (%) | Val Acc (%) | Test Acc (%) | Macro F1-Score | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RGB Baseline** (Raw) | 91.46 | 90.00 | **81.82** | 0.5000 | 37,178,952 |
| **RGB Baseline** (Emb) | 19.51 | 10.00 | **36.36** | 0.0667 | 37,178,952 |
| **Pose Only** | 20.73 | 30.00 | **9.09** | 0.0208 | 5,332,744 |
| **Late Fusion** (Raw) | 76.83 | 70.00 | **81.82** | 0.5208 | 42,511,696 |
| **Feature Concat** (Raw) | 59.76 | 60.00 | **54.55** | 0.3000 | 42,704,456 |
| **Adaptive Fusion** (Proposed) | 40.24 | 40.00 | **45.45** | 0.1409 | 19,264,650 |
| **Adaptive Fusion** (Latest Collapse)| 20.73 | 30.00 | **9.09** | 0.0208 | 19,264,650 |

---

## 2. Root-Cause Analysis of Underperformance

### A. Severe Data Scarcity vs. Model Capacity
* **The Problem**: The training dataset consists of only **82 video recordings**. 
* **Parameters**: The Adaptive Fusion model has **19,264,650 parameters**. 
* **Details**: Training a 19M parameter model (consisting of two separate Transformer layers and projection heads) from scratch on 82 samples is highly underdetermined. The joint representation space quickly overfits to training details and fails to generalize.

### B. Gating Network Collapse to a Static Bias
* **The Problem**: The Softmax-gated network output values for $\alpha$ (RGB) and $\beta$ (Pose) collapsed to constant values.
* **Evidence**: In [results/fusion_weight_analysis.csv](file:///d:/NIT%20proj/Adaptive_signlang/results/fusion_weight_analysis.csv), the weights for all test samples are identical:
  * $\alpha_{RGB} = 0.0421$ (4.2%)
  * $\beta_{Pose} = 0.9579$ (95.8%)
* **Mechanism**: During the early epochs, the RGB and Pose branch weights are random, generating noisy embeddings. The gating network quickly learns that Pose features (258 coordinates) are cleaner and normalized, and collapses into a local minimum where it assigns a large static bias to Pose. Once the gate biases towards Pose, the gradient update for the RGB branch diminishes, locking the model into a collapsed state.

### C. Low-Quality Pose Modality Dominating the Output
* **The Problem**: The Pose Only model achieves only **9.09% test accuracy** (random guessing).
* **Details**: Landmarks tracked via MediaPipe suffer from coordinate-level noise, lack of temporal alignment, and complete loss of hand shape/orientation. Because the gating network assigns $95.8\%$ weight to the Pose embedding, it forces the fused vector $\mathbf{z}_{fused}$ to be almost entirely composed of noisy Pose features, dragging the model's performance down to $9.09\%$ (predicting `happy` for every video).

### D. Lack of Pretrained Weight Loading
* **The Problem**: In [scripts/train_adaptive.py](file:///d:/NIT%20proj/Adaptive_signlang/scripts/train_adaptive.py#L118-L130), `AdaptiveMultiModalModel` is initialized randomly. Unlike Late Fusion, it **does not load** the pretrained weights from `best_model.pth` or `best_pose_model.pth`.
* **Details**: Training the transformers from scratch on 82 samples prevents the intermediate embeddings `rgb_feat` and `pose_feat` from becoming discriminative.

---

## 3. Scientific Interpretation of the Results

1. **Is the result scientifically meaningful?**
   Yes. It demonstrates that intermediate-level dynamic gating is highly unstable on small datasets and prone to modality collapse. It proves that visual appearance (RGB) remains the primary driver of accuracy in sign language recognition.
2. **Is the pose branch hurting performance?**
   Yes. The pose branch is extremely noisy (accuracy 9.09%), and because the gating network collapses to weighting Pose at 95.8%, the low-quality Pose branch directly ruins the classification capability of the fused representation.
3. **Does RGB dominance explain the result?**
   Yes. RGB features capture hand shapes, positions, and context which are critical for distinguishing adjectives (like *loud* vs *happy*). The RGB baseline achieves 81.82% accuracy on raw frames because it has access to these rich spatial cues, while the Adaptive model discards them.

---

## 4. Concrete Improvements for Future Work

1. **Gate Regularization via Modality Dropout**:
   Implement Modality Dropout (randomly zeroing out one of the modal inputs during training with probability $p$) to force the gating network to learn to rely on both branches.
2. **Load Pretrained Weights & Freeze Encoders**:
   Initialize the RGB and Pose branches in `AdaptiveMultiModalModel` using `best_model.pth` and `best_pose_model.pth` and **freeze** them. Only train the linear projections, gate MLP, and final classifier.
3. **Apply Gate Entropy Regularization**:
   Add an entropy penalty to the loss function:
   $$\mathcal{L}_{total} = \mathcal{L}_{CE} - \lambda \mathcal{H}(\alpha, \beta)$$
   This prevents the Softmax output from collapsing to a one-hot vector (like $[0.04, 0.96]$) and encourages a balanced distribution of modality weights.
4. **ResNet50 Evaluation Alignment**:
   Align the speed-optimized evaluations in `week4_compare.py` to use the fine-tuned CNN weights from `best_model.pth` rather than the default ImageNet ResNet50, which caused the artificial drop in evaluation scores.
