0# Project Explanation Guide: Adaptive Sign Language Recognition

This guide is designed for beginners to understand, run, and confidently explain this project to advisors, professors, or examiners.

---

## 1. What is this project?

The goal of this project is **Indian Sign Language (ISL) Recognition**. The system takes a video of a person performing a dynamic sign (like "happy", "sad", "loud", "quiet", "Beautiful", "Ugly", "Deaf", "Blind") and automatically predicts which sign is being performed.

### Key Terms Explained:
*   **Multi-Modal**: The model uses two different types of input data ("modalities") at the same time:
    1.  **RGB Appearance (Visuals)**: Raw video frames that capture what the signer looks like, their hand shapes, and clothing.
    2.  **Pose Coordinates (Geometry)**: Geometric landmark points ($X, Y, Z$ positions) of the body and hands tracked frame-by-frame.
*   **Fusion**: The process of combining these two modalities to make a more accurate prediction.
*   **Adaptive Gating**: A smart mechanism that dynamically calculates how much to trust the RGB visuals versus the Pose coordinates for each specific video.
*   **Transformer**: A modern type of neural network (originally famous in natural language processing, like ChatGPT) used here to capture how gestures change over time across video frames.

---

## 2. System Architecture (How it works under the hood)

The project uses a **Dual-Branch Architecture** (two parallel pipelines):

```
                     ┌──> [ResNet50] ──> [Transformer Encoder] ──┐ (RGB Embedding)
                     │                                           │
  [Input Video] ─────┤                                           ├──> [Fusion Model] ──> [Prediction]
                     │                                           │
                     └──> [MediaPipe] ──> [Transformer Encoder] ─┘ (Pose Embedding)
```

### Branch A: The RGB Branch
1.  **Input**: A sequence of 16 frames from the video.
2.  **Feature Extractor (ResNet50)**: A pre-trained convolutional neural network (CNN) that converts each image frame into a 2048-dimensional vector representing visual features (shapes, textures, hand configurations).
3.  **Temporal Transformer**: Analyzes the sequence of these vectors to learn the timing and transition of the gesture over the 16 frames.

### Branch B: The Pose Branch
1.  **Landmark Extractor (MediaPipe Holistic)**: Tracks the body and hands frame-by-frame, extracting **258 keypoint coordinates** (33 body points, 21 left-hand points, 21 right-hand points).
2.  **Temporal Transformer**: Analyzes how these 258 coordinate points move over the 16 frames.

### The Fusion Module
We evaluate 5 ways to combine these branches:
1.  **RGB Baseline**: Ignores coordinates. Predicts using only visual features.
2.  **Pose Only**: Ignores visuals. Predicts using only coordinate movements.
3.  **Late Fusion**: Trains the RGB and Pose models separately. At test time, it averages their predicted probabilities.
4.  **Feature Concatenation**: Pastes the RGB features and Pose features side-by-side to make a long vector, then uses a classifier.
5.  **Adaptive Fusion (Proposed)**: Uses a small "gating network" that looks at both embeddings and outputs two weights: $\alpha$ (for RGB) and $\beta$ (for Pose) such that $\alpha + \beta = 1.0$. The final embedding is a weighted sum:
    $$\text{Fused Embedding} = (\alpha \times \text{RGB}) + (\beta \times \text{Pose})$$

---

## 3. The Dataset

*   **Size**: 103 video recordings.
*   **Classes**: 8 distinct sign gestures: *loud, quiet, happy, sad, Beautiful, Ugly, Deaf, Blind*.
*   **Split**: 
    *   **Training**: 82 videos (~80%)
    *   **Validation**: 10 videos (~10%)
    *   **Testing**: 11 videos (~10%)

---

## 4. Key Findings & Results (What you should present)

Here is a summary of the performance results from the evaluation:

| Model Configuration | Test Accuracy | Macro F1-Score | Main Takeaway |
| :--- | :---: | :---: | :--- |
| **RGB Baseline** | **81.82%** | 0.5000 | Highly accurate; visual shapes are very descriptive. |
| **Late Fusion** | **81.82%** | **0.5208** | Best overall; uses coordinate clues to refine visual errors. |
| **Feature Concatenation** | 54.55% | 0.3000 | Overfits because of too many parameters on a small dataset. |
| **Adaptive Fusion** | 45.45% | 0.1409 | Learns to prioritize coordinates ($\beta = 85.2\%$) but overfits. |
| **Pose Only** | 9.09% | 0.0500 | Coordinate paths alone are too noisy to learn with limited data. |

### 💡 Why did the Adaptive Fusion model get lower accuracy than RGB alone?
The gating network learned that landmark coordinates are highly stable (since coordinates are invariant to changes in lighting or clothing). Thus, it converged to weighting Pose heavily ($\beta \approx 85.2\%$). However, because the Pose branch itself performs poorly due to the small size of the dataset, relying too much on Pose dragged down the overall accuracy of the Adaptive Fusion model. 

---

## 5. Explainability (Explainable AI - XAI)

Professors love when models are explainable. We implemented two types of explainability:

1.  **Spatial Explainability (Grad-CAM)**: 
    *   *What it does*: Highlights which pixels in the video frames the RGB branch focused on.
    *   *Result*: For correct predictions, the model highlights the signer's hands, arms, and face. For incorrect predictions, the activations are scattered or focus on the background.
2.  **Temporal Explainability (Transformer Self-Attention Maps)**:
    *   *What it does*: Visualizes which frames in the 16-frame sequence were deemed most important by the Transformer.
    *   *Result*: The model automatically pays attention to frames **6 to 12** (the middle of the clip). This is where the gesture reaches its "apex" (semantic peak), while ignoring the beginning and ending frames where hands are just moving up or down.

---

## 6. How to Run the Project Pipeline

Everything is automated using a single master runner script: `scripts/run_project.py`.

### Step 1: Install Dependencies
Open your command prompt or terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
Execute the master script:
```bash
python scripts/run_project.py
```

### What this script does automatically:
1.  **Stage 1**: Checks that your folders and dataset are present.
2.  **Stage 2**: Verifies your Python environment has all packages.
3.  **Stage 3**: Validates that all project scripts are in place.
4.  **Stage 4**: Sequentially executes the pipeline:
    *   Explores the dataset statistics (`explore_dataset.py`).
    *   Splits the dataset into train/val/test (`split_dataset.py`).
    *   Trains and evaluates the RGB model (`train.py`, `evaluate.py`).
    *   Extracts MediaPipe landmark coordinates (`extract_pose.py`).
    *   Trains the Pose model (`train_pose.py`).
    *   Trains the multi-modal fusion and adaptive models (`train_fusion.py`, `train_adaptive.py`).
    *   Runs comparison metrics, ablation studies, Grad-CAM, and attention visualizations.
5.  **Stage 5**: Verifies that all expected plots, reports, and checkpoints are successfully generated in the `results/` folder.
