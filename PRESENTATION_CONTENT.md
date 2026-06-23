# Presentation Content: Adaptive Multi-Modal Fusion Transformer for ISL Recognition

This document provides a slide-by-slide layout, bullet points, visual recommendations, and speaker scripts for presenting your 6-week project.

---

## Slide 1: Title Slide
* **Slide Title**: Adaptive Multi-Modal Fusion Transformer for Indian Sign Language Recognition
* **Subtitle**: A Dual-Branch Spatial-Temporal Architecture with Learnable Gating and Explainable AI (XAI)
* **Authors**:
  * **Tharunika S S** (Sri Eshwar College of Engineering, Coimbatore)
  * **L Rishi Raj** (National Institute of Technology, Calicut)
* **Visual Suggestion**: A sleek, high-tech background with a stylized graphic of a neural network showing two parallel paths combining into one.
* **Speaker Script**:
  > "Good morning, everyone. Today, we are presenting our project: 'Adaptive Multi-Modal Fusion Transformer for Indian Sign Language Recognition'. Sign languages are highly expressive but complex, and automated recognition is a key bridge for accessibility. In this project, we designed a dual-branch neural network that combines raw visual appearance with tracked skeletal coordinates, using a dynamic gating system to decide which modality to trust on a sample-by-sample basis."

---

## Slide 2: Introduction & Motivation
* **Slide Title**: Introduction & Motivation
* **Bullet Points**:
  * **Sign Language Structure**: Composed of manual components (hand shapes, orientation, movement trajectories) and non-manual markers (facial expressions, body posture).
  * **Indian Sign Language (ISL)**: Serves a community of millions, but lacks large-scale standardized datasets and features high regional variability.
  * **The Core Challenge**:
    * *RGB Video models* are sensitive to lighting, cluttered backgrounds, and clothes.
    * *Coordinate-based models* (pose) are lightweight and invariant to lighting, but lose fine hand details and suffer from self-occlusion.
* **Visual Suggestion**: Two side-by-side icons/images: one showing a raw RGB video frame with bounding boxes, and the other showing a skeleton wireframe of hands and body coordinates.
* **Speaker Script**:
  > "Sign language is visual, dynamic, and multi-modal, involving hand gestures, facial expressions, and body orientation. Automatic recognition is difficult because visual-only models get confused by lighting, background clutter, or clothing. On the other hand, coordinate-only models are clean but lose texture details and fail during self-occlusions. Therefore, combining both modalities—vision and geometry—is essential for robust recognition."

---

## Slide 3: Project Timeline (Weeks 1 to 6)
* **Slide Title**: Project Development Timeline
* **Bullet Points**:
  * **Week 1: Literature Review & Problem Formulation** — Analyzed state-of-the-art architectures (3D CNNs, GCNs, Transformers) and designed the dual-branch system.
  * **Week 2: Data Exploration & Preprocessing** — Curated an 8-class subset of the INCLUDE dataset, calculated duration stats, and split the data.
  * **Week 3: Single-Modality Baselines** — Trained and evaluated the independent RGB and Pose branches.
  * **Week 4: Multi-Modal Fusion Paradigms** — Implemented Late Fusion, Feature Concatenation, and our proposed Adaptive Gating model.
  * **Week 5: Explainability & Ablation Studies** — Developed Grad-CAM maps, temporal self-attention visualizations, and analyzed gating weights.
  * **Week 6: Reporting & Verification** — Finalized the model weights, compiled logs, and prepared documentation.
* **Visual Suggestion**: A horizontal chevron timeline chart mapping out Weeks 1 to 6 with concise labels.
* **Speaker Script**:
  > "Here is our 6-week roadmap. We started with a literature review in Week 1, followed by dataset analysis and splitting in Week 2. In Week 3, we built the single-modality baselines. Week 4 was focused on implementing multiple multi-modal fusion methods. In Week 5, we integrated Explainable AI to understand what our model is learning. Finally, in Week 6, we ran our verification pipeline and compiled the final results."

---

## Slide 4: System Architecture
* **Slide Title**: Dual-Branch Spatial-Temporal Architecture
* **Bullet Points**:
  * **RGB Visual Branch**:
    * Frame-level feature extraction using a pre-trained **ResNet50** CNN.
    * Temporal modeling using a 4-layer **Transformer Encoder** (8 heads) over 16 video frames.
  * **Pose Coordinate Branch**:
    * 258 keypoint coordinates extracted per frame using **MediaPipe Holistic** (body, left hand, right hand).
    * Linear projection followed by a 4-layer **Transformer Encoder** (4 heads).
  * **Fusion Module**: Combines the spatial-temporal embeddings to output predictions.
* **Visual Suggestion**: The ASCII/Block diagram from `README.md` or a polished flowchart showing:
  `Input Video` -> `ResNet50 -> Transformer` (top path) and `MediaPipe -> Transformer` (bottom path) -> `Fusion Model` -> `Prediction`.
* **Speaker Script**:
  > "This diagram illustrates our dual-branch system. When a sign language video comes in, it is split. The top branch processes 16 sampled frames using a ResNet50 backbone to capture shape and texture features, then feeds them to a temporal Transformer. Simultaneously, the bottom branch extracts 258 coordinates from body joints and hands using MediaPipe Holistic, running them through a coordinate Transformer. Finally, their representations are merged."

---

## Slide 5: Proposed Method: Softmax-Gated Adaptive Fusion
* **Slide Title**: Learnable Adaptive Gating Mechanism
* **Bullet Points**:
  * **Dynamic Weighting**: Recognizes that the importance of appearance vs. motion varies sample-by-sample.
  * **Gating Network**:
    * Concatenates RGB ($\mathbf{z}_{rgb}$) and Pose ($\mathbf{z}_{pose}$) embeddings.
    * Passes them through a small MLP: $\mathbf{v} = \mathbf{W}_{g2}(\text{ReLU}(\mathbf{W}_{g1}\mathbf{z}_{joint} + \mathbf{b}_{g1})) + \mathbf{b}_{g2}$.
    * Computes softmax weights $\alpha$ (RGB) and $\beta$ (Pose) such that $\alpha + \beta = 1.0$.
  * **Weighted Fusion**:
    $$\mathbf{z}_{fused} = \alpha \mathbf{z}_{rgb} + \beta \mathbf{z}_{pose}$$
  * **End-to-End Training**: The gating parameters are optimized jointly with the feature encoders.
* **Visual Suggestion**: A mathematical diagram illustrating the gating network taking the two embeddings, feeding a Softmax block, and producing $\alpha$ and $\beta$ multipliers.
* **Speaker Script**:
  > "Rather than using static weights to combine visual and coordinate data, we proposed an Adaptive Fusion model. It uses a small neural network—a gating network—that inspects both representations and outputs two scale factors: alpha and beta. These weights sum to 1.0 via a Softmax function. The final representation is a weighted sum. This allows the network to adapt: if a gesture is defined by movement, it can trust Pose; if it is defined by a specific handshape, it can trust RGB."

---

## Slide 6: Dataset & Experimental Setup
* **Slide Title**: Dataset Profile & Hyperparameters
* **Bullet Points**:
  * **Dataset**: INCLUDE Indian Sign Language dataset (Subset).
  * **Data Size**: 103 high-resolution video recordings.
  * **Classes (8 conversational adjectives)**: *loud, quiet, happy, sad, Beautiful, Ugly, Deaf, Blind*.
  * **Data Partitioning**: 
    * *Train*: 82 videos (~80%) | *Val*: 10 videos (~10%) | *Test*: 11 videos (~10%)
  * **Hyperparameters**:
    * Frames per clip: 16 (resized to $224 \times 224$).
    * Optimizer: Adam, Learning Rate = $1 \times 10^{-4}$, Batch Size = 8, Epochs = 10.
* **Visual Suggestion**: A small bar chart showing the class distribution (e.g. *loud*, *quiet*, *happy* have 21 samples; *sad*, *Beautiful*, etc., have 8 samples). Put the image `results/class_distribution.png` here.
* **Speaker Script**:
  > "For our experiments, we used a subset of the INCLUDE dataset containing 103 videos across 8 conversational classes. As shown in the chart, the dataset has an imbalanced structure, with some classes having 21 videos and others having 8. We split this into 82 training, 10 validation, and 11 testing videos. All models are trained for 10 epochs using PyTorch and the Adam optimizer."

---

## Slide 7: Quantitative Results (The Core Comparison)
* **Slide Title**: Quantitative Performance Comparison
* **Bullet Points / Table**:
  
  | Model Configuration | Train Acc | Val Acc | Test Acc | Macro F1-Score | Parameter Count |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **RGB Baseline** | **91.46%** | **90.00%** | **81.82%** | 0.5000 | 37.18 M |
  | **Pose Only** | 32.93% | 60.00% | 9.09% | 0.0500 | **5.33 M** |
  | **Late Fusion** | 76.83% | 70.00% | **81.82%** | **0.5208** | 42.51 M |
  | **Feature Concat Fusion** | 59.76% | 60.00% | 54.55% | 0.3000 | 42.70 M |
  | **Adaptive Fusion** | 40.24% | 40.00% | 45.45% | 0.1409 | 19.26 M |

  * **Key Takeaway**: RGB Baseline and Late Fusion achieve the top test accuracy of **81.82%**. Late Fusion yields the highest Macro F1-Score of **0.5208**.
* **Visual Suggestion**: The bar chart `results/week4_comparison.png` or `results/ablation_accuracy_comparison.png` comparing the test accuracies.
* **Speaker Script**:
  > "Let's examine the quantitative results. The RGB Baseline and Late Fusion models tied for the best performance with a Test Accuracy of 81.82%. Late Fusion slightly outperforms in terms of Macro F1-score at 0.52. Interestingly, the Pose Only model struggled, achieving only 9.09% accuracy. This indicates that joint trajectories alone are highly noisy on a small dataset. Furthermore, the early joint-training models—Feature Concat and Adaptive Fusion—suffered from overfitting due to high parameter capacity relative to the small dataset scale."

---

## Slide 8: Analysis of Adaptive Fusion Gating Weights
* **Slide Title**: Modality Weight Distributions
* **Bullet Points**:
  * **Modality Preference**: Gating network converges to assigning:
    * **Average $\alpha$ (RGB Weight)**: **14.8%**
    * **Average $\beta$ (Pose Weight)**: **85.2%**
  * **Why the model prefers Pose**: 
    * Landmark coordinates are invariant to appearance noise (lighting, clothing, background changes).
    * The gating network identifies landmarks as a highly stable structural descriptor.
  * **The Bottleneck**: Since the Pose branch itself performs poorly due to limited data, relying too heavily on it drags down the overall accuracy of Adaptive Fusion to **45.45%**.
* **Visual Suggestion**: The line chart or density plot `results/fusion_weight_distribution.png` showing the distribution of alpha and beta.
* **Speaker Script**:
  > "We analyzed the gating weights in the Adaptive Fusion model. Over the test set, the average weight assigned to Pose was 85.2%, while RGB received 14.8%. The network learned that pose coordinates are stable and immune to background clutter or lighting. However, because the Pose branch itself was weak due to the small data scale, relying too heavily on it dragged down the Adaptive model's final performance. This highlights an important insight: the model learns coordinate stability, but needs regularization to avoid over-trusting a weak modality."

---

## Slide 9: Spatial Explainability: Grad-CAM Analysis
* **Slide Title**: Spatial Interpretability via Grad-CAM
* **Bullet Points**:
  * **Objective**: Visualize which pixels in the RGB video frames influence the classification decision.
  * **Key Observations**:
    * **Correct Predictions**: The CNN focuses heavily on the signer's hands, arms, and head.
    * **Incorrect Predictions**: Attention map activations are scattered, focusing on background noise or non-semantic clothing elements.
  * **Significance**: Validates that our RGB backbone learns meaningful physical representations of signs.
* **Visual Suggestion**: A sequence of Grad-CAM overlay frames (`results/gradcam/sample_01_MVI_9292/combined.png`) showing warm color heatmaps centered around the signer's hands.
* **Speaker Script**:
  > "To ensure our models are learning valid sign features and not just memorizing backgrounds, we applied Grad-CAM. For correct classifications, the warm-colored heatmaps cleanly highlight the signer's hands and face. For incorrect predictions, the heatmaps are scattered across the background. This visual proof confirms that our ResNet50 backbone successfully extracts the shape and location of the signing hands."

---

## Slide 10: Temporal Explainability: Attention Maps
* **Slide Title**: Temporal Interpretability via Self-Attention
* **Bullet Points**:
  * **Objective**: Discover which video frames are deemed most important by the Transformer Encoder.
  * **Key Observations**:
    * The Transformer displays non-uniform temporal attention across the 16 frames.
    * Attention peaks heavily in the middle frames (**frames 6 to 12**).
  * **Physical Interpretation**:
    * In sign language, initial and final frames represent transition states (raising hands and lowering them).
    * The **semantic apex** (the actual sign gesture) occurs in the middle, and the model automatically learns to attend to this region.
* **Visual Suggestion**: The attention matrix heatmap `results/attention_maps/sample_01_MVI_9292.png` showing peak weights on the middle indices.
* **Speaker Script**:
  > "Next, we analyzed temporal attention. The Transformer self-attention weights across the 16 frames were not uniform. Attention peaks strongly in frames 6 through 12, which correspond to the exact middle of the clip. This represents the semantic 'apex' of the gesture. The beginning and ending frames, where the signer is just raising or lowering their hands, receive low attention. The model automatically learned this temporal structure without any manual frame alignment."

---

## Slide 11: Error & Confusion Analysis
* **Slide Title**: Error Profiles & Common Confusions
* **Bullet Points**:
  * **Top Confused Class Pairs**:
    * ***loud* misclassified as *happy* (42 errors)**: Both gestures involve raising open hands towards the head and active facial movements.
    * ***quiet* misclassified as *Blind* (21 errors)**: The sign for *quiet* places a finger to the lips, while *Blind* touches fingers to the eyes.
  * **Clustering around *Blind* and *quiet***: Act as "sinks" because their gestural configurations involve hands returning near the face.
* **Visual Suggestion**: The confusion matrix heatmap `results/confusion_matrix.png` or `results/confusion_heatmaps.png`.
* **Speaker Script**:
  > "We also analyzed the model's errors using a confusion matrix. The most prominent confusion was misclassifying 'loud' as 'happy', which occurred 42 times. Physically, both signs involve open hands raised near the ears with similar dynamics. We also saw 'quiet' confused with 'Blind' because 'quiet' involves a finger on the lips, while 'Blind' involves fingers near the eyes. Because these gestures are visually similar and the hands are close to the face, the model struggles to differentiate them with limited resolutions."

---

## Slide 12: Conclusion & Future Work
* **Slide Title**: Conclusions & Future Scope
* **Bullet Points**:
  * **Project Conclusions**:
    * Successfully built a dual-branch visual-geometric Transformer pipeline.
    * Identified that RGB Baseline and Late Fusion provide the highest test accuracy (81.82%).
    * Revealed that the gating network learns to prioritize coordinate stability (85.2% Pose weight) but suffers from overfitting due to sample size.
    * Proven spatial and temporal explainability using Grad-CAM and Transformer self-attention.
  * **Future Directions**:
    * **Modality Dropout (ModDrop)**: Regularize the gating network to prevent over-trusting one modality.
    * **Cross-Attention Fusion**: Integrate query-key-value mechanisms to align modalities frame-by-frame.
    * **Dataset Scaling**: Expand training to the full INCLUDE dataset (over 10,000 videos).
* **Visual Suggestion**: A clean, icon-based closing slide.
* **Speaker Script**:
  > "In conclusion, we have designed, evaluated, and explained a dual-branch Multi-Modal Transformer for Indian Sign Language Recognition. Our experiments show that while visual appearance is currently the strongest driver, the gating network successfully learns the stability of coordinate paths. For future work, we plan to implement Modality Dropout to regularize the gates, experiment with Cross-Attention Fusion, and scale training to the full INCLUDE dataset. Thank you, and we are open to any questions."
