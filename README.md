# EEG-model

# EEG Category Classification from Brain Signals 🧠📊

This repository presents a solution for classifying image categories based on EEG signals recorded while subjects were viewing images. The goal was to predict one of five semantic categories (animal, food, clothing, tool, vehicle) from raw EEG data.

## 🎯 Task Overview

- **Input**: EEG time series (channels × time steps)
- **Output**: Image category label (one of 5 classes)
- **Samples**:
  - Train: 118,800
  - Validation: 59,400
  - Test: 59,400
- **Metric**: Top-1 Accuracy
- **Dataset**: Modified version of [Gifford2022 EEG Dataset](https://...optional-link)

In contrast to the original dataset, this competition uses a simplified 5-class version and ensures that the test data corresponds to images *seen in the training set* to avoid zero-shot classification.

---

## 🧠 Key Innovations

### 1. Subject-wise Normalization
EEG signals differ significantly between subjects. We introduced a custom `SubjectNormalizer` class that normalizes EEG data based on subject-wise mean and variance, helping to reduce inter-subject distribution shifts.

### 2. Time-Series Specific Augmentation
Using the `tsaug` library, we applied:
- **TimeWarp**
- **AddNoise**
- **Dropout**
- **Reverse**

to improve robustness. Test-Time Augmentation (TTA) further reduced sensitivity to signal noise.

### 3. Feature Engineering with Multi-Scale Temporal CNN + Attention
- **MultiScaleConvBlock** captures both short-term and long-term EEG patterns.
- **ChannelSEAttention** learns to weigh physiologically relevant EEG channels dynamically.

### 4. Conformer Backbone
We adapted the **Conformer** architecture [Gulati et al., 2020] which combines:
- Self-attention (for long-range dependencies)
- Depthwise CNN (for local features)
- Residual and LayerNorm connections for stable learning.

This allowed the model to simultaneously learn hierarchical temporal structures in EEG.

### 5. Optuna-based Hyperparameter Tuning
Using [Optuna](https://optuna.org/), we automatically searched for:
- Network depth
- Hidden sizes
- Learning rates
- Label smoothing coefficients

to maximize validation accuracy.

### 6. Stratified K-Fold Ensemble with TTA
We used **Stratified K-Fold (Fold0, Fold1)** training and **TTA-based ensemble prediction** to maximize generalization.

---

## 🏆 Results

- Achieved **Top-1 Accuracy: 0.53406** on the public evaluation set.
- Note: The true test set is held by the competition organizers, so this score reflects validation performance based on the public split.

This performance exceeds standard CNN and EEGNet baselines, demonstrating the strength of subject-specific normalization, multi-scale features, and Conformer-based modeling.


---

## 🎓 Relevance to Graduate Research

This project demonstrates my ability to:

- Work with real EEG time-series data and domain challenges.
- Implement cutting-edge deep learning models (e.g., Conformer).
- Apply subject-specific signal processing and regularization.
- Build and evaluate robust classification pipelines for neuro-related tasks.

I aim to pursue graduate study in:
- Computational Neuroscience
- Brain-Computer Interfaces (BCI)
- Cognitive AI
- Machine Learning for Neuroimaging

---

## 🗂️ Repository Structure

