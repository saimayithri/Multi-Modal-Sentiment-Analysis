# 🧠 Multimodal Sentiment Analysis (MSA)
## Combatting Text Fragility via Adaptive Modality Balancing

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

Welcome to the official repository for our research on **Multimodal Sentiment Analysis**. 

Current state-of-the-art MSA models heavily over-rely on the language modality (text). If the text transcript contains errors (e.g., ASR failures, missing words, or noise), the entire model collapses, completely ignoring perfectly valid audio and visual cues. 

This project introduces a **3-Stage Hybrid Learning Framework** that forces the model to learn and utilize all three modalities (Text, Audio, Vision) evenly, creating a highly robust system that survives severe text degradation.

---

## 🚀 Novel Architecture Highlights

Our framework introduces three key mechanisms to prevent text-dominance:

1. **Adaptive Curriculum Modality Dropout:** Dynamically tracks the text encoder's confidence during training. When text becomes "too easy" to classify, the model dynamically drops text tokens, forcing the network to learn from audio and vision.
2. **On-the-fly Gradient Modulation (OGM-GE):** Monitors the gradients of each modality. If the text gradients dominate the backward pass, they are dynamically scaled down to prevent the text encoder from overfitting before the other modalities converge.
3. **Inter-Modal Contrastive Alignment (InfoNCE):** Pulls the dense representations of Audio and Vision closer to the highly semantic Text representation in the latent space, teaching the A/V encoders to understand sentiment without needing text.
4. **Learnable Gating Fusion:** A dynamic attention mechanism that learns to weigh the importance of each modality on a per-sample basis before final classification.

---

## 📂 Repository Structure

```text
.
├── datasets/            # Dataloaders for CMU-MOSI and alignment logic
├── models/              # Neural network architectures
│   └── msamodel.py      # Core Hybrid Fusion architecture & Gating
├── modules/             # Individual Encoders (Transformers, LSTMs)
├── run/                 # Execution scripts
│   ├── download_data.py # Google Drive dataset downloader
│   ├── train.py         # 3-Stage Training Pipeline
│   ├── robustness_test.py # Evaluation suite for text corruption
│   └── visualize.py     # t-SNE and attention weight visualizations
├── src/                 # Utilities and evaluation metrics
├── MSA_Research.ipynb   # 1-Click Google Colab Training Notebook
└── requirements.txt     # Python dependencies
```

---

## 🛠️ Quick Start (Google Colab)

The easiest way to reproduce our results is using Google Colab. 

1. Open `MSA_Research_Colab.ipynb` in Google Colab.
2. Ensure you have a GPU runtime enabled (`Runtime > Change runtime type > T4 GPU`).
3. Run **Section 1 & 2** to clone the repo and download the `CMU-MOSI` dataset.
4. Run the **3-Stage Training Pipeline** sequentially.
5. Run the **Robustness Evaluation** to generate the degradation plots.

---

## 💻 Local Setup & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/saimayithri/Multi-Modal-Sentiment-Analysis.git
cd Multi-Modal-Sentiment-Analysis
pip install -r requirements.txt
```

### 2. Download Dataset
You need the CMU-MOSI dataset (`unaligned_50.pkl` or `aligned_50.pkl`). Place the `.pkl` file in the `data/` directory.

### 3. Training Pipeline
To achieve maximum robustness, models must be trained in 3 distinct stages:

**Stage 1: Train Text-Only Teacher**
```bash
python run/train.py --stage 1 --data_path data/mosi_data.pkl --num_epochs 40 --lr 1e-4
```

**Stage 2: Knowledge Distillation (Train Audio/Vision)**
```bash
python run/train.py --stage 2 --data_path data/mosi_data.pkl --num_epochs 40 --lr 1e-4
```

**Stage 3: Full Hybrid Fine-Tuning (with Novel Modulators)**
```bash
python run/train.py --stage 3 \
    --data_path data/mosi_data.pkl \
    --use_ogm \
    --use_contrastive \
    --use_adaptive_dropout \
    --modality_dropout 0.5 \
    --final_model_path models/robust_hybrid_best.pt
```

### 4. Evaluate Robustness
To prove the model's resistance to text degradation, run our robustness suite which injects Gaussian Noise and Token Dropout at various intensities:
```bash
python run/robustness_test.py \
    --model_path models/robust_hybrid_best.pt \
    --data_path data/mosi_data.pkl
```
*This will generate benchmark plots and degradation tables in the `figures/` directory.*

---

## 📊 Results & Performance

### 1. Robustness vs SOTA
State-of-the-art models like MISA (2020) and MFM push clean-text accuracy on CMU-MOSI to ~84-85%. However, they do so by hyper-optimizing for the text modality. When text noise is introduced, these models suffer catastrophic failure.

Our framework trades a small amount of clean-text accuracy for a massive gain in real-world robustness:
* **Clean-Text Accuracy:** 81.01% (Validation) / 76.8% (Test Set)
* **Robustness:** Under **severe text corruption (50% token dropout)**, our Balanced Hybrid model degrades significantly less than traditional text-dominant baselines, relying on its strong audio and visual representations to maintain accurate sentiment predictions.

*(Note: Validation accuracy is used for early stopping and hyperparameter tuning, while Test accuracy is reported in our JSON logs for strict evaluation).*

Check the generated `figures/robustness_curves.png` for visual proof of this degradation gap.
