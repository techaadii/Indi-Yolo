# Indi-Yolo: IISERB Vehicle Multi-class Classification

A high-performance, modular machine learning pipeline for classifying vehicle types from cropped vehicle images. This project uses a fine-tuned **YOLO26s-cls** architecture to achieve state-of-the-art accuracy on the IISERB vehicle dataset.

---

## 🚀 Performance Summary

| Metric | Value |
|---------|------:|
| **Top-1 Accuracy** | **98.3%** |
| **Top-5 Accuracy** | **100.0%** |
| **Inference Speed** | **0.2 ms/image** (NVIDIA RTX A4000) |
| **Number of Classes** | **9** vehicle categories (IISERB standard) |

---

## 📂 Project Structure

The repository follows a modular design for scalability, reproducibility, and clean environment management.

```text
Indi-Yolo/
├── configs/                 # Centralized training configurations
├── src/
│   ├── data/                # Stratified data preparation & parsing
│   ├── models/              # YOLO trainer & WandB integration
│   └── main.py              # Execution entry point
├── pyproject.toml           # Environment & dependency definitions
├── uv.lock                  # Deterministic environment snapshot
└── README.md
```

---

## 🛠️ Installation & Setup

This project uses **uv** for fast and reproducible Python environment management.

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Indi-Yolo.git
cd Indi-Yolo
```

### 2. Sync the environment

```bash
uv sync
```

### 3. Login to Weights & Biases

```bash
uv run wandb login
```

---

## 📊 Dataset & Model Weights

The IISERB vehicle dataset cannot be publicly released due to institutional data ownership.

### Dataset

- **Status:** Not publicly available
- **Owner:** IISER Bhopal
- Images follow the naming convention:

```text
Camera_Date_Time_Frame_Label_Confidence.jpg
```

Example:

```text
Cam03_2024_04_12_000123_bus_0.56.jpg
```

### Fine-tuned Model

Download the trained weights (`best.pt`) from Google Drive.

- **Top-1 Accuracy:** 98.3%

---

## 🏗️ Usage

### Data Preparation

The preprocessing pipeline automatically:

- Extracts labels from filenames
- Performs an **80/20 stratified train-validation split**
- Preserves class balance for all vehicle categories

Place your raw vehicle crops in the directory specified in:

```text
configs/train_config.py
```

---

### Training

Run the complete pipeline (data preparation + training + WandB logging):

```bash
uv run train-yolo
```

---

### Inference

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")

results = model.predict("vehicle_crop.jpg")

predicted = results[0].names[results[0].probs.top1]
print(f"Predicted Class: {predicted}")
```

---

## 🧪 Methodology

### Label Extraction

Vehicle labels are automatically extracted from filenames using a custom parser targeting the second-to-last filename segment.

Example:

```text
Cam03_12345_bus_0.56.jpg
                  ↑
               Vehicle Class
```

---

### Stratified Splitting

An 80/20 stratified split ensures:

- Every vehicle class appears in both training and validation sets.
- Rare vehicle classes are preserved.
- Missing-class validation issues are avoided.

---

### Model Architecture

- **Backbone:** YOLO26s-cls
- Optimized for:
  - Low-latency inference
  - Traffic monitoring
  - Vehicle analytics
  - Edge deployment

---

## 📈 Results

| Metric | Score |
|---------|------:|
| Top-1 Accuracy | **98.3%** |
| Top-5 Accuracy | **100.0%** |
| Inference Speed | **0.2 ms/image** |

---

## 👤 Author

**Aditya Sinha**

- Initial Work
- Lead Developer

Project ownership: **IISER Bhopal**

---

## 📄 Notes

> **Dataset Availability**
>
> The IISERB vehicle dataset belongs to **IISER Bhopal** and therefore cannot be publicly distributed.
>
> This repository provides the training pipeline and fine-tuned model weights only. Users may train on their own datasets or use the released weights for inference where permitted.

---

## 📌 Model Weights

Local training output:

```text
/home/jataayu/CamSel/runs/classify/Indi-Yolo/train-4/weights/best.pt
```

Replace this local path with the Google Drive download link when releasing the repository.
