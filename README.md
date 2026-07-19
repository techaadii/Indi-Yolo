Indi-Yolo: IISERB Vehicle Multi-class Classification
![alt text](https://img.shields.io/badge/Python-3.14%2B-blue)

![alt text](https://img.shields.io/badge/managed%20by-uv-arc.svg)

![alt text](https://img.shields.io/badge/Weights%20%26%20Biases-Logged-yellow)
A high-performance, modular machine learning pipeline for classifying vehicle types from crops. This project uses a fine-tuned YOLO26s-cls architecture to achieve state-of-the-art accuracy on the IISERB vehicle dataset.
🚀 Performance Summary
Top-1 Accuracy: 98.3%
Top-5 Accuracy: 100.0%
Inference Speed: 0.2ms per image (NVIDIA RTX A4000)
Classes: 9 vehicle categories (IISERB standard)
📂 Project Structure
The project follows a modular design for scalability and clean environment management.
code
Text
Indi-Yolo/
├── configs/             # Centralized training configurations
├── src/
│   ├── data/            # Stratified data preparation & parsing
│   ├── models/          # YOLO trainer & WandB integration
│   └── main.py          # Execution entry point
├── pyproject.toml       # Environment & dependency definitions
├── uv.lock              # Deterministic environment snapshot
└── README.md            # Documentation
🛠️ Installation & Setup
This project uses uv for extremely fast and reproducible environment management.
Clone the repository:
code
Bash
git clone https://github.com/yourusername/Indi-Yolo.git
cd Indi-Yolo
Sync the environment:
code
Bash
uv sync
Login to Weights & Biases:
code
Bash
uv run wandb login
📊 Dataset & Model Weights
Due to size and privacy constraints, the raw dataset and weights are hosted externally.
Dataset (IISERB Crops): Download from Google Drive
Format: Images named as Camera_Date_Time_Frame_Label_Confidence.jpg
Fine-tuned Weights (best.pt): Download from Google Drive
Accuracy: 98.3%
🏗️ Usage
1. Data Preparation
The pipeline automatically parses class labels from the filenames and creates a stratified split (80/20) to ensure class balance.
Source Folder: Place your raw crops in the folder specified in configs/train_config.py.
2. Training
To run the full training pipeline (Preparation + Training + Logging):
code
Bash
uv run train-yolo
3. Inference / Prediction
code
Python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict("vehicle_crop.jpg")
print(f"Predicted Class: {results[0].names[results[0].probs.top1]}")
🧪 Methodology
Label Extraction: Labels are extracted using a custom regex/split parser that targets the second-to-last segment of the filename (e.g., ..._bus_0.56.jpg).
Stratified Splitting: Logic ensures that even rare vehicle classes are represented in both Training and Validation sets to prevent "missing class" errors.
Architecture: YOLO-cls backbone optimized for low-latency inference in traffic monitoring scenarios.
👤 Author
Aditya S - Initial Work / Lead Developer
Project ownership: IISERB
Links for your final steps:
Model Weights Path: /home/jataayu/CamSel/runs/classify/Indi-Yolo/train-4/weights/best.pt


**Note: Dataset belongs to IISERB hence we are sharing the finetuned model not the data
