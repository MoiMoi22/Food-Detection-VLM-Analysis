# Food Detection & Analysis System with VLM Enrichment 🍱

This project focuses on building a comprehensive pipeline for food detection, leveraging Vision-Language Models (VLM) to enrich dataset metadata and deploying a demonstration system.

## 📂 Project Structure

This repository is organized into specific notebooks tackling different stages of the MLOps pipeline:

### 1. Data Preparation & Enhancement
- **`Vison_LLM_Prompt.ipynb`**: Utilizes a Vision-Language Model (VLM) to extract rich metadata and context from food images, enhancing the dataset beyond simple class labels.
- **`Preprocess_annotation.ipynb`**: Handles COCO format cleaning, performs **stratified splitting** (Train/Val/Test) to ensure class balance, and integrates negative background samples to reduce False Positives.
- **`Data_Exploration.ipynb`**: Visualizes the distribution of food attributes and class imbalances across the merged and separated datasets.

### 2. Modeling & Training
- **`Train.ipynb`**: Contains the training loop, loss visualization, and evaluation metrics (mAP, Confusion Matrix) on the test set.

### 3. Deployment
- **`Final_System.ipynb`**: A deployment demo script that launches the model inference server using **ngrok** for remote access.

## 🛠️ Tech Stack
- **Platform**: Google Colab
- **Core**: Python, PyTorch, NumPy, Pandas
- **Computer Vision**: YOLOV11
- **VLM**: Qwen/Qwen2-VL-7B-Instruct
- **Tools**: Matplotlib (Visualization), Ngrok (Tunneling)

## 🚀 How to Run
1. Clone the repository (Should be implemented on Google Colab).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt