# Installation

Clone the repository:

```bash
git clone https://github.com/SHUBH2029/Fruit-recognition-using-vgg-16.git

cd Fruit-recognition-using-vgg-16
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# How to Run

## Option 1: Run on Kaggle (Recommended)

### Step 1: Download the Dataset

Download the Fruit Recognition Dataset from Kaggle:

https://www.kaggle.com/datasets/chrisfilo/fruit-recognition

### Step 2: Create a Kaggle Notebook

1. Login to Kaggle.
2. Click **Code → New Notebook**.
3. Enable **GPU Accelerator** from Notebook Settings.
4. Click **Add Input**.
5. Search for **Fruit Recognition Dataset**.
6. Add the dataset to the notebook.

### Step 3: Upload the Project Notebook

Upload:

```text
notebook/notebook70662a07fd.ipynb
```

or copy the notebook code into a new Kaggle notebook.

### Step 4: Run the Notebook

Run all notebook cells sequentially:

```text
Runtime → Run All
```

The notebook will automatically:

* Load the dataset
* Perform preprocessing
* Train FruitNet
* Generate evaluation metrics
* Generate GradCAM visualizations
* Perform bias analysis
* Save figures and reports

### Generated Outputs

```text
figures/
├── accuracy_curve.png
├── training_loss.png
├── validation_loss.png
├── confusion_matrix.png
├── roc_curves.png
├── final_metrics.png

gradcam/
├── gradcam_29398.png
├── gradcam_report.txt

bias_analysis/
├── class_imbalance.png
├── prediction_bias.png
├── color_bias.png
├── background_bias.png
├── top_confusions.png

reports/
├── classification_report.csv
├── final_metrics.csv
├── final_results.txt
└── project_conclusion.txt
```

---

## Option 2: Run Locally using Jupyter Notebook

### Launch Jupyter

```bash
jupyter notebook
```

### Open the Notebook

Navigate to:

```text
notebook/notebook70662a07fd.ipynb
```

### Execute the Notebook

Run all cells from top to bottom.

The notebook will:

1. Load and preprocess the dataset
2. Initialize FruitNet Architecture
3. Train using CAF and FDL
4. Evaluate model performance
5. Generate visualizations
6. Save reports and metrics

---

# Reproducibility

The project ensures reproducible results through:

* Fixed random seed initialization
* Deterministic CUDA operations
* Consistent dataset splitting
* Phase-wise transfer learning strategy
* Controlled training configuration

---

# Hardware Requirements

Recommended Configuration:

* NVIDIA GPU (T4 / RTX Series)
* CUDA-enabled PyTorch
* Minimum 8 GB GPU Memory
* 16 GB System RAM
* Python 3.10+

Training was performed using Kaggle GPU resources.
