# 🩺 Breast Cancer Classification Using Deep Learning

A deep learning project that classifies breast cancer tumours as **Malignant** or **Benign** using the Breast Cancer Wisconsin Diagnostic dataset. This project explores neural networks and techniques to improve model generalisation, including Early Stopping, Dropout, L1 Regularization, L2 Regularization, and ElasticNet (L1 + L2).

## 📌 Project Overview

Breast cancer classification is an important medical machine learning application. The goal of this project is to build, train, evaluate, and compare different neural network architectures to understand their classification performance and ability to generalise to unseen data.

The project progresses from a simple baseline model to a final combined model using regularization and overfitting-control techniques.

### 🎯 Objectives

- 📊 Explore and understand the dataset.
- 🧹 Preprocess and standardize input features.
- 🧠 Build a Single-Layer Perceptron (SLP) and Multi-Layer Perceptron (MLP).
- ⚙️ Compare ReLU, tanh, and sigmoid activation functions.
- 🛑 Apply Early Stopping to reduce unnecessary training.
- 🎲 Investigate Dropout rates of 0.1, 0.3, and 0.5.
- 🔒 Compare L1, L2, and L1-L2 regularization.
- 📈 Evaluate models using accuracy, precision, recall, and F1-score.
- 🩺 Discuss the importance of false negatives in medical diagnosis support.

## 📂 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic dataset** available through `scikit-learn`.

| Property | Description |
|---|---|
| Total samples | 569 |
| Input features | 30 |
| Target classes | 2 |
| Malignant class | `0` |
| Benign class | `1` |
| Training samples | 455 |
| Testing samples | 114 |

The 30 numerical features describe characteristics of cell nuclei derived from digitized images of breast mass samples.

### 🎯 Target Labels

- 🔴 **Malignant (`0`)** — cancerous tumour.
- 🟢 **Benign (`1`)** — non-cancerous tumour.

The dataset contains 212 malignant samples and 357 benign samples.

## 🛠️ Technologies Used

- 🐍 Python
- 📓 Jupyter Notebook
- 🧮 NumPy and Pandas
- 📊 Matplotlib and Seaborn
- 🤖 Scikit-learn
- 🧠 TensorFlow and Keras

## ⚙️ Project Workflow

### 1️⃣ Data Loading and Exploratory Data Analysis

- Loaded the dataset using `load_breast_cancer(as_frame=True)`.
- Examined the feature dimensions and target distribution.
- Visualized the class distribution.
- Created a feature correlation heatmap.
- Observed strong correlations among features such as `radius_mean`, `perimeter_mean`, and `area_mean`.

### 2️⃣ Data Preprocessing

The dataset was split into training and testing sets using an 80:20 ratio, with stratification to preserve the class distribution.

`StandardScaler` was fitted only on the training data and then applied to both training and testing features. This prevents information from the test set from influencing the scaling parameters.

### 3️⃣ Single-Layer Perceptron (SLP)

A simple neural network with one sigmoid output neuron was used as the baseline model.

**Purpose:** Establish a baseline against which more complex neural networks could be compared.

### 4️⃣ Multi-Layer Perceptron (MLP)

An MLP with two hidden layers and ReLU activation was developed to learn more complex, nonlinear relationships.

Additional experiments compared ReLU, tanh, and sigmoid activation functions.

### 5️⃣ Early Stopping

Early Stopping monitored validation loss and stopped training when it failed to improve for a specified number of epochs.

The configuration used:

- `monitor="val_loss"`
- `patience=20` for the relevant experiments
- `restore_best_weights=True`

This helps prevent unnecessary training and can reduce overfitting.

### 6️⃣ Dropout Regularization

Dropout layers randomly disable a fraction of neuron activations during training.

The project compared dropout rates of:

- `0.1` — 10% dropout
- `0.3` — 30% dropout
- `0.5` — 50% dropout

A dropout rate of `0.3` was used in the final combined architecture.

### 7️⃣ Weight Regularization

Three regularization approaches were explored:

- **L1 Regularization:** Encourages sparse weights by penalizing the absolute values of weights.
- **L2 Regularization:** Penalizes large weights, encouraging smaller parameter values.
- **L1-L2 (ElasticNet):** Combines L1 and L2 penalties.

The L2 experiments used a regularization coefficient of `0.001`. The ElasticNet experiment used `l1=0.0001` and `l2=0.001`.

### 8️⃣ Final Combined Model

The final model combines L2 regularization, Dropout, and Early Stopping.

**Architecture:**

`Dense(128, ReLU, L2) → Dropout(0.3) → Dense(64, ReLU, L2) → Dropout(0.3) → Dense(1, Sigmoid)`

Configuration:

- Optimizer: Adam
- Loss function: Binary cross-entropy
- Metric: Accuracy
- Maximum epochs: 300
- Batch size: 32
- Validation split: 10%
- Early Stopping patience: 20
- Restore best weights: Enabled

## 📊 Model Comparison Results

The following results were obtained in the notebook using the same 114-sample test set.

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|---|---:|---:|---:|---:|
| SLP (Task 2) | 93.86% | 94.52% | 95.83% | 95.17% |
| MLP-ReLU (Task 3) | 95.61% | 98.55% | 94.44% | 96.45% |
| MLP + Early Stopping (Task 4) | 95.61% | 98.55% | 94.44% | 96.45% |
| MLP + Dropout (Task 5) | 95.61% | 98.55% | 94.44% | 96.45% |
| MLP + L2 (Task 6) | 95.61% | 98.55% | 94.44% | 96.45% |
| Final Combined Model (Task 7.1) | 95.61% | 98.55% | 94.44% | 96.45% |

**Note:** Precision, recall, and F1-score in this table treat class `1` (benign) as the positive class, following the default `scikit-learn` metric settings. These metrics do not directly represent sensitivity for detecting malignant tumours.

### 🔍 Key Findings

- 📈 The MLP models achieved higher test accuracy than the baseline SLP in the recorded results.
- 🧠 Hidden layers allow the MLP to learn more complex relationships in the data.
- 🛑 Early Stopping, Dropout, and regularization provide different ways to control overfitting.
- ⚖️ The tested MLP variants produced identical rounded test metrics in this experiment.
- 🧪 These results do not establish that one regularization technique is superior. Additional validation is required to determine whether the techniques improve generalisation consistently.

## 🩺 Clinical Insight for Medical Diagnosis Support

For cancer diagnosis support, **false negatives are particularly important** because a malignant tumour classified as benign could delay further medical investigation.

Model selection should therefore consider:

- **Malignant-class recall (sensitivity):** How many malignant cases are correctly detected.
- **False-negative rate:** How many malignant cases are incorrectly classified as benign.
- **Precision:** How many cases flagged as malignant are actually malignant.
- **Specificity:** How many benign cases are correctly identified.
- **Threshold selection:** A lower threshold for predicting malignancy may improve sensitivity, but can increase false positives.

Because the model's sigmoid output represents the probability of class `1` (benign), a case can be flagged as potentially malignant when its predicted benign probability falls below a suitably validated threshold.

The threshold should be selected using validation data and clinical risk requirements, rather than choosing it arbitrarily.

⚠️ **Clinical limitation:** This project is an educational machine learning experiment. The models have not been established as clinically validated diagnostic systems. Independent evaluation, malignant-class sensitivity analysis, calibration assessment, and clinical validation would be required before any real-world medical use.

## 📁 Project Structure

```text
Breast-Cancer-Classification/
│
├── Breast_cancer_diagnostic.ipynb   # Complete analysis and experiments
└── README.md                        # Project documentation
```

## 🏁 Conclusion

This project demonstrates a complete deep learning workflow for binary breast cancer classification, from exploratory data analysis and preprocessing to model comparison and regularization.

The MLP models achieved **95.61% test accuracy** in the recorded experiment, compared with **93.86% for the SLP baseline**. The final combined model brings together L2 regularization, Dropout, and Early Stopping, but its identical test metrics do not demonstrate a measurable improvement over the other MLP variants.

The main lesson is that model evaluation should go beyond accuracy. For a medical decision-support application, reliable malignant-case detection, careful threshold selection, and independent clinical validation are essential.

---

⭐ If you find this project useful, consider starring the repository and sharing feedback!

**Disclaimer:** This project is for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment.
