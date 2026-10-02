# MNIST Handwritten Digit Recognition

A machine learning project for recognizing handwritten digits (0–9) using the **MNIST Digit Recognizer dataset**. The project implements both a **Random Forest baseline model** and a **Convolutional Neural Network (CNN)**, followed by detailed evaluation using accuracy, precision, recall, F1-score, and a confusion matrix.

This project was completed as **Task 2 of Month 1** during my **Machine Learning Internship at Arch Technologies**.

---

## 📌 Project Overview

Handwritten digit recognition is a fundamental computer vision and machine learning problem. The objective of this project is to train a model that can identify handwritten digits from grayscale images of size **28 × 28 pixels**.

Two approaches were implemented:

1. **Random Forest Classifier** — used as a baseline machine learning model.
2. **Convolutional Neural Network (CNN)** — used as the main deep learning model for image classification.

The CNN achieved a **98.94% validation accuracy** on the validation set.

---

## 🎯 Objectives

* Load and explore the MNIST handwritten digit dataset.
* Perform exploratory data analysis (EDA).
* Visualize handwritten digit samples.
* Preprocess and normalize image data.
* Split the dataset into training and validation sets.
* Build a Random Forest baseline classifier.
* Build and train a CNN for digit classification.
* Compare model performance.
* Evaluate the CNN using multiple classification metrics.
* Generate a confusion matrix.
* Analyze model predictions.
* Save the trained model and evaluation results.

---

## 📊 Dataset

The project uses the **MNIST Digit Recognizer dataset** obtained from Kaggle.

### Dataset Characteristics

* Image size: **28 × 28 pixels**
* Number of classes: **10**
* Classes: **0–9**
* Pixel values: **0–255**
* Labeled samples used: **42,000**
* Training split: **33,600 samples**
* Validation split: **8,400 samples**

The pixel values were normalized from the range **0–255** to **0–1** before training the CNN.

### Dataset Format

The training dataset contains:

* `label` — target digit
* `pixel0` to `pixel783` — grayscale pixel values

The dataset itself is **not included in this repository**.

---

## 🛠️ Technologies & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Google Colab / Jupyter Notebook

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Exploratory Data Analysis
   ↓
Image Visualization
   ↓
Data Preprocessing
   ↓
Train / Validation Split
   ↓
Random Forest Baseline
   ↓
CNN Model
   ↓
Model Training
   ↓
Evaluation
   ↓
Confusion Matrix
   ↓
Prediction Analysis
   ↓
Model & Results Export
```

---

## 🔍 Exploratory Data Analysis

The project performs several EDA steps, including:

* Checking dataset dimensions
* Checking missing values
* Checking duplicate records
* Checking pixel value range
* Analyzing the distribution of digits
* Visualizing sample handwritten digits

These steps help understand the structure and quality of the dataset before model training.

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

### 1. Separate Features and Labels

The `label` column was separated from the pixel features.

### 2. Normalize Pixel Values

Pixel values were divided by `255.0`:

```python
X = train_df.drop('label', axis=1).values / 255.0
```

This converts pixel values from:

```text
0–255
```

to:

```text
0–1
```

### 3. Reshape Images

The flattened 784-pixel vectors were reshaped into:

```text
28 × 28 × 1
```

to make them suitable for CNN processing.

### 4. Train/Validation Split

The dataset was divided using an **80/20 stratified split**:

```text
Training Set:   33,600 images
Validation Set: 8,400 images
```

---

# 🌲 Baseline Model — Random Forest

A Random Forest classifier was implemented as the baseline model.

### Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Because Random Forest expects tabular input, the images were flattened before training.

### Baseline Result

**Validation Accuracy: 96.39%**

This baseline provides a useful comparison against the CNN model.

---

# 🧠 Main Model — Convolutional Neural Network

A CNN was developed using TensorFlow/Keras for image classification.

### Architecture

```text
Input: 28 × 28 × 1
        ↓
Conv2D — 32 filters
        ↓
MaxPooling2D
        ↓
Conv2D — 64 filters
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense — 128 neurons
        ↓
Dropout — 0.3
        ↓
Dense — 10 neurons
        ↓
Softmax Output
```

### Model Compilation

The model uses:

* **Optimizer:** Adam
* **Loss:** Sparse Categorical Crossentropy
* **Metric:** Accuracy

### Early Stopping

Early stopping was used to monitor validation loss and restore the best model weights:

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

---

# 📈 Training Performance

Training and validation accuracy were monitored during CNN training.

Training and validation loss were also visualized to observe model learning and identify potential overfitting.

The notebook includes graphs for:

* Training vs. validation accuracy
* Training vs. validation loss

---

# 📊 Model Evaluation

The CNN was evaluated on the validation dataset using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

### Final CNN Results

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **98.94%** |
| Precision | **98.94%** |
| Recall    | **98.94%** |
| F1-Score  | **98.94%** |

These are **validation-set results** from the project.

---

## 🔄 Model Comparison

| Model         | Validation Accuracy |
| ------------- | ------------------: |
| Random Forest |          **96.39%** |
| CNN           |          **98.94%** |

The CNN achieved higher validation accuracy than the Random Forest baseline on this dataset split.

---

# 📉 Confusion Matrix

A confusion matrix was generated to examine the classification performance for each digit class from **0 to 9**.

It provides a visual representation of:

* Correct predictions
* Misclassified digits
* Which digit classes are more frequently confused

The confusion matrix is included in the project notebook/results.

---

# 🔎 Prediction Analysis

The project also visualizes model predictions by displaying handwritten digit images together with their:

```text
True Label
Predicted Label
```

This provides a visual way to inspect how the trained CNN performs on individual validation samples.

---

# 💾 Model & Results Export

The trained CNN model was saved using the Keras format:

```text
mnist_digit_recognition_model.keras
```

Evaluation metrics were exported to:

```text
mnist_evaluation_results.csv
```

---

# 📁 Repository Structure

```text
MNIST-Digit-Recognition/
│
├── README.md
├── requirements.txt
│
├── MNIST_Digit_Recognition.ipynb
│
├── results/
│   ├── training_accuracy_loss.png
│   ├── confusion_matrix.png
│   ├── predictions.png
│   └── mnist_evaluation_results.csv

```

> The Kaggle dataset is not included in this repository. Users should download the dataset separately and place the required CSV file in the appropriate data location.

---

# 📌 Key Project Highlights

* Implemented a complete handwritten digit classification pipeline.
* Performed exploratory data analysis and visualization.
* Applied image normalization and reshaping.
* Built a Random Forest baseline model.
* Developed a CNN using TensorFlow/Keras.
* Used early stopping during CNN training.
* Achieved **98.94% validation accuracy** with the CNN.
* Evaluated the model using multiple classification metrics.
* Generated a confusion matrix for detailed error analysis.
* Saved the trained model and evaluation results.

---

# 🔮 Future Improvements

Possible extensions of this project include:

* Data augmentation
* Experimenting with deeper CNN architectures
* Hyperparameter tuning
* Testing additional machine learning/deep learning models
* Deploying the trained model as a web application
* Creating an interactive handwritten-digit prediction interface


# 👩‍💻 Author

**Laiba Qamar**

BSCS Student | Machine Learning Intern
Interested in Artificial Intelligence, Machine Learning, Generative AI, and practical AI applications.

---

## 📚 Acknowledgements

* Kaggle — MNIST Digit Recognizer Dataset
* TensorFlow / Keras
* Scikit-learn
* Arch Technologies — Machine Learning Internship
