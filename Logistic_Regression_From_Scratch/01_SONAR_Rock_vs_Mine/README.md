# SONAR Rock vs Mine Prediction
### Logistic Regression from Scratch — No ML Libraries

---

## Overview

This is my first Machine Learning project, where I built a **Logistic Regression model completely from scratch** using only NumPy — no sklearn, no ready-made models, just pure math.

The goal is to classify sonar signals as either a **Rock** or a **Mine** based on 60 frequency features returned from sonar waves.

---

## What I Built

- Implemented the **Sigmoid function** manually
- Implemented **Binary Cross-Entropy** cost function
- Implemented **Gradient Descent** from scratch
- Applied **Min-Max Scaling** manually (without sklearn)
- Applied **Train/Test Split** manually (without sklearn)
- Added a **bias term** by appending a column of ones to the feature matrix

---

## Project Structure

```
01_SONAR_Rock_vs_Mine/
├── Model.py           # Logistic Regression class (from scratch)
├── Prep_Data.ipynb    # Data loading, EDA, preprocessing, training
└── sonar.csv          # Dataset
```

---

## Dataset

- **Source:** UCI Machine Learning Repository — SONAR Dataset
- **Samples:** 208
- **Features:** 60 (sonar frequency readings)
- **Labels:** R (Rock) or M (Mine)

---

## Results

| Metric | Value |
|--------|-------|
| Mean Test Accuracy | **83.33%** |
| Std across 10 runs | **0.0000** |
| Learning Rate (eta) | 0.01 |
| Epochs | 10,000 |

The model is fully stable — same result every run, which confirms that Logistic Regression has a **convex cost function** with a single global minimum.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/youssif-2005/Machine-Learning-Projects.git
```

2. Navigate to the project folder:
```bash
cd Machine-Learning-Projects/Logistic_Regression_From_Scratch/01_SONAR_Rock_vs_Mine
```

3. Open `Prep_Data.ipynb` in Jupyter or Google Colab and run all cells.

---

## Author

**Youssif Sherif** — Building ML from the ground up, one algorithm at a time.
