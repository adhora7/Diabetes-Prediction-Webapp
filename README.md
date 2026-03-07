# 🩺 Diabetes Prediction Web App

A machine learning web application that predicts whether a person is diabetic based on diagnostic health measurements. Built using a Support Vector Machine (SVM) classifier trained on the PIMA Indian Diabetes Dataset and deployed via Streamlit Cloud.

 **Live App:** https://diabetes-prediction-webapp-zh.streamlit.app

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Local Setup](#local-setup)
- [Full Deployment Guide](#full-deployment-guide)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

---

## About the Project

This project demonstrates a complete end-to-end machine learning workflow:

1. **Data Analysis** — Exploratory analysis on the PIMA Indian Diabetes Dataset
2. **Model Training** — SVM classifier with StandardScaler preprocessing
3. **Web App** — Interactive Streamlit interface for real-time predictions
4. **Cloud Deployment** — Live deployment on Streamlit Community Cloud via GitHub

---

## Dataset

**PIMA Indian Diabetes Dataset** — from the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function score |
| Age | Age in years |
| Outcome | 1 = Diabetic, 0 = Not Diabetic (target) |

---

## Project Structure

```
Diabetes-Prediction-Webapp/
│
└── README.md
└── Prediction_of_Diabetes.ipynb
├── diabetes.csv    # Main Streamlit web app
├── diabetes_prediction_webapp.py   # PIMA Indian Diabetes Dataset
├──  predictive_system.py           # Python dependencies
└── requirements.txt                # Project documentation
```

---

## How It Works

The app uses `@st.cache_resource` to train the SVM model automatically on first load — no pre-saved pickle file needed. Here's the full pipeline:

```
diabetes.csv  →  StandardScaler  →  SVM (linear kernel)  →  Streamlit UI  →  Prediction
```

1. Dataset is loaded from `diabetes.csv`
2. Features are standardized using `StandardScaler`
3. An SVM classifier with a linear kernel is trained (80/20 train-test split)
4. The model is cached using `@st.cache_resource` so it trains only once
5. User inputs are scaled and passed to the model for prediction

**Model Accuracy:**
- Training: ~78%
- Testing: ~77%

---

## Local Setup

### Prerequisites
- Python 3.10+
- Anaconda (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/adhora7/Diabetes-Prediction-Webapp.git
cd Diabetes-Prediction-Webapp
```

### Step 2: Create a Conda Environment

```bash
conda create -n mlapp python=3.10 -y
conda activate mlapp
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the App

```bash
streamlit run diabetes_prediction_webapp.py
```

Open your browser at `http://localhost:8501` 

---

## Full Deployment Guide

Follow these steps to deploy your own version of this app on Streamlit Community Cloud.

### Step 1: Create a GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click **"New"** (green button) to create a new repository
3. Name it (e.g. `diabetes-prediction-webapp`)
4. Set visibility to **Public**
5. Click **"Create repository"**

### Step 2: Upload Your Files to GitHub

Make sure your repo contains these files:

| File | Required |
|---|---|
| `diabetes.csv` | ✅ Yes |
| `diabetes_prediction_webapp.py` | ✅ Yes |
| `requirements.txt` | ✅ Yes |
| `README.md` | Optional but recommended |

To upload files:
1. On your repo page click **"Add file"** → **"Upload files"**
2. Drag and drop all your files
3. Click **"Commit changes"**

### Step 3: Verify `requirements.txt`

Your `requirements.txt` must contain exactly:

```
pandas
scikit-learn
streamlit==1.43.0
```

> ⚠️ **Important:** Always pin the streamlit version (e.g. `streamlit==1.43.0`). Without pinning, Streamlit Cloud may install an incompatible version and your app will crash.

### Step 4: Deploy on Streamlit Community Cloud

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Fill in the form:
   - **Repository:** `your-username/diabetes-prediction-webapp`
   - **Branch:** `main`
   - **Main file path:** `diabetes_prediction_webapp.py`
5. Click **"Deploy!"**

### Step 5: Wait for Build to Complete

Streamlit Cloud will:
- Clone your repository
- Install packages from `requirements.txt`
- Launch your app

First deployment takes **3–5 minutes**. After that, updates are reflected in seconds.

### Step 6: Share Your App

Once deployed, you'll get a public URL like:
```
https://your-app-name.streamlit.app
```

Share this link with anyone — no installation required on their end!

---

## Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'sklearn'`
Your `requirements.txt` is missing `scikit-learn`. Add it and redeploy.

### ❌ `ModuleNotFoundError: No module named 'altair.vegalite.v4'`
Streamlit version is too old. Pin it in `requirements.txt`:
```
streamlit==1.43.0
```

### ❌ `SyntaxError: invalid syntax`
Extra text (like markdown backticks) was accidentally added to your `.py` file. Open the file on GitHub, scroll to the bottom, and make sure the last line is:
```python
    main()
```

### ❌ Build stuck "in the oven" for 30+ minutes
A `.devcontainer` folder in your repo causes Streamlit Cloud to attempt a full Docker build. Delete it from your GitHub repo.

### ❌ `sklearn` version incompatibility with pickle
If you use a pre-saved `.sav` model, the sklearn version on Streamlit Cloud must match the version used to train it. The easiest fix is to **retrain the model inside the app** using `@st.cache_resource` as done in this project — no pickle file needed.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| scikit-learn | SVM model, StandardScaler |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| Streamlit | Web app framework |
| GitHub | Version control and hosting |
| Streamlit Cloud | Free app deployment |

---

## Author

**Faria Anowara Adhora** — built as part of a diabetes prediction ML project.

---

## License

This project is open source under the [MIT License](LICENSE).
