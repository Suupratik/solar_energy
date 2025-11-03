# 🌞 Solar Energy Prediction using Machine Learning

## 🧭 Overview

This project, **Solar Energy Prediction**, is developed under the **Shell-Edunet Skills4Future AI/ML Internship (Oct–Nov 2025)**.  
The main goal is to **predict both DC Power and AC Power output** from a solar plant using machine learning techniques based on environmental and sensor data.

The system helps understand how weather parameters such as temperature, irradiation, and humidity affect energy production. It enables better forecasting, maintenance planning, and efficient solar resource management.

The project is divided into **4 weekly milestones**, each representing 25% progress.  
By the end, you will have a complete **end-to-end ML model** capable of predicting power output from real-world solar energy datasets.

## ⚙️ Project Objectives

- To process and clean solar generation and weather sensor data.  
- To perform exploratory data analysis (EDA) on environmental and power factors.  
- To train a **Random Forest Regressor** to predict both DC and AC power.  
- To visualize the results and compare model performance.  
- To deploy or extend the model (optional: via Streamlit web app).

## 🗂️ Project Structure

Your main project folder is named **`solar_energy/`** and contains the following files and subfolders:

solar_energy/
│
├── data/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│
├── notebooks/
│   └── Solar_Power_Prediction.ipynb
│
├── models/
│   ├── dc_rf_model.pkl
│   ├── ac_rf_model.pkl
│
├── results/
│   ├── feature_importances.png
│   ├── predictions_vs_actual_dc.png
│   ├── predictions_vs_actual_ac.png
│
├── analysis/
│   └── model_comparison.md
│
├── requirements.txt
├── README.md
└── venv/

> The `.ipynb` notebook contains all code for **data preprocessing, feature engineering, model training, evaluation, and visualization**.  
> The other folders contain saved models, results, and supporting files.

## 💾 Dataset Information

The dataset consists of two CSV files collected from a solar power plant:

1. Plant_1_Generation_Data.csv – Contains timestamps, DC power, AC power, and daily yield.  
2. Plant_1_Weather_Sensor_Data.csv – Contains temperature, irradiation, and weather readings.

These datasets are merged on timestamps and used to build relationships between environmental features and power output.

## 🧩 Technologies Used

- Programming Language: Python  
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib  
- Environment: Jupyter Notebook / VS Code with `.ipynb` support  
- Model Used: Random Forest Regressor  
- Version Control: Git + GitHub  
- Optional Deployment: Streamlit  

## 🧰 Step-by-Step Setup Guide

Follow these steps to set up and run the project on your system.

### 1️⃣ Clone the Repository

Open your terminal (Command Prompt or PowerShell on Windows) and run:

git clone https://github.com/<your-username>/solar_energy.git
cd solar_energy

Replace <your-username> with your GitHub username.

### 2️⃣ Create a Virtual Environment

Create a Python virtual environment to isolate project dependencies.

python -m venv venv

Activate it:

- On Windows:
  venv\Scripts\activate
- On macOS/Linux:
  source venv/bin/activate

You’ll know it’s activated when you see (venv) before your terminal prompt.

### 3️⃣ Install Required Packages

Install all required dependencies listed in requirements.txt:

pip install -r requirements.txt

If you face permission errors, try:
pip install --upgrade pip
pip install -r requirements.txt

### 4️⃣ Open and Explore the Notebook

Launch Jupyter Notebook or open the .ipynb file in VS Code.

- If using Jupyter, run:
  jupyter notebook
  Then open: notebooks/Solar_Power_Prediction.ipynb

- If using VS Code, just open the project folder and click on the notebook file.  
  Ensure your Python interpreter is set to the venv environment.

### 5️⃣ Run Each Section Step-by-Step

Inside the notebook, the code is divided by Week milestones:

- Week 1: Data loading, merging, cleaning, preprocessing (✅ for submission).  
- Week 2: Model training (Random Forest) and feature importance visualization.  
- Week 3: Hyperparameter tuning, evaluation metrics, and saving models.  
- Week 4: Deployment-ready results, visualizations, and final documentation.

Each later week’s section is clearly commented — simply uncomment to activate those cells once you progress.

### 6️⃣ View Results and Outputs

After running the notebook:

- Trained models will be saved in the models/ folder (.pkl files).  
- Graphs and plots will be saved automatically in the results/ folder.  
- A summary comparison of models is available inside analysis/model_comparison.md.

You can visually inspect:
- feature_importances.png – shows which weather factors affect power most.  
- predictions_vs_actual_dc.png – compares predicted and actual DC power.  
- predictions_vs_actual_ac.png – compares predicted and actual AC power.

## 🧠 Understanding the Model

The Random Forest Regressor is used because it handles nonlinear relationships and noisy data better than simple models like Linear Regression.  
It builds multiple decision trees on subsets of data and averages their outputs for robust and stable predictions.

In this project:
- DC Power and AC Power are predicted separately.
- Both models are evaluated using metrics like R² and RMSE.

## 🚀 Optional: Streamlit Deployment (Future Scope)

You can later build a web interface using Streamlit to allow real-time prediction from input parameters.

Example steps:

pip install streamlit joblib

Create a file named app.py:

import streamlit as st
import joblib
import numpy as np

st.title("Solar Power Prediction")

model = joblib.load("models/dc_rf_model.pkl")
temp = st.number_input("Enter Temperature:")
irr = st.number_input("Enter Irradiation:")
if st.button("Predict DC Power"):
    result = model.predict(np.array([[temp, irr]]))
    st.success(f"Predicted DC Power: {result[0]:.2f} kW")

Run the app:
streamlit run app.py

## 🧾 Notes

- Always activate your venv before running or installing anything.  
- Keep dataset files in the data/ folder as used in the code.  
- If Jupyter or pandas gives a path error, ensure your working directory is set to the project root.  
- Do not upload the venv/ folder to GitHub — it’s large and unnecessary.

## 💡 Future Enhancements

- Integration of real-time IoT sensor API for live data updates.  
- Adding energy efficiency and loss estimation analytics.  
- Streamlit web dashboard deployment.  
- Comparison of multiple ML models (XGBoost, LightGBM, etc.).

## 🏁 Conclusion

This project provides a full AI/ML pipeline for solar energy prediction — from raw data cleaning to advanced machine learning models and visual insights.  
It demonstrates strong data preprocessing, feature analysis, and model interpretability skills — aligning with real-world sustainability goals.
