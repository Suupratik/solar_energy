# ☀️ **Solar Energy Power Prediction**

### *AICTE–Shell Skills4Future Internship (Energy Theme)*

**Intern Name:** Supratik Mitra
**Project Folder:** `solar_energy`

------

## 🗓️ **Project Overview**

This project aims to **predict solar power output (DC & AC)** using weather and operational parameters such as **ambient temperature, module temperature, and irradiation**.
A **Linear Regression model** is trained and visualized through an: **interactive Streamlit dashboard**.

------

## 🎯 **Objectives**

1. Clean and preprocess raw solar plant data.
2. Merge generation and weather sensor datasets.
3. Train predictive models for DC and AC power output.
4. Evaluate model accuracy using MAE, MSE, and R² metrics.
5. Build an interactive dashboard for visualization and live prediction.

------

## 📁 **Project Structure**

```
solar_energy/
│
├─ analysis/
│ └─ analysis.md # Detailed analysis and observations
│
├─ app/
│ └─ solar_dashboard.py # Streamlit/Dashboard app for visualization
│
├─ data/
│ ├─ Plant_1_Generation_Data.csv # Historical solar generation data
│ └─ Plant_1_Weather_Sensor_Data.csv # Weather sensor data
│
├─ models/
│ ├─ ac_power_model.pkl # Trained AC power prediction model
│ └─ dc_power_model.pkl # Trained DC power prediction model
│
├─ notebooks/
│ └─ Solar_Power_Prediction.ipynb # Jupyter notebook for analysis and modeling
│
├─ results/
│ ├─ cleaned_data.csv # Preprocessed/cleaned dataset
│ ├─ predictions_vs_actual_ac.png # AC power prediction vs actual plot
│ └─ predictions_vs_actual_dc.png # DC power prediction vs actual plot
│
├─ venv/ # Python virtual environment
│
├─ .gitignore # Files to ignore in git
├─ README.md # Project documentation
└─ requirements.txt # Required Python packages
```

------

## ⚙️ **Tech Stack**

| Category            | Tools Used                                                 |
| ------------------- | ---------------------------------------------------------- |
| **Language**        | Python 3.x                                                 |
| **Libraries**       | pandas, numpy, matplotlib, scikit-learn, streamlit, joblib |
| **Visualization**   | Matplotlib, Streamlit                                      |
| **IDE**             | VS Code / Jupyter Notebook                                 |
| **Version Control** | Git + GitHub                                               |
| **Dataset Source**  | AICTE–Shell Edunet Energy Theme (Plant 1 Data)             |

------

## 🧩 **Implementation Steps**

### ✅ **Week 1: Data Preprocessing**

* Imported datasets: `Plant_1_Generation_Data.csv`, `Plant_1_Weather_Sensor_Data.csv`
* Converted `DATE_TIME` columns to datetime objects.
* Merged datasets on `DATE_TIME`.
* Dropped unnecessary columns (`PLANT_ID`, `SOURCE_KEY`).
* Handled missing numeric values.
* Saved cleaned dataset → `results/cleaned_data.csv`

------

### ✅ **Week 2: Model Training and Evaluation**

* Selected features: `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, `IRRADIATION`
* Targets: `DC_POWER`, `AC_POWER`
* Split dataset (80–20) into training and testing sets.
* Trained **Linear Regression models** for DC and AC power.
* Evaluated models using:

  * **MAE (Mean Absolute Error)**
  * **MSE (Mean Squared Error)**
  * **R² Score (Accuracy %)**
* Saved models → `models/dc_power_model.pkl`, `models/ac_power_model.pkl`

------

## 📊 **Sample Results**

| Metric              | DC Power | AC Power |
| :------------------ | :------: | :------: |
| Mean Absolute Error |   23.41  |   21.56  |
| Mean Squared Error  |  1210.33 |  1050.18 |
| R² Score            |   0.996  |   0.994  |

✅ *Both models achieved above 99% accuracy, indicating excellent fit.*

------
## Final Week: Solar Energy Dashboard (Streamlit App)
## 🖥️ **Streamlit Dashboard**

The dashboard provides:

* Interactive visualizations of **real vs predicted power output**
* **Live prediction sliders** for custom input values
* Accuracy metrics and data previews

### 🚀 Run the App:

1. **Activate virtual environment**

   ```
   venv\Scripts\activate
   ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Navigate to app folder**

   ```
   cd app
   ```

4. **Run Streamlit**

   ```
   streamlit run solar_dashboard.py
   ```

5. **Access dashboard in browser:**
   🔗 [http://localhost:8501](http://localhost:8501)

------

## 🧠 **Model Features**

| Feature                 | Description                                    |
| ----------------------- | ---------------------------------------------- |
| **Ambient Temperature** | Temperature of the surrounding air (°C)        |
| **Module Temperature**  | Temperature of the solar panel surface (°C)    |
| **Irradiation**         | Amount of sunlight per m² (W/m²)               |
| **DC Power Output**     | Direct current power produced (W)              |
| **AC Power Output**     | Alternating current power after conversion (W) |

------

## 📸 **Dashboard Preview**

* 📊 Real vs Predicted DC & AC Power Scatter Plots
* ⚙️ Adjustable sliders for temperature and irradiation
* 📈 Instant prediction results with accuracy metrics

------

## 💾 **Files Generated**

| File                 | Purpose                           |
| -------------------- | --------------------------------- |
| `cleaned_data.csv`   | Final preprocessed dataset        |
| `dc_power_model.pkl` | Trained DC power prediction model |
| `ac_power_model.pkl` | Trained AC power prediction model |
| `solar_dashboard.py` | Streamlit app code                |
| `requirements.txt`   | All dependencies for quick setup  |

------

## 🏁 **Future Improvements**

* Integrate **Random Forest or XGBoost** for higher robustness.
* Add **real-time solar monitoring API** for live data updates.
* Deploy the dashboard on **Streamlit Cloud / Render**.
* Add **performance analytics (efficiency, degradation)**.

------

## 👨‍💻 **Author**

**Supratik Mitra**
- AICTE–Shell Skills4Future Internship (Oct–Nov 2025)
- **Theme:** Energy | Project: *Solar Energy Prediction*

📧 *Email:* (mailto:supratikmitracpsbp2015to16103@gmail.com) 


