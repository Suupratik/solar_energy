# Model Comparison and Analysis — Solar Energy AI Project

## Project Overview
This project predicts **DC Power** and **AC Power** output for a solar power plant using sensor and generation data. Two machine learning models were developed and evaluated:
1. Linear Regression (baseline)
2. Random Forest Regressor (final chosen model)

The goal is to determine which model better captures the nonlinear relationships between environmental factors and power generation.

---

## 1. Dataset Description
The dataset contains two CSV files:
- `Plant_1_Generation_Data.csv` — energy output, source keys, and timestamps.
- `Plant_1_Weather_Sensor_Data.csv` — environmental conditions like irradiation, ambient temperature, and module temperature.

The files were merged on the **DATE_TIME** column to create a unified dataset.  
Missing or duplicate values were handled, and new features like `hour`, `day`, and `month` were extracted from timestamps to capture temporal effects on power generation.

---

## 2. Objective
Predict two target variables:
- **DC_POWER** — Direct current output from solar panels.
- **AC_POWER** — Alternating current power delivered after inverter conversion.

Independent variables included:
- AMBIENT_TEMPERATURE  
- MODULE_TEMPERATURE  
- IRRADIATION  
- Temporal features (hour, day, month)

---

## 3. Data Preprocessing
- Converted timestamps to datetime format.
- Merged both datasets by `DATE_TIME`.
- Removed duplicates and handled missing values.
- Extracted time-based features.
- Scaled numeric data for Linear Regression (not required for Random Forest).
- Split dataset into 80% training and 20% testing sets.

```python
from sklearn.model_selection import train_test_split
X = data[['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'hour', 'day', 'month']]
y_dc = data['DC_POWER']
y_ac = data['AC_POWER']
X_train, X_test, y_train_dc, y_test_dc = train_test_split(X, y_dc, test_size=0.2, random_state=42)
_, _, y_train_ac, y_test_ac = train_test_split(X, y_ac, test_size=0.2, random_state=42)
```

---

## 4. Baseline Model — Linear Regression

### Theory
Linear Regression assumes a **linear relationship** between input variables and target.  
For solar data, this assumption is too simplistic because the relationship between irradiation, temperature, and power output is nonlinear.

### Code
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

linreg_dc = LinearRegression()
linreg_ac = LinearRegression()

linreg_dc.fit(X_train, y_train_dc)
linreg_ac.fit(X_train, y_train_ac)

y_pred_dc_lin = linreg_dc.predict(X_test)
y_pred_ac_lin = linreg_ac.predict(X_test)

r2_dc_lin = r2_score(y_test_dc, y_pred_dc_lin)
r2_ac_lin = r2_score(y_test_ac, y_pred_ac_lin)

print("Linear Regression DC Power R2:", r2_dc_lin)
print("Linear Regression AC Power R2:", r2_ac_lin)
```

### Result Interpretation
- The R² values were moderate (~0.70–0.78).  
- The model performed reasonably but failed to capture sudden fluctuations due to cloud cover or irradiation drops.
- Residual plots indicated non-linear patterns → violation of linear model assumptions.

---

## 5. Final Model — Random Forest Regressor

### Theory
Random Forest is an **ensemble of decision trees**, combining multiple weak learners to form a strong predictor.  
It handles nonlinear relationships, noise, and feature interactions better than linear models.

### Code
```python
from sklearn.ensemble import RandomForestRegressor

rf_dc = RandomForestRegressor(n_estimators=100, random_state=42)
rf_ac = RandomForestRegressor(n_estimators=100, random_state=42)

rf_dc.fit(X_train, y_train_dc)
rf_ac.fit(X_train, y_train_ac)

y_pred_dc_rf = rf_dc.predict(X_test)
y_pred_ac_rf = rf_ac.predict(X_test)

r2_dc_rf = r2_score(y_test_dc, y_pred_dc_rf)
r2_ac_rf = r2_score(y_test_ac, y_pred_ac_rf)

print("Random Forest DC Power R2:", r2_dc_rf)
print("Random Forest AC Power R2:", r2_ac_rf)
```

### Observations
- The R² improved significantly (~0.95+).
- The model handled nonlinear dependencies between temperature, irradiation, and power generation.
- Random Forest captured seasonal variations and sensor noise effectively.
- Training time was higher than Linear Regression, but inference speed remained acceptable.

---

## 6. Feature Importance Analysis

Random Forest provides insight into which features most influence predictions.

```python
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(rf_dc.feature_importances_, index=X.columns)
importance.sort_values(ascending=True).plot(kind='barh', figsize=(6,4))
plt.title("Feature Importance for DC Power Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
```

**Findings:**
- `IRRADIATION` contributed the most to DC and AC power output.
- `MODULE_TEMPERATURE` also played a significant role.
- Temporal features like `hour` captured the daily solar cycle.

---

## 7. Model Persistence
The final trained models were saved for deployment and future inference.

```python
import joblib
joblib.dump(rf_dc, "../models/dc_rf_model.pkl")
joblib.dump(rf_ac, "../models/ac_rf_model.pkl")
```

---

## 8. Comparative Discussion

Comparison: Linear Regression vs Random Forest

Nature:

Linear Regression: Follows a linear relationship between input and output.

Random Forest: Nonlinear ensemble method using multiple decision trees.

#Accuracy:

Linear Regression: Provides moderate accuracy.

Random Forest: Generally achieves higher accuracy.

#Training Speed:

Linear Regression: Trains very fast due to simple computations.

Random Forest: Slower training because it builds many trees.

#Interpretability:

Linear Regression: Easy to interpret and explain coefficients.

Random Forest: Complex and less interpretable.

#Handling of Noise:

Linear Regression: Performs poorly with noisy data.

Random Forest: More robust and handles noise effectively.

### Final Verdict
The Random Forest model clearly outperformed Linear Regression due to its nonlinear learning ability, robustness against outliers, and higher R² scores. It is therefore selected as the final predictive model for both **DC Power** and **AC Power** forecasting.

---

## 9. Future Enhancements
1. **Streamlit Integration** — Build an interactive dashboard where users can upload real-time sensor data and see predicted DC/AC outputs instantly.
2. **Hyperparameter Optimization** — Use GridSearchCV or RandomizedSearchCV to further boost accuracy.
3. **Real-Time Prediction API** — Deploy using Flask or FastAPI for live monitoring of solar power generation.
4. **Weather Forecast Integration** — Include predicted solar irradiation for future power forecasting.
5. **Visualization Improvements** — Incorporate prediction vs actual plots and feature importance charts in the UI.

---

## 10. Conclusion
This project demonstrates how machine learning can effectively model complex energy systems.  
The comparison between Linear Regression and Random Forest shows the transition from a simple analytical approach to a powerful predictive model capable of handling real-world solar energy data.

The final model achieves strong predictive accuracy for both DC and AC power and sets a solid foundation for future deployment as a Streamlit-based dashboard or real-time inference API.

---

**Author:** Supratik Mitra  
**Internship Program:** Shell-Edunet Skills4Future (AI/ML — Energy Theme)  
**Duration:** October–November 2025  
**Project Title:** Solar Energy Power Prediction Using Machine Learning
