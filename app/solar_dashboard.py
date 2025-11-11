# ==============================================================
# ☀️ SOLAR ENERGY DASHBOARD – STREAMLIT APP
# AICTE–Shell Skills4Future Internship | Theme: Energy
# Author: Supratik Mitra
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Solar Energy Dashboard", layout="wide")
st.title("🔆 Solar Energy Prediction Dashboard")
st.write("This dashboard visualizes solar power data and predicts DC & AC power output using Linear Regression models.")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    import os
    base_path = os.path.dirname(__file__)  # path to /app folder
    data_path = os.path.join(base_path, '..', 'results', 'cleaned_data.csv')
    data = pd.read_csv(os.path.abspath(data_path))
    return data


data = load_data()
st.success("✅ Cleaned data loaded successfully!")

# -----------------------------
# SHOW RAW DATA
# -----------------------------
if st.checkbox("📄 Show raw data"):
    st.dataframe(data.head(10))

# -----------------------------
# FEATURE SELECTION
# -----------------------------
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
target_dc = 'DC_POWER'
target_ac = 'AC_POWER'

X = data[features]
y_dc = data[target_dc]
y_ac = data[target_ac]

# -----------------------------
# MODEL TRAINING
# -----------------------------
X_train, X_test, y_dc_train, y_dc_test = train_test_split(X, y_dc, test_size=0.2, random_state=42)
_, _, y_ac_train, y_ac_test = train_test_split(X, y_ac, test_size=0.2, random_state=42)

dc_model = LinearRegression()
ac_model = LinearRegression()

dc_model.fit(X_train, y_dc_train)
ac_model.fit(X_train, y_ac_train)

y_dc_pred = dc_model.predict(X_test)
y_ac_pred = ac_model.predict(X_test)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mae, mse

r2_dc, mae_dc, mse_dc = evaluate(y_dc_test, y_dc_pred)
r2_ac, mae_ac, mse_ac = evaluate(y_ac_test, y_ac_pred)

st.subheader("📊 Model Evaluation Results")
col1, col2 = st.columns(2)
with col1:
    st.metric("DC Power Accuracy (R²)", f"{r2_dc*100:.2f}%")
    st.metric("Mean Absolute Error (DC)", f"{mae_dc:.2f}")
with col2:
    st.metric("AC Power Accuracy (R²)", f"{r2_ac*100:.2f}%")
    st.metric("Mean Absolute Error (AC)", f"{mae_ac:.2f}")

# -----------------------------
# VISUALIZATION
# -----------------------------
st.subheader("🔋 Real vs Predicted Power Comparison")

fig1, ax1 = plt.subplots()
ax1.scatter(y_dc_test[:100], y_dc_pred[:100], color='blue', label='DC Predicted')
ax1.plot([y_dc_test.min(), y_dc_test.max()], [y_dc_test.min(), y_dc_test.max()], 'r--', lw=2, label='Perfect Fit')
ax1.set_xlabel("Actual DC Power")
ax1.set_ylabel("Predicted DC Power")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.scatter(y_ac_test[:100], y_ac_pred[:100], color='green', label='AC Predicted')
ax2.plot([y_ac_test.min(), y_ac_test.max()], [y_ac_test.min(), y_ac_test.max()], 'r--', lw=2, label='Perfect Fit')
ax2.set_xlabel("Actual AC Power")
ax2.set_ylabel("Predicted AC Power")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# -----------------------------
# USER INPUT PREDICTION
# -----------------------------
st.subheader("⚙️ Try Predicting Your Own Values")

col1, col2, col3 = st.columns(3)
ambient = col1.slider("Ambient Temperature (°C)", 20, 45, 30)
module = col2.slider("Module Temperature (°C)", 25, 65, 45)
irradiation = col3.slider("Irradiation (W/m²)", 0, 1000, 600)

input_data = np.array([[ambient, module, irradiation]])
dc_pred = dc_model.predict(input_data)[0]
ac_pred = ac_model.predict(input_data)[0]

st.write("### 🔮 Predicted Outputs:")
st.info(f"**Predicted DC Power:** {dc_pred:.2f} W")
st.info(f"**Predicted AC Power:** {ac_pred:.2f} W")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("------")
st.caption("Developed by Supratik Mitra | AICTE–Shell Skills4Future Internship – Energy Theme")
