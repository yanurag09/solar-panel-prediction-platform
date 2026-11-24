import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("charts", exist_ok=True)
np.random.seed(42)

# ---------------------------------------------------------
# 1. GENERATE SIMULATED HOURLY DATA (2025, 5 kW, Ghaziabad)
# ---------------------------------------------------------
timestamps = pd.date_range("2025-01-01 00:00:00", "2025-12-31 23:00:00", freq="h")
n = len(timestamps)
hour = timestamps.hour.values
day_of_year = timestamps.dayofyear.values

# Solar geometry at latitude 28.7 N
lat_rad = np.radians(28.7)
declination = np.radians(23.44 * np.sin(2 * np.pi * (284 + day_of_year) / 365.0))
hour_angle = np.radians(15.0 * (hour - 12.0))
cos_zenith = (
    np.sin(lat_rad) * np.sin(declination)
    + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
)
sun_height_index = np.clip(cos_zenith, 0.0, None)

# Cloud cover (higher in monsoon) + realistic forecast error
seasonal_cloud = 0.25 + 0.35 * np.exp(-0.5 * ((day_of_year - 215) / 40.0) ** 2)
smooth_cloud = (
    pd.Series(seasonal_cloud + np.random.normal(0, 0.22, n))
    .rolling(6, min_periods=1, center=True)
    .mean()
    .values
)
cloud_true = np.clip(smooth_cloud, 0.0, 1.0)
cloud_cover_forecast = np.clip(cloud_true + np.random.normal(0, 0.08, n), 0.0, 1.0)

# Temperature, humidity, wind speed
temp_seasonal = 25.0 + 11.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.0)
temp_diurnal = 6.0 * np.sin(2 * np.pi * (hour - 8) / 24.0)
temp_true = temp_seasonal + temp_diurnal - 4.0 * cloud_true + np.random.normal(0, 1.2, n)
temperature_C = temp_true + np.random.normal(0, 1.0, n)

humidity_pct = np.clip(
    55 + 30 * cloud_true - 0.8 * (temp_true - 25) + np.random.normal(0, 5, n), 15, 98
)
wind_speed_ms = np.clip(np.random.gamma(2.2, 1.2, n), 0.3, 12.0)

# Panel physics (0.42%/C temp loss, 96% inverter eff, periodic dust buildup)
clear_sky_irr = 1080.0 * (sun_height_index ** 1.15)
irradiance_Wm2 = np.clip(clear_sky_irr * (1.0 - 0.65 * (cloud_true ** 1.6)), 0.0, None)
panel_temp = temp_true + (irradiance_Wm2 / 800.0) * 25.0 - 0.4 * wind_speed_ms
temp_factor = 1.0 - 0.0042 * np.maximum(panel_temp - 25.0, 0.0)
dust_loss = 1.0 - 0.04 * ((day_of_year % 20) / 20.0)
inverter_eff = 0.96

power_kW = np.where(
    sun_height_index > 0.05,
    5.0 * (irradiance_Wm2 / 1000.0) * temp_factor * inverter_eff * dust_loss
    + np.random.normal(0, 0.05, n),
    0.0,
)
power_kW = np.clip(power_kW, 0.0, 5.0)

# Insert 6 two-day faults (12 fault days in Oct-Dec with 30% output loss)
fault_days = [283, 284, 300, 301, 317, 318, 330, 331, 343, 344, 355, 356]
fault_flag = np.isin(day_of_year, fault_days).astype(int)
power_kW = np.where(fault_flag == 1, power_kW * 0.70, power_kW)

df = pd.DataFrame({
    "timestamp": timestamps,
    "hour": hour,
    "day_of_year": day_of_year,
    "cloud_cover_forecast": np.round(cloud_cover_forecast, 4),
    "temperature_C": np.round(temperature_C, 2),
    "humidity_pct": np.round(humidity_pct, 2),
    "wind_speed_ms": np.round(wind_speed_ms, 2),
    "irradiance_Wm2": np.round(irradiance_Wm2, 2),
    "sun_height_index": np.round(sun_height_index, 4),
    "power_kW": np.round(power_kW, 4),
    "fault_flag": fault_flag,
})
df.to_csv("solar_dataset.csv", index=False)

# Remove night hours
cleaned = df[df["sun_height_index"] >= 0.05].copy().reset_index(drop=True)
cleaned.to_csv("solar_dataset_cleaned.csv", index=False)

# Save Data Dictionary
data_dict = pd.DataFrame([
    ("timestamp", "datetime", "Hourly reading timestamp for 2025"),
    ("hour", "integer", "Hour of the day (0 to 23)"),
    ("day_of_year", "integer", "Day number in the year (1 to 365)"),
    ("cloud_cover_forecast", "float", "Forecast cloud cover (0 clear to 1 overcast)"),
    ("temperature_C", "float", "Forecast air temperature in Celsius"),
    ("humidity_pct", "float", "Relative humidity in percent"),
    ("wind_speed_ms", "float", "Wind speed in metres per second"),
    ("irradiance_Wm2", "float", "Simulated solar irradiance on panel (W/m2)"),
    ("sun_height_index", "float", "Cosine of solar zenith angle at 28.7 N"),
    ("power_kW", "float", "AC power output of the 5 kW rooftop plant (target)"),
    ("fault_flag", "integer", "1 on injected 30% loss fault days, else 0"),
], columns=["column", "type", "description"])
data_dict.to_csv("solar_data_dictionary.csv", index=False)

print(f"Created solar_dataset.csv ({len(df)} rows) & solar_dataset_cleaned.csv ({len(cleaned)} rows)")

# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (4 CHARTS)
# ---------------------------------------------------------
cleaned["month"] = cleaned["timestamp"].dt.month
season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    4: "Summer", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon"
}
cleaned["season"] = cleaned["month"].map(season_map)

# Chart 1: Seasonal hourly power curve
plt.figure(figsize=(8, 4))
for s in ["Winter", "Summer", "Monsoon"]:
    sub = cleaned[cleaned["season"] == s].groupby("hour")["power_kW"].mean()
    plt.plot(sub.index, sub.values, marker="o", label=s)
plt.title("Average Power by Hour of Day Across Seasons")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Power (kW)")
plt.legend()
plt.tight_layout()
plt.savefig("charts/1_seasonal_hourly_power.png")
plt.close()

# Chart 2: Monthly energy output
plt.figure(figsize=(8, 4))
monthly_kwh = cleaned.groupby("month")["power_kW"].sum()
plt.bar(monthly_kwh.index, monthly_kwh.values, color="#0d3b66")
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.title("Total Monthly Solar Energy Output (kWh)")
plt.ylabel("Energy (kWh)")
plt.tight_layout()
plt.savefig("charts/2_monthly_energy.png")
plt.close()

# Chart 3: Irradiance vs Power colored by temperature
plt.figure(figsize=(7, 4))
sc = plt.scatter(cleaned["irradiance_Wm2"], cleaned["power_kW"], c=cleaned["temperature_C"], cmap="viridis", s=6, alpha=0.6)
plt.colorbar(sc, label="Temperature (°C)")
plt.title("Solar Power vs. Irradiance")
plt.xlabel("Irradiance (W/m²)")
plt.ylabel("Power (kW)")
plt.tight_layout()
plt.savefig("charts/3_irradiance_vs_power.png")
plt.close()

# Chart 4: Mean power by cloud cover band
cleaned["cloud_band"] = pd.cut(
    cleaned["cloud_cover_forecast"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    include_lowest=True
)
plt.figure(figsize=(7, 4))
cleaned.groupby("cloud_band", observed=False)["power_kW"].mean().plot(kind="bar", color="#ee964b")
plt.title("Mean Power by Forecast Cloud Cover Band")
plt.xlabel("Forecast Cloud Cover")
plt.ylabel("Mean Power (kW)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/4_cloud_cover_bands.png")
plt.close()

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING & MODEL COMPARISON (NO NEURAL NET)
# ---------------------------------------------------------
cleaned["hour_sin"] = np.sin(2 * np.pi * cleaned["hour"] / 24.0)
cleaned["hour_cos"] = np.cos(2 * np.pi * cleaned["hour"] / 24.0)
cleaned["doy_sin"] = np.sin(2 * np.pi * cleaned["day_of_year"] / 365.0)
cleaned["doy_cos"] = np.cos(2 * np.pi * cleaned["day_of_year"] / 365.0)

features = [
    "cloud_cover_forecast", "temperature_C", "humidity_pct", "wind_speed_ms",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "sun_height_index"
]

# Time-ordered split: Jan-Sep train, Oct-Dec test
train_df = cleaned[cleaned["month"] <= 9]
test_df = cleaned[cleaned["month"] >= 10]
test_clean = test_df[test_df["fault_flag"] == 0]

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=42),
}

results = []
print("\n--- Model Comparison (Oct-Dec Non-Fault Hours) ---")
for name, model in models.items():
    model.fit(train_df[features], train_df["power_kW"])
    preds = model.predict(test_clean[features])
    mae = mean_absolute_error(test_clean["power_kW"], preds)
    rmse = np.sqrt(mean_squared_error(test_clean["power_kW"], preds))
    r2 = r2_score(test_clean["power_kW"], preds)
    results.append((name, mae, rmse, r2))
    print(f"{name:22s} | MAE: {mae:.3f} kW | RMSE: {rmse:.3f} kW | R2: {r2:.3f}")

# Save Random Forest as the primary model for app.py
rf_model = models["Random Forest"]
joblib.dump(rf_model, "solar_model.joblib")
print("\nSaved Random Forest model to solar_model.joblib")

# Chart 5: Model RMSE comparison
res_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
plt.figure(figsize=(8, 4))
plt.bar(res_df["Model"], res_df["RMSE"], color=["#8da9c4", "#0d3b66", "#ee964b"])
plt.title("Model Comparison — Test RMSE (Lower is Better)")
plt.ylabel("RMSE (kW)")
plt.tight_layout()
plt.savefig("charts/5_model_comparison_rmse.png")
plt.close()

# Chart 6: Random Forest Feature Importance
imp = pd.Series(rf_model.feature_importances_, index=features).sort_values()
plt.figure(figsize=(8, 4))
imp.plot(kind="barh", color="#0d3b66")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("charts/6_rf_feature_importance.png")
plt.close()

# ---------------------------------------------------------
# 4. FAULT DETECTION EVALUATION (Oct-Dec, 9 AM - 3 PM)
# ---------------------------------------------------------
test_midday = test_df[(test_df["hour"] >= 9) & (test_df["hour"] <= 15)].copy()
test_midday["pred_kW"] = rf_model.predict(test_midday[features])

daily = test_midday.groupby(test_midday["timestamp"].dt.date).agg(
    actual=("power_kW", "sum"),
    predicted=("pred_kW", "sum"),
    fault=("fault_flag", "max")
)
daily["ratio"] = daily["actual"] / daily["predicted"]
daily["alert"] = (daily["ratio"] < 0.85).astype(int)

tp = int(((daily["alert"] == 1) & (daily["fault"] == 1)).sum())
fp = int(((daily["alert"] == 1) & (daily["fault"] == 0)).sum())
fn = int(((daily["alert"] == 0) & (daily["fault"] == 1)).sum())
tn = int(((daily["alert"] == 0) & (daily["fault"] == 0)).sum())
print("\n--- Fault Detection (Ratio < 0.85) ---")
print(f"Caught: {tp}/12 | Missed: {fn} | False Alarms: {fp} | Normal Days Clear: {tn}/80")

# Chart 7: Daily Fault Detection Ratio
plt.figure(figsize=(10, 4))
plt.plot(daily.index, daily["ratio"], color="#0d3b66", label="Actual / Predicted")
fault_pts = daily[daily["fault"] == 1]
plt.scatter(fault_pts.index, fault_pts["ratio"], color="red", zorder=5, label="Inserted Fault Days")
plt.axhline(0.85, color="orange", linestyle="--", label="Alert Threshold (0.85)")
plt.title("Daily Performance Ratio & Fault Detection (Oct-Dec)")
plt.ylabel("Actual / Predicted Ratio")
plt.legend()
plt.tight_layout()
plt.savefig("charts/7_fault_detection_ratio.png")
plt.close()
print("Saved all 7 charts in charts/")