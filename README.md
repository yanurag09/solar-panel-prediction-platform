# Solar Panel Prediction System Platform

A machine learning web application that forecasts hourly AC power output for a 5 kW rooftop solar plant (Ghaziabad, 28.7° N) from weather inputs and triggers a fault alert when actual output falls below 85% of the predicted value.

## Features
- **Physics-based Hourly Dataset:** Simulates one full year (8,760 hourly rows) of solar geometry, cloud cover, temperature, humidity, wind, panel temperature loss, and injected fault days.
- **Model Comparison & Random Forest:** Compares Linear Regression, Random Forest, and Gradient Boosting on an unseen time-ordered Oct–Dec split; exports the trained Random Forest model (`solar_model.joblib`).
- **Fault Detection:** Flags under-performing days using `actual / predicted < 0.85` (caught 12/12 injected fault days with 0 false alarms).
- **Flask REST API & Dashboard:** Serves predictions via `POST /predict` and a browser form at `/`.

## Project Structure
- `solar_model_training_and_eda.py` — Generates datasets, creates 7 EDA/evaluation charts, compares models, and saves `solar_model.joblib`.
- `app.py` — Flask API (`/predict`) and interactive web dashboard (`/`).
- `solar_dataset.csv` & `solar_dataset_cleaned.csv` — Raw (8,760 rows) and daytime-filtered (4,191 rows) datasets.
- `solar_data_dictionary.csv` — Column-by-column schema description.
- `charts/` — EDA plots, model RMSE comparison, feature importance, and fault-detection charts.
- `docs/` — Project Synopsis and Project Report.
- `requirements.txt` — Python dependencies.

## Quick Start
```bash
pip install -r requirements.txt
python solar_model_training_and_eda.py
python app.py