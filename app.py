import math
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
model = joblib.load("solar_model.joblib")

FEATURES = [
    "cloud_cover_forecast", "temperature_C", "humidity_pct", "wind_speed_ms",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "sun_height_index"
]

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Solar Panel Prediction System Platform</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f4f7fa; margin: 0; padding: 30px; color: #1d3557; }
    .card { max-width: 520px; margin: auto; background: #fff; padding: 25px 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    h2 { margin-top: 0; color: #0d3b66; }
    label { display: block; margin-top: 12px; font-weight: bold; font-size: 14px; }
    input { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 5px; }
    button { margin-top: 18px; width: 100%; padding: 11px; background: #0d3b66; color: #fff; border: none; border-radius: 5px; font-size: 15px; cursor: pointer; }
    button:hover { background: #1d5c96; }
    .result { margin-top: 20px; padding: 14px; border-radius: 6px; display: none; }
    .ok { background: #e8f5e9; border-left: 5px solid #2e7d32; }
    .alert { background: #ffebee; border-left: 5px solid #c62828; }
  </style>
</head>
<body>
  <div class="card">
    <h2>☀️ Solar Panel Prediction System</h2>
    <p style="font-size:13px; color:#555;">5 kW Rooftop Plant Forecast & Fault Detection (Ghaziabad)</p>
    <form id="predictForm">
      <label>Hour of Day (0–23)</label>
      <input type="number" id="hour" value="12" min="0" max="23" required>

      <label>Day of Year (1–365)</label>
      <input type="number" id="day_of_year" value="150" min="1" max="365" required>

      <label>Forecast Cloud Cover (0 = Clear, 1 = Overcast)</label>
      <input type="number" id="cloud" value="0.3" step="0.05" min="0" max="1" required>

      <label>Air Temperature (°C)</label>
      <input type="number" id="temperature" value="32" step="0.5" required>

      <label>Humidity (%)</label>
      <input type="number" id="humidity" value="55" min="0" max="100" required>

      <label>Wind Speed (m/s)</label>
      <input type="number" id="wind" value="2.5" step="0.1" min="0" required>

      <label>Actual Power Output in kW (Optional — for Fault Check)</label>
      <input type="number" id="actual_power" step="0.01" min="0" placeholder="Leave blank for forecast only">

      <button type="submit">Predict Solar Output</button>
    </form>
    <div id="resultBox" class="result"></div>
  </div>

  <script>
    document.getElementById("predictForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const payload = {
        hour: Number(document.getElementById("hour").value),
        day_of_year: Number(document.getElementById("day_of_year").value),
        cloud: Number(document.getElementById("cloud").value),
        temperature: Number(document.getElementById("temperature").value),
        humidity: Number(document.getElementById("humidity").value),
        wind: Number(document.getElementById("wind").value),
      };
      const actualVal = document.getElementById("actual_power").value;
      if (actualVal !== "") payload.actual_power = Number(actualVal);

      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      const box = document.getElementById("resultBox");
      box.style.display = "block";

      if (!res.ok) {
        box.className = "result alert";
        box.innerHTML = `<strong>Error:</strong> ${data.error}`;
        return;
      }

      let html = `<strong>Predicted Output:</strong> ${data.predicted_power_kW} kW`;
      if (data.ratio !== null) {
        html += `<br><strong>Performance Ratio (Actual / Predicted):</strong> ${data.ratio}`;
        html += `<br><strong>Status:</strong> ${data.fault_alert ? "⚠️ FAULT ALERT (Output below 85% threshold)" : "✅ Normal Operation"}`;
      }
      box.className = data.fault_alert ? "result alert" : "result ok";
      box.innerHTML = html;
    });
  </script>
</body>
</html>
"""

def calc_sun_height(hour: float, day_of_year: int) -> float:
    lat_rad = math.radians(28.7)
    declination = math.radians(23.44 * math.sin(2 * math.pi * (284 + day_of_year) / 365.0))
    hour_angle = math.radians(15.0 * (hour - 12.0))
    cos_z = (
        math.sin(lat_rad) * math.sin(declination)
        + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle)
    )
    return max(0.0, cos_z)

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    required = ["hour", "day_of_year", "cloud", "temperature", "humidity", "wind"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"missing field: {field}"}), 400

    try:
        hour = float(data["hour"])
        doy = int(data["day_of_year"])
        cloud = float(data["cloud"])
        temp = float(data["temperature"])
        humidity = float(data["humidity"])
        wind = float(data["wind"])
    except (ValueError, TypeError):
        return jsonify({"error": "all input fields must be numeric"}), 400

    if not (0 <= hour <= 23):
        return jsonify({"error": "hour must be between 0 and 23"}), 400
    if not (1 <= doy <= 365):
        return jsonify({"error": "day_of_year must be between 1 and 365"}), 400
    if not (0.0 <= cloud <= 1.0):
        return jsonify({"error": "cloud must be a number between 0 and 1"}), 400
    if not (0.0 <= humidity <= 100.0):
        return jsonify({"error": "humidity must be between 0 and 100"}), 400
    if wind < 0:
        return jsonify({"error": "wind must be >= 0"}), 400

    sun_height = calc_sun_height(hour, doy)
    if sun_height < 0.05:
        pred_kw = 0.0
    else:
        row = pd.DataFrame([[
            cloud, temp, humidity, wind,
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * doy / 365.0),
            math.cos(2 * math.pi * doy / 365.0),
            sun_height
        ]], columns=FEATURES)
        pred_kw = max(0.0, round(float(model.predict(row)[0]), 3))

    ratio = None
    fault_alert = False
    if "actual_power" in data and data["actual_power"] is not None:
        actual = float(data["actual_power"])
        if pred_kw > 0.1:
            ratio = round(actual / pred_kw, 2)
            fault_alert = ratio < 0.85

    return jsonify({
        "predicted_power_kW": pred_kw,
        "ratio": ratio,
        "fault_alert": fault_alert
    }), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)