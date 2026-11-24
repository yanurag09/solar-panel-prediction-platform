import time
from app import app

client = app.test_client()

# 1. Home page
r = client.get("/")
assert r.status_code == 200, "Home page failed"

# 2. Valid request (12 noon, day 150, cloud 0.3, 32 C)
base = {"hour": 12, "day_of_year": 150, "cloud": 0.3, "temperature": 32, "humidity": 55, "wind": 2.5}
r = client.post("/predict", json=base)
pred = r.get_json()["predicted_power_kW"]
assert r.status_code == 200 and pred > 2.5, f"Unexpected prediction: {pred}"

# 3. Low actual output (actual 1.5 kW -> alert raised)
r = client.post("/predict", json={**base, "actual_power": 1.5})
data = r.get_json()
assert r.status_code == 200 and data["fault_alert"] is True

# 4. Slightly low output (actual 3.0 kW)
r = client.post("/predict", json={**base, "actual_power": 3.0})
assert r.status_code == 200

# 5. Missing field (wind left out -> 400 error)
no_wind = {k: v for k, v in base.items() if k != "wind"}
r = client.post("/predict", json=no_wind)
assert r.status_code == 400 and "missing field: wind" in r.get_json()["error"]

# 6. Out-of-range value (cloud 1.5 -> 400 error)
r = client.post("/predict", json={**base, "cloud": 1.5})
assert r.status_code == 400 and "cloud must be a number between 0 and 1" in r.get_json()["error"]

# 7. Night hour (2 am -> 0.0 kW)
r = client.post("/predict", json={**base, "hour": 2})
assert r.status_code == 200 and r.get_json()["predicted_power_kW"] == 0.0

# 8. Speed check over 100 calls
t0 = time.perf_counter()
for _ in range(100):
    client.post("/predict", json=base)
avg_ms = (time.perf_counter() - t0) * 1000 / 100

print(f"All 7 functional tests PASSED! | Sample Noon Prediction: {pred} kW | Avg Latency: {avg_ms:.2f} ms")