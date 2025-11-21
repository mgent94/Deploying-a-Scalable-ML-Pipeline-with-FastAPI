import requests
import json

# ==== URLs ====
URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{URL}/predict"

# ==== Test GET endpoint ====
print("Testing GET / ...")
r = requests.get(URL)
print("GET response:", r.json())

# ==== Build sample payload ====
payload = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# ==== Test POST /predict ====
print("\nTesting POST /predict ...")
headers = {"Content-Type": "application/json"}
r = requests.post(PREDICT_URL, data=json.dumps(payload), headers=headers)
print("POST response:", r.json())
