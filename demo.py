import requests

url = 'http://127.0.0.1:1234/predict'
data = {
    "X": [120000, 28, 200, 0, 0, 0, 0, 0, 0, -1, 2, 2, 2] 
}
headers = {'Content-Type': 'application/json'}
response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print(f"Error: {response.status_code}, Message: {response.json()}")
