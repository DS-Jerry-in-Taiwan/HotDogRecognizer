import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

img_path = "data/hotdog/test/hotdog/1000.png"
def test_api_predict():
    url = "http://127.0.0.1:8000/predict"
    files = {'file': open(img_path, 'rb')}
    response = requests.post(url, files=files)
    assert response.status_code == 200
    result = response.json()
    print(result)
    assert "prediction" in result