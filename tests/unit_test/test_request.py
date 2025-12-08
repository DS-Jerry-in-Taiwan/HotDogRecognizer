import requests

def test_api_predict():
    url = "http://127.0.0.1:8000/predict/"
    img_path = "/home/ubuntu/projects/HotDogRecognizer/data/hotdog/test/hotdog/1000.png"
    with open(img_path, "rb") as f:
        files = {"file": (img_path, f, "image/png")}
        response = requests.post(url, files=files)
    assert response.status_code == 200
    result = response.json()
    print("API result:", result)
    assert "prediction" in result
    assert "probabilities" in result

if __name__ == "__main__":
    test_api_predict()