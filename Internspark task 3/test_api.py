import os
import requests
from PIL import Image

# 1. Create a dummy image to test with
test_image_path = "test_image.jpg"
img = Image.new("RGB", (32, 32), color="blue")
img.save(test_image_path)
print(f"Generated test image: '{test_image_path}'")

# 2. Define target API endpoint (Localhost port 8000)
url = "http://localhost:8000/predict"

try:
    # 3. Open image file and send POST request
    with open(test_image_path, "rb") as f:
        files = {"file": (test_image_path, f, "image/jpeg")}
        print(f"Sending request to: {url}...")
        response = requests.post(url, files=files)

    # 4. Print response status and JSON payload
    print(f"\nResponse Status Code: {response.status_code}")
    print("Response JSON Payload:")
    print(response.json())

finally:
    # Cleanup temporary image file
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
