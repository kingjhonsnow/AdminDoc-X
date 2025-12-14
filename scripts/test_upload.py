"""
Simple test script to simulate file upload to Flask endpoint
"""
import requests
import os

BASE_URL = "http://127.0.0.1:5000"
TEST_PDF = os.path.join("../data", "ahmed.pdf")

# Test 1: Check if Flask is up
try:
    print("Testing connection to Flask...")
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"Health check: {resp.status_code}")
    print(resp.json())
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Test 2: Upload file
print("\nTesting file upload...")
try:
    with open(TEST_PDF, 'rb') as f:
        files = {'file': f}
        resp = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        print(f"Upload response: {resp.status_code}")
        print(resp.json())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
