#!/usr/bin/env python3
import requests
import json
import numpy as np
from tensorflow.keras.datasets import mnist
import time

def test_rest_api_final():
    """Final working test script"""
    print("🎯 Final REST API Test")
    print("=" * 50)
    
    # Wait for service
    time.sleep(3)
    
    # Check server status
    try:
        status = requests.get('http://localhost:8501/v1/models/mnist', timeout=10)
        if status.status_code == 200:
            print("✅ TensorFlow Serving is running")
        else:
            print(f"❌ TF Serving error: {status.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        return False

    # Get test data
    (_, _), (x_test, y_test) = mnist.load_data()
    sample = x_test[0:1].reshape(1, 28, 28, 1).astype('float32') / 255
    
    print(f"Testing digit: {y_test[0]}")
    
    # Try multiple formats
    test_payloads = [
        {"instances": sample.tolist()},
        {"inputs": sample.tolist()},
    ]
    
    headers = {"content-type": "application/json"}
    url = 'http://localhost:8501/v1/models/mnist:predict'
    
    for i, payload in enumerate(test_payloads):
        try:
            print(f"\n🔧 Attempt {i+1}: {list(payload.keys())[0]}")
            
            response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=15)
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS! Response keys:", list(result.keys()))
                
                # Find predictions
                for key in result.keys():
                    if key != 'model_spec' and isinstance(result[key], list):
                        predictions = result[key][0]
                        predicted = np.argmax(predictions)
                        confidence = np.max(predictions)
                        print(f"🎯 Prediction: {predicted} ({confidence:.2%} confidence)")
                        print(f"📊 Actual: {y_test[0]}")
                        print(f"✅ Correct: {predicted == y_test[0]}")
                        return True
                
                return True
            else:
                print(f"❌ Error: {response.text[:100]}...")
                
        except Exception as e:
            print(f"💥 Exception: {e}")
    
    return False

if __name__ == "__main__":
    success = test_rest_api_final()
    if success:
        print("\n🎉 🎉 🎉 SUCCESS! API is working!")
    else:
        print("\n❌ Test failed - check model and service")