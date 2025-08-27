#!/usr/bin/env python3
import requests
import docker
import time
import subprocess
import os

def check_services():
    """Check if all services are running properly"""
    print("🔍 Service Health Check")
    print("=" * 40)
    
    # Check Docker
    try:
        client = docker.from_env()
        print("✅ Docker is running")
    except:
        print("❌ Docker is not running")
        return False
    
    # Check containers
    result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                          capture_output=True, text=True)
    containers = result.stdout.split('\n')
    
    if 'tf_serving_mnist' in containers:
        print("✅ TensorFlow Serving container is running")
    else:
        print("❌ TensorFlow Serving not running")
        return False
    
    # Check model status
    try:
        response = requests.get('http://localhost:8501/v1/models/mnist', timeout=10)
        if response.status_code == 200:
            print("✅ Model is loaded and serving")
            return True
        else:
            print(f"❌ Model not loaded: {response.status_code}")
            return False
    except:
        print("❌ Cannot connect to TF Serving")
        return False

if __name__ == "__main__":
    if check_services():
        print("\n🎉 All services are healthy!")
        print("\nYou can now run: python tests/test_rest_api_final.py")
    else:
        print("\n❌ Services need attention")
        print("\nRun: docker-compose -f docker/docker-compose.yml up -d")