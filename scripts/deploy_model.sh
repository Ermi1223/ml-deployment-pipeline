import argparse
import subprocess
import time
import requests
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models' / 'mnist'




def restart_tf_serving():
# Adjust path if docker-compose located elsewhere
subprocess.run(['docker-compose', '-f', str(ROOT / 'docker' / 'docker-compose.yml'), 'restart', 'tf-serving'], check=True)




def health_check(timeout=10):
start = time.time()
url = 'http://localhost:8501/v1/models/mnist'
while time.time() - start < timeout:
try:
r = requests.get(url, timeout=2)
if r.status_code == 200:
return True
except Exception:
pass
time.sleep(1)
return False




def sample_inference():
# Send one sample to model to ensure predict works
from tensorflow.keras.datasets import mnist
import numpy as np
(_, _), (x_test, y_test) = mnist.load_data()
sample = x_test[0].reshape(28,28,1).astype('float32')/255.0
payload = {"instances": [sample.tolist()]}
r = requests.post('http://localhost:8501/v1/models/mnist:predict', json=payload, timeout=5)
return r.status_code == 200




def main(version):
print(f"Deploying version {version}")
restart_tf_serving()
time.sleep(3)
if not health_check(timeout=15):
raise RuntimeError('Healt