#!/usr/bin/env python3
import requests
import json
import time
import pandas as pd
from datetime import datetime
from prometheus_client.parser import text_string_to_metric_families

class ModelMonitor:
    def __init__(self, tf_serving_url="http://localhost:8501"):
        self.tf_serving_url = tf_serving_url
        self.metrics_url = f"{tf_serving_url}/metrics"  # ✅ Correct URL
    
    def check_metrics_available(self):
        """Check if metrics endpoint is available"""
        try:
            response = requests.get(self.metrics_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def fetch_metrics(self):
        """Fetch Prometheus metrics from TF Serving"""
        try:
            response = requests.get(self.metrics_url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching metrics: {e}")
            return None
    
    def get_basic_metrics(self):
        """Get basic metrics through the TF Serving API"""
        try:
            status_url = f"{self.tf_serving_url}/v1/models/mnist"
            status_response = requests.get(status_url, timeout=5)
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                print("Model status:", json.dumps(status_data, indent=2))
        except Exception as e:
            print(f"Error getting basic metrics: {e}")
    
    def parse_metrics(self, metrics_data):
        """Parse and summarize some key metrics"""
        summary = {}
        for family in text_string_to_metric_families(metrics_data):
            if 'tensorflow' in family.name or 'serving' in family.name:
                summary[family.name] = []
                for sample in family.samples:
                    summary[family.name].append({
                        "name": sample.name,
                        "value": sample.value,
                        "labels": sample.labels
                    })
        return summary
    
    def run(self):
        """Main monitoring loop"""
        print("Starting Model Monitor...")
        print(f"Metrics URL: {self.metrics_url}")
        
        if not self.check_metrics_available():
            print("❌ Prometheus metrics not available")
            self.get_basic_metrics()
            return
        
        print("✅ Prometheus metrics available")
        
        try:
            while True:
                metrics_data = self.fetch_metrics()
                
                if metrics_data:
                    print(f"\n📊 Metrics collected at {datetime.now()}")
                    print("=" * 50)
                    
                    metrics_summary = self.parse_metrics(metrics_data)
                    
                    # Print first 3 samples per metric family
                    for family_name, samples in metrics_summary.items():
                        print(f"{family_name}:")
                        for s in samples[:3]:
                            print(f"  {s['name']}: {s['value']} {s['labels']}")
                    
                    print("=" * 50)
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

def main():
    monitor = ModelMonitor()
    
    # Check if TensorFlow Serving is running
    try:
        response = requests.get('http://localhost:8501/v1/models/mnist', timeout=5)
        if response.status_code == 200:
            print("✅ TensorFlow Serving is running")
        else:
            print("❌ TensorFlow Serving not responding properly")
            return
    except:
        print("❌ TensorFlow Serving not running")
        print("Please start it with: docker-compose -f docker/docker-compose.yml up -d")
        return
    
    # Start monitoring
    monitor.run()

if __name__ == "__main__":
    main()
