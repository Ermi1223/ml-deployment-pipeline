import requests
import json
import random
import time
from collections import defaultdict

class CanaryTester:
    def __init__(self, base_url, traffic_percentage=0.1):
        self.base_url = base_url
        self.traffic_percentage = traffic_percentage
        self.results = defaultdict(list)
        
    def generate_test_data(self):
        # Generate random test data
        from tensorflow.keras.datasets import mnist
        (_, _), (x_test, y_test) = mnist.load_data()
        
        test_data = []
        for i in range(100):  # 100 test samples
            sample = x_test[i].reshape(28, 28, 1).astype('float32') / 255
            test_data.append({
                "data": sample.tolist(),
                "true_label": int(y_test[i])
            })
        
        return test_data
    
    def predict(self, data, version=None):
        url = f"{self.base_url}/v1/models/mnist"
        if version:
            url += f"/versions/{version}"
        url += ":predict"
        
        payload = json.dumps({"instances": [data]})
        headers = {"content-type": "application/json"}
        
        start_time = time.time()
        response = requests.post(url, data=payload, headers=headers)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            prediction = json.loads(response.text)['predictions'][0]
            predicted_class = np.argmax(prediction)
            return predicted_class, latency, True
        else:
            return None, latency, False
    
    def run_test(self, stable_version, canary_version, duration=3600):
        test_data = self.generate_test_data()
        end_time = time.time() + duration
        
        print(f"Starting canary test: {stable_version} vs {canary_version}")
        print(f"Traffic percentage: {self.traffic_percentage * 100}%")
        print(f"Test duration: {duration} seconds")
        
        while time.time() < end_time:
            for sample in test_data:
                # Route traffic based on percentage
                if random.random() < self.traffic_percentage:
                    version = canary_version
                    group = "canary"
                else:
                    version = stable_version
                    group = "stable"
                
                # Make prediction
                predicted, latency, success = self.predict(sample["data"], version)
                
                # Record results
                self.results[group].append({
                    "success": success,
                    "latency": latency,
                    "accurate": success and predicted == sample["true_label"]
                })
            
            time.sleep(1)  # Wait before next batch
        
        return self.analyze_results()
    
    def analyze_results(self):
        analysis = {}
        
        for group, results in self.results.items():
            total = len(results)
            successes = sum(1 for r in results if r["success"])
            accuracies = sum(1 for r in results if r.get("accurate", False))
            latencies = [r["latency"] for r in results if r["success"]]
            
            analysis[group] = {
                "request_count": total,
                "success_rate": successes / total if total > 0 else 0,
                "accuracy": accuracies / successes if successes > 0 else 0,
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            }
        
        return analysis

if __name__ == "__main__":
    tester = CanaryTester("http://localhost:8501", traffic_percentage=0.1)
    results = tester.run_test("1", "2", duration=600)  # 10 minute test
    
    print("Canary Test Results:")
    for group, metrics in results.items():
        print(f"\n{group.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Determine if canary should be promoted
    canary_metrics = results["canary"]
    stable_metrics = results["stable"]
    
    success_criteria = (
        canary_metrics["success_rate"] >= stable_metrics["success_rate"] * 0.95 and
        canary_metrics["accuracy"] >= stable_metrics["accuracy"] * 0.95 and
        canary_metrics["avg_latency"] <= stable_metrics["avg_latency"] * 1.1
    )
    
    if success_criteria:
        print("\n✓ Canary test PASSED - Can be promoted to production")
    else:
        print("\n✗ Canary test FAILED - Should not be promoted")