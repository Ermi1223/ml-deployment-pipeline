import time
import statistics
import concurrent.futures
import requests
import json
import numpy as np
from tensorflow.keras.datasets import mnist

class ModelPerformanceTester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.rest_url = f"{base_url}/v1/models/mnist:predict"
        
    def prepare_test_data(self, num_samples=100):
        """Prepare test data from MNIST dataset"""
        (_, _), (x_test, y_test) = mnist.load_data()
        test_data = []
        for i in range(min(num_samples, len(x_test))):
            image = x_test[i].reshape(28, 28, 1).astype('float32') / 255
            test_data.append({
                'image': image,  # Will flatten in request if needed
                'true_label': int(y_test[i])
            })
        return test_data
    
    def test_rest_api(self, test_data, num_requests=100, concurrent_requests=10):
        """Test REST API performance"""
        latencies = []
        correct_predictions = 0
        
        def make_request(data):
            payload = json.dumps({"instances": [data['image'].tolist()]})
            headers = {'Content-Type': 'application/json'}
            start_time = time.time()
            try:
                response = requests.post(self.rest_url, data=payload, headers=headers, timeout=5)
                latency = time.time() - start_time
                if response.status_code == 200:
                    predictions = response.json().get('predictions') or response.json().get('outputs')
                    predicted_class = np.argmax(predictions[0])
                    correct = predicted_class == data['true_label']
                    return latency, correct
                return latency, False
            except requests.exceptions.RequestException:
                return 0, False
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request, test_data[i % len(test_data)]) for i in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                latency, correct = future.result()
                if latency > 0:
                    latencies.append(latency)
                if correct:
                    correct_predictions += 1
        
        accuracy = correct_predictions / num_requests
        avg_latency = statistics.mean(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        throughput = num_requests / sum(latencies) if latencies else 0
        
        return {
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'p95_latency': p95_latency,
            'throughput': throughput,
            'total_requests': num_requests,
            'correct_predictions': correct_predictions
        }
    
    def test_load_performance(self, test_data, duration=60, concurrent_requests=20):
        """Test sustained load performance"""
        start_time = time.time()
        results = {
            'requests_completed': 0,
            'requests_failed': 0,
            'latencies': [],
            'correct_predictions': 0
        }
        
        def worker():
            while time.time() - start_time < duration:
                data = test_data[np.random.randint(0, len(test_data))]
                payload = json.dumps({"instances": [data['image'].tolist()]})
                headers = {'Content-Type': 'application/json'}
                try:
                    request_start = time.time()
                    response = requests.post(self.rest_url, data=payload, headers=headers, timeout=5)
                    latency = time.time() - request_start
                    results['latencies'].append(latency)
                    results['requests_completed'] += 1
                    if response.status_code == 200:
                        predictions = response.json().get('predictions') or response.json().get('outputs')
                        predicted_class = np.argmax(predictions[0])
                        if predicted_class == data['true_label']:
                            results['correct_predictions'] += 1
                    else:
                        results['requests_failed'] += 1
                except requests.exceptions.RequestException:
                    results['requests_failed'] += 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(worker) for _ in range(concurrent_requests)]
            time.sleep(duration)
            # Threads will naturally stop after duration
        
        total_requests = results['requests_completed'] + results['requests_failed']
        accuracy = results['correct_predictions'] / results['requests_completed'] if results['requests_completed'] > 0 else 0
        avg_latency = statistics.mean(results['latencies']) if results['latencies'] else 0
        p95_latency = np.percentile(results['latencies'], 95) if results['latencies'] else 0
        throughput = results['requests_completed'] / duration
        
        return {
            'duration': duration,
            'total_requests': total_requests,
            'requests_completed': results['requests_completed'],
            'requests_failed': results['requests_failed'],
            'accuracy': accuracy,
            'avg_latency': avg_latency,
            'p95_latency': p95_latency,
            'throughput': throughput,
            'error_rate': results['requests_failed'] / total_requests if total_requests > 0 else 0
        }
    
    def run_comprehensive_test(self):
        print("Preparing test data...")
        test_data = self.prepare_test_data(1000)
        
        print("Running baseline performance test...")
        baseline = self.test_rest_api(test_data, num_requests=100, concurrent_requests=10)
        
        print("Running load test...")
        load_test = self.test_load_performance(test_data, duration=120, concurrent_requests=20)
        
        report = {
            'timestamp': time.time(),
            'baseline_performance': baseline,
            'load_test_performance': load_test,
            'summary': {
                'baseline_accuracy': baseline['accuracy'],
                'baseline_throughput': baseline['throughput'],
                'load_test_throughput': load_test['throughput'],
                'load_test_error_rate': load_test['error_rate']
            }
        }
        
        with open(f"performance_report_{int(time.time())}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    tester = ModelPerformanceTester()
    print("Starting comprehensive model performance test...")
    report = tester.run_comprehensive_test()
    
    print("\n=== PERFORMANCE REPORT ===")
    print(f"Baseline Accuracy: {report['baseline_performance']['accuracy']:.3f}")
    print(f"Baseline Throughput: {report['baseline_performance']['throughput']:.1f} req/s")
    print(f"Load Test Throughput: {report['load_test_performance']['throughput']:.1f} req/s")
    print(f"Load Test Error Rate: {report['load_test_performance']['error_rate']:.3f}")
    print(f"Load Test P95 Latency: {report['load_test_performance']['p95_latency']:.3f}s")
    
    success = (
        report['baseline_performance']['accuracy'] >= 0.97 and
        report['load_test_performance']['error_rate'] <= 0.05 and
        report['load_test_performance']['p95_latency'] <= 0.5
    )
    
    print(f"\nPerformance Test Result: {'PASS' if success else 'FAIL'}")
    return success

if __name__ == "__main__":
    main()
