import subprocess
import json
import smtplib
import argparse
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def load_production_metrics():
    """Load production metrics from JSON file"""
    try:
        with open('models/mnist/production_metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"accuracy": 0.0, "version": 0}

def send_notification(current_accuracy, new_accuracy, success=True, smtp_config=None):
    """Send notification (console only for now, fix email later)"""
    if success:
        print(f"✅ SUCCESS: New model promoted!")
        print(f"   Old accuracy: {current_accuracy:.4f}")
        print(f"   New accuracy: {new_accuracy:.4f}")
    else:
        print(f"❌ FAILED: Accuracy gate not met")
        print(f"   Current accuracy: {current_accuracy:.4f}")
        print(f"   New accuracy: {new_accuracy:.4f}")
        print("   Keeping previous model version")

def run_training():
    """Run the training script"""
    # Use the correct path to the training script
    result = subprocess.run(['python', 'scripts/train_model.py'], 
                           capture_output=True, text=True, cwd='.')
    
    print("Training output:", result.stdout)
    if result.stderr:
        print("Training errors:", result.stderr)
    
    if result.returncode != 0:
        print(f"Training failed with return code: {result.returncode}")
        return False, 0.0
    
    # Extract accuracy from output
    output_lines = result.stdout.split('\n')
    accuracy = 0.0
    for line in output_lines:
        if 'Test accuracy:' in line:
            try:
                accuracy = float(line.split(': ')[1].strip())
                break
            except (IndexError, ValueError):
                continue
    
    return True, accuracy

def main():
    parser = argparse.ArgumentParser(description='Model retraining pipeline')
    parser.add_argument('--min-accuracy', type=float, default=0.97,
                       help='Minimum accuracy required for promotion')
    args = parser.parse_args()
    
    # Get current production metrics
    prod_metrics = load_production_metrics()
    current_accuracy = prod_metrics.get('accuracy', 0.0)
    
    print(f"Current production accuracy: {current_accuracy:.4f}")
    print(f"Minimum accuracy required: {args.min_accuracy:.4f}")
    
    # Run training
    success, new_accuracy = run_training()
    
    if not success:
        print("Training failed completely")
        send_notification(current_accuracy, 0.0, success=False)
        return
    
    print(f"New model accuracy: {new_accuracy:.4f}")
    
    # Check accuracy gate
    accuracy_improved = new_accuracy > current_accuracy * 1.01  # 1% improvement
    meets_min_accuracy = new_accuracy >= args.min_accuracy
    
    if accuracy_improved or meets_min_accuracy:
        print(f"✅ Accuracy gate passed!")
        send_notification(current_accuracy, new_accuracy, success=True)
    else:
        print(f"❌ Accuracy gate failed!")
        send_notification(current_accuracy, new_accuracy, success=False)

if __name__ == "__main__":
    main()