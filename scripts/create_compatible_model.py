#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import os
import json
import shutil

def create_and_save_compatible_model():
    """Create a model that's guaranteed to work with TF Serving"""
    print("🔄 Creating compatible model...")
    
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess - CRITICAL: Use proper reshaping
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    # Convert to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Create a SIMPLE model that always works
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train briefly
    print("🏋️ Training model...")
    model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), verbose=1)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'✅ Test accuracy: {test_acc:.4f}')

    # Save model in a TF Serving compatible way
    model_dir = "models/mnist"
    version_dir = f"{model_dir}/1"
    
    # Clean up
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(version_dir, exist_ok=True)

    # Save using tf.saved_model.save - THIS IS CRITICAL
    tf.saved_model.save(model, version_dir)
    
    # Verify the model can be loaded
    try:
        loaded_model = tf.saved_model.load(version_dir)
        print("✅ Model saved and can be loaded successfully")
        
        # Test prediction locally
        sample = x_test[0:1]
        serving_fn = loaded_model.signatures['serving_default']
        prediction = serving_fn(tf.constant(sample))
        
        print("📍 Local test prediction successful")
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

    # Save metadata
    metadata = {
        "accuracy": float(test_acc),
        "version": "1",
        "tf_version": tf.__version__
    }
    
    with open(f"{model_dir}/production_metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Model saved to: {version_dir}")
    print(f"📊 TensorFlow version: {tf.__version__}")
    return True

if __name__ == "__main__":
    success = create_and_save_compatible_model()
    if success:
        print("\n🎉 Model created successfully! Restart TensorFlow Serving:")
        print("docker-compose -f docker/docker-compose.yml up -d")
    else:
        print("\n❌ Model creation failed")