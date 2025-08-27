import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import os
import json
import time

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_evaluate_model():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = create_model()
    model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1, validation_data=(x_test, y_test))
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")
    return model, acc

def export_model(model, accuracy):
    export_dir = os.path.join("exported_model", "mnist")
    os.makedirs(export_dir, exist_ok=True)

    # Use integer version number (incremental)
    existing_versions = [int(d) for d in os.listdir(export_dir) if d.isdigit()]
    version = max(existing_versions, default=0) + 1

    export_path = os.path.join(export_dir, str(version))
    tf.saved_model.save(model, export_path)
    print(f"Model exported to {export_path}")

    # Track production metrics
    metrics_path = os.path.join(export_dir, "production_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {"accuracy": 0, "version": 0}

    # Promote if better
    if accuracy > metrics["accuracy"]:
        metrics["accuracy"] = float(accuracy)
        metrics["version"] = version
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        print(f"✅ New model promoted to production (version {version}, acc {accuracy:.4f})")
    else:
        print(f"⚠️ Model accuracy {accuracy:.4f} not better than production {metrics['accuracy']:.4f}")

    return version, accuracy


if __name__ == "__main__":
    model, accuracy = train_and_evaluate_model()
    version, accuracy = export_model(model, accuracy)
    print(f"Finished: version {version} with accuracy {accuracy:.4f}")
