import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

def predict_grpc():
    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    # Create PredictRequest
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mnist'
    request.model_spec.signature_name = 'serving_default'
    
    # Dummy MNIST image
    dummy_image = np.zeros((1, 28*28), dtype=np.float32)
    request.inputs['flatten_input'].CopyFrom(
        tf.make_tensor_proto(dummy_image, shape=dummy_image.shape)
    )
    
    # Send request
    response = stub.Predict(request, timeout=10.0)
    print("gRPC Prediction Result:")
    print(response)

if __name__ == "__main__":
    print("=== gRPC API Test ===")
    predict_grpc()
