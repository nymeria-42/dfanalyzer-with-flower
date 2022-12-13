from flwr.client import start_numpy_client, NumPyClient
from os import environ
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import cifar10

# Make TensorFlow Log Less Verbose
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load Model (MobileNetV2): 10 Output Classes
model = MobileNetV2(input_shape=(32, 32, 3), classes=10, weights=None)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Load Data (CIFAR-10): Popular Colored Image Classification Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Server IPv4 or IPv6 Address
server_address="127.0.0.1:8080"

# Define Flower Client
class CifarClient(NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        batch_size = config["batch_size"]
        print("[SERVER ROUND {0}] Local Epochs: {1} | Batch Size: {2}".format(server_round, local_epochs, batch_size))
        model.set_weights(parameters)
        print("Fitting Model:")
        model.fit(x_train, y_train, epochs=local_epochs, batch_size=batch_size)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        print("Evaluating Model:")
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

# Root Certificates (SSL-Enabled Secure Connection)
# root_certificates = Path("../.cache/certificates/ca.crt").read_bytes()

# Start Flower Client
start_numpy_client(server_address=server_address,
                   client=CifarClient(),
                   root_certificates=None)

