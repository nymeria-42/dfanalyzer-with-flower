from flwr.client import start_numpy_client, NumPyClient
from os import environ
from pathlib import Path
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import cifar10

import timeit
from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from dfa_lib_python.task_status import TaskStatus
from dfa_lib_python.extractor_extension import ExtractorExtension

dataflow_tag = "flower-df"

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

t1 = Task(1, dataflow_tag, "ServerConfig")
t2 = Task(2, dataflow_tag, "Strategy", dependency=t1)
t3 = Task(3, dataflow_tag, "TrainingConfig", dependency=t2)

CLIENT_NUMBER = 1

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
        t4 = Task(4, dataflow_tag, "ClientTraining", dependency = t3)

        t4_input = DataSet("iClientTraining", [Element([CLIENT_NUMBER, server_round, len(x_train)])])
        t4.add_dataset(t4_input)
        t4.begin()
        start = timeit.default_timer()

        model.fit(x_train, y_train, epochs=local_epochs, batch_size=batch_size)

        end = timeit.default_timer()
        t4_output= DataSet("oClientTraining", [Element([CLIENT_NUMBER, end - start])])
        t4.add_dataset(t4_output)
        t4.end()

        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        print("Evaluating Model:")
        t5 = Task(5, dataflow_tag, "ClientEvaluation", dependency = Task(4, dataflow_tag, "ClientTraining", dependency = t3))
        t5_input = DataSet("iClientEvaluation", [Element([CLIENT_NUMBER, len(x_test)])])
        t5.add_dataset(t5_input)
        t5.begin()
        
        start = timeit.default_timer()

        loss, accuracy = model.evaluate(x_test, y_test)

        end = timeit.default_timer()

        t5_output= DataSet("oClientEvaluation", [Element([CLIENT_NUMBER, loss, accuracy, end - start])])
        t5.add_dataset(t5_output)
        t5.end()
        return loss, len(x_test), {"accuracy": accuracy}

# Root Certificates (SSL-Enabled Secure Connection)
# root_certificates = Path("../.cache/certificates/ca.crt").read_bytes()

# Start Flower Client
start_numpy_client(server_address=server_address,
                   client=CifarClient(),
                   root_certificates=None)
