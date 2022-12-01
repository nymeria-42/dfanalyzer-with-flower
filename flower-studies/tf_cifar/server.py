from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from pathlib import Path

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
df = Dataflow(dataflow_tag)

# Training Configuration For Each Round
def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 1 if server_round < 2 else 2,
        "batch_size": 128,
    }
    return config

# FedAvg Strategy (Federated Averaging)
strategy = FedAvg(fraction_fit=1.0,  # 100% of available clients for training
                  fraction_evaluate=0.5,  # 50% of available clients for evaluation
                  min_fit_clients=1,  # Minimum of 1 client for training
                  min_evaluate_clients=1,  # Minimum of 1 client for evaluation
                  min_available_clients=1,  # Wait until 1 client is available
                  on_fit_config_fn=fit_config) # Fit config function

# Server IPv4 or IPv6 Address
server_address="0.0.0.0:8080"


tf1 = Transformation("Config")
tf1_input = Set("iConfig", SetType.INPUT, 
    [Attribute("ROUNDS", AttributeType.NUMERIC), 
    Attribute("TIMEOUT", AttributeType.NUMERIC)])
tf1_output = Set("oConfig", SetType.OUTPUT, 
    [Attribute("ROUNDS2", AttributeType.NUMERIC), 
    Attribute("TIMEOUT2", AttributeType.NUMERIC)])
tf1.set_sets([tf1_input, tf1_output])
df.add_transformation(tf1)
df.save()

# Server Config

t1 = Task(1, dataflow_tag, "Config")
t1_input = DataSet("iConfig", [Element([1, 2])])
t1.add_dataset(t1_input)
t1.begin()

num_rounds = 2
round_timeout_in_seconds = 0
server_config = ServerConfig(num_rounds=num_rounds,
                             round_timeout=round_timeout_in_seconds)

t1_output= DataSet("oConfig", [Element([3, 4])])
t1.add_dataset(t1_output)
t1.end()
# Certificates (SSL-Enabled Secure Connection)
# certificates = (Path("../.cache/certificates/ca.crt").read_bytes(),
#                 Path("../.cache/certificates/server.pem").read_bytes(),
#                 Path("../.cache/certificates/server.key").read_bytes())

# Start Flower Server
start_server(server_address=server_address,
             config=server_config,
             strategy=strategy,
             certificates=None)

