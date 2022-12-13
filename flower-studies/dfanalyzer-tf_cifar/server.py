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

tf1 = Transformation("ServerConfig")
tf1_input = Set("iServerConfig", SetType.INPUT, 
    [Attribute("num_rounds", AttributeType.NUMERIC), 
    Attribute("round_timeout_in_seconds", AttributeType.NUMERIC)])
tf1_output = Set("oServerConfig", SetType.OUTPUT, 
    [Attribute("address", AttributeType.TEXT)])
tf1.set_sets([tf1_input, tf1_output])
df.add_transformation(tf1)

tf2 = Transformation("Strategy")

tf2_input = Set("iStrategy", SetType.INPUT, 
     [Attribute("fraction_fit", AttributeType.NUMERIC), 
    Attribute("fraction_evaluate", AttributeType.NUMERIC),
    Attribute("min_fit_clients", AttributeType.NUMERIC), 
    Attribute("min_evaluate_clients", AttributeType.NUMERIC),
    Attribute("min_available_clients", AttributeType.NUMERIC)])

tf2_output = Set("oStrategy", SetType.OUTPUT, 
    [ Attribute("type", AttributeType.TEXT)])

# definição de dependencia de task
tf1_output.set_type(SetType.INPUT)
tf1_output.dependency=tf1._tag

tf2.set_sets([tf1_output, tf2_input, tf2_output])
df.add_transformation(tf2)

tf3 = Transformation("TrainingConfig")
tf3_input = Set("iTrainingConfig", SetType.INPUT, 
    [Attribute("server_round", AttributeType.NUMERIC)])
tf3_output = Set("oTrainingConfig", SetType.OUTPUT, 
    [Attribute("server_round", AttributeType.NUMERIC),
    Attribute("local_epochs", AttributeType.NUMERIC), 
    Attribute("batch_size", AttributeType.NUMERIC)])

tf2_output.set_type(SetType.INPUT)
tf2_output.dependency=tf2._tag

tf3.set_sets([tf2_output, tf3_input, tf3_output])
df.add_transformation(tf3)

df.save()


# Training Configuration For Each Round
def fit_config(server_round: int):
    t3 = Task(3, dataflow_tag, "TrainingConfig", dependency=t2)
    t3_input = DataSet("iTrainingConfig", [Element([server_round])])
    t3.add_dataset(t3_input)
    t3.begin()

    config = {
        "server_round": server_round,
        "local_epochs": 1 if server_round < 2 else 2,
        "batch_size": 128,
    }

    
    t3_output= DataSet("oTrainingConfig", [Element([config["server_round"], config["local_epochs"], config["batch_size"]])])
    t3.add_dataset(t3_output)
    t3.end()
    return config


# Server IPv4 or IPv6 Address
server_address="0.0.0.0:8080"

# Server Config
num_rounds = 2
round_timeout_in_seconds = 0

t1 = Task(1, dataflow_tag, "ServerConfig")
t1_input = DataSet("iServerConfig", [Element([num_rounds, round_timeout_in_seconds])])
t1.add_dataset(t1_input)
t1.begin()

server_config = ServerConfig(num_rounds=num_rounds,
                             round_timeout=round_timeout_in_seconds)

t1_output= DataSet("oServerConfig", [Element([server_address])])
t1.add_dataset(t1_output)
t1.end()


# FedAvg Strategy (Federated Averaging)
fraction_fit=1.0
fraction_evaluate=0.5
min_fit_clients=1
min_evaluate_clients=1
min_available_clients=1

t2 = Task(2, dataflow_tag, "Strategy", dependency=t1)
t2_input = DataSet("iStrategy", [Element([fraction_fit, fraction_evaluate, min_fit_clients, min_evaluate_clients, min_available_clients])])
t2.add_dataset(t2_input)
t2.begin()

strategy = FedAvg(fraction_fit=fraction_fit,  # 100% of available clients for training
                  fraction_evaluate=fraction_evaluate,  # 50% of available clients for evaluation
                  min_fit_clients=min_fit_clients,  # Minimum of 1 client for training
                  min_evaluate_clients=min_evaluate_clients,  # Minimum of 1 client for evaluation
                  min_available_clients=min_available_clients,  # Wait until 1 client is available
                  on_fit_config_fn=fit_config) # Fit config function

t2_output= DataSet("oStrategy", [Element([type(strategy).__name__])])
t2.add_dataset(t2_output)
t2.end()


# Certificates (SSL-Enabled Secure Connection)
# certificates = (Path("../.cache/certificates/ca.crt").read_bytes(),
#                 Path("../.cache/certificates/server.pem").read_bytes(),
#                 Path("../.cache/certificates/server.key").read_bytes())

# Start Flower Server
start_server(server_address=server_address,
             config=server_config,
             strategy=strategy,
             certificates=None)
