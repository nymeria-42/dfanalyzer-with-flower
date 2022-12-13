from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from pathlib import Path

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

# Server Config
num_rounds = 2
round_timeout_in_seconds = 0
server_config = ServerConfig(num_rounds=num_rounds,
                             round_timeout=round_timeout_in_seconds)

# Certificates (SSL-Enabled Secure Connection)
#certificates = (Path("../.cache/certificates/ca.crt").read_bytes(),
#                Path("../.cache/certificates/server.pem").read_bytes(),
#                Path("../.cache/certificates/server.key").read_bytes())

# Start Flower Server
start_server(server_address=server_address,
             config=server_config,
             strategy=strategy,
             certificates=None)
