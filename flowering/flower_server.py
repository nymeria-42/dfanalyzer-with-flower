from argparse import ArgumentParser
from configparser import ConfigParser
from logging import FileHandler, Formatter, getLevelName, Logger, StreamHandler
from flwr.common import Metrics, NDArrays, Parameters
from flwr.server import Server, ServerConfig, SimpleClientManager, start_server
from flwr.server.strategy import FedAvg, FedAvgM, Strategy
from pathlib import Path
from re import findall
from typing import Any, List, Optional, Tuple


class FlowerServer:

    def __init__(self,
                 server_id: int,
                 server_config_file: Path) -> None:
        # Server's ID and Config File.
        self.server_id = server_id
        self.server_config_file = server_config_file
        # Server's Config File Settings.
        self.general_settings = None
        self.logging_settings = None
        self.fl_settings = None
        self.ssl_settings = None
        self.grpc_settings = None
        self.training_hyper_parameters_settings = None
        self.testing_hyper_parameters_settings = None
        # Other Attributes.
        self.logger = None
        self.flower_server = None
        self.flower_server_config = None
        self.fl_round = None
        self.global_model_initial_parameters = None
        self.global_model_parameters = None

    @staticmethod
    def parse_config_section(config_parser: ConfigParser,
                             section_name: str) -> dict:
        parsed_section = {key: value for key, value in config_parser[section_name].items()}
        for key, value in parsed_section.items():
            if value == "None":
                parsed_section[key] = None
            elif value in ["True", "Yes"]:
                parsed_section[key] = True
            elif value in ["False", "No"]:
                parsed_section[key] = False
            elif value.isdigit():
                parsed_section[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                parsed_section[key] = float(value)
            elif not findall(r"%\(.*?\)s+", value) and findall(r"\[.*?]+", value):
                aux_list = value.replace("[", "").replace("]", "").replace(" ", "").split(",")
                for index, item in enumerate(aux_list):
                    if item.isdigit():
                        aux_list[index] = int(item)
                    elif item.replace(".", "", 1).isdigit():
                        aux_list[index] = float(item)
                parsed_section[key] = aux_list
            elif not findall(r"%\(.*?\)s+", value) and findall(r"\(.*?\)+", value):
                aux_list = value.replace("(", "").replace(")", "").replace(" ", "").split(",")
                for index, item in enumerate(aux_list):
                    if item.isdigit():
                        aux_list[index] = int(item)
                    elif item.replace(".", "", 1).isdigit():
                        aux_list[index] = float(item)
                parsed_section[key] = tuple(aux_list)
        return parsed_section

    def set_attribute(self,
                      attribute_name: str,
                      attribute_value: Any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> Any:
        return getattr(self, attribute_name)

    def parse_flower_server_config_file(self) -> None:
        # Get Server's Config File.
        server_config_file = self.get_attribute("server_config_file")
        # Init ConfigParser Object.
        cp = ConfigParser()
        cp.optionxform = str
        cp.read(filenames=server_config_file, encoding="utf-8")
        # Parse 'General Settings' and Set Attributes.
        general_settings = self.parse_config_section(cp, "General Settings")
        self.set_attribute("general_settings", general_settings)
        # Parse 'Logging Settings' and Set Attributes.
        logging_settings = self.parse_config_section(cp, "Logging Settings")
        self.set_attribute("logging_settings", logging_settings)
        # Parse 'FL Settings' and Set Attributes.
        fl_settings = self.parse_config_section(cp, "FL Settings")
        self.set_attribute("fl_settings", fl_settings)
        # Parse 'SSL Settings' and Set Attributes.
        ssl_settings = self.parse_config_section(cp, "SSL Settings")
        self.set_attribute("ssl_settings", ssl_settings)
        # Parse 'gRPC Settings' and Set Attributes.
        grpc_settings = self.parse_config_section(cp, "gRPC Settings")
        self.set_attribute("grpc_settings", grpc_settings)
        # Parse 'Training Hyper-parameters Settings' and Set Attributes.
        training_hyper_parameters_settings = self.parse_config_section(cp, "Training Hyper-parameters Settings")
        self.set_attribute("training_hyper_parameters_settings", training_hyper_parameters_settings)
        # Parse 'Testing Hyper-parameters Settings' and Set Attributes.
        testing_hyper_parameters_settings = self.parse_config_section(cp, "Testing Hyper-parameters Settings")
        self.set_attribute("testing_hyper_parameters_settings", testing_hyper_parameters_settings)
        # Unbind ConfigParser Object (Garbage Collector).
        del cp

    def load_logger(self) -> Optional[Logger]:
        logger = None
        general_settings = self.get_attribute("general_settings")
        if general_settings["enable_logging"]:
            logger_name = "FlowerServer_" + str(self.get_attribute("server_id"))
            logging_settings = self.get_attribute("logging_settings")
            logger = Logger(name=logger_name,
                            level=logging_settings["level"])
            formatter = Formatter(fmt=logging_settings["format"],
                                  datefmt=logging_settings["date_format"])
            if logging_settings["log_to_file"]:
                file_handler = FileHandler(filename=logging_settings["file_name"],
                                           mode=logging_settings["file_mode"],
                                           encoding=logging_settings["encoding"])
                file_handler.setLevel(logger.getEffectiveLevel())
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            if logging_settings["log_to_console"]:
                console_handler = StreamHandler()
                console_handler.setLevel(logger.getEffectiveLevel())
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        return logger

    def log_message(self,
                    message: str,
                    message_level: str) -> None:
        logger = self.get_attribute("logger")
        if logger and getLevelName(logger.getEffectiveLevel()) != "NOTSET":
            if message_level == "DEBUG":
                logger.debug(msg=message)
            elif message_level == "INFO":
                logger.info(msg=message)
            elif message_level == "WARNING":
                logger.warning(msg=message)
            elif message_level == "ERROR":
                logger.error(msg=message)
            elif message_level == "CRITICAL":
                logger.critical(msg=message)

    @staticmethod
    def load_global_model_initial_parameters() -> Optional[Parameters]:
        """Server-side parameter initialization. A powerful mechanism which can be used, for example:
        \n - To resume the training from a previously saved checkpoint;
        \n - To implement hybrid approaches, such as to fine-tune a pre-trained model using federated learning.
        \n If no parameters are returned, the server will randomly select one client and ask its parameters."""
        # TODO: To Implement (If Ever Needed)...
        return None

    def get_grpc_listen_ip_address_and_port(self) -> str:
        grpc_settings = self.get_attribute("grpc_settings")
        return grpc_settings["grpc_listen_ip_address"] + ":" + str(grpc_settings["grpc_listen_port"])

    def get_grpc_max_message_length_in_bytes(self) -> int:
        return self.get_attribute("grpc_settings")["grpc_max_message_length_in_bytes"]

    @staticmethod
    def instantiate_simple_client_manager() -> SimpleClientManager:
        return SimpleClientManager()

    @staticmethod
    def evaluate_fn(fl_round: int,
                    global_model_current_parameters: NDArrays,
                    evaluate_config: dict) -> Optional[Metrics]:
        """Server-side (Centralized) evaluation function called by Flower after every training round.
        \nRequires a server-side dataset to evaluate the newly aggregated model without sending it to the Clients.
        \nThe 'losses_centralized' and 'metrics_centralized' will only contain values using this centralized evaluation.
        \nAlternative: Client-side (Federated) evaluation."""
        # TODO: To Implement (If Ever Needed)...
        return None

    def on_fit_config_fn(self,
                         fl_round: int) -> Optional[dict]:
        """Training configuration function called by Flower before each training round."""
        # Update the Current FL Round (Necessary Workaround on Flower v1.1.0).
        self.set_attribute("fl_round", fl_round)
        # Log the Current FL Round (If Logger is Enabled for "INFO" Level).
        message = "[Server {0}] Current FL Round: {1}".format(self.get_attribute("server_id"),
                                                              self.get_attribute("fl_round"))
        self.log_message(message, "INFO")
        # Get the Training Hyper-parameters Settings.
        training_hyper_parameters_settings = self.get_attribute("training_hyper_parameters_settings")
        # Replace All Values of None Type to "None" String (Necessary Workaround on Flower v1.1.0).
        training_hyper_parameters_settings = \
            {k: ("None" if v is None else v) for k, v in training_hyper_parameters_settings.items()}
        # Set the Training Configuration to be Sent to All Participating Clients.
        fit_config = {"fl_round": fl_round}
        fit_config.update(training_hyper_parameters_settings)
        # Log the Training Configuration (If Logger is Enabled for "DEBUG" Level).
        message = "[Server {0} | FL Round {1}] Fit Config: {2}".format(self.get_attribute("server_id"),
                                                                       fit_config["fl_round"],
                                                                       fit_config)
        self.log_message(message, "DEBUG")
        # Return the Training Configuration to be Sent to All Participating Clients.
        return fit_config

    def on_evaluate_config_fn(self,
                              fl_round: int) -> Optional[dict]:
        """Testing configuration function called by Flower before each testing round."""
        # Get the Testing Hyper-parameters Settings.
        testing_hyper_parameters_settings = self.get_attribute("testing_hyper_parameters_settings")
        # Replace All Values of None Type to "None" String (Necessary Workaround on Flower v1.1.0).
        testing_hyper_parameters_settings = \
            {k: ("None" if v is None else v) for k, v in testing_hyper_parameters_settings.items()}
        # Set the Testing Configuration to be Sent to All Participating Clients.
        evaluate_config = {"fl_round": fl_round}
        evaluate_config.update(testing_hyper_parameters_settings)
        # Log the Testing Configuration (If Logger is Enabled for "DEBUG" Level).
        message = "[Server {0} | FL Round {1}] Evaluate Config: {2}".format(self.get_attribute("server_id"),
                                                                            evaluate_config["fl_round"],
                                                                            evaluate_config)
        self.log_message(message, "DEBUG")
        # Return the Testing Configuration to be Sent to All Participating Clients.
        return evaluate_config

    def fit_metrics_aggregation_fn(self,
                                   training_metrics: List[Tuple[int, Metrics]]) -> Optional[Metrics]:
        """Metrics aggregation function called by Flower after every training round."""
        # Get the Total Number of Participating Clients.
        total_num_clients = len(training_metrics)
        # Get the Training Metrics Names.
        metrics_names_list = list(training_metrics[0][1].keys())
        # Multiply Each Training Metrics Value of Each Participating Client
        # By His Number of Training Examples (Client's Contribution).
        metrics_products_list = []
        for metric_name in metrics_names_list:
            metric_product = [num_examples * metric[metric_name] for num_examples, metric in training_metrics]
            metrics_products_list.append(metric_product)
        # Get the Total Number of Training Examples (of All Participating Clients).
        total_num_examples = sum([num_examples for num_examples, _ in training_metrics])
        # Aggregate the Training Metrics (Weighted Average).
        aggregated_metrics = {}
        for metric_index in range(0, len(metrics_names_list)):
            metric_name = metrics_names_list[metric_index]
            weighted_average_metric = sum(metrics_products_list[metric_index]) / total_num_examples
            aggregated_metrics[metric_name] = weighted_average_metric
        # Log the Aggregated Training Metrics (If Logger is Enabled for "INFO" Level).
        message = "[Server {0} | FL Round {1} | {2}] Aggregated Training Metrics (Weighted Average): {3}" \
            .format(self.get_attribute("server_id"),
                    self.get_attribute("fl_round"),
                    "".join([str(total_num_clients), " Clients" if total_num_clients > 1 else " Client"]),
                    aggregated_metrics)
        self.log_message(message, "INFO")
        # Return the Aggregated Training Metrics.
        return aggregated_metrics

    def evaluate_metrics_aggregation_fn(self,
                                        testing_metrics: List[Tuple[int, Metrics]]) -> Optional[Metrics]:
        """Metrics aggregation function called by Flower after every testing round."""
        # Get the Total Number of Participating Clients.
        total_num_clients = len(testing_metrics)
        # Get the Testing Metrics Names.
        metrics_names_list = list(testing_metrics[0][1].keys())
        # Multiply Each Testing Metrics Value of Each Participating Client
        # By His Number of Testing Examples (Client's Contribution).
        metrics_products_list = []
        for metric_name in metrics_names_list:
            metric_product = [num_examples * metric[metric_name] for num_examples, metric in testing_metrics]
            metrics_products_list.append(metric_product)
        # Get the Total Number of Testing Examples (of All Participating Clients).
        total_num_examples = sum([num_examples for num_examples, _ in testing_metrics])
        # Aggregate the Testing Metrics (Weighted Average).
        aggregated_metrics = {}
        for metric_index in range(0, len(metrics_names_list)):
            metric_name = metrics_names_list[metric_index]
            weighted_average_metric = sum(metrics_products_list[metric_index]) / total_num_examples
            aggregated_metrics[metric_name] = weighted_average_metric
        # Log the Aggregated Testing Metrics (If Logger is Enabled for "INFO" Level).
        message = "[Server {0} | FL Round {1} | {2}] Aggregated Testing Metrics (Weighted Average): {3}" \
            .format(self.get_attribute("server_id"),
                    self.get_attribute("fl_round"),
                    "".join([str(total_num_clients), " Clients" if total_num_clients > 1 else " Client"]),
                    aggregated_metrics)
        self.log_message(message, "INFO")
        # Return the Aggregated Testing Metrics.
        return aggregated_metrics

    def instantiate_server_aggregation_strategy(self) -> Strategy:
        # Get Server Config File.
        server_config_file = self.get_attribute("server_config_file")
        # Init ConfigParser Object.
        cp = ConfigParser()
        cp.optionxform = str
        cp.read(filenames=server_config_file, encoding="utf-8")
        fl_settings = self.get_attribute("fl_settings")
        server_aggregation_strategy = None
        if fl_settings["server_aggregation_strategy"] == "FedAvg":
            # FedAvg - Federated Averaging Aggregation Strategy.
            server_aggregation_strategy = \
                FedAvg(fraction_fit=fl_settings["fraction_fit"],
                       fraction_evaluate=fl_settings["fraction_evaluate"],
                       min_fit_clients=fl_settings["min_fit_clients"],
                       min_evaluate_clients=fl_settings["min_evaluate_clients"],
                       min_available_clients=fl_settings["min_available_clients"],
                       evaluate_fn=self.evaluate_fn,
                       on_fit_config_fn=self.on_fit_config_fn,
                       on_evaluate_config_fn=self.on_evaluate_config_fn,
                       accept_failures=fl_settings["accept_rounds_containing_failures"],
                       initial_parameters=self.get_attribute("global_model_initial_parameters"),
                       fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
                       evaluate_metrics_aggregation_fn=self.evaluate_metrics_aggregation_fn)
        elif fl_settings["server_aggregation_strategy"] == "FedAvgM":
            # Parse 'FedAvgM Settings'.
            fed_avg_m_settings = self.parse_config_section(cp, "FedAvgM Settings")
            # FedAvgM - Federated Averaging with Server Momentum Aggregation Strategy.
            server_aggregation_strategy = \
                FedAvgM(fraction_fit=fl_settings["fraction_fit"],
                        fraction_evaluate=fl_settings["fraction_evaluate"],
                        min_fit_clients=fl_settings["min_fit_clients"],
                        min_evaluate_clients=fl_settings["min_evaluate_clients"],
                        min_available_clients=fl_settings["min_available_clients"],
                        evaluate_fn=self.evaluate_fn,
                        on_fit_config_fn=self.on_fit_config_fn,
                        on_evaluate_config_fn=self.on_evaluate_config_fn,
                        accept_failures=fl_settings["accept_rounds_containing_failures"],
                        initial_parameters=self.get_attribute("global_model_initial_parameters"),
                        fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
                        evaluate_metrics_aggregation_fn=self.evaluate_metrics_aggregation_fn,
                        server_learning_rate=fed_avg_m_settings["server_learning_rate"],
                        server_momentum=fed_avg_m_settings["server_momentum"])
        # Unbind ConfigParser Object (Garbage Collector).
        del cp
        return server_aggregation_strategy

    def instantiate_flower_server(self) -> Server:
        # Instantiate Simple Client Manager.
        simple_client_manager = self.instantiate_simple_client_manager()
        # Instantiate Server's Aggregation Strategy.
        server_aggregation_strategy = self.instantiate_server_aggregation_strategy()
        # Instantiate Flower Server.
        flower_server = Server(client_manager=simple_client_manager,
                               strategy=server_aggregation_strategy)
        return flower_server

    def instantiate_flower_server_config(self) -> ServerConfig:
        fl_settings = self.get_attribute("fl_settings")
        # Instantiate Flower Server's Config.
        flower_server_config = ServerConfig(num_rounds=fl_settings["num_rounds"],
                                            round_timeout=fl_settings["round_timeout_in_seconds"])
        return flower_server_config

    def get_ssl_certificates(self) -> Optional[Tuple[bytes]]:
        fl_settings = self.get_attribute("fl_settings")
        ssl_certificates = None
        if fl_settings["enable_ssl"]:
            ssl_settings = self.get_attribute("ssl_settings")
            prefix_path = Path("./FlowerServer_" + str(self.get_attribute("server_id")))
            ca_certificate_bytes = prefix_path.joinpath(ssl_settings["ca_certificate_file"]).read_bytes()
            server_certificate_bytes = prefix_path.joinpath(ssl_settings["server_certificate_file"]).read_bytes()
            server_rsa_private_key_bytes = \
                prefix_path.joinpath(ssl_settings["server_rsa_private_key_file"]).read_bytes()
            ssl_certificates = (ca_certificate_bytes, server_certificate_bytes, server_rsa_private_key_bytes)
        return ssl_certificates

    def start_flower_server(self) -> None:
        # Get gRPC Server's Listen IP Address and Port.
        grpc_listen_ip_address_and_port = self.get_grpc_listen_ip_address_and_port()
        # Get Flower Server.
        flower_server = self.get_attribute("flower_server")
        # Get Flower Server's Config.
        flower_server_config = self.get_attribute("flower_server_config")
        # Get gRPC Max Message Length (in Bytes).
        grpc_max_message_length_in_bytes = self.get_grpc_max_message_length_in_bytes()
        # Get Secure Socket Layer (SSL) Certificates (SSL-Enabled Secure Connection).
        ssl_certificates = self.get_ssl_certificates()
        # Start Flower Server.
        start_server(server_address=grpc_listen_ip_address_and_port,
                     server=flower_server,
                     config=flower_server_config,
                     grpc_max_message_length=grpc_max_message_length_in_bytes,
                     certificates=ssl_certificates)


def main() -> None:
    # Begin.
    # Parse Flower Server Arguments.
    ag = ArgumentParser(description="Flower Server Arguments")
    ag.add_argument("--server_id",
                    type=int,
                    required=True,
                    help="Server ID (no default)")
    ag.add_argument("--server_config_file",
                    type=Path,
                    required=True,
                    help="Server Config File (no default)")
    parsed_args = ag.parse_args()
    # Get Flower Server Arguments.
    server_id = int(parsed_args.server_id)
    server_config_file = Path(parsed_args.server_config_file)
    # Init FlowerServer Object.
    fs = FlowerServer(server_id, server_config_file)
    # Parse Flower Server Config File.
    fs.parse_flower_server_config_file()
    # Instantiate and Set Logger.
    logger = fs.load_logger()
    fs.set_attribute("logger", logger)
    # Load and Set Global Model Initial Parameters.
    global_model_initial_parameters = fs.load_global_model_initial_parameters()
    fs.set_attribute("global_model_initial_parameters", global_model_initial_parameters)
    # Instantiate and Set Flower Server.
    flower_server = fs.instantiate_flower_server()
    fs.set_attribute("flower_server", flower_server)
    # Instantiate and Set Flower Server's Config.
    flower_server_config = fs.instantiate_flower_server_config()
    fs.set_attribute("flower_server_config", flower_server_config)
    # Start Flower Server.
    fs.start_flower_server()
    # Unbind Objects (Garbage Collector).
    del ag
    del fs
    # End.
    exit(0)


if __name__ == "__main__":
    main()

