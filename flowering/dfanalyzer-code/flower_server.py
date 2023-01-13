from argparse import ArgumentParser
from configparser import ConfigParser
from logging import FileHandler, Formatter, getLevelName, Logger, StreamHandler
from flwr.common import Metrics, NDArrays, Parameters
from flwr.server import Server, ServerConfig, SimpleClientManager, start_server
from flwr.server.strategy import FedAvg, FedAvgM, Strategy
from pathlib import Path
from re import findall
from typing import Any, List, Optional, Tuple

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
from dfa_lib_python.dependency import Dependency
import time
import pymonetdb



dataflow_tag = "flower-df"


class FlowerServer:
    def __init__(self, server_id: int, server_config_file: Path) -> None:
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
    def parse_config_section(config_parser: ConfigParser, section_name: str) -> dict:
        parsed_section = {
            key: value for key, value in config_parser[section_name].items()
        }
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
                aux_list = (
                    value.replace("[", "").replace("]", "").replace(" ", "").split(",")
                )
                for index, item in enumerate(aux_list):
                    if item.isdigit():
                        aux_list[index] = int(item)
                    elif item.replace(".", "", 1).isdigit():
                        aux_list[index] = float(item)
                parsed_section[key] = aux_list
            elif not findall(r"%\(.*?\)s+", value) and findall(r"\(.*?\)+", value):
                aux_list = (
                    value.replace("(", "").replace(")", "").replace(" ", "").split(",")
                )
                for index, item in enumerate(aux_list):
                    if item.isdigit():
                        aux_list[index] = int(item)
                    elif item.replace(".", "", 1).isdigit():
                        aux_list[index] = float(item)
                parsed_section[key] = tuple(aux_list)
        return parsed_section

    def set_attribute(self, attribute_name: str, attribute_value: Any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self, attribute_name: str) -> Any:
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

        attributes = [
            "num_rounds",
            "round_timeout_in_seconds",
            "accept_rounds_containing_failures",
            "enable_ssl",
            "server_aggregation_strategy",
            "fraction_fit",
            "fraction_evaluate",
            "min_fit_clients",
            "min_evaluate_clients",
            "min_available_clients",
        ]
        to_dfanalyzer = [fl_settings.get(attr, None) for attr in attributes]
        t1 = Task(1, dataflow_tag, "ServerConfig")

        t1.begin()

        self.set_attribute("fl_settings", fl_settings)

        # Parse 'SSL Settings' and Set Attributes.
        ssl_settings = self.parse_config_section(cp, "SSL Settings")
        self.set_attribute("ssl_settings", ssl_settings)
        # Parse 'gRPC Settings' and Set Attributes.
        grpc_settings = self.parse_config_section(cp, "gRPC Settings")
        self.set_attribute("grpc_settings", grpc_settings)

        t1_output = DataSet(
            "oServerConfig",
            [
                Element(
                    [
                        self.server_id,
                        str(grpc_settings["grpc_listen_ip_address"])
                        + str(grpc_settings["grpc_listen_port"]),
                        grpc_settings["grpc_max_message_length_in_bytes"],
                    ]
                    + to_dfanalyzer
                )
            ],
        )
        t1.add_dataset(t1_output)
        t1.end()
        # Parse 'Training Hyper-parameters Settings' and Set Attributes.

        training_hyper_parameters_settings = self.parse_config_section(
            cp, "Training Hyper-parameters Settings"
        )
        self.set_attribute(
            "training_hyper_parameters_settings", training_hyper_parameters_settings
        )
        # Parse 'Testing Hyper-parameters Settings' and Set Attributes.
        testing_hyper_parameters_settings = self.parse_config_section(
            cp, "Testing Hyper-parameters Settings"
        )
        self.set_attribute(
            "testing_hyper_parameters_settings", testing_hyper_parameters_settings
        )
        # Unbind ConfigParser Object (Garbage Collector).
        del cp

    def load_logger(self) -> Optional[Logger]:
        logger = None
        general_settings = self.get_attribute("general_settings")
        if general_settings["enable_logging"]:
            logger_name = "FlowerServer_" + str(self.get_attribute("server_id"))
            logging_settings = self.get_attribute("logging_settings")
            logger = Logger(name=logger_name, level=logging_settings["level"])
            formatter = Formatter(
                fmt=logging_settings["format"], datefmt=logging_settings["date_format"]
            )
            if logging_settings["log_to_file"]:
                file_handler = FileHandler(
                    filename=logging_settings["file_name"],
                    mode=logging_settings["file_mode"],
                    encoding=logging_settings["encoding"],
                )
                file_handler.setLevel(logger.getEffectiveLevel())
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            if logging_settings["log_to_console"]:
                console_handler = StreamHandler()
                console_handler.setLevel(logger.getEffectiveLevel())
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
        return logger

    def log_message(self, message: str, message_level: str) -> None:
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
        return (
            grpc_settings["grpc_listen_ip_address"]
            + ":"
            + str(grpc_settings["grpc_listen_port"])
        )

    def get_grpc_max_message_length_in_bytes(self) -> int:
        return self.get_attribute("grpc_settings")["grpc_max_message_length_in_bytes"]

    @staticmethod
    def instantiate_simple_client_manager() -> SimpleClientManager:
        return SimpleClientManager()

    @staticmethod
    def evaluate_fn(
        fl_round: int, global_model_current_parameters: NDArrays, evaluate_config: dict
    ) -> Optional[Metrics]:
        """Server-side (Centralized) evaluation function called by Flower after every training round.
        \nRequires a server-side dataset to evaluate the newly aggregated model without sending it to the Clients.
        \nThe 'losses_centralized' and 'metrics_centralized' will only contain values using this centralized evaluation.
        \nAlternative: Client-side (Federated) evaluation."""
        # TODO: To Implement (If Ever Needed)...
        return None

    def on_fit_config_fn(self, fl_round: int) -> Optional[dict]:
        """Training configuration function called by Flower before each training round."""
        # Update the Current FL Round (Necessary Workaround on Flower v1.1.0).
        self.set_attribute("fl_round", fl_round)
        # Log the Current FL Round (If Logger is Enabled for "INFO" Level).
        message = "[Server {0}] Current FL Round: {1}".format(
            self.get_attribute("server_id"), self.get_attribute("fl_round")
        )
        self.log_message(message, "INFO")
        # Get the Training Hyper-parameters Settings.
        training_hyper_parameters_settings = self.get_attribute(
            "training_hyper_parameters_settings"
        )
        # Replace All Values of None Type to "None" String (Necessary Workaround on Flower v1.1.0).
        training_hyper_parameters_settings = {
            k: ("None" if v is None else v)
            for k, v in training_hyper_parameters_settings.items()
        }
        # Set the Training Configuration to be Sent to All Participating Clients.
        fit_config = {"fl_round": fl_round}
        fit_config.update(training_hyper_parameters_settings)
        # Log the Training Configuration (If Logger is Enabled for "DEBUG" Level).
        message = "[Server {0} | FL Round {1}] Fit Config: {2}".format(
            self.get_attribute("server_id"), fit_config["fl_round"], fit_config
        )
        self.log_message(message, "DEBUG")
        t7 = Task(7 + 6 * (fl_round - 1), dataflow_tag, "TrainingConfig")
        t7.add_dependency(Dependency(["serverconfig", "strategy"], ["1", "2"]))
        t7.begin()

        to_dfanalyzer = [fl_round, time.ctime()]

        t7_input = DataSet("iTrainingConfig", [Element(to_dfanalyzer)])
        t7.add_dataset(t7_input)

        attributes = [
            "shuffle",
            "batch_size",
            "initial_epoch",
            "epochs",
            "steps_per_epoch",
            "validation_split",
            "validation_batch_size",
        ]
        to_dfanalyzer = [fl_round, time.ctime()] + [
            training_hyper_parameters_settings.get(attr, 0) for attr in attributes
        ]

        t7_output = DataSet("oTrainingConfig", [Element(to_dfanalyzer)])
        t7.add_dataset(t7_output)
        t7.end()
        # Return the Training Configuration to be Sent to All Participating Clients.
        return fit_config

    def on_evaluate_config_fn(self, fl_round: int) -> Optional[dict]:
        """Testing configuration function called by Flower before each testing round."""
        # Get the Testing Hyper-parameters Settings.
        testing_hyper_parameters_settings = self.get_attribute(
            "testing_hyper_parameters_settings"
        )
        # Replace All Values of None Type to "None" String (Necessary Workaround on Flower v1.1.0).
        testing_hyper_parameters_settings = {
            k: ("None" if v is None else v)
            for k, v in testing_hyper_parameters_settings.items()
        }
        # Set the Testing Configuration to be Sent to All Participating Clients.
        evaluate_config = {"fl_round": fl_round}
        evaluate_config.update(testing_hyper_parameters_settings)
        # Log the Testing Configuration (If Logger is Enabled for "DEBUG" Level).
        message = "[Server {0} | FL Round {1}] Evaluate Config: {2}".format(
            self.get_attribute("server_id"),
            evaluate_config["fl_round"],
            evaluate_config,
        )
        self.log_message(message, "DEBUG")

        t9 = Task(9, dataflow_tag, "TestConfig")
        t9.begin()
        attributes = ["batch_size", "steps"]
        to_dfanalyzer = [
            testing_hyper_parameters_settings.get(attr, 0) for attr in attributes
        ]

        t9_output = DataSet("oTestConfig", [Element(to_dfanalyzer)])
        t9.add_dataset(t9_output)
        t9.end()
        # Return the Testing Configuration to be Sent to All Participating Clients.
        return evaluate_config

    def fit_metrics_aggregation_fn(
        self, training_metrics: List[Tuple[int, Metrics]]
    ) -> Optional[Metrics]:
        """Metrics aggregation function called by Flower after every training round."""
        t9 = Task(
            9 + 6 * (self.get_attribute("fl_round") - 1),
            dataflow_tag,
            "ServerTrainingAggregation",
            dependency=Task(
                8 + 6 * (self.get_attribute("fl_round") - 1),
                dataflow_tag,
                "ClientTraining",
            ),
        )
        t9.begin()

        to_dfanalyzer = [self.get_attribute("fl_round"), time.ctime()]
        t9_input = DataSet("iServerTrainingAggregation", [Element(to_dfanalyzer)])
        t9.add_dataset(t9_input)

        # Get the Total Number of Participating Clients.
        total_num_clients = len(training_metrics)
        # Get the Training Metrics Names.
        metrics_names_list = list(training_metrics[0][1].keys())
        # Multiply Each Training Metrics Value of Each Participating Client
        # By His Number of Training Examples (Client's Contribution).
        metrics_products_list = []
        for metric_name in metrics_names_list:
            metric_product = [
                num_examples * metric[metric_name]
                for num_examples, metric in training_metrics
            ]
            metrics_products_list.append(metric_product)
        # Get the Total Number of Training Examples (of All Participating Clients).
        total_num_examples = sum([num_examples for num_examples, _ in training_metrics])
        # Aggregate the Training Metrics (Weighted Average).
        aggregated_metrics = {}
        for metric_index in range(0, len(metrics_names_list)):
            metric_name = metrics_names_list[metric_index]
            weighted_average_metric = (
                sum(metrics_products_list[metric_index]) / total_num_examples
            )
            aggregated_metrics[metric_name] = weighted_average_metric

        # Log the Aggregated Training Metrics (If Logger is Enabled for "INFO" Level).
        message = "[Server {0} | FL Round {1} | {2}] Aggregated Training Metrics (Weighted Average): {3}".format(
            self.get_attribute("server_id"),
            self.get_attribute("fl_round"),
            "".join(
                [
                    str(total_num_clients),
                    " Clients" if total_num_clients > 1 else " Client",
                ]
            ),
            aggregated_metrics,
        )
        self.log_message(message, "INFO")

        to_dfanalyzer = [
            self.get_attribute("fl_round"),
            total_num_clients,
            total_num_examples,
            aggregated_metrics["sparse_categorical_accuracy"],
            aggregated_metrics["loss"],
            aggregated_metrics["fit_time"],
            time.ctime(),
        ]
        t9_output = DataSet("oServerTrainingAggregation", [Element(to_dfanalyzer)])
        t9.add_dataset(t9_output)
        t9.end()
        # Return the Aggregated Training Metrics.
        return aggregated_metrics

    def evaluate_metrics_aggregation_fn(
        self, testing_metrics: List[Tuple[int, Metrics]]
    ) -> Optional[Metrics]:
        """Metrics aggregation function called by Flower after every testing round."""
        # Get the Total Number of Participating Clients.
        t12 = Task(
            12 + 6 * (self.get_attribute("fl_round") - 1),
            dataflow_tag,
            "ServerEvaluationAggregation",
            dependency=Dependency(
                ["clientevaluation"], [11 + 6 * (self.get_attribute("fl_round") - 1)]
            ),
        )
        t12.begin()

        to_dfanalyzer = [self.get_attribute("fl_round"), time.ctime()]
        t12_input = DataSet("iServerEvaluationAggregation", [Element(to_dfanalyzer)])
        t12.add_dataset(t12_input)

        total_num_clients = len(testing_metrics)
        # Get the Testing Metrics Names.
        metrics_names_list = list(testing_metrics[0][1].keys())
        # Multiply Each Testing Metrics Value of Each Participating Client
        # By His Number of Testing Examples (Client's Contribution).
        metrics_products_list = []
        for metric_name in metrics_names_list:
            metric_product = [
                num_examples * metric[metric_name]
                for num_examples, metric in testing_metrics
            ]
            metrics_products_list.append(metric_product)
        # Get the Total Number of Testing Examples (of All Participating Clients).
        total_num_examples = sum([num_examples for num_examples, _ in testing_metrics])
        # Aggregate the Testing Metrics (Weighted Average).
        aggregated_metrics = {}
        for metric_index in range(0, len(metrics_names_list)):
            metric_name = metrics_names_list[metric_index]
            weighted_average_metric = (
                sum(metrics_products_list[metric_index]) / total_num_examples
            )
            aggregated_metrics[metric_name] = weighted_average_metric

        # Log the Aggregated Testing Metrics (If Logger is Enabled for "INFO" Level).
        message = "[Server {0} | FL Round {1} | {2}] Aggregated Testing Metrics (Weighted Average): {3}".format(
            self.get_attribute("server_id"),
            self.get_attribute("fl_round"),
            "".join(
                [
                    str(total_num_clients),
                    " Clients" if total_num_clients > 1 else " Client",
                ]
            ),
            aggregated_metrics,
        )
        self.log_message(message, "INFO")

        to_dfanalyzer = [
            self.get_attribute("fl_round"),
            total_num_clients,
            total_num_examples,
            aggregated_metrics["sparse_categorical_accuracy"],
            aggregated_metrics["loss"],
            aggregated_metrics["evaluate_time"],
            time.ctime(),
        ]
        t12_output = DataSet("oServerEvaluationAggregation", [Element(to_dfanalyzer)])
        t12.add_dataset(t12_output)
        t12.end()
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
        t2 = Task(2, dataflow_tag, "Strategy")
        t2.begin()
        if fl_settings["server_aggregation_strategy"] == "FedAvg":
            # FedAvg - Federated Averaging Aggregation Strategy.
            server_aggregation_strategy = FedAvg(
                fraction_fit=fl_settings["fraction_fit"],
                fraction_evaluate=fl_settings["fraction_evaluate"],
                min_fit_clients=fl_settings["min_fit_clients"],
                min_evaluate_clients=fl_settings["min_evaluate_clients"],
                min_available_clients=fl_settings["min_available_clients"],
                evaluate_fn=self.evaluate_fn,
                on_fit_config_fn=self.on_fit_config_fn,
                on_evaluate_config_fn=self.on_evaluate_config_fn,
                accept_failures=fl_settings["accept_rounds_containing_failures"],
                initial_parameters=self.get_attribute(
                    "global_model_initial_parameters"
                ),
                fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=self.evaluate_metrics_aggregation_fn,
            )
            t2_output = DataSet("oStrategy", [Element([0, 0])])
            t2.add_dataset(t2_output)

        elif fl_settings["server_aggregation_strategy"] == "FedAvgM":
            # Parse 'FedAvgM Settings'.
            fed_avg_m_settings = self.parse_config_section(cp, "FedAvgM Settings")
            # FedAvgM - Federated Averaging with Server Momentum Aggregation Strategy.
            server_aggregation_strategy = FedAvgM(
                fraction_fit=fl_settings["fraction_fit"],
                fraction_evaluate=fl_settings["fraction_evaluate"],
                min_fit_clients=fl_settings["min_fit_clients"],
                min_evaluate_clients=fl_settings["min_evaluate_clients"],
                min_available_clients=fl_settings["min_available_clients"],
                evaluate_fn=self.evaluate_fn,
                on_fit_config_fn=self.on_fit_config_fn,
                on_evaluate_config_fn=self.on_evaluate_config_fn,
                accept_failures=fl_settings["accept_rounds_containing_failures"],
                initial_parameters=self.get_attribute(
                    "global_model_initial_parameters"
                ),
                fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=self.evaluate_metrics_aggregation_fn,
                server_learning_rate=fed_avg_m_settings["server_learning_rate"],
                server_momentum=fed_avg_m_settings["server_momentum"],
            )
            attributes = ["server_learning_rate", "server_momentum"]
            to_dfanalyzer = [fed_avg_m_settings.get(attr, None) for attr in attributes]
            t2_output = DataSet("oStrategy", [Element(to_dfanalyzer)])
            t2.add_dataset(t2_output)
        t2.end()
        # Unbind ConfigParser Object (Garbage Collector).
        del cp
        return server_aggregation_strategy

    def instantiate_flower_server(self) -> Server:
        # Instantiate Simple Client Manager.
        simple_client_manager = self.instantiate_simple_client_manager()
        # Instantiate Server's Aggregation Strategy.
        server_aggregation_strategy = self.instantiate_server_aggregation_strategy()
        # Instantiate Flower Server.
        flower_server = Server(
            client_manager=simple_client_manager, strategy=server_aggregation_strategy
        )
        return flower_server

    def instantiate_flower_server_config(self) -> ServerConfig:
        fl_settings = self.get_attribute("fl_settings")
        # Instantiate Flower Server's Config.
        flower_server_config = ServerConfig(
            num_rounds=fl_settings["num_rounds"],
            round_timeout=fl_settings["round_timeout_in_seconds"],
        )
        return flower_server_config

    def get_ssl_certificates(self) -> Optional[Tuple[bytes]]:
        fl_settings = self.get_attribute("fl_settings")
        ssl_certificates = None
        if fl_settings["enable_ssl"]:
            ssl_settings = self.get_attribute("ssl_settings")
            prefix_path = Path("./FlowerServer_" + str(self.get_attribute("server_id")))
            ca_certificate_bytes = prefix_path.joinpath(
                ssl_settings["ca_certificate_file"]
            ).read_bytes()
            server_certificate_bytes = prefix_path.joinpath(
                ssl_settings["server_certificate_file"]
            ).read_bytes()
            server_rsa_private_key_bytes = prefix_path.joinpath(
                ssl_settings["server_rsa_private_key_file"]
            ).read_bytes()
            ssl_certificates = (
                ca_certificate_bytes,
                server_certificate_bytes,
                server_rsa_private_key_bytes,
            )
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
        start_server(
            server_address=grpc_listen_ip_address_and_port,
            server=flower_server,
            config=flower_server_config,
            grpc_max_message_length=grpc_max_message_length_in_bytes,
            certificates=ssl_certificates,
        )


def main() -> None:
    # Begin.

    ##########
    # DfAnalyzer Instrumentation
    df = Dataflow(dataflow_tag)

    tf1 = Transformation("ServerConfig")
    tf1_output = Set(
        "oServerConfig",
        SetType.OUTPUT,
        [
            Attribute("server_id", AttributeType.NUMERIC),
            Attribute("address", AttributeType.TEXT),
            Attribute("max_message_length_in_bytes", AttributeType.TEXT),
            Attribute("num_rounds", AttributeType.NUMERIC),
            Attribute("round_timeout_in_seconds", AttributeType.NUMERIC),
            Attribute("accept_rounds_with_failures", AttributeType.TEXT),
            Attribute("enable_ssl", AttributeType.TEXT),
            Attribute("server_aggregation_strategy", AttributeType.TEXT),
            Attribute("fraction_fit", AttributeType.NUMERIC),
            Attribute("fraction_evaluate", AttributeType.NUMERIC),
            Attribute("min_fit_clients", AttributeType.NUMERIC),
            Attribute("min_evaluate_clients", AttributeType.NUMERIC),
            Attribute("min_available_clients", AttributeType.NUMERIC),
        ],
    )
    tf1.set_sets([tf1_output])
    df.add_transformation(tf1)

    tf2 = Transformation("Strategy")

    tf2_output = Set(
        "oStrategy",
        SetType.OUTPUT,
        [
            Attribute("server_learning_rate", AttributeType.NUMERIC),
            Attribute("server_momentum", AttributeType.NUMERIC),
        ],
    )

    tf2.set_sets([tf2_output])
    df.add_transformation(tf2)

    tf3 = Transformation("ModelConfig")
    tf3_output = Set(
        "oModelConfig",
        SetType.OUTPUT,
        [
            Attribute("model", AttributeType.TEXT),
            Attribute("optimizer", AttributeType.TEXT),
            Attribute("loss_function", AttributeType.TEXT),
            Attribute("loss_weights", AttributeType.TEXT),
            Attribute("weighted_metrics", AttributeType.TEXT),
            Attribute("run_eagerly", AttributeType.TEXT),
            Attribute("steps_per_execution", AttributeType.NUMERIC),
            Attribute("jit_compile", AttributeType.TEXT),
            Attribute("input_shape", AttributeType.TEXT),
            Attribute("alpha", AttributeType.NUMERIC),
            Attribute("include_top", AttributeType.TEXT),
            Attribute("weights", AttributeType.TEXT),
            Attribute("input_tensor", AttributeType.TEXT),
            Attribute("pooling", AttributeType.TEXT),
            Attribute("classes", AttributeType.NUMERIC),
            Attribute("classifier_activation", AttributeType.TEXT),
        ],
    )
    tf3.set_sets([tf3_output])
    df.add_transformation(tf3)

    tf4 = Transformation("OptimizerConfig")
    tf4_output = Set(
        "oOptimizerConfig",
        SetType.OUTPUT,
        [
            Attribute("learning_rate", AttributeType.NUMERIC),
            Attribute("momentum", AttributeType.NUMERIC),
            Attribute("nesterov", AttributeType.TEXT),
            Attribute("name", AttributeType.TEXT),
        ],
    )
    tf4.set_sets([tf4_output])
    df.add_transformation(tf4)

    tf5 = Transformation("LossConfig")
    tf5_output = Set(
        "oLossConfig",
        SetType.OUTPUT,
        [
            Attribute("from_logits", AttributeType.TEXT),
            Attribute("ignore_class", AttributeType.TEXT),
            Attribute("reduction", AttributeType.TEXT),
            Attribute("name", AttributeType.TEXT),
        ],
    )
    tf5.set_sets([tf5_output])
    df.add_transformation(tf5)

    tf6 = Transformation("DatasetLoad")
    tf6_output = Set(
        "oDatasetLoad",
        SetType.OUTPUT,
        [
            Attribute("client_id", AttributeType.NUMERIC),
            Attribute("loading_time", AttributeType.TEXT),
        ],
    )
    tf6.set_sets([tf6_output])
    df.add_transformation(tf6)

    tf7 = Transformation("TrainingConfig")
    tf7_input = Set(
        "iTrainingConfig",
        SetType.INPUT,
        [
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("starting_time", AttributeType.TEXT),
        ],
    )

    tf7_output = Set(
        "oTrainingConfig",
        SetType.OUTPUT,
        [
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("ending_time", AttributeType.TEXT),
            Attribute("shuffle", AttributeType.TEXT),
            Attribute("batch_size", AttributeType.NUMERIC),
            Attribute("initial_epoch", AttributeType.NUMERIC),
            Attribute("epochs", AttributeType.NUMERIC),
            Attribute("steps_per_epoch", AttributeType.TEXT),
            Attribute("validation_split", AttributeType.NUMERIC),
            Attribute("validation_batch_size", AttributeType.TEXT),
        ],
    )

    tf2_output.set_type(SetType.INPUT)
    tf2_output.dependency = tf2._tag

    tf1_output.set_type(SetType.INPUT)
    tf1_output.dependency = tf1._tag

    tf7.set_sets([tf1_output, tf2_output, tf7_input, tf7_output])
    df.add_transformation(tf7)

    tf8 = Transformation("ClientTraining")
    tf8_input = Set(
        "iClientTraining",
        SetType.INPUT,
        [
            Attribute("client_id", AttributeType.NUMERIC),
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("size_x_train", AttributeType.NUMERIC),
            Attribute("global_current_parameters", AttributeType.TEXT),
            Attribute("starting_time", AttributeType.TEXT),
        ],
    )
    tf8_output = Set(
        "oClientTraining",
        SetType.OUTPUT,
        [
            Attribute("client_id", AttributeType.NUMERIC),
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("training_time", AttributeType.NUMERIC),
            Attribute("accuracy", AttributeType.NUMERIC),
            Attribute("local_weights", AttributeType.TEXT),
            Attribute("ending_time", AttributeType.TEXT),
        ],
    )

    tf3_output.set_type(SetType.INPUT)
    tf3_output.dependency = tf3._tag

    tf4_output.set_type(SetType.INPUT)
    tf4_output.dependency = tf4._tag

    tf5_output.set_type(SetType.INPUT)
    tf5_output.dependency = tf5._tag

    tf6_output.set_type(SetType.INPUT)
    tf6_output.dependency = tf6._tag

    tf7_output.set_type(SetType.INPUT)
    tf7_output.dependency = tf7._tag

    tf8.set_sets(
        [
            tf3_output,
            tf4_output,
            tf5_output,
            tf6_output,
            tf7_output,
            tf8_input,
            tf8_output,
        ]
    )
    df.add_transformation(tf8)

    tf9 = Transformation("ServerTrainingAggregation")
    tf9_input = Set(
        "iServerTrainingAggregation",
        SetType.INPUT,
        [
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("starting_time", AttributeType.TEXT),
        ],
    )
    tf9_output = Set(
        "oServerTrainingAggregation",
        SetType.OUTPUT,
        [
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("total_num_clients", AttributeType.NUMERIC),
            Attribute("total_num_examples", AttributeType.NUMERIC),
            Attribute("accuracy", AttributeType.NUMERIC),
            Attribute("loss", AttributeType.NUMERIC),
            Attribute("training_time", AttributeType.NUMERIC),
            Attribute("ending_time", AttributeType.TEXT),
        ],
    )

    tf8_output.set_type(SetType.INPUT)
    tf8_output.dependency = tf8._tag

    tf9.set_sets([tf8_output, tf9_input, tf9_output])
    df.add_transformation(tf9)

    tf10 = Transformation("TestConfig")
    tf10_output = Set(
        "oTestConfig",
        SetType.OUTPUT,
        [
            Attribute("batch_size", AttributeType.NUMERIC),
            Attribute("steps", AttributeType.TEXT),
        ],
    )
    tf10.set_sets([tf10_output])
    df.add_transformation(tf10)

    tf11 = Transformation("ClientEvaluation")
    tf11_input = Set(
        "iClientEvaluation",
        SetType.INPUT,
        [
            Attribute("client_id", AttributeType.NUMERIC),
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("num_testing_examples", AttributeType.NUMERIC),
            Attribute("starting_time", AttributeType.TEXT),
        ],
    )
    tf11_output = Set(
        "oClientEvaluation",
        SetType.OUTPUT,
        [
            Attribute("client_id", AttributeType.NUMERIC),
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("loss", AttributeType.NUMERIC),
            Attribute("evaluation_time", AttributeType.NUMERIC),
            Attribute("accuracy", AttributeType.NUMERIC),
            Attribute("ending_time", AttributeType.TEXT),
        ],
    )

    tf9_output.set_type(SetType.INPUT)
    tf9_output.dependency = tf9._tag

    tf10_output.set_type(SetType.INPUT)
    tf10_output.dependency = tf10._tag

    tf11.set_sets([tf9_output, tf10_output, tf11_input, tf11_output])
    df.add_transformation(tf11)

    tf12 = Transformation("ServerEvaluationAggregation")
    tf12_input = Set(
        "iServerEvaluationAggregation",
        SetType.INPUT,
        [
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("starting_time", AttributeType.TEXT),
        ],
    )
    tf12_output = Set(
        "oServerEvaluationAggregation",
        SetType.OUTPUT,
        [
            Attribute("server_round", AttributeType.NUMERIC),
            Attribute("total_num_clients", AttributeType.NUMERIC),
            Attribute("total_num_examples", AttributeType.NUMERIC),
            Attribute("accuracy", AttributeType.NUMERIC),
            Attribute("loss", AttributeType.NUMERIC),
            Attribute("evaluation_time", AttributeType.NUMERIC),
            Attribute("ending_time", AttributeType.TEXT),
        ],
    )

    tf11_output.set_type(SetType.INPUT)
    tf11_output.dependency = tf11._tag

    tf12.set_sets([tf11_output, tf12_input, tf12_output])
    df.add_transformation(tf12)

    df.save()

    ##########
    # Parse Flower Server Arguments.
    ag = ArgumentParser(description="Flower Server Arguments")
    ag.add_argument(
        "--server_id", type=int, required=True, help="Server ID (no default)"
    )
    ag.add_argument(
        "--server_config_file",
        type=Path,
        required=True,
        help="Server Config File (no default)",
    )
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
