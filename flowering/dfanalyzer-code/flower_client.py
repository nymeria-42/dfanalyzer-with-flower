from argparse import ArgumentParser
from configparser import ConfigParser
from flwr.client import NumPyClient, start_numpy_client
from flwr.common import NDArray, NDArrays
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import Model
from keras.applications import MobileNetV2
from keras.datasets import cifar10
from keras.losses import Loss, SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.optimizers import Optimizer, SGD
from logging import FileHandler, Formatter, getLevelName, Logger, StreamHandler
from pathlib import Path
from re import findall
from time import perf_counter
from typing import Any, Optional, Tuple


class Client(NumPyClient):

    def __init__(self,
                 client_id: int,
                 model: Model,
                 x_train: NDArray,
                 y_train: NDArray,
                 x_test: NDArray,
                 y_test: NDArray,
                 logger: Logger) -> None:
        self.client_id = client_id
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.logger = logger

    def log_message(self,
                    message: str,
                    message_level: str) -> None:
        logger = self.logger
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

    def get_properties(self,
                       config: dict) -> dict:
        # TODO: To Implement (If Ever Needed)...
        pass

    def get_parameters(self,
                       config: dict) -> NDArrays:
        # Return the Local Model's Current Parameters (Weights).
        return self.model.get_weights()

    def fit(self,
            global_model_current_parameters: NDArrays,
            fit_config: dict) -> Tuple[NDArrays, int, dict]:
        # Update the Local Model With the Global Model's Current Parameters (Weights).
        self.model.set_weights(global_model_current_parameters)
        # Replace All "None" String Values with None Type (Necessary Workaround on Flower v1.1.0).
        fit_config = {k: (None if v == "None" else v) for k, v in fit_config.items()}
        # Log the Training Configuration Received from the Server (If Logger is Enabled for "DEBUG" Level).
        message = "[Client {0} | FL Round {1}] Fit Config: {2}".format(self.client_id,
                                                                       fit_config["fl_round"],
                                                                       fit_config)
        self.log_message(message, "DEBUG")
        # Log the Fit Starting Time (If Logger is Enabled for "INFO" Level).
        message = "[Client {0} | FL Round {1}] Fitting the Model...".format(self.client_id,
                                                                            fit_config["fl_round"])
        self.log_message(message, "INFO")
        # Start the Fit Model Timer.
        fit_time_start = perf_counter()
        # Fit the Local Updated Model With the Local Training Dataset.
        training_metrics_history = self.model.fit(x=self.x_train,
                                                  y=self.y_train,
                                                  shuffle=fit_config["shuffle"],
                                                  batch_size=fit_config["batch_size"],
                                                  initial_epoch=fit_config["initial_epoch"],
                                                  epochs=fit_config["epochs"],
                                                  steps_per_epoch=fit_config["steps_per_epoch"],
                                                  validation_split=fit_config["validation_split"],
                                                  validation_batch_size=fit_config["validation_batch_size"])
        # Get the Fit Time in Seconds (Fitting Duration).
        fit_time_end = perf_counter() - fit_time_start
        # Log the Fit Ending Time (If Logger is Enabled for "INFO" Level).
        message = "[Client {0} | FL Round {1}] Finished Fitting the Model in {2} Seconds." \
            .format(self.client_id,
                    fit_config["fl_round"],
                    fit_time_end)
        self.log_message(message, "INFO")
        # Get the Local Model's Current Parameters (Weights).
        weight_tensors_list = self.get_parameters(fit_config)
        # Get the Number of Training Examples Used.
        num_training_examples = len(self.x_train)
        # Get the Last Epoch's Training Metrics.
        training_metrics = {}
        for metric_name in self.model.metrics_names:
            training_metrics[metric_name] = training_metrics_history.history[metric_name][-1]
        # Add the Fit Time to the Training Metrics.
        training_metrics.update({"fit_time": fit_time_end})
        # Return the Model's Local Weights, Number of Training Examples, and Training Metrics to be Sent to the Server.
        return weight_tensors_list, num_training_examples, training_metrics

    def evaluate(self,
                 global_model_current_parameters: NDArrays,
                 evaluate_config: dict) -> Tuple[float, int, dict]:
        # Update the Local Model With the Global Model's Current Parameters (Weights).
        self.model.set_weights(global_model_current_parameters)
        # Replace All "None" String Values with None Type (Necessary Workaround on Flower v1.1.0).
        evaluate_config = {k: (None if v == "None" else v) for k, v in evaluate_config.items()}
        # Log the Testing Configuration Received from the Server (If Logger is Enabled for "DEBUG" Level).
        message = "[Client {0} | FL Round {1}] Evaluate Config: {2}".format(self.client_id,
                                                                            evaluate_config["fl_round"],
                                                                            evaluate_config)
        self.log_message(message, "DEBUG")
        # Log the Evaluate Starting Time (If Logger is Enabled for "INFO" Level).
        message = "[Client {0} | FL Round {1}] Evaluating the Model...".format(self.client_id,
                                                                               evaluate_config["fl_round"])
        self.log_message(message, "INFO")
        # Start the Evaluate Model Timer.
        evaluate_time_start = perf_counter()
        # Evaluate the Local Updated Model With the Local Testing Dataset.
        testing_metrics_history = self.model.evaluate(x=self.x_test,
                                                      y=self.y_test,
                                                      batch_size=evaluate_config["batch_size"],
                                                      steps=evaluate_config["steps"])
        # Get the Evaluate Time in Seconds (Evaluating Duration).
        evaluate_time_end = perf_counter() - evaluate_time_start
        # Log the Evaluate Ending Time (If Logger is Enabled for "INFO" Level).
        message = "[Client {0} | FL Round {1}] Finished Evaluating the Model in {2} Seconds." \
            .format(self.client_id,
                    evaluate_config["fl_round"],
                    evaluate_time_end)
        self.log_message(message, "INFO")
        # Get the Number of Testing Examples Used.
        num_testing_examples = len(self.x_test)
        # Get the Testing Metrics.
        testing_metrics = {}
        metric_index = 0
        for metric_name in self.model.metrics_names:
            testing_metrics[metric_name] = testing_metrics_history[metric_index]
            metric_index += 1
        # Add the Evaluate Time to the Testing Metrics.
        testing_metrics.update({"evaluate_time": evaluate_time_end})
        # Get the Loss Metric.
        loss = testing_metrics["loss"]
        # Return the Loss Metric, Number of Testing Examples, and Testing Metrics to be Sent to the Server.
        return loss, num_testing_examples, testing_metrics


class FlowerClient:

    def __init__(self,
                 client_id: int,
                 client_config_file: Path) -> None:
        # Client's ID and Config File.
        self.client_id = client_id
        self.client_config_file = client_config_file
        # Client's Config File Settings.
        self.fl_settings = None
        self.ssl_settings = None
        self.grpc_settings = None
        self.ml_model_settings = None
        # Other Attributes.
        self.logger = None
        self.flower_client = None
        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.metrics = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

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

    def parse_flower_client_config_file(self) -> None:
        # Get Client's Config File.
        client_config_file = self.get_attribute("client_config_file")
        # Init ConfigParser Object.
        cp = ConfigParser()
        cp.optionxform = str
        cp.read(filenames=client_config_file, encoding="utf-8")
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
        # Parse 'ML Model Settings' and Set Attributes.
        ml_model_settings = self.parse_config_section(cp, "ML Model Settings")
        self.set_attribute("ml_model_settings", ml_model_settings)
        # Unbind ConfigParser Object (Garbage Collector).
        del cp

    def load_logger(self) -> Optional[Logger]:
        logger = None
        general_settings = self.get_attribute("general_settings")
        if general_settings["enable_logging"]:
            logger_name = "FlowerClient_" + str(self.get_attribute("client_id"))
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

    def get_grpc_server_ip_address_and_port(self) -> str:
        grpc_settings = self.get_attribute("grpc_settings")
        return grpc_settings["grpc_server_ip_address"] + ":" + str(grpc_settings["grpc_server_port"])

    def instantiate_ml_model(self) -> Model:
        # Get Client Config File.
        client_config_file = self.get_attribute("client_config_file")
        # Init ConfigParser Object.
        cp = ConfigParser()
        cp.optionxform = str
        cp.read(filenames=client_config_file, encoding="utf-8")
        ml_model_settings = self.get_attribute("ml_model_settings")
        ml_model = None
        if ml_model_settings["model"] == "MobileNetV2":
            # Parse 'MobileNetV2 Settings'.
            mobilenet_v2_settings = self.parse_config_section(cp, "MobileNetV2 Settings")
            # MobileNetV2 - Image Classification Model Architecture.
            ml_model = MobileNetV2(input_shape=mobilenet_v2_settings["input_shape"],
                                   alpha=mobilenet_v2_settings["alpha"],
                                   include_top=mobilenet_v2_settings["include_top"],
                                   weights=mobilenet_v2_settings["weights"],
                                   input_tensor=mobilenet_v2_settings["input_tensor"],
                                   pooling=mobilenet_v2_settings["pooling"],
                                   classes=mobilenet_v2_settings["classes"],
                                   classifier_activation=mobilenet_v2_settings["classifier_activation"])
        # Unbind ConfigParser Object (Garbage Collector).
        del cp
        return ml_model

    def instantiate_ml_model_optimizer(self) -> Optimizer:
        # Get Client Config File.
        client_config_file = self.get_attribute("client_config_file")
        # Init ConfigParser Object.
        cp = ConfigParser()
        cp.optionxform = str
        cp.read(filenames=client_config_file, encoding="utf-8")
        ml_model_settings = self.get_attribute("ml_model_settings")
        ml_model_optimizer = None
        if ml_model_settings["optimizer"] == "SGD":
            # Parse 'SGD Settings'.
            sgd_settings = self.parse_config_section(cp, "SGD Settings")
            # SGD - Stochastic Gradient Descent Optimizer (With Momentum).
            ml_model_optimizer = SGD(learning_rate=sgd_settings["learning_rate"],
                                     momentum=sgd_settings["momentum"],
                                     nesterov=sgd_settings["nesterov"],
                                     name=sgd_settings["name"])
        # Unbind ConfigParser Object (Garbage Collector).
        del cp
        return ml_model_optimizer

    def instantiate_ml_model_loss_function(self) -> Loss:
        # Get Client Config File.
        client_config_file = self.get_attribute("client_config_file")
        # Init ConfigParser Object.
        cp = ConfigParser()
        cp.optionxform = str
        cp.read(filenames=client_config_file, encoding="utf-8")
        ml_model_settings = self.get_attribute("ml_model_settings")
        ml_model_loss_function = None
        if ml_model_settings["loss_function"] == "SparseCategoricalCrossentropy":
            # Parse 'SparseCategoricalCrossentropy Settings'.
            scc_settings = self.parse_config_section(cp, "SparseCategoricalCrossentropy Settings")
            # SparseCategoricalCrossentropy - Stochastic Gradient Descent Optimizer (With Momentum).
            ml_model_loss_function = SparseCategoricalCrossentropy(from_logits=scc_settings["from_logits"],
                                                                   ignore_class=scc_settings["ignore_class"],
                                                                   reduction=scc_settings["reduction"],
                                                                   name=scc_settings["name"])
        # Unbind ConfigParser Object (Garbage Collector).
        del cp
        return ml_model_loss_function

    def instantiate_ml_model_metrics(self) -> list:
        ml_model_settings = self.get_attribute("ml_model_settings")
        metrics_list = ml_model_settings["metrics"]
        ml_model_metrics = []
        for metric in metrics_list:
            if metric == "SparseCategoricalAccuracy":
                # SparseCategoricalAccuracy - Accuracy for Integer Multi-Label Classification Model.
                ml_model_metrics.append(SparseCategoricalAccuracy())
        return ml_model_metrics

    @staticmethod
    def load_ml_model_local_data() -> NDArrays:
        # Load Data (CIFAR-10): Popular Colored Image Classification Dataset.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return [x_train, y_train, x_test, y_test]

    def compile_ml_model(self) -> None:
        # Get ML Model's Optimizer.
        optimizer = self.get_attribute("optimizer")
        # Get ML Model's Loss Function.
        loss_function = self.get_attribute("loss_function")
        # Get ML Model's Metrics.
        metrics = self.get_attribute("metrics")
        ml_model_settings = self.get_attribute("ml_model_settings")
        # Compile ML Model.
        self.model.compile(optimizer=optimizer,
                           loss=loss_function,
                           metrics=metrics,
                           loss_weights=ml_model_settings["loss_weights"],
                           weighted_metrics=ml_model_settings["weighted_metrics"],
                           run_eagerly=ml_model_settings["run_eagerly"],
                           steps_per_execution=ml_model_settings["steps_per_execution"],
                           jit_compile=ml_model_settings["jit_compile"])

    def instantiate_flower_numpy_client(self) -> NumPyClient:
        # Get Client's ID.
        client_id = self.get_attribute("client_id")
        # Get ML Model.
        model = self.get_attribute("model")
        # Get X_Train.
        x_train = self.get_attribute("x_train")
        # Get Y_Train.
        y_train = self.get_attribute("y_train")
        # Get X_Test.
        x_test = self.get_attribute("x_test")
        # Get Y_Test.
        y_test = self.get_attribute("y_test")
        # Get Logger.
        logger = self.get_attribute("logger")
        # Instantiate Flower's NumPyClient.
        flower_numpy_client = Client(client_id=client_id,
                                     model=model,
                                     x_train=x_train,
                                     y_train=y_train,
                                     x_test=x_test,
                                     y_test=y_test,
                                     logger=logger)
        return flower_numpy_client

    def get_grpc_max_message_length_in_bytes(self) -> int:
        return self.get_attribute("grpc_settings")["grpc_max_message_length_in_bytes"]

    def get_ssl_certificates(self) -> Optional[bytes]:
        fl_settings = self.get_attribute("fl_settings")
        ssl_certificates = None
        if fl_settings["enable_ssl"]:
            ssl_settings = self.get_attribute("ssl_settings")
            prefix_path = Path("./FlowerClient_" + str(self.get_attribute("client_id")))
            ca_certificate_bytes = prefix_path.joinpath(ssl_settings["ca_certificate_file"]).read_bytes()
            ssl_certificates = ca_certificate_bytes
        return ssl_certificates

    def start_flower_client(self) -> None:
        # Get gRPC Server's IP Address and Port.
        grpc_server_ip_address_and_port = self.get_grpc_server_ip_address_and_port()
        # Get Flower Client.
        flower_client = self.get_attribute("flower_client")
        # Get gRPC Max Message Length (in Bytes).
        grpc_max_message_length_in_bytes = self.get_grpc_max_message_length_in_bytes()
        # Get Secure Socket Layer (SSL) Root (CA) Certificate (SSL-Enabled Secure Connection).
        ssl_certificates = self.get_ssl_certificates()
        # Start Flower Client.
        start_numpy_client(server_address=grpc_server_ip_address_and_port,
                           client=flower_client,
                           grpc_max_message_length=grpc_max_message_length_in_bytes,
                           root_certificates=ssl_certificates)


def main() -> None:
    # Begin.
    # Parse Flower Client Arguments.
    ag = ArgumentParser(description="Flower Client Arguments")
    ag.add_argument("--client_id",
                    type=int,
                    required=True,
                    help="Client ID (no default)")
    ag.add_argument("--client_config_file",
                    type=Path,
                    required=True,
                    help="Client Config File (no default)")
    parsed_args = ag.parse_args()
    # Get Flower Client Arguments.
    client_id = int(parsed_args.client_id)
    client_config_file = Path(parsed_args.client_config_file)
    # Init FlowerClient Object.
    fc = FlowerClient(client_id, client_config_file)
    # Parse Flower Client Config File.
    fc.parse_flower_client_config_file()
    # Instantiate and Set Logger.
    logger = fc.load_logger()
    fc.set_attribute("logger", logger)
    # Instantiate and Set ML Model.
    model = fc.instantiate_ml_model()
    fc.set_attribute("model", model)
    # Instantiate and Set ML Model's Optimizer.
    optimizer = fc.instantiate_ml_model_optimizer()
    fc.set_attribute("optimizer", optimizer)
    # Instantiate and Set ML Model's Loss Function.
    loss_function = fc.instantiate_ml_model_loss_function()
    fc.set_attribute("loss_function", loss_function)
    # Instantiate and Set ML Model's Metrics.
    metrics = fc.instantiate_ml_model_metrics()
    fc.set_attribute("metrics", metrics)
    # Compile ML Model.
    fc.compile_ml_model()
    # Load and Set ML Model's Local Data (x_train, y_train, x_test, y_test).
    local_data = fc.load_ml_model_local_data()
    fc.set_attribute("x_train", local_data[0])
    fc.set_attribute("y_train", local_data[1])
    fc.set_attribute("x_test", local_data[2])
    fc.set_attribute("y_test", local_data[3])
    # Instantiate and Set Flower Client.
    flower_client = fc.instantiate_flower_numpy_client()
    fc.set_attribute("flower_client", flower_client)
    # Start Flower Client.
    fc.start_flower_client()
    # Unbind Objects (Garbage Collector).
    del ag
    del fc
    # End.
    exit(0)


if __name__ == "__main__":
    main()

