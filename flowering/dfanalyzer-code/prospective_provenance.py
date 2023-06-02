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
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path


ag = ArgumentParser(description="Flower Server Arguments")

ag.add_argument(
        "--server_config_file",
        type=Path,
        required=False,
        default="config/flower_server.cfg",
        help="Server Config File (no default)",
        dest="server_config_file"
    )

parsed_args = ag.parse_args()
cp = ConfigParser()
cp.optionxform = str
cp.read(filenames=parsed_args.server_config_file, encoding="utf-8")
monetdb_settings =  cp["MonetDB Settings"]

# DfAnalyzer Instrumentation
dataflow_tag = "flower-df"
df = Dataflow(dataflow_tag)

tf1 = Transformation("ServerConfig")
tf1_input = Set(
    "iServerConfig",
    SetType.INPUT,
    [
        Attribute("server_id", AttributeType.NUMERIC),
        Attribute("address", AttributeType.TEXT),
        Attribute("max_message_length_in_bytes", AttributeType.TEXT),
        Attribute("num_rounds", AttributeType.NUMERIC),
        Attribute("round_timeout_in_seconds", AttributeType.NUMERIC),
        Attribute("accept_rounds_with_failures", AttributeType.TEXT),
        Attribute("enable_ssl", AttributeType.TEXT),
        Attribute("enable_dynamic_adjustment", AttributeType.TEXT),
        Attribute("server_aggregation_strategy", AttributeType.TEXT),
        Attribute("fraction_fit", AttributeType.NUMERIC),
        Attribute("fraction_evaluate", AttributeType.NUMERIC),
        Attribute("min_fit_clients", AttributeType.NUMERIC),
        Attribute("min_evaluate_clients", AttributeType.NUMERIC),
        Attribute("min_available_clients", AttributeType.NUMERIC),
    ],
)
tf1_output = Set("oServerConfig", SetType.OUTPUT, [])
tf1.set_sets([tf1_input, tf1_output])
df.add_transformation(tf1)

tf2 = Transformation("LoadGlobalWeights")

tf2_input = Set(
    "iLoadGlobalWeights",
    SetType.INPUT,
    [
        Attribute("server_id", AttributeType.NUMERIC),
        Attribute("starting_time", AttributeType.TEXT),
        Attribute("ending_time", AttributeType.TEXT),
        Attribute("loading_time", AttributeType.TEXT),
    ],
)
tf2_output = Set("oLoadGlobalWeights", SetType.OUTPUT, [
    Attribute("loaded_from_mongo", AttributeType.TEXT),
])
tf2.set_sets([tf2_input, tf2_output])
df.add_transformation(tf2)


tf3 = Transformation("Strategy")

tf3_input = Set(
    "iStrategy",
    SetType.INPUT,
    [
        Attribute("server_learning_rate", AttributeType.NUMERIC),
        Attribute("server_momentum", AttributeType.NUMERIC),
    ],
)
tf3_output = Set("oStrategy", SetType.OUTPUT, [])
tf3.set_sets([tf3_input, tf3_output])
df.add_transformation(tf3)

tf4 = Transformation("DatasetLoad")
tf4_input = Set(
    "iDatasetLoad",
    SetType.INPUT,
    [
        Attribute("client_id", AttributeType.NUMERIC),
        Attribute("loading_time", AttributeType.TEXT),
    ],
)
tf4_output = Set("oDatasetLoad", SetType.OUTPUT, [])
tf4.set_sets([tf4_input, tf4_output])
df.add_transformation(tf4)

tf5 = Transformation("ModelConfig")
tf5_input = Set(
    "iModelConfig",
    SetType.INPUT,
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

tf5_output = Set(
    "oModelConfig",
    SetType.OUTPUT,
    [],
)

tf5.set_sets([tf5_input, tf5_output])
df.add_transformation(tf5)

tf6 = Transformation("OptimizerConfig")
tf6_input = Set(
    "iOptimizerConfig",
    SetType.INPUT,
    [
        Attribute("learning_rate", AttributeType.NUMERIC),
        Attribute("momentum", AttributeType.NUMERIC),
        Attribute("nesterov", AttributeType.TEXT),
        Attribute("name", AttributeType.TEXT),
    ],
)

tf6_output = Set(
    "oOptimizerConfig",
    SetType.OUTPUT,
    [],
)

tf6.set_sets([tf6_input, tf6_output])
df.add_transformation(tf6)

tf7 = Transformation("LossConfig")
tf7_input = Set(
    "iLossConfig",
    SetType.INPUT,
    [
        Attribute("from_logits", AttributeType.TEXT),
        Attribute("ignore_class", AttributeType.TEXT),
        Attribute("reduction", AttributeType.TEXT),
        Attribute("name", AttributeType.TEXT),
    ],
)
tf7_output = Set(
    "oLossConfig",
    SetType.OUTPUT,
    [],
)

tf7.set_sets([tf7_input, tf7_output])
df.add_transformation(tf7)

tf8 = Transformation("TrainingConfig")
tf8_input = Set(
    "iTrainingConfig",
    SetType.INPUT,
    [
        Attribute("server_id", AttributeType.NUMERIC),
        Attribute("server_round", AttributeType.NUMERIC),
        Attribute("starting_time", AttributeType.TEXT),
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

tf8_output = Set(
    "oTrainingConfig",
    SetType.OUTPUT,
    [
        Attribute("server_round", AttributeType.NUMERIC),
        Attribute("dynamically_adjusted", AttributeType.TEXT),
    ],
)

tf1_output.set_type(SetType.INPUT)
tf1_output.dependency = tf1._tag

tf2_output.set_type(SetType.INPUT)
tf2_output.dependency = tf2._tag

tf3_output.set_type(SetType.INPUT)
tf3_output.dependency = tf3._tag

tf8.set_sets([tf1_output, tf2_output, tf3_output, tf8_input, tf8_output])

df.add_transformation(tf8)

tf9 = Transformation("ClientTraining")

tf9_output = Set(
    "oClientTraining",
    SetType.OUTPUT,
    [
        Attribute("client_id", AttributeType.NUMERIC),
        Attribute("server_round", AttributeType.NUMERIC),
        Attribute("training_time", AttributeType.NUMERIC),
        Attribute("size_x_train", AttributeType.NUMERIC),
        Attribute("accuracy", AttributeType.NUMERIC),
        Attribute("loss", AttributeType.NUMERIC),
        Attribute("val_loss", AttributeType.NUMERIC),
        Attribute("val_accuracy", AttributeType.TEXT),
        Attribute("local_weights", AttributeType.TEXT),
        Attribute("starting_time", AttributeType.TEXT),
        Attribute("ending_time", AttributeType.TEXT),
    ],
)

tf4_output.set_type(SetType.INPUT)
tf4_output.dependency = tf4._tag

tf5_output.set_type(SetType.INPUT)
tf5_output.dependency = tf5._tag

tf6_output.set_type(SetType.INPUT)
tf6_output.dependency = tf6._tag

tf7_output.set_type(SetType.INPUT)
tf7_output.dependency = tf7._tag

tf8_output.set_type(SetType.INPUT)
tf8_output.dependency = tf8._tag

tf9.set_sets(
    [
        tf4_output,
        tf5_output,
        tf6_output,
        tf7_output,
        tf8_output,
        tf9_output,
    ]
)
df.add_transformation(tf9)

tf10 = Transformation("ServerTrainingAggregation")
tf10_output = Set(
    "oServerTrainingAggregation",
    SetType.OUTPUT,
    [
        Attribute("server_id", AttributeType.NUMERIC),
        Attribute("server_round", AttributeType.NUMERIC),
        Attribute("total_num_clients", AttributeType.NUMERIC),
        Attribute("total_num_examples", AttributeType.NUMERIC),
        Attribute("accuracy", AttributeType.NUMERIC),
        Attribute("loss", AttributeType.NUMERIC),
        Attribute("val_accuracy", AttributeType.NUMERIC),
        Attribute("val_loss", AttributeType.NUMERIC),
        Attribute("weights_mongo_id", AttributeType.TEXT),
        Attribute("insertion_time", AttributeType.TEXT),
        Attribute("training_time", AttributeType.NUMERIC),
        Attribute("starting_time", AttributeType.TEXT),
        Attribute("ending_time", AttributeType.TEXT),
    ],
)

tf9_output.set_type(SetType.INPUT)
tf9_output.dependency = tf9._tag

tf10.set_sets([tf9_output, tf10_output])
df.add_transformation(tf10)

tf11 = Transformation("EvaluationConfig")
tf11_input = Set(
    "iEvaluationConfig",
    SetType.INPUT,
    [
        Attribute("batch_size", AttributeType.NUMERIC),
        Attribute("steps", AttributeType.TEXT),
    ],
)

tf11_output = Set(
    "oEvaluationConfig",
    SetType.OUTPUT,
    [],
)

tf10_output.set_type(SetType.INPUT)
tf10_output.dependency = tf10._tag

tf11.set_sets([tf10_output, tf11_input, tf11_output])
df.add_transformation(tf11)

tf12 = Transformation("ClientEvaluation")

tf12_output = Set(
    "oClientEvaluation",
    SetType.OUTPUT,
    [
        Attribute("client_id", AttributeType.NUMERIC),
        Attribute("server_round", AttributeType.NUMERIC),
        Attribute("loss", AttributeType.NUMERIC),
        Attribute("evaluation_time", AttributeType.NUMERIC),
        Attribute("accuracy", AttributeType.NUMERIC),
        Attribute("num_testing_examples", AttributeType.NUMERIC),
        Attribute("starting_time", AttributeType.TEXT),
        Attribute("ending_time", AttributeType.TEXT),
    ],
)

tf11_output.set_type(SetType.INPUT)
tf11_output.dependency = tf11._tag

tf12.set_sets([tf11_output, tf12_output])
df.add_transformation(tf12)

tf13 = Transformation("ServerEvaluationAggregation")

tf13_output = Set(
    "oServerEvaluationAggregation",
    SetType.OUTPUT,
    [
        Attribute("server_id", AttributeType.NUMERIC),
        Attribute("server_round", AttributeType.NUMERIC),
        Attribute("total_num_clients", AttributeType.NUMERIC),
        Attribute("total_num_examples", AttributeType.NUMERIC),
        Attribute("accuracy", AttributeType.NUMERIC),
        Attribute("loss", AttributeType.NUMERIC),
        Attribute("evaluation_time", AttributeType.NUMERIC),
        Attribute("starting_time", AttributeType.TEXT),
        Attribute("ending_time", AttributeType.TEXT),
    ],
)

tf12_output.set_type(SetType.INPUT)
tf12_output.dependency = tf12._tag

tf13.set_sets([tf11_output, tf13_output])
df.add_transformation(tf13)

tf8 = Transformation("TrainingConfig")

tf13_output.set_type(SetType.INPUT)
tf13_output.dependency = tf13._tag

tf8.set_sets([tf13_output])
df.add_transformation(tf8)

df.save()
tries = 0
while tries < 100:
    try:
        conn = pymonetdb.connect(
            hostname=monetdb_settings["hostname"],
            port=monetdb_settings["port"],
            username=monetdb_settings["username"],
            password=monetdb_settings["password"],
            database=monetdb_settings["database"]
        )
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE OR REPLACE FUNCTION check_metrics (fl_round int)
        RETURNS table (training_time double, accuracy_training double, loss_training double, 
            val_accuracy double, val_loss double, accuracy_evaluation double, loss_evaluation double)
        BEGIN
            RETURN
            SELECT
                st.training_time,
                st.accuracy,
                st.loss,
                st.val_accuracy,
                st.val_loss,
                se.accuracy,
                se.loss
            FROM
                oservertrainingaggregation as st
            JOIN 
                oserverevaluationaggregation as se
            ON
                st.server_round = se.server_round
            WHERE
                st.server_round = fl_round;
        END;"""
        )

        cursor.execute(
            """CREATE OR REPLACE FUNCTION update_hyperparameters (accuracy_goal double,
        limit_training_time double,
        limit_accuracy_change double,
        fl_round int)
        RETURNS boolean
        BEGIN
            RETURN
            SELECT 
                CASE WHEN (SELECT DISTINCT dynamically_adjusted FROM otrainingconfig
                    WHERE server_round BETWEEN fl_round - 2 AND fl_round - 1 AND dynamically_adjusted = 'True') IS NOT NULL THEN 0
                    WHEN (SELECT DISTINCT
                    CASE
                        WHEN (last_value(accuracy_training) OVER () < accuracy_goal
                        AND last_value(training_time) OVER () < limit_training_time*60 
                        AND (last_value(accuracy_training) OVER () > first_value(accuracy_training) OVER ()
                        AND last_value(val_accuracy) OVER () > first_value(val_accuracy) OVER ())
                        AND last_value(accuracy_training) OVER () - first_value(accuracy_training) OVER () < limit_accuracy_change)
                        THEN 1
                        ELSE 0
                    END
                    FROM
                        (
                        SELECT * FROM check_metrics(fl_round - 2)
                        UNION 
                        SELECT * FROM check_metrics(fl_round - 1)) AS t1) THEN 1
                ELSE 0
            END;
        END;"""
        )

        cursor.execute(
            """
        CREATE OR REPLACE FUNCTION check_last_round_fl (server_id int)
        RETURNS int
        BEGIN
            RETURN
            SELECT
                MAX(server_round)
            FROM
                oservertrainingaggregation as st
            WHERE st.server_id = server_id;
        END;"""
        )

        cursor.execute(
            """
        CREATE OR REPLACE FUNCTION get_num_rounds (server_id int)
        RETURNS int
        BEGIN
            RETURN
            SELECT
                MIN(num_rounds)
            FROM 
                iServerConfig as sc
            WHERE sc.server_id = server_id;
        END;"""
        )

        cursor.execute(
            """
        CREATE OR REPLACE FUNCTION check_ending_fl (server_id int)
        RETURNS bool
        BEGIN
            RETURN
            SELECT
                check_last_round_fl(server_id) = get_num_rounds(server_id);
        END;"""
        )

        cursor.execute(
            """
        CREATE OR REPLACE FUNCTION check_max_server_id ()
        RETURNS int
        BEGIN
            RETURN
            SELECT
                MAX(server_id)
            FROM 
                iServerConfig as sc;
        END;"""
        )

        conn.commit()
        cursor.close()
        conn.close()
        break
    except Exception as e:
        time.sleep(1)
        tries += 1
