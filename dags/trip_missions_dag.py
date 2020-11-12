# Add path to modules to sys path
import sys

sys.path.insert(1, "/home/ubuntu/trip_missions")

from airflow import DAG
from airflow.contrib.operators.emr_create_job_flow_operator import (
    EmrCreateJobFlowOperator,
)
from airflow.contrib.operators.emr_terminate_job_flow_operator import (
    EmrTerminateJobFlowOperator,
)
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from config.load_config import load_yaml
from config import constants
from config.load_config import Config
from utils.send_email import notify_email
from utils.run_spark_jobs import add_spark_step
from utils.logging_framework import log
from utils.copy_app_to_s3 import copy_app_to_s3


# Load the config file
config = load_yaml(constants.config_path)

# Check the config types
try:
    Config(**config)
except TypeError as error:
    log.error(error)

with DAG(**config["dag"]) as dag:

    # Create egg file
    create_egg = BashOperator(
        task_id="create_app_egg",
        bash_command="cd /home/ubuntu/trip_missions && python /home/ubuntu/trip_missions/setup.py bdist_egg",
        run_as_user="airflow",
    )

    # Copy application files to s3
    upload_code = PythonOperator(
        task_id="upload_app_to_s3", python_callable=copy_app_to_s3, op_args=[config]
    )

    # Start the cluster
    cluster_creator = EmrCreateJobFlowOperator(
        task_id="create_job_flow",
        job_flow_overrides=config["emr"],
        aws_conn_id="aws_default",
        emr_conn_id="emr_default",
        on_failure_callback=notify_email,
    )

    # Stage the transaction and time DataFrames
    # task = "stage_data"
    # stage_data, staging_step_sensor = add_spark_step(
    #    task=task,
    #    path_to_egg=config["s3"]["egg"],
    #    runner=config["s3"]["StageRunner"],
    #    bucket=config["s3"]["Bucket"],
    #    data_folder=config["s3"]["DataFolder"],
    #    staging_path=config["s3"]["StagingDataPath"],
    # )

    # Preprocess the data for input to LDA
    # task = "preprocess_data"
    # preprocess_data, preprocessing_step_sensor = add_spark_step(
    #     task=task,
    #     path_to_egg=config["s3"]["egg"],
    #     runner=config["s3"]["PreprocessRunner"],
    #     staging_path=config["s3"]["StagingDataPath"],
    #     sample=config["preprocessing"]["sample"],
    #     sample_rate=config["preprocessing"]["sample_rate"],
    #     train_frac=config["preprocessing"]["train_frac"],
    #     model_path=config["s3"]["SavedModels"],
    # )
    #
    # # Tune the LDA model
    # task = "tune_model"
    # tune_model, tune_model_step_sensor = add_spark_step(
    #     task=task,
    #     path_to_egg=config["s3"]["egg"],
    #     runner=config["s3"]["TuneModelRunner"],
    #     bucket=config["s3"]["Bucket"],
    #     staging_path=config["s3"]["StagingDataPath"],
    #     max_iterations=config["model_tune"]["MaxIterations"],
    #     model_path=config["s3"]["SavedModels"],
    # )

    # Run profiling
    task = "profiling"
    profiling, profiling_step_sensor = add_spark_step(
        task=task,
        path_to_egg=config["s3"]["egg"],
        runner=config["s3"]["ProfilingRunner"],
        bucket=config["s3"]["Bucket"],
        staging_path=config["s3"]["StagingDataPath"],
        model_path=config["s3"]["SavedModels"],
        profiling_path=config["s3"]["ProfilingDataPath"],
    )

    # Remove the cluster
    cluster_remover = EmrTerminateJobFlowOperator(
        task_id="remove_cluster",
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    create_egg >> upload_code >> cluster_creator >> \
        profiling >> profiling_step_sensor >> \
        cluster_remover

#stage_data >> staging_step_sensor >> \
#preprocess_data >> preprocessing_step_sensor >> \
#tune_model >> tune_model_step_sensor >> \