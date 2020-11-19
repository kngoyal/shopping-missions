# Add path to modules to sys path
import sys

sys.path.insert(1, "/home/ubuntu/trip_missions")

from airflow import DAG
from airflow.contrib.operators.emr_create_job_flow_operator import (
    EmrCreateJobFlowOperator,
)
from airflow.contrib.sensors.emr_step_sensor import EmrStepSensor
from airflow.contrib.operators.emr_add_steps_operator import EmrAddStepsOperator
from airflow.contrib.operators.emr_terminate_job_flow_operator import (
    EmrTerminateJobFlowOperator,
)
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from config.load_config import load_yaml
from config import constants
from config.load_config import Config
from utils.send_email import notify_email
from utils.logging_framework import log
from utils.copy_app_to_s3 import copy_app_to_s3


# Load the config file
config = load_yaml(constants.config_path)

# Check the config types
try:
    Config(**config)
except TypeError as error:
    log.error(error)

# ====== Functions ======


def score_only():
    """check if DAG should only score a DataFrame
    """

    score = False
    run_scoring_config = config["app"]["ScoreOnly"]

    if run_scoring_config.lower() == "yes":
        score = True

    return score


# Determine if only scoring should be executed
run_scoring = score_only()

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

    # Determine if only scoring is to be run
    branching = BranchPythonOperator(
        task_id="branching",
        dag=dag,
        python_callable=lambda: "add_step_model_scoring" if run_scoring else "add_step_stage_data",
    )

    # ========== MODEL SCORING ==========
    # Run scoring depending on branching decision
    task = "model_scoring"
    model_scoring = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run model scoring step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["ScoreRunner"],
                        task,
                        config["s3"]["StagingDataPath"],
                        config["s3"]["SavedModels"],
                        config["s3"]["ScoringDataPath"],
                        config["s3"]["ScoringFileName"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    model_scoring_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # ========== STAGE DATA ==========

    # Stage the transaction and time DataFrames
    task = "stage_data"
    stage_data = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run stage data step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["StageRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["s3"]["DataFolder"],
                        config["s3"]["StagingDataPath"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    staging_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # ========== PRE-PROCESS DATA ==========

    # Pre-process the data for input to LDA
    task = "preprocess_data"
    preprocess_data = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run pre-process data step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["PreprocessRunner"],
                        task,
                        config["s3"]["StagingDataPath"],
                        config["preprocessing"]["sample"],
                        config["preprocessing"]["sample_rate"],
                        config["preprocessing"]["train_frac"],
                        config["s3"]["SavedModels"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    preprocessing_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # ========== TUNE MODEL ==========

    # Tune the LDA model
    task = "train_tune_model"
    train_tune_model = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run LDA training and tuning step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["TuneModelRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["s3"]["StagingDataPath"],
                        config["model"]["MaxIterations"],
                        config["s3"]["SavedModels"],
                        config["model"]["Tune"],
                        config["model"]["k"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    train_tune_model_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # ========== PROFILING ==========

    # Run profiling
    task = "profiling"
    profiling = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=[
            {
                "Name": "Run profiling step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": [
                        "spark-submit",
                        "--deploy-mode",
                        "cluster",
                        "--py-files",
                        config["s3"]["egg"],
                        config["s3"]["ProfilingRunner"],
                        task,
                        config["s3"]["Bucket"],
                        config["s3"]["StagingDataPath"],
                        config["s3"]["SavedModels"],
                        "{{ execution_date }}",
                    ],
                },
            }
        ],
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    profiling_step_sensor = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    # Remove the cluster
    cluster_remover = EmrTerminateJobFlowOperator(
        task_id="remove_cluster",
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    create_egg >> upload_code >> cluster_creator >> branching >> \
        stage_data >> staging_step_sensor >> \
        preprocess_data >> preprocessing_step_sensor >> \
        train_tune_model >> train_tune_model_step_sensor >> \
        profiling >> profiling_step_sensor >> cluster_remover

    branching >> model_scoring >> model_scoring_step_sensor >> cluster_remover
