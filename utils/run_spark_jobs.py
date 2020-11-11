from airflow.contrib.sensors.emr_step_sensor import EmrStepSensor
from airflow.contrib.operators.emr_add_steps_operator import EmrAddStepsOperator
from utils.aws_utils import add_step_to_emr
from utils.send_email import notify_email
from utils.logging_framework import log
from typing import Tuple


def add_spark_step(
    task: str, path_to_egg: str, runner: str, **kwargs
) -> Tuple[EmrAddStepsOperator, EmrStepSensor]:

    """ Function to add a Spark step to emr

    Parameters
    ----------
    task : str
        Name of task to execute
    path_to_egg : str
        Path to the egg file containing the main Spark application
    runner : str
        Name of the main runner file

    """

    # dum_is a default value if not specified - not all tasks require all kwargs
    bucket = kwargs.get("bucket", "dum_str")
    staging_path = kwargs.get("staging_path", "dum_str")
    data_folder = kwargs.get("data_folder", "dum_str")
    sample = kwargs.get("sample", "True")
    sample_rate = kwargs.get("sample_rate", "1")
    train_frac = kwargs.get("train_frac", "0.8")
    model_path = kwargs.get("model_path", "dum_str")
    max_iterations = kwargs.get("max_iterations", "10")

    # Add the Spark step
    spark_step = add_step_to_emr(
        task_id=task,
        egg=path_to_egg,
        runner=runner,
        bucket=bucket,
        data_folder=data_folder,
        staging_path=staging_path,
        execution_date="{{ execution_date }}",
        sample=sample,
        sample_rate=sample_rate,
        train_frac=train_frac,
        max_iterations=max_iterations,
        model_path = model_path,
    )

    step_adder = EmrAddStepsOperator(
        task_id="add_step_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_job_flow', key='return_value') }}",
        aws_conn_id="aws_default",
        steps=spark_step,
        on_failure_callback=notify_email,
    )

    step_name = "add_step_{}".format(task)
    step_checker = EmrStepSensor(
        task_id="watch_{}".format(task),
        job_flow_id="{{ task_instance.xcom_pull('create_job_flow', key='return_value') }}",
        step_id="{{{{ task_instance.xcom_pull(task_ids='{}', key='return_value')[0] }}}}".format(
            step_name
        ),
        aws_conn_id="aws_default",
        on_failure_callback=notify_email,
    )

    log.info("Step sensor added for task {}".format(task))
    log.info("Step added for task {}".format(task))

    return step_adder, step_checker
