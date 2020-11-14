import sys
import boto3
from io import StringIO
from pyspark.sql import SparkSession
from src.model_train.model_train_helpers import *
from utils.logging_framework import log

if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[2]
    staging_path = sys.argv[4]
    max_iterations = int(sys.argv[9])
    model_path = sys.argv[10]
    tune = sys.argv[11]
    k = int(sys.argv[12])

    spark = SparkSession.builder.appName("trip_missions").getOrCreate()

    log.info("Running task {}".format(task))

    # ========== Import training and test DataFrames ==========
    train_path = os.path.join(staging_path, "train-df/")
    log.info("Reading training DataFrame from {}".format(train_path))
    train_df = spark.read.parquet(train_path)

    test_path = os.path.join(staging_path, "test-df/")
    log.info("Reading test DataFrame from {}".format(test_path))
    test_df = spark.read.parquet(test_path)

    # Should the model be tuned?
    if tune == 'Tune':

        # ========== Tune the LDA ==========
        model_iterator = create_lda_iterator(
            train_df=train_df, max_iterations=max_iterations
        )

        iteration, log_likelihood, log_perplexity, num_topics = tune_lda_models(
            test_df=test_df,
            model_iterator=model_iterator,
            num_iters=5,
            model_path=model_path,
        )

        # ========== Get performance metrics ==========
        model_performance = pd.DataFrame(
            [iteration, num_topics, log_perplexity, log_likelihood]
        ).T
        model_performance.columns = [
            "iteration",
            "num_topics",
            "log_perplexity",
            "log_likelihood",
        ]
        model_performance.loc[:, "max_iterations"] = max_iterations

        # Write the performance metrics
        performance_path = os.path.join(
            staging_path, "model-performance/model_performance.csv"
        )
        log.info("Saving model performance DataFrame to {}".format(performance_path))

        bucket = bucket
        csv_buffer = StringIO()
        model_performance.to_csv(csv_buffer)

        s3_resource = boto3.resource("s3")
        s3_resource.Object(bucket, "model-performance/model_performance.csv").put(
            Body=csv_buffer.getvalue()
        )

        # ========== Load and save the best model ==========
        log.info("Loading best performing model")
        model = load_best(model_performance=model_performance, model_path=model_path)

        save_best_path = os.path.join(model_path, "best-model/")
        log.info("Saving the best performing model to {}".format(save_best_path))
        model.write().overwrite().save(save_best_path)

    # If model does not need to be tuned fit with fixed k
    else:

        log.info("Training LDA model with max_iterations = {} and k = {}".format(max_iterations, k))
        model = train_lda_model(train_df=train_df,
                                max_iterations=max_iterations,
                                k=k)

        save_path = os.path.join(model_path, "best-model/")
        log.info("Saving the model to {}".format(save_path))
        model.write().overwrite().save(save_path)

