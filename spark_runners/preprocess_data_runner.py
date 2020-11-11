import sys
import os
from pyspark.sql import SparkSession
from src.preprocess_data.preprocessing_helpers import *
from utils.logging_framework import log

if __name__ == "__main__":

    task = sys.argv[1]
    staging_path = sys.argv[4]
    sample = sys.argv[6]
    sample_rate = float(sys.argv[7])
    train_frac = float(sys.argv[8])
    model_path = sys.argv[10]

    spark = SparkSession.builder.appName("trip_missions").getOrCreate()

    log.info("Running task {}".format(task))

    # ========== Import transaction data from staging ==========
    staging_trans_path = os.path.join(staging_path, "trans-data/")
    log.info("Importing transaction staging data from")
    trans_df = spark.read.parquet(staging_trans_path)

    # ========== Sample DataFrame if requested ==========

    if sample == "True":
        log.info(
            "Sampling transaction DataFrame with sample rate {}".format(sample_rate)
        )
        trans_df = trans_df.sample(withReplacement=False, fraction=sample_rate, seed=42)

    # ========== Generate item lists for each basket ==========

    log.info("Creating basket and item list")
    basket_item_list = create_bask_list(trans_df)

    # ========== Split the DataFrame into training and test sets ==========

    log.info(
        "Splitting DataFrame into training and test sets based on training percentage {}".format(
            train_frac
        )
    )
    test_frac = 1 - train_frac
    train, test = basket_item_list.randomSplit([train_frac, test_frac], seed=1)
    log.info(
        "Training set contains {} rows and test set contains {} rows".format(
            train.count(), test.count()
        )
    )

    # ========== Create TFIDF vectors ==========

    log.info("Creating TFIDF vectors")
    num_unique_prods = trans_df.select("PROD_CODE").dropDuplicates().count()
    cvmodel, result_tfidf_train, result_tfidf_test = create_tfidf(
        train, test, num_unique_prods
    )

    # Save the count vectorizer model
    cv_save_path = os.path.join(model_path, "cv-model")
    log.info("Saving model to {}".format(cv_save_path))
    cvmodel.write().overwrite().save(cv_save_path)

    # ========== Save training and test DataFrames to s3 ==========

    # Rename "BASKET_ID" as "id" on training and test sets as the LDA algorithm expects these columns
    train_df = result_tfidf_train.select("BASKET_ID", "features").withColumnRenamed(
        "BASKET_ID", "id"
    )

    test_df = result_tfidf_test.select("BASKET_ID", "features").withColumnRenamed(
        "BASKET_ID", "id"
    )

    train_path = os.path.join(staging_path, "train-df/")
    log.info("Saving training DataFrame to {}".format(train_path))
    train_df.write.parquet(train_path, mode="overwrite")

    test_path = os.path.join(staging_path, "test-df/")
    log.info("Saving test DataFrame to {}".format(test_path))
    test_df.write.parquet(test_path, mode="overwrite")
