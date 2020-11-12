import sys
import os
import boto3
from io import StringIO
from pyspark.sql import SparkSession
from pyspark.ml.clustering import DistributedLDAModel
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel
from src.profiling.profiling_helpers import *
from utils.logging_framework import log

if __name__ == "__main__":

    task = sys.argv[1]
    bucket = sys.argv[2]
    staging_path = sys.argv[4]
    model_path = sys.argv[10]

    spark = SparkSession.builder.appName("trip_missions").getOrCreate()

    log.info("Running task {}".format(task))

    # ========== Import training and transaction DataFrames ==========
    train_path = os.path.join(staging_path, "train-df/")
    log.info("Reading training DataFrame from {}".format(train_path))
    train_df = spark.read.parquet(train_path)

    staging_trans_path = os.path.join(staging_path, "trans-data/")
    log.info("Importing transaction staging data from")
    trans_df = spark.read.parquet(staging_trans_path)

    # ========== Load the LDA and Countvectorizer models to use for profiling ==========
    best_path = os.path.join(model_path, "best-model/")
    log.info("Loading the best performing LDA model from {}".format(best_path))
    model = DistributedLDAModel.load(best_path)

    cv_load_path = os.path.join(model_path, "cv-model")
    log.info("Loading the Countvectorizer model from {}".format(cv_load_path))
    cvmodel = CountVectorizerModel.load(cv_load_path)

    # ========== Run the profiling ==========

    log.info("Running profiling")
    # Create the profiling object
    profiler = ProfileModel(
        df=train_df, trans_df=trans_df, lda_model=model, cv_model=cvmodel
    )

    # Get top 'terms' (products for each shopping mission)
    log.info("Profiling - top terms executing")
    top_terms = profiler.get_top_terms(num_terms=20)

    # Dominant basket mission (Fresh, Grocery, Mixed)
    log.info("Profiling - dominant basket mission executing")
    dom_miss_profile = profiler.run_profiles("BASKET_DOMINANT_MISSION")

    # Basket size (Small, Medium, Large)
    log.info("Profiling - basket size executing")
    bask_size_profile = profiler.run_profiles("BASKET_SIZE")

    # Basket type (Small Shop, Full Shop, Top Up Shop)
    log.info("Profiling - basket type executing")
    bask_type_profile = profiler.run_profiles("BASKET_TYPE")

    # Store format (SS, MS, LS, XLS - small store, medium store, large store, extra large store)
    log.info("Profiling - store format executing")
    store_format_profile = profiler.run_profiles("STORE_FORMAT")

    # Basket price sensitivity - LA (low affluence), MM (mid-market), UM (up-market)
    log.info("Profiling - basket price sensitivity executing")
    basket_ps_profile = profiler.run_profiles("BASKET_PRICE_SENSITIVITY")

    # Customer price sensitivty - LA (low affluence), MM (mid-market), UM (up-market)
    log.info("Profiling - customer price sensitivity executing")
    cust_ps_profile = profiler.run_profiles("CUST_PRICE_SENSITIVITY")

    # Day of week shopped
    log.info("Profiling - day of week shopped executing")
    day_profile = profiler.run_profiles("SHOP_WEEKDAY")

    # Hour shopped
    log.info("Profiling - hour shopped executing")
    hour_profile = profiler.run_profiles("SHOP_HOUR")

    # Prod code 40
    log.info("Profiling - prod code 40")
    prod_code_40_profile = profiler.run_profiles("PROD_CODE_40")

    # Prod code 30
    log.info("Profiling - prod code 30")
    prod_code_30_profile = profiler.run_profiles("PROD_CODE_30")

    # Set the profiles together
    all_profiles = (
        dom_miss_profile.union(bask_size_profile)
        .union(bask_type_profile)
        .union(store_format_profile)
        .union(basket_ps_profile)
        .union(cust_ps_profile)
        .union(day_profile)
        .union(hour_profile)
        .union(prod_code_40_profile)
        .union(prod_code_30_profile)
    )

    # ========== Save profiles ==========
    bucket = bucket
    csv_buffer = StringIO()
    all_profiles = all_profiles.toPandas()
    all_profiles.to_csv(csv_buffer)

    s3_resource = boto3.resource("s3")
    s3_resource.Object(bucket, "model-profiles/model_profiles.csv").put(
        Body=csv_buffer.getvalue()
    )
