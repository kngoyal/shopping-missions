import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.clustering import DistributedLDAModel
from src.preprocess_data.preprocessing_helpers import *
from utils.logging_framework import log
from pyspark.sql.types import (
    StringType,
    IntegerType,
    StructField,
    FloatType,
    StructType
)

if __name__ == "__main__":

    task = sys.argv[1]
    staging_path = sys.argv[2]
    model_path = sys.argv[3]
    scoring_path = sys.argv[4]
    scoring_file = sys.argv[5]

    spark = SparkSession.builder.appName("trip_missions").getOrCreate()

    log.info("Running task {}".format(task))

    # ========== Import data to score ==========
    log.info("Importing scoring data from {}".format(scoring_path))
    log.info("Scoring data file {}".format(scoring_file))
    path_to_score = os.path.join(scoring_path, scoring_file)

    trans_field = [
        StructField("SHOP_WEEK", IntegerType(), True),
        StructField("SHOP_DATE", IntegerType(), True),
        StructField("SHOP_WEEKDAY", IntegerType(), True),
        StructField("SHOP_HOUR", IntegerType(), True),
        StructField("QUANTITY", IntegerType(), True),
        StructField("SPEND", FloatType(), True),
        StructField("PROD_CODE", StringType(), True),
        StructField("PROD_CODE_10", StringType(), True),
        StructField("PROD_CODE_20", StringType(), True),
        StructField("PROD_CODE_30", StringType(), True),
        StructField("PROD_CODE_40", StringType(), True),
        StructField("CUST_CODE", StringType(), True),
        StructField("CUST_PRICE_SENSITIVITY", StringType(), True),
        StructField("CUST_LIFESTAGE", StringType(), True),
        StructField("BASKET_ID", StringType(), True),
        StructField("BASKET_SIZE", StringType(), True),
        StructField("BASKET_PRICE_SENSITIVITY", StringType(), True),
        StructField("BASKET_TYPE", StringType(), True),
        StructField("BASKET_DOMINANT_MISSION", StringType(), True),
        StructField("STORE_CODE", StringType(), True),
        StructField("STORE_FORMAT", StringType(), True),
        StructField("STORE_REGION", StringType(), True),
    ]

    trans_schema = StructType(trans_field)
    trans_df = spark.read.csv(path_to_score, header=True, schema=trans_schema)

    # ========== Prepare data for scoring ==========

    log.info("Creating basket and item list")
    scoring_df = create_bask_list(trans_df)

    log.info("Creating TFIDF vector")
    cv_load_path = os.path.join(model_path, "cv-model")
    log.info("Loading the Countvectorizer model from {}".format(cv_load_path))
    cvmodel = CountVectorizerModel.load(cv_load_path)

    # TF
    tf = cvmodel.transform(scoring_df)

    # IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    idfModel = idf.fit(tf)
    scoring_df_tfidf = idfModel.transform(tf)

    scoring_df_tfidf = scoring_df_tfidf.select(
        "BASKET_ID", "features"
    ).withColumnRenamed("BASKET_ID", "id")

    # ========== Load the LDA model and score ==========
    best_path = os.path.join(model_path, "best-model/")
    log.info("Loading the best performing LDA model from {}".format(best_path))
    model = DistributedLDAModel.load(best_path)

    scored_baskets = model.transform(scoring_df_tfidf)
    scored_baskets = scored_baskets.withColumnRenamed("id", "BASKET_ID")

    log.info("Obtain the shopping mission")

    # Get the mission based on the index of the maximum value of the topicDistribution
    max_index = F.udf(lambda x: x.tolist().index(max(x)), IntegerType())
    tagged_mission = scored_baskets.withColumn(
        "mission", max_index("topicDistribution")
    ).drop("features", "topicDistribution")

    # Join back to the original scoring data
    trans_df_scored = trans_df.join(tagged_mission, "BASKET_ID")

    # ========== Write the scored file ==========
    scoring_path = os.path.join(scoring_path, "scored-df/")
    log.info("Write the scored file to ".format(scoring_path))
    trans_df_scored.write.parquet(scoring_path, mode="overwrite")

    # ========== Print distribution of scored baskets for QA ==========
    scored_basket_dist = trans_df_scored.groupBy("mission").agg(
        F.countDistinct("BASKET_ID").alias("NUM_BASKETS")
    )
    scored_basket_dist.show(scored_basket_dist.count())
