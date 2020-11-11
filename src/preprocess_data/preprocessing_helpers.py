from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import IntegerType, ArrayType
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizer, IDF


def create_bask_list(df: SparkDataFrame) -> SparkDataFrame:
    """Function to generate a DataFrame with a unique row for each basket with a column containing
    a list of the products purchased in the basket - this is similar to a 'document' where the products
    are the words

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame containing the raw transactions

    Returns
    -------
    basket_item_list : pyspark.sql.DataFrame
       DataFrame containing the unique basket identifier and a list of items purchased in the basket
       as a feature column

    """

    explode_df = df.select("BASKET_ID", "PROD_CODE", "QUANTITY")

    n_to_array = F.udf(lambda n: [n] * n, ArrayType(IntegerType()))
    explode_df2 = explode_df.withColumn("n", n_to_array(explode_df.QUANTITY))
    explode_df3 = explode_df2.withColumn("n", F.explode(explode_df2.n))

    basket_item_list = explode_df3.groupBy("BASKET_ID").agg(
        F.collect_list("PROD_CODE").alias("BASKET_ITEM_LIST")
    )

    return basket_item_list


def create_tfidf(
    train: SparkDataFrame, test: SparkDataFrame, num_unique_prods: int
) -> tuple:
    """Function to generate tf-idf vector for input to LDA

    Parameters
    ----------
    train : pyspark.sql.DataFrame
        DataFrame containing the training set
    test : pyspark.sql.DataFrame
        DataFrame containing the test set
    num_unique_prods : int
        The number of unique products across the train and test set - drives the total 'vocabulary'

    Returns
    -------
    result_tfidf_train : pyspark.sql.DataFrame
       DataFrame containing the tf-idf vector for the training set
    result_tfidf_test : pyspark.sql.DataFrame
       DataFrame containing the tf-idf vector for the test set
    cvmodel : pyspark.ml.feature.CountVectorizer
        The fitted count vectorizer model

    """

    # TF
    cv = CountVectorizer(
        inputCol="BASKET_ITEM_LIST",
        outputCol="raw_features",
        vocabSize=num_unique_prods,
        minDF=100,
    )

    cvmodel = cv.fit(train)
    result_cv_train = cvmodel.transform(train)
    result_cv_test = cvmodel.transform(test)

    # IDF
    idf = IDF(inputCol="raw_features", outputCol="features")
    idfModel = idf.fit(result_cv_train)
    result_tfidf_train = idfModel.transform(result_cv_train)
    result_tfidf_test = idfModel.transform(result_cv_test)

    return cvmodel, result_tfidf_train, result_tfidf_test
