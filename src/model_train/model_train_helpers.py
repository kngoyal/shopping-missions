import pandas as pd
import os
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.ml.clustering import LDA, DistributedLDAModel


def create_lda_iterator(train_df: SparkDataFrame, max_iterations: int) -> LDA:
    """Function to create the LDA iterator object to fit multiple models

    Parameters
    ----------
    train_df : pyspark.sql.DataFrame
        DataFrame containing the training set
    max_iterations : int
        Maximum number of iterations for the LDA model

    Returns
    -------
    model_iterator : pyspark.ml.clustering.LDA
        Model iterator object

    """

    lda = LDA(seed=1, optimizer="em")
    lda.setMaxIter(max_iterations)

    params = [{lda.k: 10},
              ]

    model_iterator = lda.fitMultiple(train_df, params)

    return model_iterator


def run_lda_models(test_df: SparkDataFrame, model_iterator: iter, num_iters: int, model_path: str) -> tuple:
    """Function to fit the LDA models

    Parameters
    ----------
    test_df : pyspark.sql.DataFrame
        DataFrame containing the test set
    model_iterator : iter
        LDA model iterator
    num_iters : int
        The number of times to iterate e.g. number of parameter combinations to fit
    model_path : str
        The path on s3 to save the LDA models

    Returns
    -------
    log_likelihood : list
        list containing log likelihood on test set
    log_perplexity : list
        list containing log perplexity on the test set
    num_topics : list
        list containing the number of topics for the iteration

    """

    log_likelihood = []
    log_perplexity = []
    num_topics = []
    iteration = []

    for i in range(0, num_iters):
        model = next(model_iterator)

        log_likelihood_model = model[1].logLikelihood(test_df)
        log_perplexity_model = model[1].logPerplexity(test_df)
        num_topics_model = model[1].describeTopics().count()
        iteration_model = i

        log_likelihood.append(log_likelihood_model)
        log_perplexity.append(log_perplexity_model)
        num_topics.append(num_topics_model)
        iteration.append(iteration_model)

        save_path = os.path.join(model_path, "model_{}".format(i))
        model[1].write().overwrite().save(save_path)

    return iteration, log_likelihood, log_perplexity, num_topics


def load_best(model_performance: pd.DataFrame, model_path: str) -> DistributedLDAModel:
    """Function to load the model with the lowest perplexity

    Parameters
    ----------
    model_performance : pandas.DataFrame
        Pandas DataFrame containing the model performance output
    model_path : str
        Path to saved models

    Returns
    -------
    model : pyspark.clustering.DistributedLDAModel

    """

    model_performance.sort_values("log_perplexity", inplace=True)
    best_model = int(model_performance.iloc[0][0])
    load_path = os.path.join(model_path, "model_{}".format(best_model))
    model = DistributedLDAModel.load(load_path)

    return model
