from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
import pandas as pd


class ProfileModel:
    """Method to score missions and create profiles"""

    def __init__(self, df, trans_df, lda_model, cv_model):
        """
        Parameters
        ----------
        df : pyspark.sql.DataFrame
            DataFrame containing basket ID and TFIDF transformation
        trans_df : pyspark.sql.DataFrame
            DataFrame containing basket ID and profiling variables
        lda_model : pyspark.clustering.DistributedLDAModel
            Fitted LDA model
        cv_model :  pyspark.ml.feature.CountVectorizer
        The fitted count vectorizer model

        """

        self.df = df
        self.trans_df = trans_df
        self.model = lda_model
        self.cvmodel = cv_model
        self.df_all_profile_trans = self.tag_mission_to_basket()
        self.tot_basks_by_mission = self.get_mission_counts()
        self.tot_basks = self.get_tot_counts()

    def tag_mission_to_basket(self):
        """ Function to score DataFrame with TFIDF transformation on fitted LDA model

        Returns
        ----------
        df_all_profile_trans : pyspark.sql.DataFrame
            DataFrame containing basket ID, mission and all profiling variables

        """

        # score the DataFrame to tag a basket with a mission
        df_mission = self.model.transform(self.df)
        df_mission = df_mission.withColumnRenamed("id", "BASKET_ID")

        # Get the mission based on the index of the maximum value of the topicDistribution
        max_index = F.udf(lambda x: x.tolist().index(max(x)), IntegerType())
        df_mission = df_mission.withColumn("mission", max_index("topicDistribution")) \
            .drop('features', 'topicDistribution')

        # Join back to the main transaction file to add the mission
        df_all_profile_trans = self.trans_df.join(df_mission, 'BASKET_ID')

        df_all_profile_trans.persist()

        return df_all_profile_trans

    def get_mission_counts(self):
        """ Function to get counts of baskets by mission

        Returns
        ----------
        tot_basks_by_mission : pyspark.sql.DataFrame
            DataFrame containing total basket counts by mission

        """

        # Get the count of total baskets by mission
        tot_basks_by_mission = self.df_all_profile_trans.groupBy('mission') \
            .agg(F.countDistinct('BASKET_ID').alias('TOT_BASKS_MISSION'))

        tot_basks_by_mission.persist()

        return tot_basks_by_mission

    def get_tot_counts(self):
        """ Function to get count of total baskets

        Returns
        ----------
        tot_basks : int
            Count of total baskets

        """

        # Get the count of total baskets
        tot_basks = self.df_all_profile_trans.agg(F.countDistinct('BASKET_ID').alias('TOT_BASKS')).collect()[0][0]

        return tot_basks

    def get_top_terms(self, num_terms):
        """ Function to get the top 'terms' (products) by mission
        Parameters
        ----------
        num_terms : int
            The number of top 'terms' (products) to return per mission

        Returns
        ----------
        mission_prod_profile : pyspark.sql.DataFrame
            DataFrame containing the mission, the top 'terms' (products) and the 'term weights'

        """

        # Get the top terms by mission from the model
        model_describe = self.model.describeTopics(num_terms)

        # Melt the data to get one row per topic and term
        top_terms = model_describe.select('topic', 'termIndices') \
            .withColumn('termIndices', F.explode('termIndices')) \
            .withColumn("id", F.monotonically_increasing_id())

        top_weights = model_describe.select('termWeights') \
            .withColumn('termWeights', F.explode('termWeights')) \
            .withColumn("id", F.monotonically_increasing_id())

        model_describe_long = top_terms.join(top_weights, ['id']).drop('id')

        # Map the column termIndices to a product
        def translate(mapping):
            def translate_(col):
                return mapping.get(col)

            return F.udf(translate_, StringType())

        vocab_dict = pd.DataFrame(self.cvmodel.vocabulary).to_dict()[0]
        mission_prod_profile = model_describe_long.withColumn("PROD_CODE", translate(vocab_dict)("termIndices")) \
            .withColumnRenamed("topic", "mission") \
            .drop('termIndices')

        return mission_prod_profile

    def run_profiles(self, var):
        """ Function to create a profile by the specified variable

        Parameters
        ----------
        var  : str
            The name of the variable to profile

        Returns
        ----------
        trans_profile : pyspark.sql.DataFrame
            DataFrame containing the profile by mission

        """

        # Get the count of baskets by mission and profile variable
        trans_profile = self.df_all_profile_trans.groupby('mission', var) \
            .agg(F.countDistinct('BASKET_ID').alias('NUM_BASKS_PROFILE'))

        trans_profile = trans_profile.join(self.tot_basks_by_mission, 'mission')

        # Get percentage of baskets by mission and profile variable
        trans_profile = trans_profile.withColumn("PERC_MISSION",
                                                 F.col("NUM_BASKS_PROFILE") / F.col("TOT_BASKS_MISSION"))

        # Get percentage of baskets over all missions
        tot_basks_profile = self.df_all_profile_trans.groupby(var) \
            .agg(F.countDistinct('BASKET_ID').alias('NUM_BASKS_OVERALL_PROFILE'))

        # Calculate the index
        index_perc = tot_basks_profile.withColumn("PERC_OVERALL", F.col("NUM_BASKS_OVERALL_PROFILE") / self.tot_basks)

        trans_profile = trans_profile.join(index_perc, var)
        trans_profile = trans_profile.withColumn("INDEX", (F.col("PERC_MISSION") / F.col("PERC_OVERALL")) * 100)
        trans_profile = trans_profile.drop("NUM_BASKS_PROFILE", "TOT_BASKS_MISSION", "NUM_BASKS_OVERALL_PROFILE",
                                           "PERC_OVERALL")

        # Rename the columns to allow profiles to be set together
        trans_profile = trans_profile.withColumnRenamed(var, "var_detail")
        trans_profile = trans_profile.withColumn("var", F.lit(var))

        trans_profile = trans_profile.orderBy(["mission", var])

        return trans_profile
