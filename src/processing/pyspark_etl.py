"""
AirRoute AI - PySpark ETL Pipeline
Distributed data cleaning, text normalization, and Parquet serialization.
"""

import os
import time
import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType

from src.ingestion.load_data import get_spark_session, load_raw_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_airline_data(raw_df: DataFrame) -> DataFrame:
    """
    Cleans raw airline tweets/reviews using PySpark native DataFrame operations:
    1. Removes records with null or missing review text or sentiment.
    2. Cleans text: removes URLs, Twitter handles (@mentions), HTML entities, and special characters.
    3. Normalizes text to lowercase and trims extra whitespaces.
    4. Encodes sentiment labels into numeric target IDs:
       - negative: 0
       - neutral:  1
       - positive: 2
    5. Fills missing negativereason with 'None/Positive' for positive/neutral tweets.
    6. Computes review metadata (character count, word count).
    """
    logger.info("Starting PySpark data cleaning and transformation pipeline...")

    # Step 1: Filter out rows where essential fields are missing
    filtered_df = raw_df.filter(
        F.col("text").isNotNull() & 
        F.col("airline_sentiment").isNotNull() & 
        (F.trim(F.col("text")) != "")
    )

    # Step 2: Text cleaning using PySpark SQL regex functions
    cleaned_df = filtered_df.withColumn(
        "clean_text",
        # 1. Replace HTML entities: &amp; -> and
        F.regexp_replace(F.col("text"), r"&amp;", " and ")
    ).withColumn(
        "clean_text",
        # 2. Replace &lt; &gt;
        F.regexp_replace(F.col("clean_text"), r"&lt;|&gt;", " ")
    ).withColumn(
        "clean_text",
        # 3. Remove URLs (http://..., https://..., www....)
        F.regexp_replace(F.col("clean_text"), r"https?://\S+|www\.\S+", " ")
    ).withColumn(
        "clean_text",
        # 4. Remove user mentions (@username)
        F.regexp_replace(F.col("clean_text"), r"@\w+", " ")
    ).withColumn(
        "clean_text",
        # 5. Remove non-alphanumeric characters except basic punctuation
        F.regexp_replace(F.col("clean_text"), r"[^a-zA-Z0-9\s.,!?'-]", " ")
    ).withColumn(
        "clean_text",
        # 6. Normalize whitespace (collapse multiple spaces into single space)
        F.trim(F.regexp_replace(F.col("clean_text"), r"\s+", " "))
    ).withColumn(
        "clean_text",
        # 7. Lowercase
        F.lower(F.col("clean_text"))
    )

    # Step 3: Remove records where cleaned text became empty
    cleaned_df = cleaned_df.filter(F.length(F.col("clean_text")) > 2)

    # Step 4: Map sentiment to numeric labels (0: negative, 1: neutral, 2: positive)
    sentiment_indexer = (
        F.when(F.lower(F.col("airline_sentiment")) == "negative", 0)
        .when(F.lower(F.col("airline_sentiment")) == "neutral", 1)
        .when(F.lower(F.col("airline_sentiment")) == "positive", 2)
        .otherwise(-1)
    )
    cleaned_df = cleaned_df.withColumn("label", sentiment_indexer)
    cleaned_df = cleaned_df.filter(F.col("label") >= 0)

    # Step 5: Fill missing negative reasons & handle confidence defaults
    cleaned_df = cleaned_df.withColumn(
        "negativereason_clean",
        F.when(F.col("negativereason").isNull(), "Not Specified / Non-Negative")
        .otherwise(F.col("negativereason"))
    ).withColumn(
        "airline_clean",
        F.when(F.col("airline").isNull(), "Unknown").otherwise(F.col("airline"))
    )

    # Step 6: Add review metadata
    cleaned_df = cleaned_df.withColumn(
        "char_count", F.length(F.col("clean_text"))
    ).withColumn(
        "word_count", F.size(F.split(F.col("clean_text"), r"\s+"))
    )

    # Step 7: Select and order relevant columns for downstream ML & routing
    final_cols = [
        "tweet_id",
        "airline_clean",
        "clean_text",
        "text",
        "airline_sentiment",
        "label",
        "airline_sentiment_confidence",
        "negativereason_clean",
        "char_count",
        "word_count",
        "tweet_created"
    ]
    
    # Check which of the target columns exist in the DataFrame
    selected_cols = [c for c in final_cols if c in cleaned_df.columns or c in ["label", "airline_clean", "clean_text", "negativereason_clean", "char_count", "word_count"]]
    
    return cleaned_df.select(selected_cols)


def run_etl_pipeline(
    raw_path: str = "data/raw/Tweets.csv",
    processed_dir: str = "data/processed/airline_reviews.parquet"
) -> DataFrame:
    """
    Executes the end-to-end PySpark ETL pipeline and saves the result as Parquet.
    """
    start_time = time.time()
    spark = get_spark_session("AirRouteAI-PySparkETL")

    logger.info(f"--- Starting ETL: {raw_path} ---> {processed_dir} ---")
    raw_df = load_raw_dataset(spark, raw_path)
    raw_count = raw_df.count()

    cleaned_df = clean_airline_data(raw_df)
    
    # Cache dataframe for fast multi-aggregation & write
    cleaned_df.cache()
    cleaned_count = cleaned_df.count()

    logger.info(f"Saving cleaned dataset to Parquet: {processed_dir}")
    os.makedirs(os.path.dirname(processed_dir), exist_ok=True)
    
    # Save as Columnar Parquet with Snappy compression (Spark default)
    cleaned_df.write.mode("overwrite").parquet(processed_dir)

    elapsed_time = time.time() - start_time
    logger.info(f"ETL completed in {elapsed_time:.2f} seconds!")
    logger.info(f"Raw rows: {raw_count} | Cleaned rows: {cleaned_count} (Retained: {cleaned_count/raw_count*100:.2f}%)")

    # Display dataset statistics
    print("\n" + "=" * 60)
    print("AIRROUTE AI - PYSPARK ETL SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Raw Records:        {raw_count:,}")
    print(f"Total Cleaned Records:    {cleaned_count:,}")
    print(f"Execution Time:           {elapsed_time:.2f} s")
    print(f"Storage Format:           Apache Parquet (Columnar)")
    print(f"Target Output Path:       {processed_dir}")
    print("-" * 60)
    
    print("\nSentiment Distribution:")
    cleaned_df.groupBy("airline_sentiment", "label").count().orderBy("label").show()

    print("\nAirline Review Distribution:")
    cleaned_df.groupBy("airline_clean").count().orderBy(F.col("count").desc()).show()

    print("\nTop 5 Complaint Categories:")
    cleaned_df.filter(F.col("airline_sentiment") == "negative") \
        .groupBy("negativereason_clean").count().orderBy(F.col("count").desc()).show(5)

    print("\nSample Cleaned Transformation (Original vs Clean):")
    cleaned_df.select("text", "clean_text", "airline_sentiment").show(3, truncate=False)
    print("=" * 60 + "\n")

    return cleaned_df


if __name__ == "__main__":
    run_etl_pipeline()
