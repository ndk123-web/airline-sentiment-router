"""
AirRoute AI - Data Ingestion Module
Handles loading, environment setup, and basic validation of raw airline review datasets.
"""

import os
import sys
import logging
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_hadoop_windows():
    """
    Automatically detects and sets HADOOP_HOME for Windows execution.
    This resolves the common Windows 'winutils.exe / HADOOP_HOME unset' issue.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    hadoop_dir = project_root / "hadoop"
    if hadoop_dir.exists() and (hadoop_dir / "bin" / "winutils.exe").exists():
        hadoop_path = str(hadoop_dir)
        os.environ["HADOOP_HOME"] = hadoop_path
        bin_path = str(hadoop_dir / "bin")
        if bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"Configured Windows HADOOP_HOME -> {hadoop_path}")


def get_spark_session(app_name: str = "AirRouteAI-DataIngestion") -> SparkSession:
    """
    Initializes and returns a configured SparkSession for local processing.
    """
    setup_hadoop_windows()
    logger.info("Initializing SparkSession...")
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse")
        .getOrCreate()
    )
    # Set log level to WARN to reduce excessive Spark CLI logs
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_raw_dataset(
    spark: SparkSession,
    file_path: str = "data/raw/Tweets.csv"
) -> DataFrame:
    """
    Loads raw CSV dataset into a PySpark DataFrame with schema inference.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {file_path}")

    logger.info(f"Loading raw dataset from {file_path} using PySpark...")
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .load(file_path)
    )
    logger.info(f"Loaded dataset with {df.count()} rows and {len(df.columns)} columns.")
    return df


if __name__ == "__main__":
    spark_session = get_spark_session()
    raw_df = load_raw_dataset(spark_session)
    raw_df.printSchema()
    raw_df.show(5, truncate=False)
    spark_session.stop()
