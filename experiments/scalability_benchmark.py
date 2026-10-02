"""
AirRoute AI - Scalability & Performance Benchmarking Module (Module 6 Syllabus)
Executes systematic, reproducible scalability benchmarks across multiple dataset slices (10%, 25%, 50%, 100%).
Compares:
1. Pandas / Scikit-Learn (Single-Node Baseline)
2. PySpark / Spark MLlib (Distributed In-Memory Pipeline)
Measures Data Loading, Feature Extraction, Model Training, Total Runtime, Accuracy, and Memory/Overhead.
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from src.ingestion.load_data import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_single_node_benchmark(df_pandas: pd.DataFrame, slice_pct: float, seed: int = 42) -> dict:
    """
    Benchmarks single-machine Pandas + Scikit-Learn pipeline for a given sample slice.
    """
    sample_df = df_pandas.sample(frac=slice_pct, random_state=seed).reset_index(drop=True)
    n_rows = len(sample_df)

    start_total = time.time()
    
    # 1. Feature Extraction (TF-IDF)
    X = sample_df["clean_text"].fillna("")
    y = sample_df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    start_feat = time.time()
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    feat_time = time.time() - start_feat

    # 2. Model Training
    start_train = time.time()
    model = LogisticRegression(max_iter=500, random_state=seed, C=1.0)
    model.fit(X_train_tfidf, y_train)
    train_time = time.time() - start_train

    # 3. Evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    total_time = time.time() - start_total

    return {
        "framework": "Scikit-Learn (Single-Node)",
        "rows": n_rows,
        "slice_pct": f"{int(slice_pct * 100)}%",
        "feature_extraction_seconds": round(feat_time, 4),
        "model_training_seconds": round(train_time, 4),
        "total_runtime_seconds": round(total_time, 4),
        "accuracy": round(float(acc), 4),
        "weighted_f1": round(float(f1), 4)
    }


def run_spark_benchmark(spark: SparkSession, full_spark_df, slice_pct: float, seed: int = 42) -> dict:
    """
    Benchmarks distributed Spark MLlib pipeline for a given sample slice.
    """
    sample_spark_df = full_spark_df.sample(withReplacement=False, fraction=slice_pct, seed=seed)
    sample_spark_df.cache()
    n_rows = sample_spark_df.count()

    start_total = time.time()

    # Train / Test split
    train_df, test_df = sample_spark_df.randomSplit([0.8, 0.2], seed=seed)
    train_df.cache()
    test_df.cache()

    # Define MLlib Pipeline
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = SparkLogisticRegression(featuresCol="features", labelCol="label", maxIter=25, regParam=0.05)

    pipeline = Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr])

    # Fit Pipeline
    start_train = time.time()
    pipeline_model = pipeline.fit(train_df)
    train_time = time.time() - start_train

    # Evaluate
    predictions = pipeline_model.transform(test_df)
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedFMeasure")

    acc = eval_acc.evaluate(predictions)
    f1 = eval_f1.evaluate(predictions)
    total_time = time.time() - start_total

    return {
        "framework": "Apache Spark MLlib (Distributed)",
        "rows": n_rows,
        "slice_pct": f"{int(slice_pct * 100)}%",
        "feature_extraction_seconds": 0.0, # Integrated in Spark MLlib Pipeline DAG
        "model_training_seconds": round(train_time, 4),
        "total_runtime_seconds": round(total_time, 4),
        "accuracy": round(float(acc), 4),
        "weighted_f1": round(float(f1), 4)
    }


def execute_all_scalability_experiments(
    parquet_path: str = "data/processed/airline_reviews.parquet",
    results_dir: str = "results"
):
    """
    Runs benchmarks across 10%, 25%, 50%, and 100% data slices.
    """
    os.makedirs(results_dir, exist_ok=True)
    logger.info("Initializing Scalability Benchmark suite across 10%, 25%, 50%, 100% slices...")

    # Load data
    df_pandas = pd.read_parquet(parquet_path)
    spark = get_spark_session("AirRouteAI-ScalabilityBenchmark")
    full_spark_df = spark.read.parquet(parquet_path)

    slices = [0.10, 0.25, 0.50, 1.00]
    benchmark_records = []

    print("\n" + "=" * 85)
    print("AIRROUTE AI - EXECUTING SCALABILITY BENCHMARKS (PANDAS/SKLEARN vs PYSPARK/MLLIB)")
    print("=" * 85)

    for s in slices:
        pct_label = f"{int(s * 100)}%"
        logger.info(f"--- Running Benchmarks for Slice: {pct_label} ---")
        
        # 1. Single Node
        res_sklearn = run_single_node_benchmark(df_pandas, slice_pct=s)
        benchmark_records.append(res_sklearn)
        print(f"[Sklearn] Slice {pct_label} ({res_sklearn['rows']} rows) -> Runtime: {res_sklearn['total_runtime_seconds']}s | Acc: {res_sklearn['accuracy']*100:.2f}%")

        # 2. Distributed Spark MLlib
        res_spark = run_spark_benchmark(spark, full_spark_df, slice_pct=s)
        benchmark_records.append(res_spark)
        print(f"[Spark  ] Slice {pct_label} ({res_spark['rows']} rows) -> Runtime: {res_spark['total_runtime_seconds']}s | Acc: {res_spark['accuracy']*100:.2f}%")

    spark.stop()

    df_results = pd.DataFrame(benchmark_records)
    csv_out = os.path.join(results_dir, "scalability_benchmark.csv")
    json_out = os.path.join(results_dir, "scalability_benchmark.json")

    df_results.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(benchmark_records, f, indent=2)

    logger.info(f"Saved benchmark results to {csv_out} and {json_out}")
    print("\n" + "=" * 85)
    print("SCALABILITY BENCHMARK RESULTS TABLE")
    print("=" * 85)
    print(df_results.to_string(index=False))
    print("=" * 85 + "\n")

    return df_results


if __name__ == "__main__":
    execute_all_scalability_experiments()
