# AirRoute AI: Scalable Airline Passenger Sentiment Analysis & Intelligent Review Routing System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.3-orange.svg)](https://spark.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Pytest](https://img.shields.io/badge/Pytest-Passing-brightgreen.svg)](https://docs.pytest.org/)

---

## 1. Project Overview

**AirRoute AI** is an academic-grade, end-to-end Big Data Analytics and Scalable Machine Learning project developed for the **Scalable ML and Big Data Analytics (BDA)** course.

The system processes real-world airline passenger feedback, performs distributed data cleaning and feature engineering using **Apache PySpark**, trains both **Scikit-learn baseline** and **Spark MLlib distributed** sentiment classification models, segments compound customer feedback into **multi-issue sub-clauses**, intelligently routes complaints to corresponding **airline operational departments**, and generates actionable **support tickets** managed through a **FastAPI backend** and an interactive **Streamlit dashboard**.

---

## 2. Dataset Information

The project utilizes the benchmark **Twitter US Airline Sentiment Dataset** (sourced from CrowdFlower / Kaggle):
- **Raw File Location:** `data/raw/Tweets.csv`
- **Total Records:** 14,640 valid passenger tweets/reviews
- **Primary Columns:**
  - `tweet_id`: Unique identifier for each tweet/review
  - `airline`: Target airline (*United, American, Delta, Southwest, US Airways, Virgin America*)
  - `airline_sentiment`: Ground-truth sentiment label (`positive`, `neutral`, `negative`)
  - `airline_sentiment_confidence`: Confidence score of the label annotation
  - `negativereason`: Operational complaint category (*Late Flight, Lost Luggage, Customer Service Issue, Flight Booking Problems, Cancelled Flight, etc.*)
  - `text`: Passenger review text

---

## 3. Big Data Architecture & PySpark Internals

```
┌──────────────────────────────────────────────────────────┐
│                   Python / PySpark API                   │
│   User Code: df = spark.read.csv("data/raw/Tweets.csv")  │
└────────────────────────────┬─────────────────────────────┘
                             │
                      (Py4J Gateway) ◄── IPC / Local Sockets
                             │
┌────────────────────────────▼─────────────────────────────┐
│                 JVM (Java Virtual Machine)               │
│   Spark Driver (Catalyst Optimizer & Tungsten Engine)    │
│   - Distributed In-Memory Processing                     │
│   - Partition-Level Text Tokenization & TF-IDF           │
│   - MLlib Distributed Logistic Regression / Naive Bayes  │
│   - Columnar Parquet File Serialization                  │
└──────────────────────────────────────────────────────────┘
```

### Why PySpark + Java (Py4J)?
- **Apache Spark Core** is built in **Scala/Java** to run on high-performance JVMs with multithreading and cluster distribution capabilities.
- **PySpark** communicates with the JVM using **Py4J** over local sockets. When Python calls Spark DataFrame APIs, the underlying transformation DAG (Directed Acyclic Graph) is optimized by the **Catalyst Optimizer** and executed inside the JVM.
- **Academic Insight**: On small datasets (<50,000 rows), single-machine Pandas runs quickly without socket overhead. As dataset size scales into gigabytes/terabytes, PySpark's partitioned in-memory computation avoids out-of-memory (OOM) crashes and provides linear horizontal scaling.

---

## 4. End-to-End System Workflow

```
[Raw Tweets CSV (3.42 MB)] 
       │
       ▼
[PySpark ETL & Cleaning Engine]
  - Regex Noise / URL / @Handle Removal
  - HTML Entity Normalization (&amp; -> and)
  - Null Filtering & Schema Validation
  - Sentiment Indexing (0: Negative, 1: Neutral, 2: Positive)
       │
       ▼
[Apache Parquet Storage (2.07 MB - 39.4% Compression)] (data/processed/airline_reviews.parquet)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Baseline Single-Node ML]               [Distributed Spark MLlib]
(Pandas + TF-IDF + LogReg/NB)            (Spark Tokenizer + IDF + MLlib)
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
           [Multi-Issue Segmentation Engine]
       (Extracts Sub-Clauses & Complaint Categories)
                            │
                            ▼
          [Rule-Based Department Router]
  (Routes to Flight Ops, Baggage, Ticketing, etc.)
                            │
                            ▼
           [SQLite / SQLAlchemy Ticket DB]
       (Open -> In Progress -> Resolved -> Closed)
                            │
                            ▼
              [FastAPI REST API Service]
            (/predict, /tickets, /analytics)
                            │
                            ▼
             [Streamlit Interactive Dashboard]
 (Overview | Realtime Prediction | Analytics | Tickets | Scalability)
```

---

## 5. PySpark ETL Pipeline & Real Dataset Statistics

### Transformation Logic (`src/processing/pyspark_etl.py`):
1. **URL & Handle Stripping**: Removes links (`https?://\S+`) and airline handles (`@united`, `@AmericanAir`).
2. **Entity Decoding**: Replaces HTML entities (e.g., `&amp;` $\rightarrow$ `and`).
3. **Special Character Cleaning**: Strips non-alphanumeric noise while preserving basic punctuation needed for clause segmentation.
4. **Label Encoding**: Maps sentiment to numerical labels:
   - `0`: **Negative** (9,178 reviews / 62.7%)
   - `1`: **Neutral** (3,099 reviews / 21.2%)
   - `2`: **Positive** (2,363 reviews / 16.1%)
5. **Columnar Persistence**: Cleaned dataset is saved as snappy-compressed Apache Parquet format.

### Actual ETL Execution Summary:
- **Total Input Records:** 14,640
- **Total Cleaned Records Retained:** 14,640 (100.0%)
- **Raw CSV Size:** 3.42 MB
- **Processed Parquet Size:** 2.07 MB (39.4% storage compression)
- **ETL Execution Runtime:** ~8.63 seconds (including Spark JVM boot and DataFrame optimizations)

#### Airline Distribution in Processed Data:
| Airline | Cleaned Review Count | Proportion |
| :--- | :--- | :--- |
| **United** | 3,822 | 26.1% |
| **US Airways** | 2,913 | 19.9% |
| **American** | 2,759 | 18.8% |
| **Southwest** | 2,420 | 16.5% |
| **Delta** | 2,222 | 15.2% |
| **Virgin America** | 504 | 3.4% |

#### Top Complaint Categories:
1. **Customer Service Issue:** 2,910 complaints
2. **Late Flight:** 1,665 complaints
3. **Can't Tell / Unspecified:** 1,190 complaints
4. **Cancelled Flight:** 847 complaints
5. **Lost Luggage:** 724 complaints

---

## 6. Machine Learning Models: Baseline vs. Distributed MLlib

AirRoute AI implements a dual-paradigm Machine Learning architecture to compare traditional single-machine workflows against distributed pipelines:

### Model Performance Comparison:

| Metric / Attribute | Scikit-Learn Baseline (Single-Machine) | Apache Spark MLlib (Distributed Pipeline) |
| :--- | :--- | :--- |
| **Framework & Engine** | Scikit-Learn (v1.7.2) / In-Memory RAM | Apache Spark MLlib (v3.5.3) / JVM DAG |
| **Feature Extraction** | Scikit-learn `TfidfVectorizer` (N-gram 1-2, 5k vocab) | Spark `Tokenizer` $\rightarrow$ `HashingTF` (5k bins) $\rightarrow$ `IDF` |
| **Classifier** | Multiclass Logistic Regression (L-BFGS) | Distributed Logistic Regression (L-BFGS / Bound Optimization) |
| **Accuracy** | **77.08%** | **72.55%** |
| **Weighted Precision** | **0.7608** | **0.7176** |
| **Weighted Recall** | **0.7708** | **0.7255** |
| **Weighted F1-Score** | **0.7531** | **0.7189** |
| **Data Load Time** | 0.1556 s | 3.1127 s |
| **Training Time** | 0.2372 s | 4.4826 s |
| **Inference Time** | 0.0006 s | 0.5535 s |
| **Total Pipeline Time** | **0.7356 s** | **7.5953 s** |
| **Target Scale** | Datasets fitting in single-node RAM (< 2 GB) | Massive datasets (100 GB to Terabytes across Clusters) |

### Key Academic Insights (For Viva & Presentation):
1. **Why Baseline Scikit-Learn is faster on 14.6k rows**:
   - Single-machine Python processes 14.6k rows entirely in L3 cache/RAM in under 1 second.
   - Apache Spark requires **JVM initialization, socket communication (Py4J), Spark context graph compilation, and partition serialization overhead**, which takes ~7-8 seconds regardless of dataset size.
2. **Why Scikit-Learn TF-IDF has slightly higher accuracy (77% vs 72.5%)**:
   - `TfidfVectorizer` builds an exact in-memory vocabulary dictionary and extracts both unigrams + bigrams.
   - Spark `HashingTF` uses Murmur3 hash trick to project arbitrary text onto fixed 5,000 buckets without storing a dictionary across nodes, which introduces slight hash collisions on small datasets in exchange for unlimited distributed scalability.

---

## 7. Multi-Issue Extraction & Intelligent Department Routing

Passenger feedback frequently contains compound sentiments addressing multiple functional areas of an airline simultaneously:
> *"@united My flight was delayed by four hours and you lost my baggage in Chicago, but the cabin crew was very polite and helpful!"*

### How the Router Dissects Compound Reviews (`src/routing/router.py`):
1. **Sentence & Clause Tokenization**: Identifies contrastive conjunctions (*"but", "however", "although", "while", "and"*) and punctuation boundaries.
2. **Dual-Layer Sentiment Scoring**: Evaluates global review sentiment via the ML model and local clause polarity via sentiment lexicon scoring.
3. **Department Mapping & Keyword Extraction**:
   - **Flight Operations**: Delays, cancellations, diversions, missed connections, tarmac wait times.
   - **Baggage Services**: Lost luggage, damaged bags, carousel delays, missing items.
   - **Reservations & Ticketing**: Booking errors, double charges, refund requests, seat changes, overbooking.
   - **Customer Experience**: Flight attendant behavior, gate agent service, responsiveness, professionalism.
   - **In-flight Services**: Meals, snacks, seat comfort, legroom, in-flight WiFi, screens/entertainment.
   - **Digital Support**: Website crashes, mobile app bugs, login errors, kiosk failures.
4. **Transparent Priority Engine**:
   - **`URGENT`**: Stranded passengers, cancelled flights without rebooking, safety concerns.
   - **`HIGH`**: Lost luggage, damaged baggage, double charges, missed connections.
   - **`MEDIUM`**: Seat discomfort, meal quality, minor website glitch.
   - **`LOW`**: Positive compliments, general non-actionable queries.

### Support Ticket Management (`src/tickets/ticket_manager.py`):
- Persistent **SQLite** storage (`data/tickets.db`) using SQLAlchemy ORM.
- **Ticket Lifecycle State Machine**:
  $$\text{Open} \longrightarrow \text{In Progress} \longrightarrow \text{Resolved} \longrightarrow \text{Closed}$$
- Support for operator status updates, notes appending, and multi-parameter filtering (by status, department, priority, airline).

---

## 8. Scalability & Performance Benchmarks (Module 6 Syllabus)

To empirically evaluate single-node versus distributed execution, systematic benchmarks were conducted across **10%, 25%, 50%, and 100% dataset slices** (`experiments/scalability_benchmark.py`):

| Slice | Rows | Framework / Engine | Feature Extraction | Model Training | Total Runtime | Accuracy | Weighted F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10%** | 1,464 | Scikit-Learn (Single-Node) | 0.0164 s | 0.0242 s | **0.0456 s** | 69.28% | 0.6275 |
| **10%** | 1,558 | Apache Spark MLlib | Included in DAG | 3.6877 s | **4.9878 s** | 70.30% | 0.6859 |
| **25%** | 3,660 | Scikit-Learn (Single-Node) | 0.0495 s | 0.1104 s | **0.1657 s** | 74.59% | 0.7063 |
| **25%** | 3,761 | Apache Spark MLlib | Included in DAG | 1.1690 s | **1.5522 s** | 68.29% | 0.6691 |
| **50%** | 7,320 | Scikit-Learn (Single-Node) | 0.0742 s | 0.1482 s | **0.2307 s** | 75.75% | 0.7307 |
| **50%** | 7,455 | Apache Spark MLlib | Included in DAG | 1.1996 s | **1.5136 s** | 67.25% | 0.6586 |
| **100%** | 14,640 | Scikit-Learn (Single-Node) | 0.1504 s | 0.2747 s | **0.4333 s** | 79.06% | 0.7757 |
| **100%** | 14,640 | Apache Spark MLlib | Included in DAG | 1.3675 s | **1.7289 s** | 72.55% | 0.7189 |

---

## 9. FastAPI REST API & Streamlit Dashboard

### FastAPI Backend Endpoints (`backend/main.py`):
- `POST /api/predict`: Real-time sentiment prediction and multi-issue clause breakdown.
- `GET /api/tickets`: Filterable list of support tickets (by status, department, priority, airline).
- `PATCH /api/tickets/{ticket_id}`: Update ticket lifecycle (`Open` $\rightarrow$ `In Progress` $\rightarrow$ `Resolved` $\rightarrow$ `Closed`) and attach resolution notes.
- `GET /api/analytics/tickets`: Summary metrics on open vs resolved tickets and departmental queues.
- `GET /api/models/evaluation`: Comparative evaluation summary between Scikit-learn and Spark MLlib.

### Streamlit Web Dashboard Modules (`frontend/app.py`):
1. **Executive Overview & Analytics**: KPI metric cards, Plotly sentiment donut charts, airline review volume breakdown, top complaint bar charts, and dataset explorer.
2. **Live Multi-Issue Review Router**: Real-time review analyzer with instant probability distribution and sub-clause ticket generation.
3. **Support Ticket Operations Center**: Complete ticket management board with status update modals, priority pills, and department queue filters.
4. **Distributed ML & Scalability Benchmarks**: Side-by-side metric comparisons, runtime curves across data sizes, and CSV vs Parquet storage efficiency visualizers.

---

## 10. Course Syllabus Alignment Matrix

| Module | Syllabus Topic | AirRoute AI Implementation | Output / Evidence |
| :--- | :--- | :--- | :--- |
| **Module 1** | Big Data Foundations & Spark ETL | `src/processing/pyspark_etl.py` | PySpark ETL cleaning, null handling, CSV vs Parquet benchmarks |
| **Module 2** | Distributed Machine Learning | `src/models/train_spark.py` | Spark MLlib Pipeline (Tokenizer $\rightarrow$ HashingTF $\rightarrow$ IDF $\rightarrow$ LogReg) |
| **Module 3** | NLP & Text Feature Engineering | `src/features/` & `src/routing/` | TF-IDF matrices, discourse clause segmentation, sentiment scoring |
| **Module 4** | Model Serving & MLOps | `backend/main.py` & `src/tickets/` | FastAPI REST endpoints, ticket state machine, Swagger docs (`/docs`) |
| **Module 5** | Advanced Topics (Heuristics) | `src/routing/router.py` | Priority scoring engine (`URGENT`, `HIGH`, `MEDIUM`, `LOW`) |
| **Module 6** | Performance & Scalability Case Study| `experiments/scalability_benchmark.py` | Benchmark curves comparing runtime across 10%, 25%, 50%, 100% slices |

---

## 11. Project Directory Structure

```text
Passenger Review/
├── data/
│   ├── raw/                       # Original raw dataset (Tweets.csv)
│   ├── processed/                 # Parquet-formatted cleaned datasets
│   └── tickets.db                 # SQLite database for support tickets
├── src/
│   ├── ingestion/                 # Dataset loader and Windows HADOOP configuration
│   ├── processing/                # PySpark ETL and text normalization
│   ├── models/                    # Baseline (Sklearn) and Spark MLlib training scripts
│   ├── routing/                   # Multi-issue extraction & department router
│   └── tickets/                   # Ticket lifecycle, DB schema, and CRUD
├── backend/                       # FastAPI REST API application
├── frontend/                      # Streamlit interactive web dashboard
├── experiments/                   # Scalability benchmarks (10%, 25%, 50%, 100%)
├── models/                        # Serialized model artifacts (joblib & MLlib)
├── results/                       # Scalability benchmark CSVs, JSONs, and metrics
├── tests/                         # Pytest test suite (100% passing)
├── hadoop/                        # Local Windows winutils.exe and hadoop.dll
├── requirements.txt               # Project Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Comprehensive project documentation
```

---

## 12. Verification & Execution Commands

### 1. Run Automated Test Suite:
```powershell
pytest tests/
```

### 2. Run PySpark ETL Pipeline:
```powershell
python -m src.processing.pyspark_etl
```

### 3. Train Baseline & Distributed MLlib Models:
```powershell
python -m src.models.train_baseline
python -m src.models.train_spark
python -m src.models.evaluate
```

### 4. Run Scalability Experiments:
```powershell
python -m experiments.scalability_benchmark
```

### 5. Launch FastAPI Backend:
```powershell
uvicorn backend.main:app --reload --port 8000
```
*Swagger UI docs available at:* `http://localhost:8000/docs`

### 6. Launch Streamlit Web Dashboard:
```powershell
streamlit run frontend/app.py
```
*Interactive dashboard available at:* `http://localhost:8501`

---

## 13. Academic License & Acknowledgments
- Dataset provided by **CrowdFlower** via **Kaggle** under Open Data License.
- Built for academic evaluation in **Scalable Machine Learning and Big Data Analytics**.
