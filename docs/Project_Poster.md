# AIRROUTE AI

## Scalable Airline Passenger Sentiment Analysis & Intelligent Review Routing System

---

| | |
| :--- | :--- |
| **Subject Name** | Scalable Machine Learning and Big Data Analytics (BDA) |
| **Faculty Name** | _________________________ |

### Group Members

| Sr. No. | Name                | Roll Number    | Division |
| :-----: | :------------------ | :------------- | :------- |
| 1       | Navnath Kadam       | 23102B0061     | CMPN-B   |
| 2       | Prathmesh Nitnaware | 23102B0060     | CMPN-B   |
| 3       | Sahil Khilari       | 23102B0061     | CMPN-B   |

---

## Problem Statement

Airlines receive massive volumes of passenger reviews daily across digital channels. The system addresses three challenges:

- **Manual Triage Delay:** Human agents manually read and route complaints, causing delayed responses for critical operational failures (flight cancellations, lost luggage).
- **Compound Feedback Blindspot:** Reviews often mix multiple issues and sentiments in a single sentence (*"Flight delayed 4 hrs and lost my bag, but crew was polite!"*). Traditional classifiers miss individual issues.
- **Big Data Scale:** Enterprise-scale feedback streams exceed single-node processing capacity, requiring Apache Spark distributed computing.

---

## Objectives

1. Build a **PySpark Distributed ETL** pipeline with Parquet compression.
2. Train and benchmark **Scikit-Learn baseline** vs. **Spark MLlib distributed** sentiment classifiers.
3. Implement **Discourse Clause Segmentation** to decompose compound reviews into atomic sub-clauses.
4. Deploy **Meta NLI Zero-Shot Classification** for automatic department routing with confidence scoring.
5. Build a **Full-Stack System** with FastAPI REST API, SQLite ticket database, and Streamlit analytics dashboard.

---

## Methodology / Workflow Diagram

```
[Raw Passenger Reviews (CSV)]
        │
        ▼
[PySpark Distributed ETL Engine]
        │ Text Cleaning, Null Removal, Schema Enforcement
        ▼
[Apache Parquet Storage (Snappy Compressed)]
        │
        ├─────────────────────────────┐
        ▼                             ▼
[Scikit-Learn Baseline]       [Spark MLlib Pipeline]
 TF-IDF → Logistic Reg.       Tokenizer → HashingTF → IDF → LogReg
 Accuracy: 77.08%              Accuracy: 72.55%
        │
        ▼
[Discourse Clause Segmentation]
 Splits compound reviews into individual clauses
        │
        ├── Clause 1 → TF-IDF → Sentiment ML → Meta NLI → Department
        ├── Clause 2 → TF-IDF → Sentiment ML → Meta NLI → Department
        └── Clause N → TF-IDF → Sentiment ML → Meta NLI → Department
        │
        ▼
[Dynamic Priority Engine + SQLite Ticket DB]
        │
        ▼
[FastAPI REST API] ←→ [Streamlit Dashboard]
```

---

## Technologies and Tools Used

| Category               | Technology                                              |
| :--------------------- | :------------------------------------------------------ |
| Big Data               | Apache Spark 3.5.3, PySpark, Apache Parquet             |
| Machine Learning       | Scikit-Learn, Spark MLlib, TF-IDF Vectorization         |
| Zero-Shot NLP          | Hugging Face Transformers (DistilBART-MNLI)             |
| Backend API            | FastAPI, Uvicorn                                        |
| Frontend               | Streamlit, Plotly                                       |
| Database               | SQLite, SQLAlchemy ORM                                  |
| Testing                | Pytest                                                  |
| Version Control        | Git & GitHub                                            |

---

## Results

### ETL Performance
- Raw CSV: **3.42 MB** → Processed Parquet: **2.07 MB** (**39.4% compression**)
- 14,640 records processed with 100% retention

### Model Comparison

| Metric              | Scikit-Learn      | Spark MLlib       |
| :------------------ | :---------------- | :---------------- |
| **Accuracy**        | **77.08%**        | **72.55%**        |
| **Weighted F1**     | **0.7531**        | **0.7189**        |
| **Pipeline Time**   | **0.74 s**        | **7.60 s**        |
| **Target Scale**    | < 2 GB (RAM)      | 100 GB+ (Cluster) |

### Scalability Benchmarks

| Data Slice | Scikit-Learn Accuracy | Spark MLlib Accuracy |
| :--------- | :-------------------- | :------------------- |
| 10%        | 69.28%                | 70.30%               |
| 25%        | 74.59%                | 68.29%               |
| 50%        | 75.75%                | 67.25%               |
| 100%       | 79.06%                | 72.55%               |

### Intelligent Routing
- Compound reviews successfully decomposed into per-clause analysis
- Meta NLI Zero-Shot achieves **82-95% confidence** in department classification
- Automated ticket generation with priority scoring (HIGH/MEDIUM/LOW)

---

## Conclusion

AirRoute AI demonstrates an end-to-end scalable Big Data and NLP system for airline passenger sentiment analysis. The system successfully:
- Compresses airline review data via PySpark ETL with Parquet storage (39.4% reduction)
- Achieves 77.08% sentiment classification accuracy with dual ML paradigms
- Decomposes compound reviews into clause-level analysis for multi-issue resolution
- Routes feedback to correct airline departments via Zero-Shot NLI with high confidence
- Automates ticket lifecycle management with a production-ready FastAPI + Streamlit stack

---

## Signatures

| | |
| :------------------------------------------- | :------------------------------------------- |
| | |
| | |
| | |
| **Faculty Signature** | **External Examiner's Signature** |
| | |
| **Name:** _________________________ | **Name:** _________________________ |
| **Date:** _________________________ | **Date:** _________________________ |

---
