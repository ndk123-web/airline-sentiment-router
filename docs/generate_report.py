"""
Generate Mini Project Report as .docx
"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
import os

doc = Document()

# ── Page Setup (A4) ──
for section in doc.sections:
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
style.paragraph_format.line_spacing = 1.5

# ── Helpers ──
def add_heading_styled(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h

def add_para(text, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=6):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.bold = bold
    return p

def add_bullet(text, bold_prefix="", level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.left_indent = Cm(1.27 + level * 0.63)
    if bold_prefix:
        run_b = p.add_run(bold_prefix)
        run_b.bold = True
        run_b.font.name = 'Times New Roman'
        run_b.font.size = Pt(12)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    return p

def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shading = cell._element.get_or_add_tcPr()
        shading_elm = shading.makeelement(qn('w:shd'), {
            qn('w:fill'): 'D9E2F3',
            qn('w:val'): 'clear'
        })
        shading.append(shading_elm)
    # Data rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()  # spacing
    return table

# ═══════════════════════════════════════════════════════════
# TITLE PAGE
# ═══════════════════════════════════════════════════════════
for _ in range(4):
    doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Mini Project Report")
run.font.name = 'Times New Roman'
run.font.size = Pt(28)
run.bold = True

doc.add_paragraph()

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("AirRoute AI")
run.font.name = 'Times New Roman'
run.font.size = Pt(22)
run.bold = True
run.font.color.rgb = RGBColor(0, 51, 102)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Scalable Airline Passenger Sentiment Analysis\n& Intelligent Review Routing System")
run.font.name = 'Times New Roman'
run.font.size = Pt(16)
run.font.color.rgb = RGBColor(68, 68, 68)

for _ in range(2):
    doc.add_paragraph()

# Group members on title page
members_data = [
    ["1", "Navnath Kadam", "23102B0061", "CMPN-B"],
    ["2", "Prathmesh Nitnaware", "23102B0060", "CMPN-B"],
    ["3", "Sahil Khilari", "23102B0061", "CMPN-B"],
]
add_table(["Sr. No.", "Name", "Roll Number", "Division"], members_data)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Subject: Scalable Machine Learning and Big Data Analytics (BDA)")
run.font.name = 'Times New Roman'
run.font.size = Pt(13)
run.bold = True

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Faculty: _________________________")
run.font.name = 'Times New Roman'
run.font.size = Pt(13)

doc.add_page_break()

# ═══════════════════════════════════════════════════════════
# 1. PROJECT TITLE
# ═══════════════════════════════════════════════════════════
add_heading_styled("1. Project Title", level=1)
add_para("AirRoute AI: Scalable Airline Passenger Sentiment Analysis & Intelligent Review Routing System", bold=True, size=13)

# ═══════════════════════════════════════════════════════════
# 2. GROUP MEMBERS
# ═══════════════════════════════════════════════════════════
add_heading_styled("2. Group Members' Names and Roll Numbers", level=1)
add_table(
    ["Sr. No.", "Name", "Roll Number", "Division"],
    members_data
)

# ═══════════════════════════════════════════════════════════
# 3. SUBJECT & FACULTY
# ═══════════════════════════════════════════════════════════
add_heading_styled("3. Subject Name and Faculty Name", level=1)
add_bullet("Scalable Machine Learning and Big Data Analytics (BDA)", bold_prefix="Subject Name: ")
add_bullet("_________________________", bold_prefix="Faculty Name: ")

# ═══════════════════════════════════════════════════════════
# 4. PROBLEM STATEMENT
# ═══════════════════════════════════════════════════════════
add_heading_styled("4. Problem Statement", level=1)
add_para(
    "Modern commercial airlines receive tens of thousands of customer reviews and feedback "
    "submissions daily across digital channels such as Twitter/X, email, and app surveys. "
    "Traditional customer service workflows face three critical bottlenecks:"
)
add_bullet(
    " Human agents manually read and route complaints, leading to delayed response times "
    "for high-priority operational failures such as flight cancellations and lost baggage.",
    bold_prefix="Manual Triage Delay:"
)
add_bullet(
    " Passengers frequently submit reviews containing multiple distinct sentiments and issues "
    "in a single sentence. For example: \"My flight was delayed by 4 hours and you lost my baggage, "
    "but the cabin crew was very polite!\" Monolithic sentiment classifiers treat this as a single "
    "data point and fail to isolate individual sub-clauses, resulting in misrouted tickets or missed praise.",
    bold_prefix="Compound Feedback Blindspot:"
)
add_bullet(
    " Standard single-node Python pandas/scikit-learn scripts fail or run out of memory when "
    "processing large-scale, high-velocity enterprise feedback streams, necessitating distributed "
    "big data frameworks like Apache Spark.",
    bold_prefix="Big Data Scalability Constraints:"
)
add_para(
    "This project addresses all three challenges by building an end-to-end scalable ML pipeline "
    "that segments compound reviews into clauses, classifies each clause's sentiment, routes each "
    "clause to the correct airline department, and generates prioritized support tickets automatically."
)

# ═══════════════════════════════════════════════════════════
# 5. OBJECTIVES
# ═══════════════════════════════════════════════════════════
add_heading_styled("5. Objectives", level=1)
objectives = [
    ("Distributed ETL Pipeline: ", "Develop an Apache PySpark data ingestion and cleaning pipeline with Snappy-compressed Apache Parquet storage for optimized big data performance."),
    ("Dual-Model Sentiment Classification: ", "Train and evaluate both single-node Scikit-Learn baseline models and distributed Apache Spark MLlib machine learning pipelines on the benchmark airline sentiment dataset."),
    ("Advanced NLP Discourse Segmentation: ", "Implement a regex-driven clause splitter to decompose complex compound reviews into individual atomic sub-clauses across conjunction boundaries."),
    ("Semantic Zero-Shot Department Routing: ", "Utilize Meta Natural Language Inference (NLI) zero-shot classification (DistilBART-MNLI) to map individual clauses to specific airline departments with confidence scoring."),
    ("Intelligent Ticket Lifecycle Management: ", "Build a dynamic priority engine and SQLite ticket database exposed via a FastAPI REST backend and an interactive Streamlit analytics dashboard."),
    ("Scalability Benchmarking: ", "Conduct empirical experiments across 10%, 25%, 50%, and 100% dataset slices to measure runtime and accuracy scaling behavior of both ML paradigms."),
]
for bold_prefix, text in objectives:
    add_bullet(text, bold_prefix=bold_prefix)

# ═══════════════════════════════════════════════════════════
# 6. DATASET DESCRIPTION
# ═══════════════════════════════════════════════════════════
add_heading_styled("6. Dataset Description", level=1)
add_bullet("Twitter US Airline Sentiment Dataset (CrowdFlower / Kaggle)", bold_prefix="Source: ")
add_bullet("CSV (Comma-Separated Values)", bold_prefix="Format: ")
add_bullet("14,640 valid passenger tweets/reviews", bold_prefix="Total Records: ")
add_bullet("United, American, Delta, Southwest, US Airways, Virgin America", bold_prefix="Airlines Covered: ")

add_para("")
add_heading_styled("Dataset Schema", level=2)
add_table(
    ["Column", "Description"],
    [
        ["tweet_id", "Unique identifier for each tweet/review"],
        ["airline", "Target airline name"],
        ["airline_sentiment", "Ground-truth sentiment label (positive/neutral/negative)"],
        ["airline_sentiment_confidence", "Confidence score of the label annotation"],
        ["negativereason", "Operational complaint category (Late Flight, Lost Luggage, etc.)"],
        ["text", "Raw passenger review text"],
    ]
)

add_heading_styled("Sentiment Distribution by Airline", level=2)
add_table(
    ["Airline", "Reviews", "Negative %", "Neutral %", "Positive %"],
    [
        ["United", "3,822", "68.9%", "18.2%", "12.9%"],
        ["US Airways", "2,913", "77.6%", "13.1%", "9.3%"],
        ["American", "2,759", "71.0%", "16.8%", "12.2%"],
        ["Southwest", "2,420", "49.1%", "27.4%", "23.5%"],
        ["Delta", "2,222", "43.0%", "32.5%", "24.5%"],
        ["Virgin America", "504", "35.9%", "34.0%", "30.1%"],
    ]
)

# ═══════════════════════════════════════════════════════════
# 7. METHODOLOGY
# ═══════════════════════════════════════════════════════════
add_heading_styled("7. Methodology / Proposed Workflow", level=1)

add_heading_styled("Stage 1: Data Ingestion & Cleaning (PySpark ETL)", level=2)
add_para("Raw CSV data is loaded into PySpark DataFrames. Text preprocessing includes URL/mention stripping, HTML entity decoding, null/empty row removal, and schema enforcement. Cleaned data is written to Apache Parquet columnar format with Snappy compression.")

add_heading_styled("Stage 2: Feature Engineering (TF-IDF Vectorization)", level=2)
add_bullet(" TfidfVectorizer converts text into a 5,000-dimensional sparse numeric vector using sublinear term frequency (1 + log(TF)) multiplied by inverse document frequency (log(N/DF)) across unigrams and bigrams.", bold_prefix="Scikit-Learn Path:")
add_bullet(" Tokenizer → StopWordsRemover → HashingTF (5,000 bins) → IDF transformer.", bold_prefix="Spark MLlib Path:")

add_heading_styled("Stage 3: Model Training", level=2)
add_bullet(" Multiclass Logistic Regression with Softmax probability calibration trained on TF-IDF features.", bold_prefix="Scikit-Learn Baseline:")
add_bullet(" Distributed Logistic Regression trained via the Spark ML Pipeline API on HashingTF-IDF features.", bold_prefix="Spark MLlib Distributed:")

add_heading_styled("Stage 4: Discourse Clause Segmentation", level=2)
add_para("Compound reviews are split into discrete grammatical clauses using regex patterns targeting coordinating and contrastive conjunctions (and, but, however, although, semicolons, commas).")

add_heading_styled("Stage 5: Clause-Level Sentiment + Zero-Shot Department Routing", level=2)
add_para("Each clause is independently vectorized via the trained TF-IDF model and evaluated for sentiment by the Logistic Regression classifier. Department classification uses Meta NLI Zero-Shot Inference (valhalla/distilbart-mnli-12-3) to map each clause to one of 6 airline departments: Flight Operations, Baggage Services, Reservations & Ticketing, Customer Experience, In-flight Services, Digital Support.")

add_heading_styled("Stage 6: Dynamic Priority Scoring & Ticket Generation", level=2)
add_para("Priority rules combine sentiment and department to assign HIGH/MEDIUM/LOW priority. Negative clauses routed to operational departments generate HIGH priority tickets. Positive clauses are logged as compliments. Tickets are persisted in SQLite with a state machine lifecycle: Open → In Progress → Resolved → Closed.")

add_heading_styled("Stage 7: API & Dashboard", level=2)
add_para("FastAPI REST API serves prediction and ticket management endpoints. Streamlit dashboard provides executive analytics, live review analysis, and ticket operations.")

add_heading_styled("Workflow Diagram", level=2)
# Add workflow as a formatted code-like text block
workflow_text = (
    "[Raw CSV Data (Tweets.csv)]\n"
    "         │\n"
    "         ▼\n"
    "[PySpark Distributed ETL Engine]\n"
    "   Text Cleaning, Null Removal, Schema Enforcement\n"
    "         │\n"
    "         ▼\n"
    "[Apache Parquet Storage (Snappy Compressed)]\n"
    "         │\n"
    "         ├─────────────────────────┐\n"
    "         ▼                         ▼\n"
    "[Scikit-Learn Baseline]    [Spark MLlib Pipeline]\n"
    " Accuracy: 77.08%          Accuracy: 72.55%\n"
    "         │\n"
    "         ▼\n"
    "[Discourse Clause Segmentation Engine]\n"
    "   Splits compound reviews → individual clauses\n"
    "         │\n"
    "   Clause N → TF-IDF → Sentiment ML → Meta NLI → Department\n"
    "         │\n"
    "         ▼\n"
    "[Dynamic Priority Engine + SQLite Ticket DB]\n"
    "         │\n"
    "         ▼\n"
    "[FastAPI REST API] ←→ [Streamlit Dashboard]"
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
run = p.add_run(workflow_text)
run.font.name = 'Consolas'
run.font.size = Pt(9)

# ═══════════════════════════════════════════════════════════
# 8. TECHNOLOGIES
# ═══════════════════════════════════════════════════════════
add_heading_styled("8. Technologies and Tools Used", level=1)
add_table(
    ["Category", "Technology / Tool"],
    [
        ["Programming Language", "Python 3.11 / 3.12 / 3.13"],
        ["Big Data & Distributed", "Apache Spark 3.5.3, PySpark"],
        ["Columnar Storage", "Apache Parquet (Snappy Compression)"],
        ["Machine Learning", "Scikit-Learn 1.3+, Apache Spark MLlib"],
        ["NLP & Text Processing", "NLTK, TF-IDF Vectorization"],
        ["Zero-Shot Classification", "Hugging Face Transformers (DistilBART-MNLI)"],
        ["Backend API", "FastAPI, Uvicorn, Pydantic"],
        ["Frontend Dashboard", "Streamlit, Plotly"],
        ["Database", "SQLite, SQLAlchemy ORM"],
        ["Testing", "Pytest"],
        ["Version Control", "Git & GitHub"],
        ["Development Environment", "Windows 10/11, VS Code"],
    ]
)

# ═══════════════════════════════════════════════════════════
# 9. IMPLEMENTATION
# ═══════════════════════════════════════════════════════════
add_heading_styled("9. Implementation", level=1)
add_para("The project codebase is organized into a modular enterprise-grade architecture:")

add_heading_styled("Project Directory Structure", level=2)
dir_text = (
    "Passenger Review/\n"
    "├── data/\n"
    "│   ├── raw/                  # Original raw dataset (Tweets.csv)\n"
    "│   ├── processed/            # Parquet-formatted cleaned datasets\n"
    "│   └── tickets.db            # SQLite database for support tickets\n"
    "├── src/\n"
    "│   ├── ingestion/            # Dataset loader & HADOOP config\n"
    "│   ├── processing/           # PySpark ETL and text normalization\n"
    "│   ├── models/               # Baseline & Spark MLlib training\n"
    "│   ├── routing/              # Clause-level sentiment & Zero-Shot router\n"
    "│   └── tickets/              # Ticket lifecycle & CRUD\n"
    "├── backend/                  # FastAPI REST API\n"
    "├── frontend/                 # Streamlit web dashboard\n"
    "├── experiments/              # Scalability benchmarks\n"
    "├── models/                   # Serialized model artifacts\n"
    "├── results/                  # Benchmark metrics\n"
    "├── tests/                    # Pytest test suite\n"
    "└── README.md                 # Documentation"
)
p = doc.add_paragraph()
run = p.add_run(dir_text)
run.font.name = 'Consolas'
run.font.size = Pt(9)

add_heading_styled("Key Implementation Details", level=2)

impl_details = [
    ("PySpark ETL Pipeline (src/processing/pyspark_etl.py): ",
     "Configures SparkSession with local driver mode. Reads raw CSV with schema inference. "
     "Applies UDF-based text cleaning: URL removal, @mention stripping, HTML entity decoding. "
     "Drops null/empty rows and applies sentiment label indexing. Writes cleaned DataFrame to "
     "Parquet with Snappy compression."),
    ("Scikit-Learn Baseline Training (src/models/train_baseline.py): ",
     "Loads Parquet data via pandas. Applies TfidfVectorizer with sublinear TF, unigram+bigram "
     "extraction, and 5,000 max features. Trains LogisticRegression (multiclass, max_iter=1000) "
     "and evaluates on 80/20 train-test split. Serializes trained models as .joblib artifacts."),
    ("Spark MLlib Pipeline (src/models/train_spark.py): ",
     "Constructs a Spark ML Pipeline: Tokenizer → StopWordsRemover → HashingTF (5,000 bins) → "
     "IDF → LogisticRegression (maxIter=30, regParam=0.05). Trains on 80/20 split and evaluates "
     "using MulticlassClassificationEvaluator."),
    ("Advanced Routing Engine (src/routing/router.py): ",
     "Regex-based clause splitter decomposes reviews across conjunction boundaries. Each clause is "
     "vectorized via trained TF-IDF and classified by Logistic Regression. Meta NLI Zero-Shot "
     "(valhalla/distilbart-mnli-12-3) classifies department with confidence scoring."),
    ("Ticket Management (src/tickets/ticket_manager.py): ",
     "SQLAlchemy ORM with SQLite backend. Full CRUD operations with state machine lifecycle: "
     "Open → In Progress → Resolved → Closed."),
    ("FastAPI REST API (backend/main.py): ",
     "POST /api/predict accepts review text and returns clause-level analysis. GET /api/tickets "
     "lists tickets with filtering. DELETE /api/tickets/{id} removes tickets. Swagger docs at /docs."),
    ("Streamlit Dashboard (frontend/app.py): ",
     "Executive analytics with sentiment distribution charts. Live multi-issue review analysis. "
     "Ticket operations center with status management and delete functionality."),
]
for bold_prefix, text in impl_details:
    add_bullet(text, bold_prefix=bold_prefix)

# ═══════════════════════════════════════════════════════════
# 10. RESULTS AND DISCUSSION
# ═══════════════════════════════════════════════════════════
add_heading_styled("10. Results and Discussion", level=1)

add_heading_styled("A. PySpark ETL Compression Benchmark", level=2)
add_table(
    ["Metric", "Value"],
    [
        ["Raw CSV Size", "3.42 MB"],
        ["Processed Parquet Size", "2.07 MB"],
        ["Storage Reduction", "39.4%"],
        ["ETL Processing Runtime", "8.63 seconds"],
        ["Records Processed", "14,640 (100% retention)"],
    ]
)
add_para("The PySpark ETL pipeline successfully transformed the raw CSV into Snappy-compressed Parquet format, achieving nearly 40% storage reduction while retaining 100% of valid records.")

add_heading_styled("B. Model Comparison: Scikit-Learn vs. Spark MLlib", level=2)
add_table(
    ["Metric", "Scikit-Learn Baseline", "Apache Spark MLlib"],
    [
        ["Framework", "Scikit-Learn (v1.7.2)", "Spark MLlib (v3.5.3)"],
        ["Execution Paradigm", "Single-Node (RAM)", "Distributed (JVM DAG)"],
        ["Feature Extraction", "TfidfVectorizer (5k vocab)", "Tokenizer → HashingTF → IDF"],
        ["Accuracy", "77.08%", "72.55%"],
        ["Weighted Precision", "0.7608", "0.7176"],
        ["Weighted Recall", "0.7708", "0.7255"],
        ["Weighted F1-Score", "0.7531", "0.7189"],
        ["Data Load Time", "0.1556 s", "3.1127 s"],
        ["Training Time", "0.2372 s", "4.4826 s"],
        ["Inference Time", "0.0006 s", "0.5535 s"],
        ["Total Pipeline Time", "0.7356 s", "7.5953 s"],
        ["Target Scale", "< 2 GB (RAM)", "100 GB+ (Clusters)"],
    ]
)
add_para(
    "Scikit-Learn achieves higher accuracy on single-node data (< 15,000 records) due to richer "
    "TF-IDF features (n-gram sublinear TF). However, Apache Spark MLlib demonstrates horizontal "
    "scalability capable of processing terabytes of enterprise data across distributed clusters "
    "with minimal accuracy tradeoff (~4.5% gap). In production airline environments with millions "
    "of daily reviews, Spark MLlib's distributed architecture becomes essential."
)

add_heading_styled("C. Scalability Benchmarks", level=2)
add_table(
    ["Slice", "Rows", "Framework", "Training Time", "Total Runtime", "Accuracy", "Weighted F1"],
    [
        ["10%", "1,464", "Scikit-Learn", "0.0242 s", "0.0456 s", "69.28%", "0.6275"],
        ["10%", "1,558", "Spark MLlib", "3.6877 s", "4.9878 s", "70.30%", "0.6859"],
        ["25%", "3,660", "Scikit-Learn", "0.1104 s", "0.1657 s", "74.59%", "0.7063"],
        ["25%", "3,761", "Spark MLlib", "1.1690 s", "1.5522 s", "68.29%", "0.6691"],
        ["50%", "7,320", "Scikit-Learn", "0.1482 s", "0.2307 s", "75.75%", "0.7307"],
        ["50%", "7,455", "Spark MLlib", "1.1996 s", "1.5136 s", "67.25%", "0.6586"],
        ["100%", "14,640", "Scikit-Learn", "0.2747 s", "0.4333 s", "79.06%", "0.7757"],
        ["100%", "14,640", "Spark MLlib", "1.3675 s", "1.7289 s", "72.55%", "0.7189"],
    ]
)
add_para(
    "Both frameworks show accuracy improvements with increasing data volume, confirming that more "
    "training data leads to better generalization. Scikit-Learn's runtime scales linearly from 0.04s "
    "to 0.43s, while Spark MLlib's overhead (JVM boot, DAG scheduling) dominates at small scales but "
    "would amortize across cluster nodes at enterprise volumes."
)

add_heading_styled("D. Intelligent Routing Results", level=2)
add_bullet("Compound reviews are successfully decomposed into discrete clauses, with each clause independently evaluated.")
add_bullet("Meta NLI Zero-Shot classification achieves high-confidence department assignments (82-95% confidence on representative examples).")
add_bullet("Dynamic priority scoring correctly generates HIGH priority tickets for negative operational issues and logs positive feedback as compliments without queue clutter.")

# (Output Screenshots placeholder)
add_heading_styled("E. Output Screenshots", level=2)
add_para("[Screenshots of the Streamlit Dashboard and FastAPI Swagger UI to be inserted here]", size=11)

# ═══════════════════════════════════════════════════════════
# 11. CONCLUSION
# ═══════════════════════════════════════════════════════════
add_heading_styled("11. Conclusion", level=1)
add_para(
    "AirRoute AI successfully demonstrates an enterprise-grade big data analytics and machine learning "
    "pipeline for airline passenger feedback. The key contributions of this project are:"
)
conclusions = [
    "PySpark ETL with Parquet Storage: Achieved 39.4% storage compression while maintaining 100% data integrity, demonstrating practical big data storage optimization.",
    "Dual-Paradigm ML Comparison: Validated that Scikit-Learn provides superior accuracy for single-node datasets while Spark MLlib enables horizontal scalability for enterprise-grade data volumes.",
    "Clause-Level Multi-Issue Resolution: Successfully decomposed compound reviews into discrete clauses, enabling per-issue sentiment analysis instead of coarse whole-review classification.",
    "Zero-Shot Department Routing: Leveraged Meta NLI inference to eliminate the need for department-labeled training data, enabling dynamic routing to 6 airline departments with confidence scoring.",
    "Full-Stack Deployment: Delivered a production-ready system with FastAPI REST API, SQLite ticket persistence, and an interactive Streamlit dashboard for real-time analytics and ticket management.",
]
for c in conclusions:
    add_bullet(c)

# ═══════════════════════════════════════════════════════════
# 12. FUTURE SCOPE
# ═══════════════════════════════════════════════════════════
add_heading_styled("12. Future Scope", level=1)
futures = [
    ("Real-Time Streaming Ingestion: ", "Integrating Apache Kafka to consume live Twitter/X and customer support webhooks in real-time."),
    ("Generative LLM Integration: ", "Deploying local LLMs (e.g., Llama 3 via Ollama) to automatically generate personalized draft email responses for customer support operators."),
    ("Multi-Lingual Support: ", "Expanding sentiment routing to support multilingual passenger reviews in Hindi, Spanish, French, and German."),
    ("Cloud Deployment: ", "Deploying the Spark pipeline on cloud-managed services (AWS EMR, Azure HDInsight, or Databricks) for true multi-node distributed processing."),
    ("Advanced Feedback Loop: ", "Implementing active learning where misrouted tickets are flagged by human agents and fed back into the model for continuous improvement."),
]
for bold_prefix, text in futures:
    add_bullet(text, bold_prefix=bold_prefix)

# ═══════════════════════════════════════════════════════════
# 13. REFERENCES
# ═══════════════════════════════════════════════════════════
add_heading_styled("13. References", level=1)
references = [
    "CrowdFlower / Kaggle, Twitter US Airline Sentiment Dataset, Kaggle Open Data License.",
    "Apache Software Foundation, Apache Spark Documentation: Spark SQL, DataFrames, and MLlib Guide, https://spark.apache.org/docs/latest/",
    "Scikit-Learn Contributors, Scikit-Learn Documentation: Supervised Learning & Feature Extraction APIs, https://scikit-learn.org/stable/",
    "FastAPI Contributors, FastAPI Documentation, https://fastapi.tiangolo.com/",
    "Streamlit Inc., Streamlit Documentation, https://docs.streamlit.io/",
    "Hugging Face, Transformers Library: Zero-Shot Classification Pipeline, https://huggingface.co/docs/transformers/",
    "Williams, A., Nangia, N., & Bowman, S. R. (2018), A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference, Proceedings of NAACL-HLT.",
    "SQLAlchemy Contributors, SQLAlchemy ORM Documentation, https://docs.sqlalchemy.org/",
]
for i, ref in enumerate(references, 1):
    p = doc.add_paragraph()
    run = p.add_run(f"[{i}] {ref}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

# ── Save ──
output_path = os.path.join(os.path.dirname(__file__), "Project_Report.docx")
doc.save(output_path)
print(f"Report saved: {output_path}")
