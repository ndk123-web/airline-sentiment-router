"""
Generate Mini Project Poster as A3-sized .docx
"""
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import os

doc = Document()

# ── A3 Landscape Page Setup ──
for section in doc.sections:
    section.page_width = Cm(42.0)    # A3 width
    section.page_height = Cm(29.7)   # A3 height (landscape)
    section.top_margin = Cm(1.5)
    section.bottom_margin = Cm(1.5)
    section.left_margin = Cm(1.5)
    section.right_margin = Cm(1.5)

style = doc.styles['Normal']
font = style.font
font.name = 'Arial'
font.size = Pt(10)

# ── Helpers ──
def add_heading_poster(text, level=1, color=RGBColor(0, 51, 102)):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Arial'
        run.font.color.rgb = color
        if level == 1:
            run.font.size = Pt(16)
        elif level == 2:
            run.font.size = Pt(13)
    h.paragraph_format.space_before = Pt(6)
    h.paragraph_format.space_after = Pt(4)
    return h

def add_para(text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, size=10, space_after=3, color=None):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color
    return p

def add_bullet(text, bold_prefix="", size=10):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    if bold_prefix:
        run_b = p.add_run(bold_prefix)
        run_b.bold = True
        run_b.font.name = 'Arial'
        run_b.font.size = Pt(size)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(size)
    return p

def set_cell_shading(cell, color_hex):
    shading = cell._element.get_or_add_tcPr()
    shading_elm = shading.makeelement(qn('w:shd'), {
        qn('w:fill'): color_hex,
        qn('w:val'): 'clear'
    })
    shading.append(shading_elm)

def add_table(headers, rows, header_color='1F4E79'):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.name = 'Arial'
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, header_color)
    # Data
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.name = 'Arial'
            run.font.size = Pt(9)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            # Alternate row shading
            if r_idx % 2 == 0:
                set_cell_shading(cell, 'E8F0FE')
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    return table

# ═══════════════════════════════════════════════════════════
# POSTER TITLE BLOCK
# ═══════════════════════════════════════════════════════════

# Title
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
run = p.add_run("AIRROUTE AI")
run.font.name = 'Arial'
run.font.size = Pt(36)
run.bold = True
run.font.color.rgb = RGBColor(0, 51, 102)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Scalable Airline Passenger Sentiment Analysis & Intelligent Review Routing System")
run.font.name = 'Arial'
run.font.size = Pt(18)
run.font.color.rgb = RGBColor(68, 68, 68)

# Thin separator line
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
run = p.add_run("━" * 120)
run.font.size = Pt(6)
run.font.color.rgb = RGBColor(0, 51, 102)

# Student & Subject info (in a table for layout)
info_table = doc.add_table(rows=1, cols=2)
info_table.alignment = WD_TABLE_ALIGNMENT.CENTER

# Left cell: Members
left_cell = info_table.rows[0].cells[0]
left_cell.text = ""
p = left_cell.paragraphs[0]
lines = [
    ("Group Members:\n", True),
    ("1. Navnath Kadam (23102B0061, CMPN-B)\n", False),
    ("2. Prathmesh Nitnaware (23102B0060, CMPN-B)\n", False),
    ("3. Sahil Khilari (23102B0061, CMPN-B)", False),
]
for text, is_bold in lines:
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(10)
    run.bold = is_bold

# Right cell: Subject/Faculty
right_cell = info_table.rows[0].cells[1]
right_cell.text = ""
p = right_cell.paragraphs[0]
p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
subj_lines = [
    ("Subject: ", True), ("Scalable ML and Big Data Analytics (BDA)\n", False),
    ("Faculty: ", True), ("_________________________", False),
]
for text, is_bold in subj_lines:
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(10)
    run.bold = is_bold

# Remove borders from info table
for row in info_table.rows:
    for cell in row.cells:
        tc = cell._element
        tcPr = tc.get_or_add_tcPr()
        tcBorders = tcPr.makeelement(qn('w:tcBorders'), {})
        for border_name in ['top', 'left', 'bottom', 'right']:
            border_el = tcBorders.makeelement(qn(f'w:{border_name}'), {
                qn('w:val'): 'none', qn('w:sz'): '0', qn('w:space'): '0', qn('w:color'): 'auto'
            })
            tcBorders.append(border_el)
        tcPr.append(tcBorders)

doc.add_paragraph().paragraph_format.space_after = Pt(2)

# ═══════════════════════════════════════════════════════════
# CONTENT IN 2-COLUMN LAYOUT (using a 2-column table)
# ═══════════════════════════════════════════════════════════
# Since python-docx doesn't natively support columns easily,
# we'll use a borderless 2-column table for layout

layout_table = doc.add_table(rows=1, cols=2)
layout_table.alignment = WD_TABLE_ALIGNMENT.CENTER

# Set column widths
for row in layout_table.rows:
    row.cells[0].width = Cm(19)
    row.cells[1].width = Cm(19)

left = layout_table.rows[0].cells[0]
right = layout_table.rows[0].cells[1]

# Remove borders
for row in layout_table.rows:
    for cell in row.cells:
        tc = cell._element
        tcPr = tc.get_or_add_tcPr()
        tcBorders = tcPr.makeelement(qn('w:tcBorders'), {})
        for border_name in ['top', 'left', 'bottom', 'right']:
            border_el = tcBorders.makeelement(qn(f'w:{border_name}'), {
                qn('w:val'): 'none', qn('w:sz'): '0', qn('w:space'): '0', qn('w:color'): 'auto'
            })
            tcBorders.append(border_el)
        tcPr.append(tcBorders)

def cell_heading(cell, text, color_hex='003366'):
    p = cell.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(13)
    run.bold = True
    run.font.color.rgb = RGBColor(0, 51, 102)
    return p

def cell_text(cell, text, size=10, bold=False):
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(size)
    run.bold = bold
    return p

def cell_bullet(cell, text, bold_prefix="", size=9):
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.left_indent = Cm(0.5)
    run_dot = p.add_run("• ")
    run_dot.font.name = 'Arial'
    run_dot.font.size = Pt(size)
    if bold_prefix:
        run_b = p.add_run(bold_prefix)
        run_b.bold = True
        run_b.font.name = 'Arial'
        run_b.font.size = Pt(size)
    run = p.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(size)
    return p

def cell_table(cell, headers, rows, header_color='1F4E79'):
    """Add a table inside a cell"""
    inner_table = cell.add_table(rows=1 + len(rows), cols=len(headers))
    inner_table.style = 'Table Grid'
    inner_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = inner_table.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        run.bold = True
        run.font.name = 'Arial'
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(255, 255, 255)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(c, header_color)
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            c = inner_table.rows[r_idx + 1].cells[c_idx]
            c.text = ""
            run = c.paragraphs[0].add_run(str(val))
            run.font.name = 'Arial'
            run.font.size = Pt(8)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if r_idx % 2 == 0:
                set_cell_shading(c, 'E8F0FE')
    return inner_table


# ═══════════════ LEFT COLUMN ═══════════════

# 1. Problem Statement
cell_heading(left, "1. Problem Statement")
cell_bullet(left, " Airlines receive massive volumes of reviews daily across digital channels.", bold_prefix="Scale:")
cell_bullet(left, " Human agents manually read and route complaints, causing delayed responses for critical operational failures.", bold_prefix="Manual Triage Delay:")
cell_bullet(left, " Reviews often mix multiple issues in one sentence (e.g., \"Flight delayed 4 hrs and lost bag, but crew was polite!\"). Traditional classifiers fail to isolate individual issues.", bold_prefix="Compound Feedback:")
cell_bullet(left, " Enterprise-scale feedback exceeds single-node processing capacity, requiring Apache Spark.", bold_prefix="Big Data:")

# 2. Objectives
cell_heading(left, "2. Objectives")
cell_bullet(left, " PySpark ETL pipeline with Parquet compression", bold_prefix="1.")
cell_bullet(left, " Scikit-Learn vs. Spark MLlib dual-model comparison", bold_prefix="2.")
cell_bullet(left, " Discourse clause segmentation for compound reviews", bold_prefix="3.")
cell_bullet(left, " Meta NLI Zero-Shot department routing with confidence", bold_prefix="4.")
cell_bullet(left, " Full-stack: FastAPI + SQLite + Streamlit dashboard", bold_prefix="5.")

# 3. Methodology / Workflow
cell_heading(left, "3. Methodology / Workflow")
workflow = (
    "[Raw CSV Data (Tweets.csv)]\n"
    "         ▼\n"
    "[PySpark ETL Engine] → [Parquet Storage]\n"
    "         ▼\n"
    "   ┌─────────────────────────┐\n"
    "   ▼                         ▼\n"
    "[Scikit-Learn]          [Spark MLlib]\n"
    " Acc: 77.08%             Acc: 72.55%\n"
    "         ▼\n"
    "[Discourse Clause Segmentation]\n"
    "   Clause N → TF-IDF → Sentiment ML\n"
    "            → Meta NLI → Department\n"
    "         ▼\n"
    "[Priority Engine + SQLite Tickets]\n"
    "         ▼\n"
    "[FastAPI] ←→ [Streamlit Dashboard]"
)
p = left.add_paragraph()
p.paragraph_format.space_after = Pt(3)
run = p.add_run(workflow)
run.font.name = 'Consolas'
run.font.size = Pt(8)

# 4. Technologies
cell_heading(left, "4. Technologies & Tools")
cell_table(left,
    ["Category", "Technology"],
    [
        ["Big Data", "Apache Spark 3.5.3, PySpark, Parquet"],
        ["Machine Learning", "Scikit-Learn, Spark MLlib, TF-IDF"],
        ["Zero-Shot NLP", "Hugging Face (DistilBART-MNLI)"],
        ["Backend", "FastAPI, Uvicorn"],
        ["Frontend", "Streamlit, Plotly"],
        ["Database", "SQLite, SQLAlchemy"],
        ["Testing & VCS", "Pytest, Git & GitHub"],
    ]
)


# ═══════════════ RIGHT COLUMN ═══════════════

# 5. Results
cell_heading(right, "5. Results")

cell_text(right, "A. ETL Performance", size=11, bold=True)
cell_bullet(right, " 3.42 MB → 2.07 MB (39.4% compression)", bold_prefix="Storage:")
cell_bullet(right, " 14,640 records, 100% retention", bold_prefix="Records:")
cell_bullet(right, " 8.63 seconds", bold_prefix="Runtime:")

cell_text(right, "B. Model Comparison", size=11, bold=True)
cell_table(right,
    ["Metric", "Scikit-Learn", "Spark MLlib"],
    [
        ["Accuracy", "77.08%", "72.55%"],
        ["Weighted Precision", "0.7608", "0.7176"],
        ["Weighted Recall", "0.7708", "0.7255"],
        ["Weighted F1-Score", "0.7531", "0.7189"],
        ["Pipeline Time", "0.74 s", "7.60 s"],
        ["Target Scale", "< 2 GB", "100 GB+"],
    ]
)

cell_text(right, "C. Scalability Benchmarks", size=11, bold=True)
cell_table(right,
    ["Slice", "Sklearn Acc.", "Spark Acc.", "Sklearn Time", "Spark Time"],
    [
        ["10%", "69.28%", "70.30%", "0.05 s", "4.99 s"],
        ["25%", "74.59%", "68.29%", "0.17 s", "1.55 s"],
        ["50%", "75.75%", "67.25%", "0.23 s", "1.51 s"],
        ["100%", "79.06%", "72.55%", "0.43 s", "1.73 s"],
    ]
)

cell_text(right, "D. Intelligent Routing", size=11, bold=True)
cell_bullet(right, "Compound reviews decomposed into per-clause analysis")
cell_bullet(right, "Meta NLI Zero-Shot: 82-95% confidence in department classification")
cell_bullet(right, "Automated HIGH/MEDIUM/LOW priority ticket generation")

# 6. Conclusion
cell_heading(right, "6. Conclusion")
cell_text(right,
    "AirRoute AI demonstrates an end-to-end scalable Big Data and NLP system "
    "for airline passenger sentiment analysis. The system compresses data via PySpark ETL "
    "(39.4% reduction), achieves 77.08% sentiment accuracy with dual ML paradigms, "
    "decomposes compound reviews into clause-level analysis, routes feedback to correct "
    "departments via Zero-Shot NLI, and automates ticket lifecycle management with "
    "FastAPI + Streamlit.", size=9)


# ═══════════════════════════════════════════════════════════
# SIGNATURE BLOCK (outside the layout table)
# ═══════════════════════════════════════════════════════════
doc.add_paragraph().paragraph_format.space_after = Pt(4)

# Separator
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("━" * 120)
run.font.size = Pt(6)
run.font.color.rgb = RGBColor(0, 51, 102)

# Signature table
sig_table = doc.add_table(rows=3, cols=2)
sig_table.alignment = WD_TABLE_ALIGNMENT.CENTER

# Remove borders
for row in sig_table.rows:
    for cell in row.cells:
        tc = cell._element
        tcPr = tc.get_or_add_tcPr()
        tcBorders = tcPr.makeelement(qn('w:tcBorders'), {})
        for border_name in ['top', 'left', 'bottom', 'right']:
            border_el = tcBorders.makeelement(qn(f'w:{border_name}'), {
                qn('w:val'): 'none', qn('w:sz'): '0', qn('w:space'): '0', qn('w:color'): 'auto'
            })
            tcBorders.append(border_el)
        tcPr.append(tcBorders)

# Row 0: Empty space for signature
for cell in sig_table.rows[0].cells:
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(30)

# Row 1: Lines
for cell in sig_table.rows[1].cells:
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("_" * 40)
    run.font.name = 'Arial'
    run.font.size = Pt(10)

# Row 2: Labels
labels = ["Faculty Signature", "External Examiner's Signature"]
for i, label in enumerate(labels):
    cell = sig_table.rows[2].cells[i]
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(label)
    run.font.name = 'Arial'
    run.font.size = Pt(11)
    run.bold = True


# ── Save ──
output_path = os.path.join(os.path.dirname(__file__), "Project_Poster_A3.docx")
doc.save(output_path)
print(f"Poster saved: {output_path}")
