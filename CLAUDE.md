# StudyAssistant - Claude Project Notes

## Project Overview
StudyAssistant is a RAG (Retrieval Augmented Generation) system for studying course materials. It ingests PDFs, chunks and embeds them, and allows querying with AI-powered answers that include citations. Designed specifically for Visual Analytics coursework with strong image analysis capabilities.

## Tech Stack
- **Vector Database:** ChromaDB
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **LLM:** Anthropic Claude Sonnet
- **PDF Processing:** PyMuPDF (fitz)
- **Image Processing:** Pillow (for resizing large images)
- **Web Interface:** Streamlit
- **Language:** Python 3.13

## File Structure
```
StudyAssistant/
├── ingest.py        # PDF ingestion - extracts text/images, chunks, embeds, stores
├── query.py         # CLI query interface with interactive mode
├── app.py           # Streamlit web interface
├── run.sh           # Launcher menu script
├── requirements.txt # Python dependencies
├── venv/            # Virtual environment
├── chroma_db/       # ChromaDB vector database (generated)
└── .processed_files.json  # Tracks ingested files (generated)
```

## Configuration
- **PDF Source:** `~/Library/Mobile Documents/com~apple~CloudDocs/StudyPDFs/`
- **Database:** `~/Desktop/StudyAssistant/chroma_db/`
- **Chunk Size:** 1000 characters
- **Chunk Overlap:** 200 characters
- **Top-K Retrieval:** 5 chunks
- **Max Image Size:** 4.5MB (auto-resized for Claude API)

## Quick Start

**Using the alias (recommended):**
```bash
study
```
This launches the Streamlit web interface at http://localhost:8501

**Or manually:**
```bash
cd ~/Desktop/StudyAssistant
source venv/bin/activate
streamlit run app.py
```

## Commands

**Ingest new PDFs:**
```bash
cd ~/Desktop/StudyAssistant
source venv/bin/activate
python ingest.py
```

**Force re-ingestion (to update image descriptions):**
```bash
rm .processed_files.json
python ingest.py
```

**CLI query (single question):**
```bash
python query.py "What is a treemap?"
```

**CLI query (interactive mode):**
```bash
python query.py -i
```

---

## Current State (Feb 2, 2026)

### Working Features

**Core RAG:**
- PDF text extraction with page numbers
- Text chunking with overlap (1000 chars, 200 overlap)
- Vector search with ChromaDB
- CLI and web interfaces

**Image Processing:**
- Image extraction from PDFs during ingestion
- Claude-powered image descriptions stored as searchable chunks
- Automatic image resizing for images >4.5MB (Claude API limit)
- 186 image descriptions extracted from course slides

**Visual Analysis (NEW):**
- Upload charts/visualizations and ask questions about them
- System analyzes uploaded image FIRST, then uses course materials
- Identifies visual encodings (position, color, shape, size)
- Classifies data attributes as quantitative vs categorical
- Connects analysis to course concepts with citations

**Citation Format:**
- IEEE/Vancouver style: numbered [1], [2] inline
- Full References section at bottom of answers

### Example Use Case (Working!)
- Upload a scatterplot showing Income vs Limit with shape (gender) and color (marital status)
- Ask: "How many attributes are shown in this scatterplot?"
- System correctly identifies:
  - 4 attributes total
  - 2 Quantitative (Income, Limit) encoded by x/y position
  - 2 Categorical (Gender, Marital Status) encoded by shape/color
- Provides theoretical explanation with course citations

---

## How It Works

### Ingestion Pipeline (ingest.py)
1. Reads PDFs from iCloud StudyPDFs folder
2. Extracts text page-by-page using PyMuPDF
3. Extracts significant images (>10KB, >100px)
4. Resizes large images to fit Claude API limits
5. Sends images to Claude for visual descriptions
6. Chunks text (1000 chars, 200 overlap) with page attribution
7. Generates embeddings with all-MiniLM-L6-v2
8. Stores in ChromaDB with metadata (filename, pages, type)
9. Tracks file hashes to skip unchanged PDFs on re-run

### Query Pipeline (app.py)
1. User enters question and optionally uploads an image
2. Question is embedded and top-k similar chunks retrieved
3. If image uploaded: Claude analyzes the image FIRST
4. Course materials provide theoretical context and citations
5. Response uses IEEE citation format with References section

---

## Future Improvements

**Potential Enhancements:**
- Automatic retrieval of visual encoding concepts when image uploaded
- Slide-aware chunking (keep slide title + content together)
- Multi-modal retrieval using CLIP embeddings
- OCR for text within charts (axis labels, legends)

---

## Environment Setup

**Required:** ANTHROPIC_API_KEY in ~/.zshrc
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**Install dependencies:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Shell alias (in ~/.zshrc):**
```bash
alias study="cd ~/Desktop/StudyAssistant && source venv/bin/activate && streamlit run app.py --server.address 0.0.0.0"
```

---

## Course Materials
Currently ingested for: **BAIS:6140 Visual Analytics** (Spring 2026)
- BAIS6140 Visual Analytics Slides.pdf
- BAIS6140 - Session 3.pdf
- BAIS6140 - Session 4.pdf
- Modules 3.1-3.5 (Tableau)
- Modules 4.1-4.4 (Exploratory Analysis, Clustering, Predictive, Forecasting)
- Modules 5.1-5.3 (Interactivity, Dashboards, Power BI)
- 515 total chunks in database

**Add new materials to:** `~/Library/Mobile Documents/com~apple~CloudDocs/StudyPDFs/`
Then run `python ingest.py` to process them.

---

## Repository
GitHub: https://github.com/SteveKosbau/StudyAssistant
