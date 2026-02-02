# StudyAssistant - Claude Project Notes

## Project Overview
StudyAssistant is a RAG (Retrieval Augmented Generation) system for studying course materials. It ingests PDFs, chunks and embeds them, and allows querying with AI-powered answers that include citations.

## Tech Stack
- **Vector Database:** ChromaDB
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **LLM:** Anthropic Claude Sonnet
- **PDF Processing:** PyMuPDF (fitz)
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

## How It Works

### Ingestion Pipeline (ingest.py)
1. Reads PDFs from iCloud StudyPDFs folder
2. Extracts text page-by-page using PyMuPDF
3. Extracts significant images and sends to Claude for descriptions
4. Chunks text (1000 chars, 200 overlap) with page attribution
5. Generates embeddings with all-MiniLM-L6-v2
6. Stores in ChromaDB with metadata (filename, pages, type)
7. Tracks file hashes to skip unchanged PDFs on re-run

### Query Pipeline (query.py / app.py)
1. Embed the user's question
2. Retrieve top-k similar chunks from ChromaDB
3. Send chunks + question to Claude Sonnet
4. System prompt enforces inline citations [Source: filename, Page(s): X]

## Commands

**Activate environment:**
```bash
cd ~/Desktop/StudyAssistant
source venv/bin/activate
```

**Ingest new PDFs:**
```bash
python ingest.py
```

**Force re-ingestion (with images):**
```bash
rm .processed_files.json
python ingest.py
```

**Run CLI query:**
```bash
python query.py "What is a treemap?"
python query.py -i  # Interactive mode
```

**Run web interface:**
```bash
streamlit run app.py
```

**Or use the launcher:**
```bash
./run.sh
```

---

## Current State (Feb 2, 2026)

### Working Features
- PDF text extraction with page numbers
- Text chunking with overlap
- Image extraction and Claude-powered descriptions
- Vector search with ChromaDB
- CLI and web interfaces
- Citation-based answers
- Image upload in web interface for visual questions

### Known Limitations
- Image descriptions stored separately from surrounding text context
- Text-only retrieval (no visual similarity search)
- No automatic retrieval of visual analytics concepts when images uploaded

---

## PLANNED: Visual Analytics Improvements

**Goal:** Better handle questions that require analyzing charts/visualizations alongside course theory.

**Example Use Case:**
- User uploads a scatterplot showing Income vs Limit with shape (gender) and color (marital status)
- Question: "How many attributes are shown?"
- System should: identify visual encodings in the image AND retrieve course content about data types, marks, channels

### Planned Changes

#### 1. Enhanced Image Query Retrieval
When an image is uploaded, automatically retrieve chunks about:
- "visual encodings"
- "quantitative vs categorical"
- "marks and channels"
- "data attributes"

#### 2. Improved Query Prompt for Visual Analysis
Update the Claude prompt to:
1. First identify all visual encodings in the uploaded image
2. Map each encoding to course concepts
3. Answer using both image analysis and course materials

#### 3. Context-Aware Image Descriptions (ingest.py)
When storing image descriptions during ingestion:
- Include the slide/page title
- Include surrounding text context
- Tag with visual analytics keywords for better retrieval

#### 4. Visual Analytics Keyword Boosting
When query mentions visualization terms or includes an image:
- Boost retrieval of chunks tagged as visual analytics concepts
- Ensure foundational theory is always included

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

---

## Course Materials
Currently ingested for: **BAIS:6140 Visual Analytics** (Spring 2026)
- BAIS6140 Visual Analytics Slides.pdf (228 pages, 75+ chunks)

Add new materials to: `~/Library/Mobile Documents/com~apple~CloudDocs/StudyPDFs/`
