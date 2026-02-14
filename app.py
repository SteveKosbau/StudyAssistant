#!/usr/bin/env python3
"""
Streamlit Web Interface for Study Assistant
Interactive Q&A with course materials including citations.
Supports image uploads and clipboard paste for visual analysis.
"""

import os
import base64
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
import streamlit.components.v1 as components
from dotenv import load_dotenv
import fitz  # PyMuPDF

# Load environment variables from .env file
load_dotenv()

# Configuration
CHROMA_PATH = os.path.expanduser("~/Desktop/StudyAssistant/chroma_db")
COLLECTION_NAME = "study_materials"
TOP_K = 5


# JavaScript for clipboard paste functionality
PASTE_IMAGE_JS = """
<style>
    #paste-zone {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background: #f9f9f9;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    #paste-zone:hover, #paste-zone.drag-over {
        border-color: #4CAF50;
        background: #e8f5e9;
    }
    #paste-zone.has-image {
        border-color: #2196F3;
        background: #e3f2fd;
    }
    #pasted-image {
        max-width: 100%;
        max-height: 300px;
        margin-top: 10px;
        border-radius: 5px;
    }
    .paste-instructions {
        color: #666;
        font-size: 14px;
    }
    .paste-success {
        color: #4CAF50;
        font-weight: bold;
    }
</style>

<div id="paste-zone" tabindex="0">
    <p class="paste-instructions">📋 Click here and press <strong>Cmd+V</strong> to paste a screenshot</p>
    <p class="paste-instructions" style="font-size: 12px; color: #999;">Or drag & drop an image</p>
    <img id="pasted-image" style="display: none;" />
</div>

<script>
    const pasteZone = document.getElementById('paste-zone');
    const pastedImage = document.getElementById('pasted-image');

    // Handle paste event
    pasteZone.addEventListener('paste', function(e) {
        e.preventDefault();
        const items = e.clipboardData.items;

        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                const blob = items[i].getAsFile();
                const reader = new FileReader();

                reader.onload = function(event) {
                    const base64Data = event.target.result;
                    pastedImage.src = base64Data;
                    pastedImage.style.display = 'block';
                    pasteZone.classList.add('has-image');
                    pasteZone.querySelector('.paste-instructions').innerHTML = '<span class="paste-success">✓ Image pasted!</span> Paste again to replace.';

                    // Send to Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: base64Data
                    }, '*');
                };

                reader.readAsDataURL(blob);
                break;
            }
        }
    });

    // Handle drag and drop
    pasteZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        pasteZone.classList.add('drag-over');
    });

    pasteZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        pasteZone.classList.remove('drag-over');
    });

    pasteZone.addEventListener('drop', function(e) {
        e.preventDefault();
        pasteZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.indexOf('image') !== -1) {
            const reader = new FileReader();

            reader.onload = function(event) {
                const base64Data = event.target.result;
                pastedImage.src = base64Data;
                pastedImage.style.display = 'block';
                pasteZone.classList.add('has-image');
                pasteZone.querySelector('.paste-instructions').innerHTML = '<span class="paste-success">✓ Image added!</span> Drop again to replace.';

                // Send to Streamlit
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: base64Data
                }, '*');
            };

            reader.readAsDataURL(files[0]);
        }
    });

    // Focus on click
    pasteZone.addEventListener('click', function() {
        pasteZone.focus();
    });

    // Also listen for paste on the whole document (backup)
    document.addEventListener('paste', function(e) {
        if (document.activeElement === pasteZone) return; // Already handled

        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                pasteZone.dispatchEvent(new ClipboardEvent('paste', {
                    clipboardData: e.clipboardData,
                    bubbles: true
                }));
                break;
            }
        }
    });
</script>
"""


@st.cache_resource
def load_models():
    """Load embedding model (cached)."""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder


@st.cache_resource
def load_database():
    """Load ChromaDB (cached)."""
    if not os.path.exists(CHROMA_PATH):
        return None, None

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        return client, collection
    except:
        return client, None


def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF file."""
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text:
            pages.append(f"[Page {i}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


def get_relevant_chunks(query: str, embedder, collection, top_k: int = TOP_K) -> list:
    """Retrieve relevant chunks from ChromaDB."""
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results['ids'][0])):
        chunks.append({
            "text": results['documents'][0][i],
            "filename": results['metadatas'][0][i]['filename'],
            "pages": results['metadatas'][0][i]['pages'],
            "distance": results['distances'][0][i]
        })

    return chunks


def query_claude(question: str, chunks: list, image_data: bytes = None, image_type: str = None, pdf_text: str = None) -> str:
    """Send question and context to Claude, get answer with citations. Optionally include an image and/or PDF text."""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['filename']}, Page(s): {chunk['pages']}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # IEEE/Vancouver citation style
    system_prompt = """You are a study assistant for a Visual Analytics course.

WHEN THE USER UPLOADS AN IMAGE:
1. FIRST, directly analyze the uploaded image - describe what you see (chart type, axes, marks, colors, shapes, legends, etc.)
2. THEN, answer the user's question about that image
3. FINALLY, use the course materials context to explain WHY - connect your analysis to course concepts

For example, if asked "How many attributes are shown?":
- Identify each visual encoding (x-position, y-position, color, shape, size, etc.)
- Each encoding typically represents one data attribute
- Classify each as quantitative or categorical based on course concepts

WHEN THE USER UPLOADS A PDF DOCUMENT:
1. Read the uploaded document content carefully
2. Answer questions about it using both the document AND your course material knowledge
3. For homework/assignments: guide the student through the problems, explain concepts, help them understand — but encourage their own thinking
4. Still cite course materials where relevant

CRITICAL RULES:
1. If an image is uploaded, analyze IT directly - don't confuse it with image descriptions in the context
2. Use course materials to provide theoretical backing for your analysis
3. Use IEEE/Vancouver citation style: numbered references [1], [2], etc.
4. Be precise and educational
5. If context doesn't cover the topic, still analyze the image and explain using general visual analytics principles
6. If a PDF document is uploaded, reference it as [Uploaded Document] in your response

FORMAT YOUR RESPONSE:
- Start with direct analysis of the uploaded image (if any)
- Answer the specific question asked
- Explain using course concepts with numbered citations [1], [2], etc.
- End with a "References" section:
  [1] Filename, Page(s): X
  [2] Filename, Page(s): Y"""

    # Build the uploaded document section if PDF was provided
    pdf_section = ""
    if pdf_text:
        pdf_section = f"""

UPLOADED DOCUMENT CONTENT:
{pdf_text}
"""

    user_text = f"""I have a question about Visual Analytics. I may have attached an image and/or a PDF document for you to analyze.

MY QUESTION: {question}
{pdf_section}
RELEVANT COURSE MATERIAL EXCERPTS (use these for theoretical context and citations):
{context}

INSTRUCTIONS:
- If I attached an image, analyze THAT image directly to answer my question
- If I attached a PDF document, use its content to answer my question
- Use the course material excerpts above to explain concepts and provide citations
- The excerpts labeled [IMAGE/FIGURE...] are from my course slides, not my uploaded image"""

    client = anthropic.Anthropic()

    # Build message content
    if image_data and image_type:
        # Include image in the message
        if isinstance(image_data, bytes):
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        else:
            image_base64 = image_data  # Already base64

        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_type,
                    "data": image_base64
                }
            },
            {
                "type": "text",
                "text": user_text
            }
        ]
    else:
        message_content = user_text

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {"role": "user", "content": message_content}
        ]
    )

    return response.content[0].text


def main():
    st.set_page_config(
        page_title="Study Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Study Assistant")
    st.markdown("*Ask questions about your course materials with AI-powered search and citations*")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("⚠️ ANTHROPIC_API_KEY not found in environment variables.")
        st.info("Please set your API key in ~/.zshrc and restart the terminal.")
        st.stop()

    # Load resources
    with st.spinner("Loading models..."):
        embedder = load_models()
        client, collection = load_database()

    # Check database status
    if collection is None:
        st.warning("⚠️ No documents found in the database.")
        st.info("""
        **To get started:**
        1. Add your PDF files to: `~/Library/Mobile Documents/com~apple~CloudDocs/StudyPDFs/`
        2. Run the ingestion script: `python ingest.py`
        3. Refresh this page
        """)
        st.stop()

    # Show database stats in sidebar
    with st.sidebar:
        st.header("📊 Database Info")
        doc_count = collection.count()
        st.metric("Total Chunks", doc_count)

        # Get unique filenames
        all_metadata = collection.get(include=["metadatas"])
        if all_metadata and all_metadata['metadatas']:
            filenames = set(m['filename'] for m in all_metadata['metadatas'])
            st.metric("Documents", len(filenames))

            st.subheader("📄 Indexed Files")
            for fname in sorted(filenames):
                st.text(f"• {fname}")

        st.divider()
        st.subheader("⚙️ Settings")
        top_k = st.slider("Number of sources to retrieve", 3, 10, TOP_K)
        show_sources = st.checkbox("Show retrieved sources", value=False)

        st.divider()
        st.subheader("🔄 Re-index")
        if st.button("Run Ingestion"):
            st.info("Run `python ingest.py` in terminal to re-index PDFs")

    # Main query interface
    st.divider()

    # Query input
    question = st.text_input(
        "🔍 Ask a question about your course materials:",
        placeholder="e.g., What is the difference between a treemap and a sunburst chart?"
    )

    # File input sections
    col_img, col_pdf = st.columns(2)

    with col_img:
        st.markdown("**Upload an image (optional):**")
        uploaded_image = st.file_uploader(
            "Chart, diagram, or visualization",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            help="Upload an image and ask questions about it"
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded image", width=300)

    with col_pdf:
        st.markdown("**Upload a PDF (optional):**")
        uploaded_pdf = st.file_uploader(
            "Homework, assignment, or reading",
            type=["pdf"],
            help="Upload a PDF document to ask questions about it"
        )
        if uploaded_pdf:
            st.success(f"Attached: {uploaded_pdf.name}")

    if st.button("Get Answer", type="primary"):
        if question:
            st.session_state['last_question'] = question

            with st.spinner("Searching course materials..."):
                chunks = get_relevant_chunks(question, embedder, collection, top_k)

            if show_sources:
                with st.expander("Retrieved Sources", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**[{i}] {chunk['filename']}** (Page(s): {chunk['pages']})")
                        st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
                        st.divider()

            # Prepare image data from uploaded file
            image_data = None
            image_type = None

            if uploaded_image:
                image_data = uploaded_image.getvalue()
                ext = uploaded_image.name.split('.')[-1].lower()
                mime_map = {
                    'png': 'image/png',
                    'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg',
                    'gif': 'image/gif',
                    'webp': 'image/webp'
                }
                image_type = mime_map.get(ext, 'image/png')
                st.info(f"Image attached: {uploaded_image.name}")

            # Extract PDF text if uploaded
            pdf_text = None
            if uploaded_pdf:
                with st.spinner("Reading PDF..."):
                    pdf_text = extract_pdf_text(uploaded_pdf)
                st.info(f"PDF attached: {uploaded_pdf.name} ({len(pdf_text):,} characters)")

            with st.spinner("Generating answer..."):
                answer = query_claude(question, chunks, image_data, image_type, pdf_text)

            st.divider()
            st.subheader("Answer")
            st.markdown(answer)

    # Example questions
    st.divider()
    st.subheader("💭 Example Questions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("What visualization is best for hierarchical data?"):
            st.session_state['example_q'] = "What visualization is best for hierarchical data?"
            st.rerun()

    with col2:
        if st.button("Explain the principles of visual encoding"):
            st.session_state['example_q'] = "Explain the principles of visual encoding"
            st.rerun()


if __name__ == "__main__":
    main()
