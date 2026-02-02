#!/usr/bin/env python3
"""
PDF Ingestion Script for Study Assistant
Extracts text and images from PDFs, chunks with overlap, and stores in ChromaDB with citations.
Images are analyzed by Claude and their descriptions become searchable.
"""

import os
import io
import json
import hashlib
import base64
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import anthropic
from PIL import Image

# Maximum image size for Claude API (5MB, use 4.5MB for safety)
MAX_IMAGE_BYTES = 4_500_000

# Configuration
PDF_FOLDER = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/StudyPDFs")
CHROMA_PATH = os.path.expanduser("~/Desktop/StudyAssistant/chroma_db")
PROCESSED_LOG = os.path.expanduser("~/Desktop/StudyAssistant/.processed_files.json")
COLLECTION_NAME = "study_materials"

# Chunking parameters
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

# Image extraction parameters
MIN_IMAGE_SIZE = 10000  # Minimum bytes for an image to be worth analyzing
MIN_IMAGE_DIMENSION = 100  # Minimum width/height in pixels


def get_file_hash(filepath: str) -> str:
    """Generate MD5 hash of file for change detection."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def load_processed_files() -> Dict[str, str]:
    """Load record of previously processed files."""
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, 'r') as f:
            return json.load(f)
    return {}


def save_processed_files(processed: Dict[str, str]):
    """Save record of processed files."""
    with open(PROCESSED_LOG, 'w') as f:
        json.dump(processed, f, indent=2)


def resize_image_if_needed(image_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Resize image if it exceeds the maximum size for Claude API."""
    if len(image_bytes) <= max_bytes:
        return image_bytes

    # Open image with PIL
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary (for PNG with transparency)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    # Iteratively reduce size until under limit
    quality = 85
    scale = 1.0

    while len(image_bytes) > max_bytes and scale > 0.1:
        # Reduce dimensions
        scale *= 0.8
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)

        # Save to bytes
        output = io.BytesIO()
        resized.save(output, format='JPEG', quality=quality, optimize=True)
        image_bytes = output.getvalue()

        # Also try reducing quality
        if quality > 60:
            quality -= 5

    return image_bytes


def describe_image_with_claude(image_bytes: bytes, filename: str, page_num: int) -> str:
    """Use Claude to describe an image for searchability."""
    try:
        client = anthropic.Anthropic()

        # Resize if too large for Claude API
        image_bytes = resize_image_if_needed(image_bytes)

        # Determine media type (might be JPEG after resizing)
        media_type = "image/jpeg" if image_bytes[:2] == b'\xff\xd8' else "image/png"

        # Encode image as base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Describe this image from course materials in detail for a visual analytics class.
Include:
- Type of visualization or diagram (e.g., bar chart, treemap, flowchart, etc.)
- What data or concepts it represents
- Key visual elements and their meanings
- Any text, labels, or legends visible
- Educational insights it conveys

Be concise but thorough. This description will be used for text search."""
                        }
                    ]
                }
            ]
        )

        description = response.content[0].text
        return f"[IMAGE/FIGURE from {filename}, Page {page_num}]\n{description}"

    except Exception as e:
        print(f"      Warning: Could not analyze image: {e}")
        return None


def extract_images_from_pdf(pdf_path: str, filename: str) -> List[Dict[str, Any]]:
    """Extract significant images from PDF and get Claude descriptions."""
    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # Check if image is significant enough
                if len(image_bytes) < MIN_IMAGE_SIZE:
                    continue

                # Check dimensions
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                    continue

                # Convert to PNG if needed for Claude
                if base_image["ext"] != "png":
                    # Use fitz to convert
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    image_bytes = pix.tobytes("png")

                images.append({
                    "page_num": page_num + 1,
                    "image_bytes": image_bytes,
                    "img_index": img_index
                })

            except Exception as e:
                continue

    doc.close()
    return images


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with page numbers.
    Handles tables by extracting text in reading order.
    Returns list of {page_num, text} dicts.
    """
    pages = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract text - use "text" mode which handles tables reasonably
        text = page.get_text("text")

        # Also try to extract any tables as structured text
        tables = page.find_tables()
        table_text = ""
        if tables:
            for table in tables:
                try:
                    df = table.to_pandas()
                    table_text += "\n" + df.to_string(index=False) + "\n"
                except:
                    pass

        combined_text = text.strip()
        if table_text.strip():
            combined_text += "\n\n[TABLE DATA]\n" + table_text.strip()

        if combined_text:
            pages.append({
                "page_num": page_num + 1,
                "text": combined_text
            })

    doc.close()
    return pages


def chunk_text(pages: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
    """
    Chunk text with overlap, preserving page number attribution.
    """
    chunks = []

    full_text = ""
    page_positions = []

    for page_data in pages:
        page_positions.append({
            "page_num": page_data["page_num"],
            "start_pos": len(full_text)
        })
        full_text += page_data["text"] + "\n\n"

    if page_positions:
        page_positions.append({
            "page_num": page_positions[-1]["page_num"],
            "start_pos": len(full_text)
        })

    start = 0
    chunk_id = 0

    while start < len(full_text):
        end = start + CHUNK_SIZE

        if end < len(full_text):
            for sep in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:
                last_sep = full_text[start:end].rfind(sep)
                if last_sep != -1 and last_sep > CHUNK_SIZE // 2:
                    end = start + last_sep + len(sep)
                    break

        chunk_text_content = full_text[start:end].strip()

        if chunk_text_content:
            chunk_pages = set()
            for i, pos in enumerate(page_positions[:-1]):
                next_pos = page_positions[i + 1]["start_pos"]
                if start < next_pos and end > pos["start_pos"]:
                    chunk_pages.add(pos["page_num"])

            chunks.append({
                "id": f"{filename}_{chunk_id}",
                "text": chunk_text_content,
                "filename": filename,
                "pages": sorted(list(chunk_pages)),
                "page_str": ", ".join(map(str, sorted(chunk_pages))),
                "type": "text"
            })
            chunk_id += 1

        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
        if start >= len(full_text):
            break

    return chunks


def ingest_pdfs(extract_images: bool = True):
    """Main ingestion function."""
    print("=" * 60)
    print("Study Assistant - PDF Ingestion")
    print("=" * 60)

    # Check for API key (needed for image analysis)
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if extract_images and not has_api_key:
        print("\n⚠️  ANTHROPIC_API_KEY not found. Image extraction will be skipped.")
        print("   Set your API key in ~/.zshrc to enable image analysis.")
        extract_images = False

    # Check if PDF folder exists
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"\nCreated PDF folder: {PDF_FOLDER}")
        print("Please add your PDFs to this folder and run again.")
        return

    # Get list of PDFs
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    if not pdf_files:
        print(f"\nNo PDFs found in: {PDF_FOLDER}")
        print("Please add your PDFs and run again.")
        return

    print(f"\nFound {len(pdf_files)} PDF(s) in {PDF_FOLDER}")
    if extract_images:
        print("Image extraction: ENABLED (images will be analyzed by Claude)")
    else:
        print("Image extraction: DISABLED")

    # Load processed files record
    processed_files = load_processed_files()

    # Check which files need processing
    files_to_process = []
    for pdf_path in pdf_files:
        file_hash = get_file_hash(str(pdf_path))
        if str(pdf_path) not in processed_files or processed_files[str(pdf_path)] != file_hash:
            files_to_process.append((pdf_path, file_hash))
        else:
            print(f"  Skipping (unchanged): {pdf_path.name}")

    if not files_to_process:
        print("\nAll files already processed. No updates needed.")
        return

    print(f"\nProcessing {len(files_to_process)} new/modified file(s)...")

    # Initialize embedding model
    print("\nLoading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB
    print("Initializing vector database...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Study materials for RAG"}
    )

    # Process each PDF
    all_chunks = []
    for pdf_path, file_hash in tqdm(files_to_process, desc="Processing PDFs"):
        filename = pdf_path.name
        print(f"\n  Extracting: {filename}")

        # Extract text
        pages = extract_text_from_pdf(str(pdf_path))
        print(f"    - {len(pages)} pages extracted")

        # Chunk text
        chunks = chunk_text(pages, filename)
        print(f"    - {len(chunks)} text chunks created")

        # Extract and analyze images
        if extract_images:
            print(f"    - Extracting images...")
            images = extract_images_from_pdf(str(pdf_path), filename)
            print(f"    - {len(images)} significant images found")

            if images:
                print(f"    - Analyzing images with Claude...")
                for img_data in tqdm(images, desc="      Analyzing", leave=False):
                    description = describe_image_with_claude(
                        img_data["image_bytes"],
                        filename,
                        img_data["page_num"]
                    )
                    if description:
                        chunks.append({
                            "id": f"{filename}_img_{img_data['page_num']}_{img_data['img_index']}",
                            "text": description,
                            "filename": filename,
                            "pages": [img_data["page_num"]],
                            "page_str": str(img_data["page_num"]),
                            "type": "image"
                        })
                print(f"    - {len([c for c in chunks if c.get('type') == 'image'])} image descriptions added")

        # Remove old chunks for this file
        try:
            existing = collection.get(where={"filename": filename})
            if existing and existing['ids']:
                collection.delete(ids=existing['ids'])
                print(f"    - Removed {len(existing['ids'])} old chunks")
        except:
            pass

        all_chunks.extend(chunks)
        processed_files[str(pdf_path)] = file_hash

    # Generate embeddings and store
    if all_chunks:
        print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True)

        print("Storing in vector database...")
        collection.add(
            ids=[chunk["id"] for chunk in all_chunks],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[{
                "filename": chunk["filename"],
                "pages": chunk["page_str"],
                "page_list": ",".join(map(str, chunk["pages"])),
                "type": chunk.get("type", "text")
            } for chunk in all_chunks]
        )

    # Save processed files record
    save_processed_files(processed_files)

    # Summary
    text_chunks = len([c for c in all_chunks if c.get("type", "text") == "text"])
    image_chunks = len([c for c in all_chunks if c.get("type") == "image"])

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  - Text chunks added: {text_chunks}")
    print(f"  - Image descriptions added: {image_chunks}")
    print(f"  - Total chunks in database: {collection.count()}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDFs into Study Assistant")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction and analysis")
    args = parser.parse_args()

    ingest_pdfs(extract_images=not args.no_images)
