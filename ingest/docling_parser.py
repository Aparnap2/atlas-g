"""PDF ingestion pipeline using IBM Docling + FAISS vector store."""
import os
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import track

console = Console()

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_store")


def _build_docling_converter() -> DocumentConverter:
    """
    Configure Docling for high-fidelity PDF parsing.
    Enables table structure recognition and OCR for scanned pages.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True          # ⭐ Critical for error-code tables
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_page_images = False       # Save memory
    pipeline_options.generate_picture_images = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _docling_to_langchain_docs(pdf_path: str) -> list[Document]:
    """
    Convert a PDF into LangChain Documents.
    Each section / table cell group becomes a separate Document with
    rich metadata (page number, section heading, element type).
    """
    console.print(f"[cyan]Docling parsing:[/cyan] {pdf_path}")
    converter = _build_docling_converter()
    result = converter.convert(pdf_path)
    doc = result.document

    langchain_docs: list[Document] = []

    # — Extract text sections —
    for item, level in doc.iterate_items():
        text = getattr(item, "text", None) or ""
        if not text.strip():
            continue

        label = getattr(item, "label", "text")
        page_no = (
            item.prov[0].page_no
            if hasattr(item, "prov") and item.prov
            else 0
        )
        heading = getattr(item, "heading", "")

        langchain_docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": pdf_path,
                    "page": page_no,
                    "element_type": str(label),
                    "section_heading": str(heading),
                },
            )
        )

    # — Extract tables as Markdown —
    for table in doc.tables:
        md_table = table.export_to_markdown()
        if not md_table.strip():
            continue
        page_no = (
            table.prov[0].page_no
            if hasattr(table, "prov") and table.prov
            else 0
        )
        langchain_docs.append(
            Document(
                page_content=md_table,
                metadata={
                    "source": pdf_path,
                    "page": page_no,
                    "element_type": "table",
                    "section_heading": "TABLE",
                },
            )
        )

    console.print(
        f"[green]Extracted {len(langchain_docs)} document chunks "
        f"({sum(1 for d in langchain_docs if d.metadata['element_type'] == 'table')} tables)[/green]"
    )
    return langchain_docs


def ingest_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> int:
    """
    Full ingest pipeline:
    1. Parse PDF with Docling (table-aware)
    2. Chunk with RecursiveCharacterTextSplitter
    3. Embed with nomic-embed-text via Ollama
    4. Persist FAISS vector store to disk

    Returns the number of chunks stored.
    """
    docs = _docling_to_langchain_docs(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    console.print(f"[yellow]Split into {len(chunks)} chunks for embedding...[/yellow]")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    store_path = Path(VECTOR_STORE_PATH)

    if store_path.exists():
        # Merge into existing store
        vectorstore = FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vectorstore.add_documents(chunks)
        console.print("[blue]Merged into existing vector store.[/blue]")
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        console.print("[blue]Created new vector store.[/blue]")

    vectorstore.save_local(str(store_path))
    console.print(f"[bold green]✓ Vector store saved to {store_path} ({len(chunks)} chunks)[/bold green]")
    return len(chunks)
