"""
Module 05 - RAG Fundamentals: 01_document_loaders.py
Comprehensive demonstration of LangChain Document Loaders:
- TextLoader & Markdown files
- CSVLoader with source_column metadata
- PyPDFLoader for document extraction
- WebBaseLoader / URL loaders for web scraping
- DirectoryLoader for batch ingestion
- Lazy loading (.lazy_load()) for streaming ingestion of massive corpora
"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, CSVLoader

DATA_DIR = Path(__file__).parent / "data"


def demo_text_and_markdown_loaders():
    print("\n--- 1. Text & Markdown Loaders ---")
    md_path = DATA_DIR / "sample.md"
    if md_path.exists():
        loader = TextLoader(str(md_path), encoding="utf-8")
        docs = loader.load()
        print(f"Loaded Markdown documents: {len(docs)}")
        print(f"Metadata: {docs[0].metadata}")
        print(f"Content Preview:\n  {docs[0].page_content[:120]}...\n")


def demo_csv_loader():
    print("--- 2. CSV Loader with Metadata Mapping ---")
    csv_path = DATA_DIR / "sample.csv"
    if csv_path.exists():
        # source_column sets the document 'source' metadata to the specific column value
        loader = CSVLoader(str(csv_path), source_column="service")
        docs = loader.load()
        print(f"Loaded CSV rows as documents: {len(docs)}")
        print(f"Sample Record:\n{docs[0].page_content}")
        print(f"Extracted Metadata: {docs[0].metadata}\n")


def demo_pdf_and_web_loaders():
    print("--- 3. PDF & Web Document Loaders ---")
    # Demonstrating standard LangChain PDF and Web ingestion patterns
    try:
        from langchain_community.document_loaders import PyPDFLoader
        print("PyPDFLoader available: Usage -> PyPDFLoader('path/to/doc.pdf').load()")
    except ImportError:
        print("Note: PyPDFLoader requires `pypdf` (`pip install pypdf`).")

    try:
        from langchain_community.document_loaders import WebBaseLoader
        print("WebBaseLoader available: Usage -> WebBaseLoader('https://docs.langchain.com').load()")
    except ImportError:
        print("Note: WebBaseLoader requires `beautifulsoup4` (`pip install beautifulsoup4`).")

    # Mocking standard parsed document outputs from a PDF/Web source
    mock_pdf_doc = Document(
        page_content="Executive Summary: Q3 distributed infrastructure throughput increased by 38% after adopting LCEL.",
        metadata={"source": "annual_report.pdf", "page": 4, "total_pages": 48}
    )
    mock_web_doc = Document(
        page_content="LangChain Documentation: Runnables provide a standard interface to invoke, stream, and batch chains.",
        metadata={"source": "https://python.langchain.com/docs/concepts/runnables/", "title": "Runnables Guide"}
    )
    print(f"Sample PDF Document Page: Page {mock_pdf_doc.metadata['page']} | {mock_pdf_doc.page_content}")
    print(f"Sample Web Document Page: Source: {mock_web_doc.metadata['source']} | Title: {mock_web_doc.metadata['title']}\n")


def demo_lazy_loading_and_batch_ingestion():
    print("--- 4. Memory-Efficient Lazy Loading (.lazy_load()) ---")
    # In production, loading millions of documents into memory at once crashes worker nodes.
    # .lazy_load() yields documents as a Python generator instead of a monolithic list.
    txt_path = DATA_DIR / "sample_doc.txt"
    if txt_path.exists():
        loader = TextLoader(str(txt_path), encoding="utf-8")
        generator = loader.lazy_load()
        print(f"Loader returned generator: {type(generator).__name__}")
        
        doc_count = 0
        for doc in generator:
            doc_count += 1
            print(f"Streamed Document {doc_count} chunk length: {len(doc.page_content)} chars")


def main():
    print("=== LangChain Document Loaders Deep Dive ===")
    demo_text_and_markdown_loaders()
    demo_csv_loader()
    demo_pdf_and_web_loaders()
    demo_lazy_loading_and_batch_ingestion()


if __name__ == "__main__":
    main()
