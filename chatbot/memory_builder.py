import argparse
import sys
from pathlib import Path

from bot.memory.embedder import Embedder
from bot.memory.vector_database.chroma import Chroma
from document_loader.format import Format
from document_loader.loader import DirectoryLoader
from document_loader.text_splitter import create_recursive_text_splitter
from entities.document import Document
from helpers.log import get_logger


import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


logger = get_logger(__name__)


def load_documents(doc_path: Path) -> list[Document]:
    loader = DirectoryLoader(
        path=doc_path, glob="**/*.md", show_progress=True
    )
    return loader.load()


def split_chunks(sources: list, chunk_size: int = 512, chunk_overlap: int = 25) -> list:
    """
    Splits a list of sources into chunks of the given size.
    """
    chunks = []
    splitter = create_recursive_text_splitter(
        format=Format.MARKDOWN.value, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def build_memory_index(doc_path: Path, vector_store_path: str, chunk_size: int, chunk_overlap: int):
    logger.info(f"Loading documents from: {doc_path}")
    sources = load_documents(doc_path)
    logger.info(f"Number of loaded documents: {len(sources)}")

    logger.info("Chunking documents...")
    chunks = split_chunks(sources, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"Number of generated chunks: {len(chunks)}")

    logger.info("Creating memory index...")
    embedding = Embedder()
    vector_database = Chroma(persist_directory=str(vector_store_path), embedding=embedding)
    vector_database.from_chunks(chunks)
    logger.info("Memory Index has been created successfully!")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory Builder")
    parser.add_argument(
        "--chunk-size", type=int, required=False, default=512,
        help="The maximum size of each chunk. Defaults to 512.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int, required=False, default=25,
        help="The amount of overlap between consecutive chunks. Defaults to 25.",
    )

    return parser.parse_args()


def main(parameters):
    root_folder = Path(__file__).resolve().parent.parent
    doc_path = root_folder / "docs"
    vector_store_path = root_folder / "vector_store" / "docs_index"

    build_memory_index(
        doc_path, str(vector_store_path),
        parameters.chunk_size,
        parameters.chunk_overlap,
    )


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True, stack_info=True)
        sys.exit(1)
