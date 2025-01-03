import concurrent.futures
from pathlib import Path
from typing import Any
from unstructured.partition.auto import partition

from entities.document import Document
from helpers.log import get_logger
from tqdm import tqdm


logger = get_logger(__name__)


class DirectoryLoader:
    """
    Class for loading documents from a directory.
    """
    def __init__(
            self, path: Path, glob: str = "**/[!.]*", recursive: bool = False,
            show_progress: bool = False, use_multithreading: bool = False,
            max_concurrency: int = 4, **partition_kwargs: Any,
    ):
        self.path = path
        self.glob = glob
        self.recursive = recursive
        self.show_progress = show_progress
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency
        self.partition_kwargs = partition_kwargs
    
    def load(self) -> list[Document]:
        if not self.path.exists():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")
        if not self.path.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.path}'")
        
        docs = []
        items = list(self.path.rglob(self.glob) if self.recursive else self.path.glob(self.glob))

        pbar = None
        if self.show_progress:
            pbar = tqdm(total=len(items))

        if self.use_multithreading:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                executor.map(lambda item: self.load_file(item, docs, pbar), items)
        else:
            for i in items:
                self.load_file(i, docs, pbar)

        if pbar:
            pbar.close()

        return docs
    
    def load_file(self, doc_path: Path, docs: list[Document], pbar: Any | None) -> None:
        if doc_path.is_file():
            try:
                logger.debug(f"Processing file: {str(doc_path)}")
                elements = partition(filename=str(doc_path), **self.partition_kwargs)
                text = "\n\n".join([str(el) for el in elements])
                docs.extend([Document(page_content=text, metadata={"source": str(doc_path)})])
            finally:
                if pbar:
                    pbar.update(1)


if __name__ == "__main__":
    root_folder = Path(__file__).resolve().parent.parent.parent
    docs_path = root_folder / "docs"
    loader = DirectoryLoader(
        path=docs_path, glob="*.md", recursive=True,
        use_multithreading=True, show_progress=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
