from typing import Any
import sentence_transformers


class Embedder:
    """
    Class for computing document embeddings.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str | None = None, **kwargs: Any) -> None:
        self.model = sentence_transformers.SentenceTransformer(model_name, cache_folder=cache_folder, **kwargs)
    
    def embed_documents(self, texts: list[str], multi_process: bool = False, **encode_kwargs: Any) -> list[list[float]]:
        """
        Computes embeddings for the given list of texts. Returns a list of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))

        if multi_process:
            pool = self.model.start_multi_process_pool()
            embeddings = self.model.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True, **encode_kwargs)
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """
        Computes embeddings for the given text.
        """
        return self.embed_documents([text])[0]
