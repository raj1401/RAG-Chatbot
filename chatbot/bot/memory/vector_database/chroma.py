import logging
import uuid
from typing import Any, Callable, Iterable
from cleantext import clean

import chromadb
import chromadb.config
from chromadb.utils.batch_utils import create_batches

from bot.memory.embedder import Embedder
from bot.memory.vector_database.distance_metric import DistanceMetric, get_relevance_score_fn
from entities.document import Document


logger = logging.getLogger(__name__)


class Chroma:
    def __init__(
            self, client: chromadb.Client = None, embedding: Embedder | None = None,
            persist_directory: str | None = None, collection_name: str = "default",
            collection_metadata: dict | None = None, is_persistent: bool = True,
    ) -> None:
        client_settings = chromadb.config.Settings(is_persistent=is_persistent)
        client_settings.persist_directory = persist_directory

        if client is None:
            self.client = chromadb.Client(settings=client_settings)
        else:
            self.client = client
        
        self.embedding = embedding

        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=None, metadata=collection_metadata
        )
    
    @property
    def embeddings(self) -> Embedder | None:
        return self.embedding
    
    def query_collection(
            self, query_texts: list[str] | None = None, query_embeddings: list[list[float]] | None = None,
            n_results: int = 5, where: dict[str, str] | None = None,
            where_document: dict[str, str] | None = None, **kwargs: Any,
    ):
        """
        This function queries the collection with the given query texts or embeddings.
        Uses the same notation as the Chroma Client's query method.
        """
        return self.collection.query(
            query_texts=query_texts, query_embeddings=query_embeddings, n_results=n_results,
            where=where, where_document=where_document, **kwargs
        )
    
    def add_texts(
            self, texts: Iterable[str], metadatas: list[dict] | None = None,
            ids: list[str] | None = None,
    ) -> list[str]:
        """
        Runs the given texts through embeddings and adds to the vectorstore.
        Returns a list of IDs of the added texts.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        embeddings = None
        texts = list(texts)
        if self.embedding is not None:
            embeddings = self.embedding.embed_documents(texts)
        
        if metadatas:
            diff_in_length = len(texts) - len(metadatas)
            if diff_in_length > 0:
                metadatas.extend([{}] * diff_in_length)
            empty_ids, non_empty_ids = [], []
            for idx, m in enumerate(metadatas):
                if not m:
                    empty_ids.append(idx)
                else:
                    non_empty_ids.append(idx)
            if empty_ids:
                texts_without_metadatas = [texts[i] for i in empty_ids]
                embeddings_without_metadatas = [embeddings[i] for i in empty_ids] if embeddings else None
                ids_without_metadatas = [ids[i] for i in empty_ids]
                self.collection.upsert(
                    embeddings=embeddings_without_metadatas, documents=texts_without_metadatas, ids=ids_without_metadatas
                )
            if non_empty_ids:
                metadatas = [metadatas[i] for i in non_empty_ids]
                texts_with_metadatas = [texts[i] for i in non_empty_ids]
                embeddings_with_metadatas = [embeddings[i] for i in non_empty_ids] if embeddings else None
                ids_with_metadatas = [ids[i] for i in non_empty_ids]
                try:
                    self.collection.upsert(
                        embeddings=embeddings_with_metadatas, documents=texts_with_metadatas, ids=ids_with_metadatas, metadatas=metadatas
                    )
                except Exception as e:
                    raise e
        else:
            self.collection.upsert(embeddings=embeddings, documents=texts, ids=ids)
        
        return ids
    
    def from_texts(
            self, texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None,
    ) -> None:
        """
        Adds the given texts to the collection batchwise.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        for batch in create_batches(api=self.client, ids=ids, documents=texts, metadatas=metadatas):
            self.add_texts(
                texts=batch[3] if batch[3] else [],
                metadatas=batch[2] if batch[2] else None,
                ids=batch[0],
            )
    
    def from_chunks(self, chunks: list) -> None:
        """
        Adds a batch of documents to the collection.
        chunks is a list of Document objects
        """
        texts = [clean(doc.page_content, no_emoji=True) for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        self.from_texts(texts, metadatas)
    
    def similarity_search_w_threshold(self, query: str, k: int = 5, threshold: float | None = 0.2):
        """
        Similarity searches on a given query with a threshold.
        k is the number of retrievals to consider.
        Returns a tuple of list of matched documents and their sources.
        """
        docs_and_scores = self.similarity_search_with_relevance_scores(query, k=k)

        if threshold is not None:
            docs_and_scores = [doc for doc in docs_and_scores if doc[1] > threshold]
            if len(docs_and_scores) < 1:
                logger.warning(f"No documents found with a relevance score above {threshold}.")
            
            docs_and_scores = sorted(docs_and_scores, key=lambda x: x[1], reverse=True)
        
        retrieved_contents = [doc[0] for doc in docs_and_scores]
        sources = []
        for doc, score in docs_and_scores:
            sources.append(
                {
                    "score": round(score, 3),
                    "document": doc.metadata.get("source"),
                    "content_preview": f"{doc.page_content[0:256]}...",
                }
            )
        
        return retrieved_contents, sources
    
    def similarity_search(self, query: str, k: int = 5, filter: dict[str, str] | None = None):
        docs_and_scores = self.similarity_search_with_score(query, k=k, filter=filter)
        return [doc[0] for doc in docs_and_scores]
    
    def similarity_search_with_score(
            self, query: str, k: int = 5, filter: dict[str, str] | None = None, where_document: dict[str, str] | None = None,
    ):
        """
        This function runs a similarity search on the given query and returns the top k results with their cosine distances.
        """
        if self.embedding is None:
            results = self.query_collection(query_texts=[query], n_results=k, where=filter, where_document=where_document)
        else:
            query_embedding = self.embedding.embed_query(query)
            results = self.query_collection(query_embeddings=[query_embedding], n_results=k, where=filter, where_document=where_document)
        
        return [
            (Document(page_content=result[0], metadata=result[1] or {}), result[2])
            for result in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
    
    def select_relevance_score_fn(self) -> Callable[[float], float]:
        distance = DistanceMetric.L2
        distance_key = "hnsw:space"
        metadata = self.collection.metadata

        if metadata and distance_key in metadata:
            distance = metadata[distance_key]
        return get_relevance_score_fn(distance)
    
    def similarity_search_with_relevance_scores(self, query: str, k: int = 5):
        """
        Returns docs and relevance scores in the range [0, 1].
        Higher scores indicate higher relevance.
        """
        relevance_score_fn = self.select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query, k=k)
        docs_and_similarities = [
            (doc, relevance_score_fn(score)) for doc, score in docs_and_scores
        ]
        if any(similarity < 0.0 or similarity > 1.0 for _, similarity in docs_and_similarities):
            logger.warning("Relevance scores must be between" f" 0 and 1, got {docs_and_similarities}")
        return docs_and_similarities
