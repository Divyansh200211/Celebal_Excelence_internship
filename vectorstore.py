from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cohere
import PyPDF2
from pinecone import Pinecone

from src.config import AppConfig, sanitize_api_key
from src.logger import get_logger
from src.utils import chunk_text, clean_text, safe_read_text

logger = get_logger("vectorstore")


class DocumentStore:
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.client = cohere.Client(sanitize_api_key(self.config.cohere_api_key))
        self.pinecone_client = Pinecone(api_key=sanitize_api_key(self.config.pinecone_api_key))
        self.index_name = self.config.pinecone_index_name
        self.index = None
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_dim = 0

    def _load_text_from_pdf(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        reader = PyPDF2.PdfReader(str(file_path))
        text_parts: List[str] = []
        metadata = {
            "filename": file_path.name,
            "pages": len(reader.pages),
            "source_type": "pdf",
        }
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"\n[Page {page_num}]\n{page_text}")
        text = "\n".join(text_parts)
        return clean_text(text), metadata

    def _load_text_from_txt(self, file_path: Path) -> Tuple[str, Dict[str, Any]]:
        text = safe_read_text(file_path)
        metadata = {
            "filename": file_path.name,
            "pages": 1,
            "source_type": "txt",
        }
        return clean_text(text), metadata

    def load_documents(self, uploaded_files: List[Path]) -> Tuple[List[Dict[str, Any]], int, int]:
        all_documents: List[Dict[str, Any]] = []
        total_chars = 0
        total_pages = 0

        for file_path in uploaded_files:
            try:
                if file_path.suffix.lower() == ".pdf":
                    text, metadata = self._load_text_from_pdf(file_path)
                elif file_path.suffix.lower() == ".txt":
                    text, metadata = self._load_text_from_txt(file_path)
                else:
                    logger.warning("Unsupported file type: %s", file_path.name)
                    continue

                if not text.strip():
                    logger.warning("Empty document after extraction: %s", file_path.name)
                    continue

                chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
                for chunk in chunks:
                    all_documents.append({
                        "text": chunk,
                        "source": file_path.name,
                        "metadata": metadata,
                    })
                total_chars += len(text)
                total_pages += metadata.get("pages", 0)
                logger.info("Loaded document: %s | chars=%d | chunks=%d", file_path.name, len(text), len(chunks))
            except Exception as exc:
                logger.exception("Failed to load %s: %s", file_path.name, exc)

        self.chunks = [item["text"] for item in all_documents]
        self.metadata = [item["metadata"] for item in all_documents]
        return all_documents, total_chars, total_pages

    def build_index(self, documents: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        if not documents:
            raise ValueError("No valid documents available to build index")
        if not self.config.cohere_api_key or not self.config.pinecone_api_key:
            raise ValueError("Cohere and Pinecone API keys must be configured")

        texts = [doc["text"] for doc in documents]
        response = self.client.embed(
            model=self.config.embedding_model,
            texts=texts,
            input_type="search_document",
        )
        embeddings = response.embeddings
        self.embedding_dim = len(embeddings[0])

        if self.index_name not in [index.name for index in self.pinecone_client.list_indexes()]:
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": self.config.pinecone_environment or "us-east-1"}},
            )

        self.index = self.pinecone_client.Index(self.index_name)
        vectors = []
        for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc_metadata = documents[idx].get("metadata", {}) or {}
            flattened_metadata = {
                "chunk_text": text,
                "source": documents[idx].get("source", "unknown"),
                "filename": doc_metadata.get("filename", "unknown"),
                "source_type": doc_metadata.get("source_type", "unknown"),
                "pages": str(doc_metadata.get("pages", 0)),
            }
            vectors.append({
                "id": f"chunk-{idx}",
                "values": embedding,
                "metadata": flattened_metadata,
            })

        self.index.upsert(vectors=vectors)

        logger.info("Pinecone index created/updated: %s", self.index_name)
        logger.info("Embedding dimension: %d", self.embedding_dim)
        return {"chunks": texts, "metadata": [doc["metadata"] for doc in documents]}, 0.0

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            self.index = self.pinecone_client.Index(self.index_name)

        response = self.client.embed(
            model=self.config.embedding_model,
            texts=[query],
            input_type="search_query",
        )
        query_embedding = response.embeddings[0]
        start = time.time()
        result = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        retrieval_time = time.time() - start

        results: List[Dict[str, Any]] = []
        for match in result.get("matches", []):
            metadata = match.get("metadata", {})
            results.append({
                "chunk": metadata.get("chunk_text", ""),
                "score": float(match.get("score", 0.0)),
                "metadata": {
                    "filename": metadata.get("filename", "unknown"),
                    "source_type": metadata.get("source_type", "unknown"),
                    "pages": metadata.get("pages", "0"),
                },
                "retrieval_time": retrieval_time,
            })
        logger.info("Similarity search completed in %.4fs", retrieval_time)
        return results
