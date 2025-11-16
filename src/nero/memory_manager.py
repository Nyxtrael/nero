"""Conversation memory manager backed by SQLite and embeddings."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, config: dict):
        self.db_path = Path(config.get("database_path", "memory.db"))
        self.results_limit = config.get("results_limit", 5)
        self.score_threshold = config.get("score_threshold", 0.5)
        self._connection = sqlite3.connect(self.db_path)
        self._ensure_schema()
        model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)

    def _ensure_schema(self) -> None:
        cursor = self._connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT,
                vector TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        self._connection.commit()

    def save_memory(self, text: str, metadata: Optional[dict] = None) -> None:
        embedding = self._embed(text)
        cursor = self._connection.cursor()
        cursor.execute(
            "INSERT INTO memories (text, metadata, vector, created_at) VALUES (?, ?, ?, ?)",
            (text, json.dumps(metadata or {}), json.dumps(embedding.tolist()), time.time()),
        )
        self._connection.commit()
        LOGGER.debug("Saved memory: %s", text)

    def search_memories(self, query: str, limit: Optional[int] = None) -> list[dict]:
        limit = limit or self.results_limit
        query_vec = self._embed(query)
        cursor = self._connection.cursor()
        cursor.execute("SELECT text, metadata, vector, created_at FROM memories")
        rows = cursor.fetchall()
        results = []
        for text, metadata, vector, created_at in rows:
            stored_vec = np.array(json.loads(vector))
            score = self._cosine_similarity(query_vec, stored_vec)
            if score >= self.score_threshold:
                results.append(
                    {
                        "text": text,
                        "metadata": json.loads(metadata),
                        "score": float(score),
                        "created_at": created_at,
                    }
                )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.embedder.encode(text, normalize_embeddings=True))

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    def close(self) -> None:
        self._connection.close()