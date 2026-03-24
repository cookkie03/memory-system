"""Search Engine - Ricerca semantica con reranking e threshold adattivo.

Logica estratta dal vecchio mcp_server.py, senza dipendenze MCP.
Usabile sia dalla chat interattiva sia da CLI/agenti AI.
"""

import time
import math
import logging
from typing import Optional
from functools import wraps

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from utils import setup_logging, get_active_embedding_config

logger = setup_logging("rag.search")

# Costanti
MAX_QUERY_LENGTH = 2000
MIN_QUERY_LENGTH = 3
MAX_LIMIT = 50


def _retry(max_attempts=2, delay=0.5):
    """Decorator retry per operazioni con errori transitori"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_error
        return wrapper
    return decorator


def _sigmoid(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _jaccard_ngram(t1: str, t2: str, n: int = 3) -> float:
    """N-gram Jaccard similarity per deduplicazione"""
    s1 = set(t1.lower()[i:i+n] for i in range(max(0, len(t1)-n+1)))
    s2 = set(t2.lower()[i:i+n] for i in range(max(0, len(t2)-n+1)))
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


class SearchEngine:
    """Motore di ricerca semantica con reranking e threshold adattivo."""

    def __init__(self, config: dict):
        self.config = config
        search_cfg = config.get('search', {})
        qdrant_cfg = config.get('qdrant', {})

        self.collection_name = qdrant_cfg.get('collection', 'documents')
        self.top_k = search_cfg.get('top_k', 10)
        self.similarity_threshold = search_cfg.get('similarity_threshold', 0.7)
        self.adaptive_threshold = search_cfg.get('adaptive_threshold', False)
        self.adaptive_min = search_cfg.get('adaptive_threshold_min', 0.5)
        self.adaptive_max = search_cfg.get('adaptive_threshold_max', 0.9)

        # Embedding model
        emb_cfg = get_active_embedding_config(config)
        logger.info(f"Caricamento embedding: {emb_cfg.get('model', '?')}")
        model_kwargs = {"trust_remote_code": emb_cfg.get('trust_remote_code', False)}
        self.model = SentenceTransformer(emb_cfg['model'], **model_kwargs)
        self.embedding_task_query = emb_cfg.get('task_query')

        # Qdrant
        mode = qdrant_cfg.get('mode', 'http')
        if mode == 'http':
            self.qdrant = QdrantClient(
                host=qdrant_cfg.get('host', 'localhost'),
                port=qdrant_cfg.get('port', 6333),
                timeout=30
            )
        else:
            self.qdrant = QdrantClient(path=qdrant_cfg.get('local_path', './qdrant_storage'))

        # Reranking (Jina Reranker v2)
        self.rerank_enabled = search_cfg.get('rerank_enabled', False)
        self.rerank_model = None
        self.rerank_top_n = search_cfg.get('rerank_top_n', 30)
        self.rerank_alpha = search_cfg.get('rerank_alpha', 0.4)

        if self.rerank_enabled:
            from sentence_transformers import CrossEncoder
            rerank_model_name = search_cfg.get('rerank_model', 'jinaai/jina-reranker-v2-base-multilingual')
            logger.info(f"Caricamento reranker: {rerank_model_name}")
            self.rerank_model = CrossEncoder(rerank_model_name, trust_remote_code=True)

        logger.info("SearchEngine pronto")

    def _validate(self, query: str, limit: int) -> Optional[str]:
        if not query or not isinstance(query, str):
            return "Query deve essere una stringa non vuota"
        q = query.strip()
        if len(q) < MIN_QUERY_LENGTH:
            return f"Query troppo corta (min {MIN_QUERY_LENGTH} char)"
        if len(q) > MAX_QUERY_LENGTH:
            return f"Query troppo lunga (max {MAX_QUERY_LENGTH} char)"
        if not isinstance(limit, int) or limit <= 0 or limit > MAX_LIMIT:
            return f"Limit deve essere intero 1-{MAX_LIMIT}"
        return None

    def _encode(self, query: str) -> list:
        kwargs = {"show_progress_bar": False}
        if self.embedding_task_query:
            kwargs["task"] = self.embedding_task_query
        vector = self.model.encode(query.strip(), **kwargs)
        return vector.tolist() if hasattr(vector, 'tolist') else list(vector)

    @_retry(max_attempts=2, delay=0.5)
    def _query_qdrant(self, vector: list, limit: int):
        return self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            timeout=30
        )

    def _deduplicate(self, results, get_score, threshold=0.85):
        if len(results) <= 1:
            return results
        sorted_results = sorted(results, key=get_score, reverse=True)
        unique = []
        for candidate in sorted_results:
            cand_text = candidate.payload.get('text', '')
            is_dup = any(
                _jaccard_ngram(cand_text, acc.payload.get('text', '')) >= threshold
                for acc in unique
            )
            if not is_dup:
                unique.append(candidate)
        return unique

    def _compute_threshold(self, scores: list[float]) -> float:
        if not self.adaptive_threshold or len(scores) < 3:
            return self.similarity_threshold

        sorted_scores = sorted(scores, reverse=True)
        best_gap = 0
        threshold_at_gap = self.similarity_threshold

        for i in range(len(sorted_scores) - 1):
            gap = sorted_scores[i] - sorted_scores[i + 1]
            if gap >= 0.08 and gap > best_gap:
                best_gap = gap
                threshold_at_gap = sorted_scores[i + 1] + 0.01

        if best_gap >= 0.08:
            return max(self.adaptive_min, min(self.adaptive_max, threshold_at_gap))

        top_half_avg = sum(sorted_scores[:len(sorted_scores)//2 + 1]) / (len(sorted_scores)//2 + 1)
        return max(self.adaptive_min, min(self.adaptive_max, top_half_avg - 0.12))

    def search(self, query: str, limit: int = None) -> list[dict]:
        """Ricerca semantica con reranking e threshold adattivo.

        Returns:
            Lista di dict con: text, source, source_path, score, chunk_id, char_start, char_end
        """
        limit = limit or self.top_k
        err = self._validate(query, limit)
        if err:
            logger.warning(f"Validazione: {err}")
            return []

        vector = self._encode(query)
        search_limit = self.rerank_top_n if self.rerank_enabled and self.rerank_model else limit
        results = self._query_qdrant(vector, search_limit)

        if not results.points:
            return []

        # Reranking
        if self.rerank_enabled and self.rerank_model:
            pairs = [(query.strip(), r.payload.get('text', '')) for r in results.points]
            rerank_logits = self.rerank_model.predict(pairs, show_progress_bar=False)
            alpha = self.rerank_alpha
            for i, result in enumerate(results.points):
                normalized = _sigmoid(float(rerank_logits[i]))
                result.rerank_score = alpha * result.score + (1 - alpha) * normalized
            results.points.sort(key=lambda x: x.rerank_score, reverse=True)
            results.points = results.points[:limit]

        # Score getter
        use_rerank = self.rerank_enabled and self.rerank_model and hasattr(results.points[0], 'rerank_score')
        get_score = (lambda r: r.rerank_score) if use_rerank else (lambda r: r.score)

        # Threshold
        scores = [get_score(r) for r in results.points]
        threshold = self._compute_threshold(scores)

        # Filter + dedup
        filtered = [r for r in results.points if get_score(r) >= threshold]
        filtered = self._deduplicate(filtered, get_score)

        if not filtered:
            return []

        # Format output
        output = []
        for r in filtered:
            output.append({
                'text': r.payload.get('text', '').strip(),
                'source': r.payload.get('source', '?'),
                'source_path': r.payload.get('source_path', r.payload.get('source', '?')),
                'score': round(get_score(r), 4),
                'chunk_id': r.payload.get('chunk_id', -1),
                'char_start': r.payload.get('char_start', -1),
                'char_end': r.payload.get('char_end', -1),
            })
        return output

    def search_formatted(self, query: str, limit: int = None) -> str:
        """Ricerca con output formattato per CLI/display"""
        results = self.search(query, limit)
        if not results:
            return "Nessun risultato rilevante trovato."

        lines = [f"Trovati {len(results)} risultati:"]
        for i, r in enumerate(results, 1):
            cite = f"char:{r['char_start']}-{r['char_end']}" if r['char_start'] >= 0 else f"chunk:{r['chunk_id']}"
            lines.append(f"\n[{i}] score:{r['score']} src:{r['source_path']} ({cite})\n{r['text']}")
        return '\n'.join(lines)

    def get_stats(self) -> str:
        """Statistiche del sistema"""
        try:
            info = self.qdrant.get_collection(self.collection_name)
            return (
                f"Collection: {self.collection_name}\n"
                f"Documenti: {info.points_count}\n"
                f"Dimensione vettori: {info.config.params.vectors.size}\n"
                f"Status: {info.status.name}\n"
                f"Reranking: {'attivo' if self.rerank_enabled else 'disattivo'}"
            )
        except Exception as e:
            return f"Errore: {e}"

    def list_sources(self) -> list[str]:
        """Lista file sorgente indicizzati"""
        try:
            sources = set()
            offset = None
            while True:
                result = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=["source_path"]
                )
                points, offset = result
                for p in points:
                    sources.add(p.payload.get('source_path', '?'))
                if offset is None:
                    break
            return sorted(sources)
        except Exception as e:
            logger.error(f"Errore list_sources: {e}")
            return []
