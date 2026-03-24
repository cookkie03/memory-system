"""Memory Manager - Integrazione mem0 per memoria conversazionale persistente.

Il sistema decide autonomamente quando usare RAG (documenti) vs mem0 (memoria):
- RAG: informazioni dai documenti caricati
- mem0: preferenze utente, contesto conversazioni passate, fatti appresi

mem0 usa storage locale (Qdrant) e LLM locale (Ollama) - zero cloud.
"""

import logging
from utils import setup_logging

logger = setup_logging("rag.memory")


class MemoryManager:
    """Gestione memoria persistente via mem0."""

    def __init__(self, config: dict):
        memory_cfg = config.get('memory', {})
        self.enabled = memory_cfg.get('enabled', False)
        self.user_id = memory_cfg.get('user_id', 'default')

        if not self.enabled:
            logger.info("Memoria disabilitata in config")
            self.mem0 = None
            return

        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError("pip install mem0ai")

        mem0_config = memory_cfg.get('mem0_config', {})
        self.mem0 = Memory.from_config(mem0_config)
        logger.info(f"mem0 inizializzato (user_id: {self.user_id})")

    def add(self, text: str, metadata: dict = None) -> None:
        """Salva informazione in memoria"""
        if not self.mem0:
            return
        try:
            kwargs = {"user_id": self.user_id}
            if metadata:
                kwargs["metadata"] = metadata
            self.mem0.add(text, **kwargs)
            logger.debug(f"Memoria aggiunta: {text[:80]}...")
        except Exception as e:
            logger.error(f"Errore salvataggio memoria: {e}")

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Cerca nella memoria"""
        if not self.mem0:
            return []
        try:
            results = self.mem0.search(query, user_id=self.user_id, limit=limit)
            memories = []
            if isinstance(results, dict) and 'results' in results:
                results = results['results']
            for r in results:
                if isinstance(r, dict):
                    memories.append({
                        'text': r.get('memory', r.get('text', str(r))),
                        'score': r.get('score', 0),
                        'metadata': r.get('metadata', {}),
                    })
                else:
                    memories.append({'text': str(r), 'score': 0, 'metadata': {}})
            return memories
        except Exception as e:
            logger.error(f"Errore ricerca memoria: {e}")
            return []

    def get_all(self, limit: int = 50) -> list[dict]:
        """Restituisce tutte le memorie dell'utente"""
        if not self.mem0:
            return []
        try:
            results = self.mem0.get_all(user_id=self.user_id, limit=limit)
            if isinstance(results, dict) and 'results' in results:
                results = results['results']
            return [
                {'text': r.get('memory', r.get('text', str(r))), 'metadata': r.get('metadata', {})}
                for r in results
            ] if results else []
        except Exception as e:
            logger.error(f"Errore get_all memoria: {e}")
            return []

    def clear(self) -> bool:
        """Cancella tutta la memoria dell'utente"""
        if not self.mem0:
            return False
        try:
            self.mem0.delete_all(user_id=self.user_id)
            logger.info(f"Memoria cancellata per user_id: {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Errore cancellazione memoria: {e}")
            return False

    def format_context(self, memories: list[dict]) -> str:
        """Formatta memorie come contesto per il LLM"""
        if not memories:
            return ""
        lines = ["MEMORIA (conversazioni/fatti precedenti):"]
        for i, m in enumerate(memories, 1):
            lines.append(f"  [{i}] {m['text']}")
        return '\n'.join(lines)
