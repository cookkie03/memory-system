"""Document Ingestion - Indicizzazione con dual embedding + mem0.

Flusso per ogni documento:
1. Estrazione testo (dots.ocr per PDF/immagini, Whisper per audio, ecc.)
2. Chunking semantico con tracciamento posizioni
3. Embedding (Jina v4 o Nemotron, in base a config)
4. Upsert in Qdrant (vettore + testo nel payload) per RAG
5. Aggiunta chunk a mem0 (knowledge base condivisa)
"""

import sys
import uuid
import time
import json
import hashlib
from pathlib import Path

from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

from utils import load_config, ensure_directory, get_supported_extensions, get_file_category, get_active_embedding_config
from extractors import extract_text
from memory import MemoryManager


def _map_protected_to_original(pos: int, protected: str, dot_placeholder: str) -> int:
    count = protected[:pos].count(dot_placeholder)
    return pos - count * (len(dot_placeholder) - 1)


console = Console()
REGISTRY_DIR = Path(__file__).parent / ".cache"


class Ingester:
    def __init__(self, config):
        self.config = config
        self.docs_path = ensure_directory(config['documents_path'])
        self.stats = {'processed': 0, 'chunks': 0, 'deleted': 0, 'skipped': 0, 'errors': []}

        qdrant_cfg = config.get('qdrant', {})
        self.collection_name = qdrant_cfg.get('collection', 'documents')
        self.min_text_length = config.get('min_text_length', 50)
        self.hash_buffer_size = config.get('hash_buffer_size', 8192)
        self.all_extensions = get_supported_extensions(config)

        # Embedding model
        emb_cfg = get_active_embedding_config(config)
        console.print(f"[yellow]Caricamento embedding: {emb_cfg.get('model', '?')}[/yellow]")
        model_kwargs = {"trust_remote_code": emb_cfg.get('trust_remote_code', False)}
        self.model = SentenceTransformer(emb_cfg['model'], **model_kwargs)
        self.embedding_dim = emb_cfg.get('dimension', 1024)
        self.embedding_task = emb_cfg.get('task_passage')

        # Qdrant
        mode = qdrant_cfg.get('mode', 'http')
        if mode == 'http':
            self.qdrant = QdrantClient(
                host=qdrant_cfg.get('host', 'localhost'),
                port=qdrant_cfg.get('port', 6333)
            )
        else:
            self.qdrant = QdrantClient(path=qdrant_cfg.get('local_path', './qdrant_storage'))

        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in existing:
            self.qdrant.create_collection(
                self.collection_name,
                VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )

        # mem0 - knowledge base condivisa
        self.memory = MemoryManager(config)

        # Registry
        REGISTRY_DIR.mkdir(exist_ok=True)
        self.registry_file = REGISTRY_DIR / "registry.json"
        self.registry = json.loads(self.registry_file.read_text()) if self.registry_file.exists() else {}

    def _hash(self, path):
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(self.hash_buffer_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _delete_vectors(self, source_path):
        try:
            self.qdrant.delete(
                self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))])
            )
        except Exception:
            pass

    def _chunk_text(self, text, size=None, overlap=None):
        """Chunking semantico con protezione abbreviazioni e tracciamento posizioni."""
        import re

        size = size or self.config.get('chunk_size', 1024)
        overlap = overlap or self.config.get('chunk_overlap', 200)
        mode = self.config.get('chunking_mode', 'sentence')

        if mode == 'character':
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    real_start = text.find(chunk, start)
                    real_end = real_start + len(chunk)
                    chunks.append((chunk, real_start, real_end))
                start += size - overlap
            return chunks

        DOT = '\x00DOT\x00'
        protected = text

        protected = re.sub(r'https?://[^\s]+', lambda m: m.group(0).replace('.', DOT), protected)
        protected = re.sub(r'[\w.-]+@[\w.-]+\.\w+', lambda m: m.group(0).replace('.', DOT), protected)
        protected = re.sub(r'(\d)\.(\d)', rf'\1{DOT}\2', protected)
        protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Dott|Ing|Avv|Sig|Sig\.ra)\.', rf'\1{DOT}', protected, flags=re.IGNORECASE)
        protected = re.sub(r'\b(Fig|Tab|Eq|Vol|No|Ch|Sec|App|Ref|Rev)\.', rf'\1{DOT}', protected, flags=re.IGNORECASE)
        protected = re.sub(r'\b(vs|etc|al|Jr|Sr|Inc|Ltd|Corp|Co)\.', rf'\1{DOT}', protected, flags=re.IGNORECASE)
        protected = re.sub(r'\b(i\.e|e\.g|et al|cf|ibid|op\.cit|viz|approx|ca)\.?',
                          lambda m: m.group(0).replace('.', DOT), protected, flags=re.IGNORECASE)
        protected = re.sub(r'(^|\n)(\d+)\.(\s)', rf'\1\2{DOT}\3', protected)
        protected = re.sub(r'\b([A-Z]\.){2,}', lambda m: m.group(0).replace('.', DOT), protected)
        protected = re.sub(r'(\d{4})\.\s*(?=[A-Z])', rf'\1{DOT} ', protected)
        protected = re.sub(r'\b(pp|vol|no|art)\.', rf'\1{DOT}', protected, flags=re.IGNORECASE)
        protected = re.sub(r'\b(Eq|Tab|Fig|Sec)\.?\s*\(?\d',
                          lambda m: m.group(0).replace('.', DOT), protected, flags=re.IGNORECASE)
        protected = re.sub(r'(\d)\.(\d{2,})\b', rf'\1{DOT}\2', protected)

        sentence_pattern = re.compile(r'(?<=[.!?])\s+|\n{2,}')
        sentences_with_pos = []
        last_end = 0

        for match in sentence_pattern.finditer(protected):
            sentence = protected[last_end:match.start()].strip()
            if sentence:
                clean_sentence = sentence.replace(DOT, '.')
                orig_start = _map_protected_to_original(last_end, protected, DOT)
                orig_end = _map_protected_to_original(match.start(), protected, DOT)
                sentences_with_pos.append((clean_sentence, orig_start, orig_end))
            last_end = match.end()

        if last_end < len(protected):
            sentence = protected[last_end:].strip()
            if sentence:
                clean_sentence = sentence.replace(DOT, '.')
                orig_start = _map_protected_to_original(last_end, protected, DOT)
                orig_end = _map_protected_to_original(len(protected), protected, DOT)
                sentences_with_pos.append((clean_sentence, orig_start, orig_end))

        if not sentences_with_pos:
            return []

        def split_long(segment, start_offset, max_size):
            if len(segment) <= max_size:
                return [(segment, start_offset, start_offset + len(segment))]
            result = []
            parts = re.split(r'(?<=[;])\s*', segment)
            if len(parts) > 1:
                offset = start_offset
                for part in parts:
                    part = part.strip()
                    if part:
                        result.extend(split_long(part, offset, max_size))
                    offset += len(part) + 1
                if result:
                    return [r for r in result if r[0]]

            parts = re.split(r'(?<=[,])\s*', segment)
            if len(parts) > 1:
                current = ""
                chunk_start = start_offset
                for part in parts:
                    if len(current) + len(part) + 1 <= max_size:
                        current = (current + " " + part).strip() if current else part
                    else:
                        if current:
                            result.append((current, chunk_start, chunk_start + len(current)))
                        chunk_start += len(current) + 1
                        current = part
                if current:
                    result.append((current, chunk_start, chunk_start + len(current)))
                if result:
                    return [r for r in result if r[0]]

            words = segment.split()
            current = ""
            chunk_start = start_offset
            for word in words:
                if len(current) + len(word) + 1 <= max_size:
                    current = (current + " " + word).strip() if current else word
                else:
                    if current:
                        result.append((current, chunk_start, chunk_start + len(current)))
                    chunk_start += len(current) + 1
                    current = word
            if current:
                result.append((current, chunk_start, chunk_start + len(current)))

            final = []
            for text_part, s, e in result:
                if len(text_part) > max_size:
                    for i in range(0, len(text_part), max_size):
                        chunk = text_part[i:i+max_size]
                        final.append((chunk, s + i, s + i + len(chunk)))
                else:
                    final.append((text_part, s, e))
            return [r for r in final if r[0]]

        chunks = []
        current_chunk = []
        current_len = 0

        for sentence, sent_start, sent_end in sentences_with_pos:
            sent_len = len(sentence)
            if sent_len > size:
                if current_chunk:
                    chunk_text = ' '.join([c[0] for c in current_chunk])
                    chunks.append((chunk_text, current_chunk[0][1], current_chunk[-1][2]))
                    current_chunk = []
                    current_len = 0
                chunks.extend(split_long(sentence, sent_start, size))
                continue

            if current_len + sent_len + 1 > size and current_chunk:
                chunk_text = ' '.join([c[0] for c in current_chunk])
                chunks.append((chunk_text, current_chunk[0][1], current_chunk[-1][2]))
                overlap_chunk = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item[0]) + 1 <= overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += len(item[0]) + 1
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = overlap_len

            current_chunk.append((sentence, sent_start, sent_end))
            current_len += sent_len + 1

        if current_chunk:
            chunk_text = ' '.join([c[0] for c in current_chunk])
            chunks.append((chunk_text, current_chunk[0][1], current_chunk[-1][2]))

        return chunks

    def _ingest(self, path):
        _t0 = time.time()
        console.print(f"[dim]  → {path.name}[/dim]")

        if path.suffix.lower() not in self.all_extensions:
            return 0, f"Formato non supportato: {path.suffix}"

        try:
            source_path = str(path.relative_to(self.docs_path))
        except ValueError:
            source_path = path.name

        self._delete_vectors(source_path)

        text, err = extract_text(path, self.config)
        if err:
            return 0, f"Estrazione: {err}"
        if not text or len(text) < self.min_text_length:
            return 0, "File troppo corto"

        console.print(f"[dim]    Estratti {len(text):,} char ({time.time()-_t0:.1f}s)[/dim]")

        chunks = self._chunk_text(text)
        if not chunks:
            return 0, "Nessun chunk generato"

        chunk_texts = [c[0] for c in chunks]

        try:
            encode_kwargs = {"show_progress_bar": False}
            if self.embedding_task:
                encode_kwargs["task"] = self.embedding_task
            raw_embeddings = self.model.encode(chunk_texts, **encode_kwargs)
            vectors = [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in raw_embeddings]
        except Exception as e:
            return 0, f"Embedding: {e}"

        if len(vectors) != len(chunks):
            return 0, f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"

        # Upsert in Qdrant (RAG)
        points = []
        for i, (chunk_data, vector) in enumerate(zip(chunks, vectors)):
            if len(vector) != self.embedding_dim:
                continue
            chunk_text, char_start, char_end = chunk_data
            if char_start < 0 or char_end <= char_start or char_end > len(text):
                char_start, char_end = -1, -1

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    'text': chunk_text,
                    'source': path.name,
                    'source_path': source_path,
                    'chunk_id': i,
                    'char_start': char_start,
                    'char_end': char_end,
                    'category': get_file_category(str(path), self.config),
                }
            ))

        if not points:
            return 0, "Nessun punto valido"

        try:
            self.qdrant.upsert(self.collection_name, points)
        except Exception as e:
            return 0, f"Upsert: {e}"

        # Aggiungi chunk a mem0 (knowledge base condivisa)
        for chunk_text, _, _ in chunks:
            self.memory.add(
                chunk_text,
                metadata={"source": source_path, "type": "document"}
            )

        console.print(f"[green]  ✓ {path.name}: {len(points)} chunks ({time.time()-_t0:.1f}s)[/green]")
        return len(points), None

    def run(self, clean=False):
        start = time.time()

        if clean:
            try:
                self.qdrant.delete_collection(self.collection_name)
            except Exception:
                pass
            self.qdrant.create_collection(
                self.collection_name,
                VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
            self.registry = {}
            self.memory.clear()
            console.print("[yellow]Storage + memoria puliti[/yellow]")

        files = {}
        for p in Path(self.docs_path).rglob("*"):
            if p.is_file() and p.suffix.lower() in self.all_extensions:
                key = str(p.relative_to(self.docs_path))
                files[key] = p

        new, mod, deleted = [], [], []
        for k in self.registry:
            if k not in files:
                deleted.append(k)
        for key, path in files.items():
            h = self._hash(path)
            if key not in self.registry:
                new.append((key, path, h))
            elif self.registry[key] != h:
                mod.append((key, path, h))

        console.print(Panel(
            f"File: {len(files)} | Nuovi: {len(new)} | Modificati: {len(mod)} | Eliminati: {len(deleted)}",
            title="Analisi"
        ))

        if not (new or mod or deleted):
            console.print("[green]Tutto aggiornato![/green]")
            return

        with Progress() as prog:
            if deleted:
                task = prog.add_task("[red]Eliminazione...", total=len(deleted))
                for key in deleted:
                    self._delete_vectors(key)
                    del self.registry[key]
                    self.stats['deleted'] += 1
                    prog.advance(task)

            to_do = new + mod
            if to_do:
                task = prog.add_task("[green]Ingestione...", total=len(to_do))
                for key, path, h in to_do:
                    n, err = self._ingest(path)
                    if err:
                        self.stats['errors'].append(f"{path.name}: {err}")
                        self.stats['skipped'] += 1
                    else:
                        self.registry[key] = h
                        self.stats['processed'] += 1
                        self.stats['chunks'] += n
                    prog.advance(task)

        self.registry_file.write_text(json.dumps(self.registry))

        elapsed = int(time.time() - start)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
        tempo = f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")
        console.print(Panel(
            f"Processati: {self.stats['processed']} | Chunks: {self.stats['chunks']} | "
            f"Skipped: {self.stats['skipped']} | Tempo: {tempo}",
            title="Completato", border_style="green"
        ))

        if self.stats['errors']:
            console.print("\n[bold red]Errori:[/bold red]")
            for e in self.stats['errors']:
                console.print(f"  • {e}")


if __name__ == "__main__":
    Ingester(load_config()).run(clean="--clean" in sys.argv)
