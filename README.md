# RAG System - Local Multimodal RAG + Memory

Sistema RAG locale con embedding multimodali, memoria conversazionale (mem0) e tool calling nativo.

## Architettura

```
main.py (entry point)
  ├── chat.py       → Chat ricorsiva con tool calling (LLM sceglie i tool)
  │     ├── llm.py  → Client OpenAI-compatible (Ollama, LM Studio, vLLM, ecc.)
  │     ├── search.py → Ricerca semantica + reranking (Jina Reranker v2)
  │     ├── memory.py → mem0 (memoria conversazionale + knowledge base)
  │     └── markdown_writer.py → Generazione/editing file .md
  ├── ingest.py     → Indicizzazione documenti (RAG + mem0)
  │     └── extractors.py → dots.ocr (PDF/img), Whisper (audio), ecc.
  └── config.yaml   → Configurazione centralizzata unica
```

## Features

- **Tutto locale**: LLM, embedding, dati, memoria - zero dipendenze cloud
- **Tool Calling nativo**: il LLM decide autonomamente quando usare RAG, memoria, o creare file
- **Dual embedding multimodale**: Jina v4 (primario) + NVIDIA Nemotron (secondario)
- **mem0 Memory**: memoria persistente condivisa con la knowledge base RAG
- **dots.ocr**: unico strumento per PDF, immagini, documenti
- **Whisper turbo**: trascrizione audio
- **Reranking**: Jina Reranker v2 con threshold adattivo
- **Generazione Markdown**: crea/modifica file .md su richiesta
- **Config centralizzata**: un solo file `config.yaml`
- **Docker Compose**: avvio completo con un comando

## Quick Start

### 1. Avvio con Docker Compose

```bash
# Avvia Qdrant + Ollama + RAG
docker-compose up -d qdrant ollama

# Pull del modello LLM
docker exec -it mcp-rag-system-ollama-1 ollama pull qwen3:8b
docker exec -it mcp-rag-system-ollama-1 ollama pull nomic-embed-text

# Indicizza documenti (metti i file in ./data/)
docker-compose run rag ingest

# Avvia chat
docker-compose run -it rag chat
```

### 2. Avvio locale (senza Docker)

```bash
# Prerequisiti: Qdrant e Ollama in esecuzione
# Qdrant: docker run -p 6333:6333 qdrant/qdrant
# Ollama: ollama serve

# Setup Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Configura (modifica config.yaml se necessario)
# In locale, cambia qdrant.host a "localhost" e llm.base_url a "http://localhost:11434/v1"

# Indicizza documenti
python main.py ingest

# Chat
python main.py chat
```

## Uso

### Comandi CLI

```bash
python main.py chat             # Chat interattiva con tool calling
python main.py ingest           # Indicizza documenti da ./data/
python main.py ingest --clean   # Re-indicizza da zero
python main.py search "query"   # Ricerca veloce
python main.py status           # Statistiche sistema
python main.py sources          # Lista documenti indicizzati
```

### Comandi Chat

| Comando      | Descrizione |
|-------------|-------------|
| `/help`     | Mostra comandi |
| `/exit`     | Esci |
| `/stats`    | Statistiche sistema |
| `/sources`  | Lista documenti indicizzati |
| `/save <f>` | Salva conversazione in markdown |
| `/files`    | Lista file markdown generati |
| `/memory`   | Mostra memorie salvate |
| `/clear`    | Reset contesto conversazione |
| `/clearmem` | Cancella tutta la memoria |

### Tool Calling

Il LLM decide autonomamente quali tool usare ad ogni richiesta:

| Tool | Quando viene usato |
|------|-------------------|
| `search_documents` | Domande su documenti/contenuti caricati |
| `search_memory` | Contesto da conversazioni passate, preferenze, fatti appresi |
| `save_memory` | L'utente condivide info da ricordare |
| `create_file` | Richiesta di creare/scrivere un file |
| `edit_file` | Richiesta di modificare un file esistente |

## Configurazione

Tutto in `config.yaml`. Sezioni principali:

- **llm**: URL + API key per qualsiasi endpoint OpenAI-compatible
- **embeddings**: Dual model (Jina v4 / Nemotron), modalita' auto/primary/secondary
- **qdrant**: Vector database (Docker o locale)
- **search**: Top-k, threshold adattivo, reranking Jina v2
- **memory**: mem0 con Qdrant + Ollama locale
- **extensions**: Formati file supportati

## Stack Tecnologico

| Componente | Tecnologia |
|-----------|-----------|
| LLM | Qualsiasi OpenAI-compatible (Ollama, LM Studio, vLLM) |
| Embedding | Jina v4 (1024D) / NVIDIA Nemotron (4096D) |
| Reranking | Jina Reranker v2 Multilingual |
| Vector DB | Qdrant |
| Memoria | mem0 |
| OCR | dots.ocr (rednote-hilab) |
| Audio | Whisper turbo |
| Framework | DataPizza |

## Formati Supportati

| Categoria | Estensioni | Estrazione |
|-----------|-----------|-----------|
| Testo/Codice | .txt, .md, .py, .js, .ts, .json, .yaml, .html, .css, .csv, .java, .cpp, .go, .rb, .php, .sh, .sql | Lettura diretta |
| PDF | .pdf | dots.ocr |
| Immagini | .png, .jpg, .jpeg, .gif, .bmp, .tiff, .webp | dots.ocr |
| Audio | .mp3, .m4a, .opus, .wav, .ogg, .flac | Whisper turbo |
| Video | .mp4, .mkv, .avi, .mov, .webm | Whisper (audio) + dots.ocr (keyframes) |
| Notebook | .ipynb | Parsing JSON celle |
| Excel | .xlsx, .xls | openpyxl |
| Presentazioni | .pptx | python-pptx |
