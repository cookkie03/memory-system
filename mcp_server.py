"""MCP Server - Espone il RAG system come server Model Context Protocol.

Permette a qualsiasi client MCP (Claude Desktop, Cursor, VS Code, ecc.)
di usare i tool del RAG system: ricerca documenti, memoria, file markdown.

Uso:
    python mcp_server.py                    # stdio (default, per Claude Desktop)
    python mcp_server.py --transport sse    # SSE (per accesso remoto)
    python mcp_server.py --transport sse --port 8080
"""

import os
import sys
from mcp.server.fastmcp import FastMCP

from utils import load_config
from search import SearchEngine
from memory import MemoryManager
from markdown_writer import MarkdownWriter

# === Init ===

config = load_config()
search_engine = SearchEngine(config)
memory_manager = MemoryManager(config)
writer = MarkdownWriter(config.get('output_path', './output'))

mcp = FastMCP(
    "RAG System",
    description="RAG system con ricerca semantica, memoria persistente e generazione markdown",
)


# === Tools ===

@mcp.tool()
def search_documents(query: str, limit: int = 10) -> str:
    """Cerca nei documenti indicizzati nella knowledge base.

    Usa questo tool per trovare informazioni nei documenti caricati.
    Restituisce i risultati piu' rilevanti con fonte e score.

    Args:
        query: La query di ricerca semantica
        limit: Numero massimo di risultati (default 10)
    """
    results = search_engine.search(query, limit=limit)
    if not results:
        return "Nessun documento rilevante trovato."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] (fonte: {r['source_path']}, score: {r['score']})\n{r['text']}"
        )
    return '\n\n'.join(lines)


@mcp.tool()
def search_memory(query: str) -> str:
    """Cerca nella memoria conversazionale e nella knowledge base condivisa.

    Usa questo tool per recuperare contesto da conversazioni passate,
    preferenze dell'utente, fatti appresi, o contenuti dei documenti.

    Args:
        query: La query di ricerca nella memoria
    """
    memories = memory_manager.search(query)
    if not memories:
        return "Nessuna memoria rilevante trovata."
    lines = []
    for i, m in enumerate(memories, 1):
        lines.append(f"[{i}] {m['text']}")
    return '\n'.join(lines)


@mcp.tool()
def save_memory(text: str) -> str:
    """Salva un'informazione importante nella memoria persistente.

    Usa quando l'utente condivide preferenze, fatti da ricordare,
    o contesto utile per conversazioni future.

    Args:
        text: L'informazione da salvare in memoria
    """
    memory_manager.add(text)
    return f"Salvato in memoria: {text[:100]}..."


@mcp.tool()
def create_file(filename: str, content: str) -> str:
    """Crea un nuovo file markdown nella directory output.

    Usa quando serve creare un riassunto, note, report, o qualsiasi
    documento strutturato in formato Markdown.

    Args:
        filename: Nome del file (senza estensione .md)
        content: Contenuto del file in formato Markdown
    """
    path = writer.create(filename, content, title=filename.replace('_', ' ').title())
    return f"File creato: {path}"


@mcp.tool()
def list_sources() -> str:
    """Lista tutti i documenti indicizzati nella knowledge base.

    Restituisce i nomi dei file sorgente che sono stati processati
    e sono disponibili per la ricerca.
    """
    sources = search_engine.list_sources()
    if not sources:
        return "Nessun documento indicizzato. Esegui prima l'ingestione."
    return '\n'.join(f"- {s}" for s in sources)


@mcp.tool()
def get_stats() -> str:
    """Mostra statistiche del sistema RAG.

    Restituisce informazioni sulla collection Qdrant:
    numero documenti, dimensione vettori, stato.
    """
    return search_engine.get_stats()


@mcp.tool()
def ingest_documents(clean: bool = False) -> str:
    """Avvia l'ingestione dei documenti dalla cartella data/.

    Scansiona la cartella data/, estrae testo, genera embedding,
    e indicizza i documenti in Qdrant per la ricerca.

    Args:
        clean: Se True, cancella tutto e re-indicizza da zero
    """
    from ingest import Ingester
    ingester = Ingester(config)
    ingester.run(clean=clean)
    stats = ingester.stats
    return (
        f"Ingestione completata: {stats['processed']} processati, "
        f"{stats['chunks']} chunks, {stats['skipped']} skippati, "
        f"{len(stats['errors'])} errori"
    )


# === Resources ===

@mcp.resource("rag://stats")
def resource_stats() -> str:
    """Statistiche correnti del sistema RAG"""
    return search_engine.get_stats()


@mcp.resource("rag://sources")
def resource_sources() -> str:
    """Lista documenti indicizzati"""
    sources = search_engine.list_sources()
    return '\n'.join(sources) if sources else "Nessun documento indicizzato"


# === Entry Point ===

def main():
    transport = "stdio"
    host = "0.0.0.0"
    port = 8000

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--transport" and i + 1 < len(args):
            transport = args[i + 1]
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        else:
            i += 1

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse", host=host, port=port)
    else:
        print(f"Transport non supportato: {transport}. Usa 'stdio' o 'sse'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
