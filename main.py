"""RAG System - Entry Point Unico

Uso:
    python main.py                  # Avvia chat (default)
    python main.py chat             # Avvia chat interattiva
    python main.py ingest           # Indicizza documenti da ./data
    python main.py ingest --clean   # Re-indicizza da zero
    python main.py search "query"   # Ricerca veloce da CLI
    python main.py status           # Mostra statistiche sistema
    python main.py sources          # Lista documenti indicizzati
"""

import sys
from utils import load_config


def main():
    config = load_config()
    command = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if command == "chat":
        from chat import Chat
        Chat(config).run()

    elif command == "ingest":
        from ingest import Ingester
        Ingester(config).run(clean="--clean" in sys.argv)

    elif command == "search":
        if len(sys.argv) < 3:
            print("Uso: python main.py search \"la tua query\"")
            sys.exit(1)
        from search import SearchEngine
        query = " ".join(sys.argv[2:])
        engine = SearchEngine(config)
        print(engine.search_formatted(query))

    elif command == "status":
        from search import SearchEngine
        engine = SearchEngine(config)
        print(engine.get_stats())

    elif command == "sources":
        from search import SearchEngine
        engine = SearchEngine(config)
        sources = engine.list_sources()
        if sources:
            for s in sources:
                print(f"  {s}")
        else:
            print("Nessun documento indicizzato. Esegui: python main.py ingest")

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
