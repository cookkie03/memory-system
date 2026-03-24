"""Chat - Interfaccia conversazionale con Tool Calling nativo.

Il LLM (via API OpenAI-compatible) decide autonomamente quali tool usare:
- search_documents: RAG search nei documenti
- search_memory: Cerca nella memoria mem0 (conversazioni + documenti)
- save_memory: Salva in memoria mem0
- create_file: Crea file markdown
- edit_file: Modifica file markdown

Supporta chat ricorsiva con contesto conversazione persistente.
"""

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from llm import LLMClient
from search import SearchEngine
from memory import MemoryManager
from markdown_writer import MarkdownWriter
from utils import load_config

console = Console()


class Chat:
    def __init__(self, config: dict):
        self.config = config
        console.print("[dim]Inizializzazione sistema...[/dim]")

        self.llm = LLMClient(config)
        self.search = SearchEngine(config)
        self.memory = MemoryManager(config)
        self.writer = MarkdownWriter(config.get('output_path', './output'))

        self.messages = []  # Storico conversazione
        self.max_history = 20

        # Tool handlers: il LLM chiama questi tramite tool calling
        self.tool_handlers = {
            "search_documents": self._tool_search_documents,
            "search_memory": self._tool_search_memory,
            "save_memory": self._tool_save_memory,
            "create_file": self._tool_create_file,
            "edit_file": self._tool_edit_file,
        }

        console.print("[green]Sistema pronto.[/green]")

    # === Tool Implementations ===

    def _tool_search_documents(self, query: str) -> str:
        """RAG search nei documenti indicizzati"""
        results = self.search.search(query)
        if not results:
            return "Nessun documento rilevante trovato."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] (fonte: {r['source_path']}, score: {r['score']})\n{r['text']}")
        return '\n\n'.join(lines)

    def _tool_search_memory(self, query: str) -> str:
        """Cerca nella memoria mem0 (conversazioni + documenti)"""
        memories = self.memory.search(query)
        if not memories:
            return "Nessuna memoria rilevante trovata."
        lines = []
        for i, m in enumerate(memories, 1):
            lines.append(f"[{i}] {m['text']}")
        return '\n'.join(lines)

    def _tool_save_memory(self, text: str) -> str:
        """Salva informazione in memoria"""
        self.memory.add(text)
        return f"Salvato in memoria: {text[:100]}..."

    def _tool_create_file(self, filename: str, content: str) -> str:
        """Crea file markdown"""
        path = self.writer.create(filename, content, title=filename.replace('_', ' ').title())
        return f"File creato: {path}"

    def _tool_edit_file(self, filename: str, instructions: str) -> str:
        """Modifica file markdown con LLM"""
        path = self.writer.edit_with_llm(filename, instructions, self.llm)
        if path:
            return f"File modificato: {path}"
        return f"File '{filename}' non trovato."

    # === Comandi Slash ===

    def _handle_command(self, cmd: str):
        """Gestisce comandi /slash. Ritorna None per /exit, True se gestito, False altrimenti."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command in ('/exit', '/quit'):
            return None

        if command == '/help':
            console.print(Panel(
                "/exit       - Esci\n"
                "/stats      - Statistiche sistema\n"
                "/sources    - Lista documenti indicizzati\n"
                "/save <f>   - Salva conversazione in markdown\n"
                "/files      - Lista file markdown generati\n"
                "/memory     - Mostra memorie salvate\n"
                "/clear      - Reset contesto conversazione\n"
                "/clearmem   - Cancella tutta la memoria",
                title="Comandi"
            ))
            return True

        if command == '/stats':
            console.print(f"[dim]{self.search.get_stats()}[/dim]")
            return True

        if command == '/sources':
            sources = self.search.list_sources()
            for s in (sources or ["Nessun documento indicizzato"]):
                console.print(f"  [dim]{s}[/dim]")
            return True

        if command == '/save':
            filename = arg or "conversazione"
            content = self._format_conversation()
            path = self.writer.create(filename, content, title="Conversazione RAG")
            console.print(f"[green]Salvato: {path}[/green]")
            return True

        if command == '/files':
            files = self.writer.list_files()
            for f in (files or []):
                console.print(f"  [dim]{f.name}[/dim]")
            if not files:
                console.print("[dim]Nessun file generato[/dim]")
            return True

        if command == '/memory':
            memories = self.memory.get_all()
            for i, m in enumerate(memories or [], 1):
                console.print(f"  [dim][{i}] {m['text']}[/dim]")
            if not memories:
                console.print("[dim]Nessuna memoria[/dim]")
            return True

        if command == '/clear':
            self.messages = []
            console.print("[yellow]Contesto conversazione resettato[/yellow]")
            return True

        if command == '/clearmem':
            if self.memory.clear():
                console.print("[yellow]Memoria cancellata[/yellow]")
            else:
                console.print("[red]Errore cancellazione[/red]")
            return True

        return False

    def _format_conversation(self) -> str:
        lines = []
        for msg in self.messages:
            if msg['role'] == 'user':
                lines.append(f"**Utente:** {msg['content']}\n")
            elif msg['role'] == 'assistant':
                lines.append(f"**Assistente:** {msg['content']}\n")
        return '\n'.join(lines)

    # === Chat Loop ===

    def process(self, user_input: str) -> str:
        """Processa input utente con tool calling.

        Il LLM decide autonomamente quali tool usare:
        1. Riceve il messaggio utente + storico conversazione
        2. Decide se chiamare tool (search, memory, markdown) o rispondere direttamente
        3. Se chiama tool, riceve i risultati e genera risposta finale
        """
        self.messages.append({"role": "user", "content": user_input})

        # Invia storico conversazione (max ultimi N messaggi)
        chat_messages = self.messages[-self.max_history:]

        # Tool calling loop: il LLM decide cosa fare
        response = self.llm.chat_with_tools(chat_messages, self.tool_handlers)

        self.messages.append({"role": "assistant", "content": response})

        # Salva contesto conversazione in mem0 (per memoria a lungo termine)
        self.memory.add(
            f"Utente: {user_input}\nAssistente: {response[:300]}",
            metadata={"type": "conversation"}
        )

        return response

    def run(self):
        """REPL loop - Chat interattiva"""
        console.print(Panel(
            "RAG Chat con Tool Calling - Il modello sceglie autonomamente i tool\n"
            "Scrivi una domanda o /help per i comandi",
            title="RAG System", border_style="cyan"
        ))

        while True:
            try:
                user_input = console.input("[bold]> [/bold]").strip()
                if not user_input:
                    continue

                if user_input.startswith('/'):
                    result = self._handle_command(user_input)
                    if result is None:
                        console.print("[dim]Ciao![/dim]")
                        break
                    if result:
                        continue

                response = self.process(user_input)
                console.print()
                console.print(Markdown(response))
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Ciao![/dim]")
                break
            except Exception as e:
                console.print(f"[red]Errore: {e}[/red]")


if __name__ == "__main__":
    Chat(load_config()).run()
