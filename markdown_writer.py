"""Markdown Writer - Generazione e editing di file Markdown su richiesta."""

import logging
from pathlib import Path
from datetime import datetime

from utils import setup_logging, ensure_directory

logger = setup_logging("rag.markdown")


class MarkdownWriter:
    """Crea e modifica file Markdown su richiesta dell'utente."""

    def __init__(self, output_path: str = "./output"):
        self.output_path = ensure_directory(output_path)
        logger.info(f"MarkdownWriter: output in {self.output_path}")

    def create(self, filename: str, content: str, title: str = None) -> Path:
        """Crea nuovo file Markdown"""
        if not filename.endswith('.md'):
            filename += '.md'

        filepath = self.output_path / filename

        lines = []
        if title:
            lines.append(f"# {title}\n")
        lines.append(content)
        lines.append(f"\n\n---\n*Generato il {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        filepath.write_text('\n'.join(lines), encoding='utf-8')
        logger.info(f"Creato: {filepath}")
        return filepath

    def append(self, filepath: str, content: str) -> Path:
        """Appende contenuto a file Markdown esistente"""
        path = Path(filepath)
        if not path.exists():
            path = self.output_path / filepath

        if not path.exists():
            return self.create(path.name, content)

        existing = path.read_text(encoding='utf-8')
        path.write_text(existing.rstrip() + '\n\n' + content + '\n', encoding='utf-8')
        logger.info(f"Aggiornato: {path}")
        return path

    def edit_with_llm(self, filepath: str, instructions: str, llm) -> Path:
        """Modifica file Markdown basandosi su istruzioni in linguaggio naturale.

        Args:
            filepath: Path del file da modificare
            instructions: Istruzioni in linguaggio naturale (es. "aggiungi una sezione su X")
            llm: Istanza LocalLLM per generare le modifiche
        """
        path = Path(filepath)
        if not path.exists():
            path = self.output_path / filepath

        if not path.exists():
            logger.warning(f"File non trovato: {path}")
            return None

        current_content = path.read_text(encoding='utf-8')
        prompt = f"""Ecco il contenuto attuale di un file Markdown:

---
{current_content}
---

Istruzioni di modifica: {instructions}

Rispondi SOLO con il file Markdown completo modificato, senza spiegazioni o commenti aggiuntivi."""

        new_content = llm.generate(prompt)

        # Pulisci eventuale markdown wrapping dalla risposta LLM
        if new_content.startswith('```markdown'):
            new_content = new_content[len('```markdown'):].strip()
        if new_content.startswith('```'):
            new_content = new_content[3:].strip()
        if new_content.endswith('```'):
            new_content = new_content[:-3].strip()

        path.write_text(new_content + '\n', encoding='utf-8')
        logger.info(f"Modificato con LLM: {path}")
        return path

    def list_files(self) -> list[Path]:
        """Lista tutti i file .md nella directory output"""
        return sorted(self.output_path.glob('*.md'))

    def read(self, filepath: str) -> str:
        """Legge contenuto di un file Markdown"""
        path = Path(filepath)
        if not path.exists():
            path = self.output_path / filepath
        if path.exists():
            return path.read_text(encoding='utf-8')
        return ""
