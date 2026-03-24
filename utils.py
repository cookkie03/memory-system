"""Utility functions - Configurazione e helper condivisi"""

import logging
import yaml
from pathlib import Path


def load_config(config_path="config.yaml") -> dict:
    """Carica configurazione YAML centralizzata"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config non trovata: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(name: str = "rag", level: str = "INFO") -> logging.Logger:
    """Configura e restituisce logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def ensure_directory(path: str) -> Path:
    """Crea directory se non esiste, restituisce Path"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_supported_extensions(config: dict) -> set:
    """Ottiene tutte le estensioni supportate da config"""
    extensions = set()
    for category in config.get('extensions', {}).values():
        if isinstance(category, list):
            extensions.update(category)
    return extensions


def get_extensions_by_category(config: dict, category: str) -> set:
    """Ottiene estensioni per categoria specifica"""
    return set(config.get('extensions', {}).get(category, []))


def is_supported_file(file_path: str, config: dict) -> bool:
    """Verifica se il file e' supportato"""
    return Path(file_path).suffix.lower() in get_supported_extensions(config)


def get_file_category(file_path: str, config: dict) -> str:
    """Restituisce la categoria del file (text, pdf, audio, image, video, etc.)"""
    ext = Path(file_path).suffix.lower()
    for category, exts in config.get('extensions', {}).items():
        if isinstance(exts, list) and ext in exts:
            return category
    return "unknown"


def get_active_embedding_config(config: dict, file_category: str = None) -> dict:
    """Restituisce la config del modello embedding attivo.

    In modalita' 'auto', sceglie il modello in base alla categoria del file:
    - primary (Jina v4): text, pdf, code, image, notebook
    - secondary (Nemotron): audio, video, excel, presentation
    """
    emb_config = config.get('embeddings', {})
    active = emb_config.get('active', 'primary')

    if active == 'auto' and file_category:
        primary_categories = {'text', 'pdf', 'image', 'notebook'}
        if file_category in primary_categories:
            return emb_config.get('primary', {})
        else:
            return emb_config.get('secondary', {})

    return emb_config.get(active, emb_config.get('primary', {}))
