"""LLM - Client OpenAI-compatible con Tool Calling nativo.

Funziona con qualsiasi API OpenAI-compatible:
Ollama, LM Studio, vLLM, OpenAI, Together, Groq, ecc.

Il LLM decide autonomamente quali tool usare ad ogni richiesta:
- search_documents: RAG search nei documenti
- search_memory: Cerca nella memoria mem0
- save_memory: Salva in memoria mem0
- create_file: Crea file markdown
- edit_file: Modifica file markdown
"""

import json
import logging
import os
from openai import OpenAI

from utils import setup_logging

logger = setup_logging("rag.llm")

# === Tool Definitions (OpenAI function calling schema) ===
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Cerca nei documenti indicizzati nella knowledge base. "
                           "Usa questo tool quando l'utente chiede informazioni su contenuti, "
                           "dati, documenti caricati, o vuole risposte basate su materiale specifico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La query di ricerca semantica nei documenti"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Cerca nella memoria conversazionale e nella knowledge base condivisa. "
                           "Usa questo tool per recuperare contesto da conversazioni passate, "
                           "preferenze dell'utente, fatti appresi, o contenuti dei documenti.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La query di ricerca nella memoria"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Salva un'informazione importante nella memoria persistente. "
                           "Usa quando l'utente condivide preferenze, fatti da ricordare, "
                           "o contesto utile per conversazioni future.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "L'informazione da salvare in memoria"
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Crea un nuovo file markdown. Usa quando l'utente chiede di "
                           "scrivere, creare, generare un file, un riassunto, delle note, ecc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Nome del file (senza estensione .md)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Contenuto del file in formato Markdown"
                    }
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Modifica un file markdown esistente. Usa quando l'utente chiede "
                           "di aggiornare, modificare, aggiungere contenuto a un file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Nome del file da modificare"
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Istruzioni di modifica in linguaggio naturale"
                    }
                },
                "required": ["filename", "instructions"]
            }
        }
    }
]


class LLMClient:
    """Client LLM OpenAI-compatible con tool calling."""

    def __init__(self, config: dict):
        llm_cfg = config.get('llm', {})
        self.model = llm_cfg.get('model', 'qwen3:8b')
        self.temperature = llm_cfg.get('temperature', 0.3)
        self.max_tokens = llm_cfg.get('max_tokens', 4096)
        self.system_prompt = llm_cfg.get('system_prompt', '')

        base_url = llm_cfg.get('base_url', 'http://localhost:11434/v1')
        # API key: SOLO da env var LLM_API_KEY (vedi .env.example)
        api_key = os.environ.get('LLM_API_KEY', 'ollama')

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"LLM: {base_url} (modello: {self.model})")

    def chat_with_tools(self, messages: list[dict], tool_handlers: dict,
                        max_tool_rounds: int = 5) -> str:
        """Chat con tool calling loop.

        Il LLM decide autonomamente quali tool chiamare.
        Il loop continua finche' il LLM non produce una risposta finale (senza tool call).

        Args:
            messages: Lista messaggi [{"role": ..., "content": ...}]
            tool_handlers: Dict {nome_tool: callable} per eseguire i tool
            max_tool_rounds: Max iterazioni tool calling (safety limit)

        Returns:
            Risposta finale testuale del LLM
        """
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(messages)

        for _ in range(max_tool_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                tools=TOOLS,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            choice = response.choices[0]

            # Se non ci sono tool call, restituisci la risposta
            if not choice.message.tool_calls:
                return choice.message.content or ""

            # Aggiungi il messaggio dell'assistente con le tool calls
            full_messages.append(choice.message)

            # Esegui ogni tool call
            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info(f"Tool call: {fn_name}({fn_args})")

                # Esegui il tool handler
                if fn_name in tool_handlers:
                    try:
                        result = tool_handlers[fn_name](**fn_args)
                    except Exception as e:
                        result = f"Errore esecuzione tool {fn_name}: {e}"
                        logger.error(result)
                else:
                    result = f"Tool '{fn_name}' non disponibile"

                # Aggiungi il risultato come messaggio tool
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

        # Safety: se troppi round, restituisci ultimo contenuto
        return choice.message.content or "Raggiunto limite iterazioni tool calling."

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generazione semplice senza tool calling"""
        messages = []
        if system_prompt or self.system_prompt:
            messages.append({"role": "system", "content": system_prompt or self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
