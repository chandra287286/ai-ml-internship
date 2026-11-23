import os

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/chat"
)

SYSTEM_PROMPT = """
You are a helpful local AI assistant.
Answer clearly and accurately.
Help users with general questions and career guidance.
If you do not know something, say so.
"""