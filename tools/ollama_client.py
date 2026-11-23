import json
import urllib.request
import urllib.error

from config.settings import MODEL_NAME, OLLAMA_URL

def ask_ollama(messages):
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["message"]["content"]

    except urllib.error.URLError as error:
        return f"Could not connect to Ollama: {error}"

    except (KeyError, json.JSONDecodeError):
        return "Could not understand the response from Ollama."