
from config.settings import SYSTEM_PROMPT
from tools.ollama_client import ask_ollama
from memory.conversation_memory import ConversationMemory


class AssistantAgent:

    def __init__(self, system_prompt=None):
        self.memory = ConversationMemory()

        # Use the supplied prompt or the default system prompt
        prompt = system_prompt or SYSTEM_PROMPT
        self.memory.add_message("system", prompt)

    def chat(self, question):
        self.memory.add_message("user", question)

        response = ask_ollama(
            self.memory.get_messages()
        )

        self.memory.add_message("assistant", response)

        return response
