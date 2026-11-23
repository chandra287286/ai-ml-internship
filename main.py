from agents.assistant_agent import AssistantAgent

def main():
    assistant = AssistantAgent()

    print("Local AI Assistant")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYou: ")

        if question.lower() == "exit":
            print("Assistant: Goodbye!")
            break

        if not question.strip():
            continue

        response = assistant.chat(question)

        print("\nAssistant:", response)

if __name__ == "__main__":
    main()