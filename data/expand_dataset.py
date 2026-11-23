
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "expanded_dataset.csv"

# Existing 10 categories and their subject areas
categories = {
    "Artificial Intelligence": {
        "topic": "artificial intelligence",
        "context": "intelligent systems that perform tasks requiring human-like reasoning",
        "tasks": [
            "explain its applications",
            "describe its benefits and limitations",
            "compare rule-based and intelligent approaches",
            "give a practical real-world example",
            "explain how it supports decision making"
        ]
    },
    "Machine Learning": {
        "topic": "machine learning",
        "context": "algorithms that learn patterns from training data",
        "tasks": [
            "explain supervised and unsupervised learning",
            "describe how a model learns from data",
            "compare classification and regression",
            "explain overfitting and underfitting",
            "give an example of model evaluation"
        ]
    },
    "Python": {
        "topic": "Python programming",
        "context": "Python code, syntax, functions and programming concepts",
        "tasks": [
            "explain functions with a simple example",
            "describe lists and dictionaries",
            "write a program using loops",
            "explain exception handling",
            "compare lists, tuples and sets"
        ]
    },
    "Data Structures": {
        "topic": "data structures",
        "context": "arrays, linked lists, stacks, queues, trees and graphs",
        "tasks": [
            "explain stack operations with an example",
            "compare arrays and linked lists",
            "describe queue implementation",
            "explain binary tree traversal",
            "compare stack and queue applications"
        ]
    },
    "LLM": {
        "topic": "large language models",
        "context": "language models, tokens, transformers and text generation",
        "tasks": [
            "explain transformer architecture",
            "describe tokenization and embeddings",
            "explain how text generation works",
            "compare fine-tuning and prompting",
            "describe common LLM limitations"
        ]
    },
    "Ollama": {
        "topic": "Ollama",
        "context": "running and managing language models locally with Ollama",
        "tasks": [
            "explain how to run a model locally",
            "show how to download a language model",
            "describe the Ollama API",
            "explain how to list installed models",
            "show how to connect a Python application"
        ]
    },
    "System Prompts": {
        "topic": "system prompts",
        "context": "instructions that define an AI assistant's role and behavior",
        "tasks": [
            "write a prompt for a coding assistant",
            "explain how to define an assistant's role",
            "describe how to add output constraints",
            "compare system prompts and user prompts",
            "improve an unclear assistant instruction"
        ]
    },
    "Conversation Memory": {
        "topic": "conversation memory",
        "context": "storing and retrieving information across assistant interactions",
        "tasks": [
            "explain short-term conversation memory",
            "describe how to store chat history",
            "explain how previous messages can be retrieved",
            "compare temporary and persistent memory",
            "describe memory management in an assistant"
        ]
    },
    "SQL": {
        "topic": "SQL databases",
        "context": "relational databases, queries, tables and data management",
        "tasks": [
            "write a query to retrieve records",
            "explain primary and foreign keys",
            "describe SQL joins with examples",
            "compare WHERE and HAVING",
            "explain database normalization"
        ]
    },
    "Career Guidance": {
        "topic": "career guidance",
        "context": "career planning, resumes, skills and interview preparation",
        "tasks": [
            "suggest a career path for a beginner",
            "explain how to improve a technical resume",
            "prepare questions for a job interview",
            "identify skills needed for a software role",
            "create a learning roadmap for a student"
        ]
    }
}

question_formats = [
    "Can you {task} related to {topic}?",
    "I am learning about {topic}. Please {task}.",
    "As a beginner, how can I {task}?",
    "I need help with {topic}. Could you {task}?",
    "Provide a clear explanation and example to {task}."
]

prompt_formats = [
    "You are a helpful tutor specializing in {topic}. Explain concepts clearly. Your context is {context}.",
    "Act as an experienced {topic} assistant. Give accurate, beginner-friendly answers. Focus on {context}.",
    "You are a technical mentor for {topic}. Use simple explanations and practical examples. Consider {context}.",
    "Answer as a knowledgeable specialist in {topic}. Keep responses relevant and structured. The subject covers {context}."
]

records = []
record_id = 1

for category, details in categories.items():
    for prompt_index, prompt_format in enumerate(prompt_formats):
        for task_index, task in enumerate(details["tasks"]):
            question_format = question_formats[
                (prompt_index + task_index) % len(question_formats)
            ]

            question = question_format.format(
                task=task,
                topic=details["topic"]
            )

            system_prompt = prompt_format.format(
                topic=details["topic"],
                context=details["context"]
            )

            records.append({
                "id": record_id,
                "category": category,
                "system_prompt": system_prompt,
                "test_question": question
            })

            record_id += 1

df = pd.DataFrame(records)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("========== DATASET EXPANSION ==========")
print("Total records:", len(df))
print("Total categories:", df["category"].nunique())
print("\nRecords per category:")
print(df["category"].value_counts())
print("\nDataset saved at:", OUTPUT_PATH)
print("\nDataset preview:")
print(df.head())
