# Week 3 – Design Enhancement Plan
## AI Capstone: Local AI Assistant with Ollama

### 1. Enhancement Title

**Persistent Vector Memory Store for Cross-Session Conversation Recall**

### 2. Goal

The current AI assistant uses in-memory conversation storage. This means conversation information may be lost when the application restarts.

The proposed enhancement is to introduce a persistent vector memory store that allows the assistant to:

- Remember important facts from previous conversations.
- Store user information as searchable vector embeddings.
- Retrieve relevant memories using semantic similarity.
- Use retrieved memories as context when generating responses.
- Retain useful information across application restarts.

The objective is to make the local AI assistant more context-aware and improve its ability to answer questions using information from previous sessions.

### 3. Proposed Technique and Architecture

**Technique:** Retrieval-Augmented Generation (RAG) with persistent vector storage.

**Proposed components:**

1. **Memory extraction:** Identify useful facts from conversations, such as user preferences, learning goals and project details.
2. **Embedding generation:** Convert each memory into a numerical vector using the Ollama embedding model `nomic-embed-text`.
3. **Vector storage:** Store embeddings, original memory text and metadata in ChromaDB.
4. **Semantic retrieval:** Convert a new user query into an embedding and retrieve the top 3 relevant memories using vector similarity.
5. **Context injection:** Add retrieved memories to the assistant's prompt before sending it to the local LLM.
6. **Persistent storage:** Store the vector database locally so memories remain available after restarting the application.

**Proposed workflow:**

User Conversation → Memory Extraction → Ollama Embeddings → ChromaDB Storage

New User Query → Query Embedding → Similarity Search → Relevant Memories → LLM Context → Response

### 4. Models, Libraries and Data Sources

**Existing model:**
- Ollama Llama 3.2 – response generation.

**New embedding model:**
- Ollama `nomic-embed-text` – converts memories and queries into embeddings.

**New libraries:**

| Library / Tool | Purpose |
|---|---|
| ChromaDB (`chromadb`) | Persistent vector database |
| Ollama | Local embedding generation |
| Existing Python libraries | Memory processing and application integration |

**Data sources:**
- Existing conversation history.
- User-provided facts and preferences.
- Project-related information shared during conversations.
- A small manually prepared memory evaluation dataset.

All memory data and vector storage will remain local.

### 5. Expected Metric Improvement

The enhancement will be evaluated using a dedicated memory-retrieval test dataset.

**Evaluation dataset:**
- 30 manually written memory test cases.
- Each test case contains a stored fact, a related query and the expected memory identifier.
- Queries will include direct questions and semantically rephrased questions.
- Tests will include information from previous sessions.

**Primary metric: Recall@3**

Recall@3 measures whether the expected relevant memory appears among the top 3 retrieved memories.

Formula:

Recall@3 = Relevant queries with the expected memory in the top 3 / Total evaluation queries

**Expected target:**

- Target Recall@3: at least 80%.
- Expected improvement: at least 20 percentage points over a separately measured keyword-based retrieval baseline, if the baseline is below the target.
- Cross-session recall: relevant stored facts should remain retrievable after restarting the application.

These are project targets, not guaranteed results. Actual performance will be reported after testing.

The existing classification accuracy is not expected to improve directly because this enhancement targets conversation memory retrieval, not the category classifier.

### 6. Files to Create or Modify

| File | Planned change |
|---|---|
| `memory/vector_memory.py` | Create persistent vector memory functionality |
| `memory/conversation_memory.py` | Integrate existing conversation memory with persistent memory |
| `tools/ollama_client.py` | Add embedding-generation support |
| `config/settings.py` | Add embedding model and vector database settings |
| `main.py` | Integrate memory retrieval into the assistant workflow |
| `data/memory_test_cases.jsonl` | Create memory evaluation test cases |
| `tests/test_vector_memory.py` | Add storage, retrieval and persistence tests |
| `requirements.txt` | Add ChromaDB dependency |
| `docs/enhancement_plan.md` | Document the enhancement plan |

### 7. Implementation Timeline

The enhancement will be divided into three sub-tasks.

| Sub-task | Activities | Estimated time |
|---|---|---|
| 1. Vector Memory Setup | Install ChromaDB, configure Ollama embeddings and implement persistent storage | 1.5 hours |
| 2. Memory Integration | Connect memory storage and semantic retrieval with the existing assistant | 1.5 hours |
| 3. Testing and Evaluation | Create test cases, measure Recall@3, verify restart persistence and document results | 1.5 hours |

**Total estimated time: 4.5 hours.**

Each sub-task is planned to take less than 2 hours.

### 8. Expected Outcome

After implementation, the AI assistant should:

- Retain selected user facts across sessions.
- Retrieve relevant information using semantic similarity.
- Provide responses with relevant historical context.
- Work locally without requiring an external cloud AI API.
- Demonstrate measurable memory-retrieval performance.

### 9. Risks and Limitations

- Incorrect memory extraction may store inaccurate information.
- Similar queries may retrieve irrelevant memories.
- Vector database storage will require disk space.
- Embedding generation may increase response latency.
- Retrieval quality depends on the embedding model and stored memory quality.
- Sensitive information should not be stored without appropriate user consent.
- The system must support updating and deleting stored memories.

### 10. Mentor Approval

**Status: Pending Mentor Review**

The proposed enhancement will be implemented only after mentor approval.

Mentor feedback:

________________________________________

Approval status: Pending

Mentor name: ____________________________

Review date: ____________________________