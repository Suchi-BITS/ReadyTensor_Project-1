Universal Semantic-First Adaptive RAG for FinOps Knowledge Systems
Abstract

Cloud FinOps knowledge systems require high factual grounding, domain awareness, and adaptive retrieval strategies. Traditional Retrieval-Augmented Generation (RAG) pipelines rely on static top-k retrieval, which often leads to irrelevant context injection or hallucinated responses.

This project presents a Universal Semantic-First Adaptive RAG architecture designed specifically for FinOps and Cloud Cost Intelligence. The system integrates semantic query rewriting, Self-RAG, Adaptive-RAG, and Corrective-RAG, combined with robust retrieval and generation evaluation metrics and explainability, while remaining LangGraph-free and production-ready.

Key Contributions

Semantic-first retrieval (cosine similarity over keyword heuristics)

Query rewriting using semantic expansion and synonym enrichment

Self-RAG, Adaptive-RAG, and Corrective-RAG in a single pipeline

Automatic escalation between RAG strategies

Fine-grained retrieval and generation evaluation metrics

Human-readable explainability for every answer

Domain-agnostic design validated on FinOps knowledge corpora

System Architecture
High-Level Architecture
flowchart LR
    U[User]
    QI[Query Intake]
    QR[Semantic Query Rewriter]
    QC[Query Complexity Estimator]

    SR[Self-RAG]
    AR[Adaptive-RAG]
    CR[Corrective-RAG]

    RR[Retriever + Reranker]
    VS[(Vector Store)]
    LLM[LLM Generator]

    EV[Evaluation Engine]
    EX[Explainability Engine]
    RESP[Final Response]

    U --> QI
    QI --> QR
    QR --> QC

    QC -->|Simple| SR
    QC -->|Moderate| AR
    QC -->|Complex| CR

    SR --> RR
    AR --> RR
    CR --> RR

    RR --> VS
    VS --> RR

    RR --> LLM
    LLM --> EV
    EV --> EX
    EX --> RESP

End-to-End Query Processing Flow
sequenceDiagram
    participant User
    participant Rewriter
    participant Classifier
    participant RAGController
    participant Retriever
    participant VectorDB
    participant LLM
    participant Evaluator
    participant Explainer

    User->>Rewriter: User Query
    Rewriter->>Classifier: Rewritten Query
    Classifier->>RAGController: Complexity Label

    RAGController->>Retriever: Retrieve Context
    Retriever->>VectorDB: Semantic Search
    VectorDB-->>Retriever: Relevant Chunks
    Retriever-->>RAGController: Reranked Context

    RAGController->>LLM: Generate Answer
    LLM-->>Evaluator: Draft Answer
    Evaluator->>Evaluator: Grounding & Coverage Check

    alt Supported
        Evaluator->>Explainer: Accept Answer
    else Unsupported
        Evaluator->>RAGController: Escalate Strategy
    end

    Explainer-->>User: Answer + Metrics + Explanation

Retrieval Strategy Selection (Adaptive Logic)
flowchart TD
    Q[User Query]
    C[Semantic Complexity Estimation]

    SR[Self-RAG]
    AR[Adaptive-RAG]
    CR[Corrective-RAG]

    Q --> C
    C -->|Low Complexity| SR
    C -->|Moderate Complexity| AR
    C -->|High Complexity| CR

RAG Techniques Implemented
1. Self-RAG

Self-RAG verifies whether retrieval and generation are sufficient and grounded before accepting an answer.

flowchart TD
    Q[Query]
    R[Retrieve Context]
    G[Generate Answer]
    E[Grounding Check]

    Q --> R --> G --> E
    E -->|Grounded| ACCEPT[Accept Answer]
    E -->|Not Grounded| ESCALATE[Escalate to Corrective-RAG]

2. Adaptive-RAG

Adaptive-RAG dynamically adjusts retrieval depth based on semantic complexity.

Simple → minimal retrieval

Moderate → multi-chunk retrieval

Complex → escalation with correction

flowchart TD
    Q[Query]
    C[Complexity Estimation]
    R1[Single-Step Retrieval]
    R2[Multi-Step Retrieval]

    Q --> C
    C -->|Simple| R1
    C -->|Complex| R2

3. Corrective-RAG

Corrective-RAG activates when retrieval is weak or grounding fails.

flowchart TD
    Q[Query]
    R1[Initial Retrieval]
    G1[Generate]
    E1[Grounding Check]

    RW[Semantic Query Rewrite]
    R2[Corrective Retrieval]
    G2[Regenerate]
    E2[Final Check]

    Q --> R1 --> G1 --> E1
    E1 -->|Fail| RW --> R2 --> G2 --> E2
    E2 -->|Pass| ACCEPT
    E2 -->|Fail| REJECT

Query Rewriting Strategy

Query rewriting is semantic, not rule-based:

Synonym expansion

Domain context enrichment

Conceptual reformulation

Example:

User Query: "Workload Behavior Forecasting"
Rewritten: "Predictive analysis of workload behavior for cost forecasting and scaling"

Evaluation Metrics
Retrieval Quality Metrics
Metric	Description
Cosine Similarity	Semantic alignment between query and chunks
Coverage Score	Overlap between query terms and retrieved context
Chunk Count	Number of evidence chunks used
Generation Quality Metrics
Metric	Description
Grounding Score	Semantic overlap between answer and context
Semantic Support	Whether answer is supported by retrieved evidence
Hallucination Risk	Inferred from grounding + coverage
Cost Metrics
Metric	Description
Token Count	Estimated tokens used
Cost (USD)	Approximate inference cost
Explainability

Each response includes a structured explanation:

Selected RAG strategy

Number of query rewrites

Retrieval confidence

Grounding decision

Acceptance or rejection reason

Example:

RAG strategy used: CORRECTIVE_RAG
Query rewrites applied: 2
Retrieved evidence chunks: 3
Grounding score: 0.62
Answer accepted based on semantic support

Design Considerations

Semantic-first over keyword heuristics

Reject only when semantically unsupported

No reliance on section headers

LangGraph-free for production simplicity

Modular RAG strategies

Explainability as a first-class citizen

Use Cases

FinOps knowledge assistants

Cloud cost governance copilots

Internal engineering knowledge systems

Enterprise documentation Q&A

Cost anomaly and forecasting explanation engines

Conclusion

This project demonstrates that semantic-first adaptive RAG pipelines significantly improve reliability, explainability, and robustness for domain-specific knowledge systems like FinOps. By integrating Self-RAG, Adaptive-RAG, and Corrective-RAG without orchestration frameworks, the system remains both powerful and deployable.


---

## Features

- RAG pipeline with vector search and semantic reranking
- Automatic ingestion of .txt documents from the data folder
- Persistent ChromaDB vector database
- High-quality HuggingFace MiniLM embeddings
- Token-based and paragraph-based text chunking
- LLM fallback chain: OpenAI → Groq → Retrieval-only
- Clean architecture with separate modules for vector DB and app logic
- Environment-driven configuration through .env

---

## Project Structure

```
project/
│
├── data/                      # Input documents (.txt files)
│       └── *.txt
│
├── chroma_store/              # Auto-persisted ChromaDB files
│
├── src/
│       ├── app.py             # Main RAG application
│       └── vectordb.py        # Vector DB wrapper (Chroma + HF + reranker)
│
├── .env                       # Configuration keys and model selection
└── README.md
```

---

## Architecture Overview

sequenceDiagram
    participant User
    participant QueryRewriter
    participant ComplexityClassifier
    participant RAGController
    participant Retriever
    participant VectorDB
    participant LLM
    participant Evaluator
    participant Explainer

    User->>QueryRewriter: Submit Query
    QueryRewriter->>ComplexityClassifier: Rewritten Query
    ComplexityClassifier->>RAGController: Complexity Label

    RAGController->>Retriever: Retrieval Request
    Retriever->>VectorDB: Similarity Search
    VectorDB-->>Retriever: Relevant Chunks
    Retriever-->>RAGController: Reranked Context

    RAGController->>LLM: Generate Answer
    LLM-->>Evaluator: Draft Answer
    Evaluator->>Evaluator: Grounding & Coverage Check

    alt Supported
        Evaluator->>Explainer: Accepted Answer
    else Unsupported
        Evaluator->>RAGController: Escalate / Rewrite
    end

    Explainer-->>User: Answer + Metrics + Explanation


### 1. Document Ingestion
- Loads all .txt files from the data directory.
- Splits documents into meaningful chunks:
  - Paragraph-based chunking
  - Token-based chunking using RecursiveCharacterTextSplitter

### 2. Embedding Layer
- Generates normalized embeddings using HuggingFace MiniLM.
- Implemented via langchain-huggingface.

### 3. Vector Store
- Stores embeddings using Chroma from langchain-chroma.
- Data persists automatically inside the chroma_store directory.

### 4. Reranking Layer
- Uses a CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) to reorder retrieved documents and improve answer accuracy.

### 5. LLM Response Layer
- Priority order:
  1. OpenAI (gpt-4o-mini)
  2. Groq (llama-3.1-8b-instant)
  3. Retrieval-only fallback if no API keys are available

### 6. Console Interface
- Simple REPL-style question-and-answer interface.
- Displays answer, retrieved context, and document sources.

---

## Required .env Configuration

Create a .env file in the project root:

```
# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

# Groq (optional fallback)
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant

# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chroma collection name
CHROMA_COLLECTION_NAME=rag_documents
```

---

## Installation

### 1. Clone the repository
```
git clone <your_repo_url>
cd <project_folder>
```

### 2. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Upgrade pip
```
python -m pip install --upgrade pip
```

### 4. Install dependencies

Create requirements.txt containing:

```
langchain
langchain-core
langchain-community
langchain-chroma
langchain-huggingface
langchain-text-splitters

chromadb
sentence-transformers
numpy
tiktoken

openai
groq

python-dotenv
requests
pydantic
typing-extensions
```

Install them:

```
pip install -r requirements.txt
```

---

## Running the Application

```
python src/app.py
```

---

## How It Works

1. Loads all .txt documents from the data folder
2. Splits them into optimized chunks
3. Generates embeddings using HuggingFace
4. Stores vectors in ChromaDB
5. Retrieves relevant chunks for a query
6. Reranks them using CrossEncoder
7. Constructs final context
8. Generates an answer using available LLMs

---

## Resetting the Vector Store

To clear Chroma:

```
from vectordb import VectorDB
v = VectorDB("rag_documents", "sentence-transformers/all-MiniLM-L6-v2")
v.clear_collection()
```

Or delete the chroma_store folder manually.

---

## Notes

- Only .txt files are supported for ingestion.
- Restart the application after adding new documents.
- All components run on CPU by default.
- You can switch to larger embedding models if required.
