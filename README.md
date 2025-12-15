Universal Semantic-First Adaptive RAG for FinOps Knowledge Systems
Abstract

Traditional Retrieval-Augmented Generation (RAG) systems retrieve a fixed number of documents regardless of relevance, query complexity, or the necessity of retrieval itself. This often leads to irrelevant context injection, hallucinations, and poor explainability.

This project presents a Universal Semantic-First Adaptive RAG system designed for FinOps and Cloud Cost Knowledge Systems. The system integrates query rewriting, Self-RAG, Adaptive-RAG, and Corrective-RAG in a single pipeline. Retrieval and generation are governed by semantic similarity, grounding checks, and adaptive escalation, without relying on LangGraph.

The result is a robust, explainable, production-ready RAG architecture that prioritizes semantic correctness over keyword heuristics.

Features

Semantic-first retrieval using cosine similarity

Intelligent query rewriting with semantic expansion

Self-RAG, Adaptive-RAG, and Corrective-RAG in one pipeline

Automatic escalation between RAG strategies

Fine-grained retrieval and generation evaluation metrics

Human-readable explainability for every answer

Domain-agnostic design validated on FinOps knowledge corpora

LangGraph-free, production-friendly implementation

High-Level System Architecture
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

End-to-End Process Flow
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

    User->>Rewriter: Submit Query
    Rewriter->>Classifier: Rewritten Query
    Classifier->>RAGController: Complexity Label

    RAGController->>Retriever: Retrieval Request
    Retriever->>VectorDB: Semantic Similarity Search
    VectorDB-->>Retriever: Candidate Chunks
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

RAG Techniques Implemented
1. Query Rewriting

Query rewriting expands the user query semantically using:

Synonyms

Domain context

Conceptual reformulation

Example:

User Query: Workload Behavior Forecasting
Rewritten: Predictive analysis of workload behavior for cost forecasting and scaling

2. Self-RAG

Self-RAG validates whether the retrieved context and generated answer are sufficiently grounded.

flowchart TD
    Q[Query]
    R[Retrieve Context]
    G[Generate Answer]
    E[Grounding Check]

    Q --> R --> G --> E
    E -->|Grounded| ACCEPT[Accept Answer]
    E -->|Not Grounded| ESCALATE[Escalate]

3. Adaptive-RAG

Adaptive-RAG dynamically adjusts retrieval depth based on semantic complexity.

flowchart TD
    Q[Query]
    C[Complexity Estimation]
    R1[Light Retrieval]
    R2[Deep Retrieval]

    Q --> C
    C -->|Simple| R1
    C -->|Complex| R2

4. Corrective-RAG

Corrective-RAG activates when grounding fails, performing query rewriting and re-retrieval.

flowchart TD
    Q[Query]
    R1[Initial Retrieval]
    G1[Generate]
    E1[Grounding Check]

    RW[Rewrite Query]
    R2[Corrective Retrieval]
    G2[Regenerate]
    E2[Final Check]

    Q --> R1 --> G1 --> E1
    E1 -->|Fail| RW --> R2 --> G2 --> E2
    E2 -->|Pass| ACCEPT
    E2 -->|Fail| REJECT

Evaluation Metrics
Retrieval Metrics

Cosine similarity score

Coverage score (query vs context)

Number of evidence chunks used

Generation Metrics

Grounding score (answer vs context)

Semantic support validation

Hallucination rejection

Cost Metrics

Estimated token usage

Estimated cost in USD

Explainability

Each response includes:

Selected RAG strategy

Number of query rewrites

Retrieval confidence

Grounding score

Acceptance or rejection reason

Sources used

Example:

RAG strategy used: CORRECTIVE_RAG
Query rewrites applied: 2
Retrieved evidence chunks: 3
Grounding score: 0.62
Answer accepted based on semantic support

Project Structure
ReadyTensor_Project/
│
├── src/
│   ├── app.py               # Main RAG application
│   ├── vectordb.py          # Vector DB abstraction
│
├── data/
│   ├── finops.txt
│   ├── cloud_cost_analysis.txt
│   ├── cloud_predictive_analysis.txt
│   └── ...
│
├── chroma_store/             # Persistent vector store
├── .env                      # Environment variables
├── requirements.txt
└── README.md

Required .env Configuration

Create a .env file in the project root:

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

Installation
1. Clone the repository
git clone <your_repo_url>
cd <project_folder>

2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows

3. Upgrade pip
python -m pip install --upgrade pip

4. Install dependencies

Create requirements.txt:

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


Install:

pip install -r requirements.txt

Running the Application
python src/app.py

Conclusion

This project demonstrates how semantic-first adaptive RAG pipelines significantly improve reliability, explainability, and factual correctness for domain-specific knowledge systems. By combining query rewriting, Self-RAG, Adaptive-RAG, and Corrective-RAG — without orchestration frameworks — the system remains both powerful and deployable.
