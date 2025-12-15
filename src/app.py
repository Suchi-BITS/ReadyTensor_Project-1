import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectordb import VectorDB

# ================== ENV ==================

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
INTENT_EMBEDDING_MODEL = "text-embedding-3-small"

COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MODEL_COST_PER_1K = 0.00015

# Semantic acceptance threshold (core fix)
SEMANTIC_SUPPORT_THRESHOLD = 0.50

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

# ================== METRICS ==================

def tokenize(text: str) -> set:
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))

def grounding(answer: str, context: str) -> float:
    a = tokenize(answer)
    c = tokenize(context)
    return len(a & c) / max(len(a), 1)

def coverage(query: str, context: str) -> float:
    q = tokenize(query)
    c = tokenize(context)
    return len(q & c) / max(len(q), 1)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def estimate_cost(tokens: int) -> float:
    return round((tokens / 1000) * MODEL_COST_PER_1K, 6)

# ================== EXPLAINABILITY ==================

def explain_decision(
    strategy: str,
    query_complexity: str,
    rewrites_used: int,
    retrieved_chunks: int,
    semantic_support: float,
    grounding_score: float,
    coverage_score: float,
    accepted: bool,
    tokens: int,
    cost_usd: float,
    reason: Optional[str] = None
) -> List[str]:

    lines = []
    lines.append(f"RAG strategy used: {strategy}")
    lines.append(f"Query classified as: {query_complexity}")
    lines.append(f"Query rewrites applied: {rewrites_used}")
    lines.append(f"Retrieved evidence chunks: {retrieved_chunks}")

    lines.append(f"Semantic support score: {round(semantic_support, 3)}")
    lines.append(f"Grounding score (diagnostic): {round(grounding_score, 3)}")
    lines.append(f"Coverage score (diagnostic): {round(coverage_score, 3)}")

    if accepted:
        lines.append("Answer accepted because semantic support exceeded threshold")
    else:
        lines.append(f"Answer rejected: {reason}")

    lines.append(f"Estimated tokens used: {tokens}")
    lines.append(f"Estimated cost (USD): {round(cost_usd, 6)}")

    return lines

# ================== RAG APP ==================

class RAGApp:
    def __init__(self):
        print("Initializing Universal Semantic-First Adaptive RAG")

        self.vdb = VectorDB(COLLECTION, EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
        self.embedder = OpenAIEmbeddings(model=INTENT_EMBEDDING_MODEL)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        self.prompt = ChatPromptTemplate.from_template(
            """You are a domain expert assistant.

Use ONLY the context below to answer the question.
If the context does not support the answer, say so clearly.

Context:
{context}

Question:
{question}

Answer:"""
        )

        self.adaptive_classifier_prompt = ChatPromptTemplate.from_template(
            """Classify the complexity of this query.

SIMPLE:
- Can be answered without documents

MODERATE:
- Needs one document or section

COMPLEX:
- Needs synthesis, enumeration, or multiple documents

Query:
{question}

Respond with exactly one word:
SIMPLE, MODERATE, COMPLEX
"""
        )

    # ---------------- INGEST ----------------

    def ingest(self):
        docs = []

        for f in DATA_DIR.glob("*.txt"):
            content = f.read_text(encoding="utf-8").strip()
            if not content:
                continue

            chunks = self.splitter.split_text(content)

            for i, ch in enumerate(chunks):
                docs.append({
                    "content": ch,
                    "metadata": {
                        "filename": f.name,
                        "chunk_id": i
                    }
                })

        self.vdb.add_documents(docs)
        print(f"Ingested {len(docs)} chunks")

    # ---------------- QUERY REWRITE ----------------

    def rewrite_query(self, query: str) -> List[str]:
        rewrites = [query]

        rewrite_prompt = ChatPromptTemplate.from_template(
            """Rewrite the question using synonyms and alternative phrasing
while preserving meaning.

Question:
{question}

Return up to 2 variants."""
        )

        chain = rewrite_prompt | self.llm | StrOutputParser()
        try:
            out = chain.invoke({"question": query})
            for line in out.split("\n"):
                if line.strip():
                    rewrites.append(line.strip())
        except Exception:
            pass

        return list(dict.fromkeys(rewrites))[:3]

    # ---------------- RETRIEVAL ----------------

    def retrieve(self, queries: List[str], k: int = 6) -> Tuple[str, int]:
        chunks = []
        seen = set()

        for q in queries:
            res = self.vdb.search(q, n_results=12)
            for i in range(len(res["documents"])):
                doc = res["documents"][i]
                sig = doc[:120]
                if sig not in seen:
                    seen.add(sig)
                    chunks.append((doc, res["metadatas"][i]))

        context = "\n\n".join(
        f"Source: {meta.get('filename', 'unknown_source')}\n{doc}"
        for doc, meta in chunks[:k]
        )

        return context[:5000], len(chunks[:k])

    # ---------------- SEMANTIC SUPPORT ----------------

    def semantic_support(self, answer: str, context: str) -> float:
        a_emb = self.embedder.embed_query(answer)
        c_emb = self.embedder.embed_query(context)
        return cosine(a_emb, c_emb)

    # ---------------- GENERATION ----------------

    def generate(self, query: str, context: str) -> str:
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query}).strip()

    # ---------------- RUN ----------------

    def run(self, query: str) -> Dict:
        classifier = self.adaptive_classifier_prompt | self.llm | StrOutputParser()
        complexity = classifier.invoke({"question": query}).strip().upper()

        rewrites = self.rewrite_query(query)

        # ---------- SELF / ADAPTIVE / CORRECTIVE ----------
        attempted_strategy = None

        context, chunks = self.retrieve(rewrites)
        answer = self.generate(query, context)

        sem = self.semantic_support(answer, context)
        grd = grounding(answer, context)
        cov = coverage(query, context)

        if sem >= SEMANTIC_SUPPORT_THRESHOLD:
            attempted_strategy = "SELF_RAG"
            accepted = True
        else:
            # Adaptive escalation
            attempted_strategy = "ADAPTIVE_RAG"
            rewrites = self.rewrite_query(f"Detailed explanation of {query}")
            context, chunks = self.retrieve(rewrites)
            answer = self.generate(query, context)

            sem = self.semantic_support(answer, context)
            grd = grounding(answer, context)
            cov = coverage(query, context)

            if sem >= SEMANTIC_SUPPORT_THRESHOLD:
                accepted = True
            else:
                attempted_strategy = "CORRECTIVE_RAG"
                accepted = False

        tokens = estimate_tokens(context + answer) if accepted else 0
        cost = estimate_cost(tokens)

        explanation = explain_decision(
            strategy=attempted_strategy,
            query_complexity=complexity,
            rewrites_used=len(rewrites),
            retrieved_chunks=chunks,
            semantic_support=sem,
            grounding_score=grd,
            coverage_score=cov,
            accepted=accepted,
            tokens=tokens,
            cost_usd=cost,
            reason="Semantically unsupported answer" if not accepted else None
        )

        if not accepted:
            return {
                "answer": "Unable to provide a reliable answer based on the available documents.",
                "metrics": {
                    "strategy": attempted_strategy,
                    "semantic_support": sem,
                    "grounding": grd,
                    "coverage": cov,
                    "tokens": 0,
                    "cost_usd": 0.0
                },
                "explainability": explanation
            }

        return {
            "answer": answer,
            "metrics": {
                "strategy": attempted_strategy,
                "semantic_support": sem,
                "grounding": grd,
                "coverage": cov,
                "tokens": tokens,
                "cost_usd": cost
            },
            "explainability": explanation
        }

# ================== MAIN ==================

def main():
    app = RAGApp()
    app.ingest()

    print("\nReady. Type a query or 'exit'.\n")

    while True:
        q = input("Enter your query: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        result = app.run(q)

        print("\n--- ANSWER ---")
        print(result["answer"])

        print("\n--- METRICS ---")
        for k, v in result["metrics"].items():
            print(f"{k}: {v}")

        print("\n--- EXPLAINABILITY ---")
        for line in result["explainability"]:
            print(f"- {line}")

if __name__ == "__main__":
    main()
