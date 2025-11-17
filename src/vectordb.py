"""
vectordb.py — Smart Vector Database Wrapper for RAG
Uses Chroma + token-based chunking + cross-encoder reranking.
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_store"


class VectorDB:
    def __init__(self, collection_name: str, embedding_model_name: str):
        """Initialize vector database with embeddings and cross-encoder"""
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        print(f" Loading embedding model: {embedding_model_name}")
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Important for cosine similarity
        )
        
        print(" Loading cross-encoder for reranking...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        CHROMA_DIR.mkdir(exist_ok=True)
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_fn,
            persist_directory=str(CHROMA_DIR)
        )
        print(f" Vector database initialized: {CHROMA_DIR}")

    def _split_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into smaller, focused chunks by paragraphs/sections"""
        chunks = []
        
        for doc in docs:
            # First split by double newlines (paragraphs/sections)
            sections = doc["content"].split("\n\n")
            
            for section in sections:
                section = section.strip()
                if len(section) < 50:  # Skip very short sections
                    continue
                
                # If section is still too long (>800 chars), split it further
                if len(section) > 800:
                    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        model_name="gpt-4o-mini",
                        chunk_size=300,
                        chunk_overlap=50
                    )
                    pieces = splitter.split_text(section)
                    for piece in pieces:
                        chunks.append({
                            "content": piece,
                            "metadata": doc["metadata"]
                        })
                else:
                    chunks.append({
                        "content": section,
                        "metadata": doc["metadata"]
                    })
        
        print(f"  Split into {len(chunks)} focused chunks")
        return chunks

    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add documents to vector store"""
        print(f" Processing {len(docs)} documents...")
        chunks = self._split_documents(docs)
        
        texts = [c["content"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        
        # Add to Chroma
        self._store.add_texts(texts=texts, metadatas=metas)
        self._store.persist()
        
        print(f" Stored {len(texts)} chunks in collection '{self.collection_name}'")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for relevant documents using vector similarity + cross-encoder reranking
        Returns documents with normalized scores between 0-1
        """
        # Initial retrieval with more candidates for reranking
        initial_k = min(n_results * 3, 20)
        
        try:
            # Use similarity_search_with_score for proper cosine similarity
            results = self._store.similarity_search_with_score(query, k=initial_k)
        except Exception as e:
            print(f" Search error: {e}")
            return {"documents": [], "metadatas": [], "scores": []}
        
        if not results:
            return {"documents": [], "metadatas": [], "scores": []}
        
        # Extract documents and convert distance to similarity
        docs = []
        metas = []
        initial_scores = []
        
        for doc, distance in results:
            docs.append(doc.page_content)
            metas.append(doc.metadata)
            # Convert L2 distance to similarity score (0-1 range)
            # For normalized embeddings, cosine similarity = 1 - (L2_distance^2 / 2)
            similarity = max(0, 1 - (distance ** 2) / 2)
            initial_scores.append(similarity)
        
        # Rerank using cross-encoder for better precision
        if docs:
            print(f" Initial retrieval: {len(docs)} documents")
            print(f"  Reranking with cross-encoder...")
            
            pairs = [(query, d) for d in docs]
            rerank_scores = self.cross_encoder.predict(pairs)
            
            # Normalize rerank scores to 0-1 range using min-max scaling
            if len(rerank_scores) > 1:
                min_score = float(np.min(rerank_scores))
                max_score = float(np.max(rerank_scores))
                score_range = max_score - min_score
                
                if score_range > 0:
                    rerank_scores = [(s - min_score) / score_range for s in rerank_scores]
                else:
                    rerank_scores = [0.5] * len(rerank_scores)
            else:
                rerank_scores = [0.5]
            
            # Sort by reranked scores
            order = np.argsort(rerank_scores)[::-1]
            
            # Take top n_results
            top_indices = order[:n_results]
            docs = [docs[i] for i in top_indices]
            metas = [metas[i] for i in top_indices]
            scores = [rerank_scores[i] for i in top_indices]
            
            print(f" Reranked, returning top {len(docs)} documents")
            print(f"  Top score: {scores[0]:.3f}")
        else:
            scores = []
        
        return {
            "documents": docs,
            "metadatas": metas,
            "scores": scores
        }

    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self._store.delete_collection()
            print(f" Cleared collection '{self.collection_name}'")
        except Exception as e:
            print(f" Error clearing collection: {e}")