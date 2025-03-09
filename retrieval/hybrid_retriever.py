import os
import numpy as np
import faiss
import pickle
from rank_bm25 import BM25Okapi
import re
import string
from openai import OpenAI


class HybridRetriever:
    def __init__(self, source="arxiv", model="text-embedding-ada-002"):
        self.source = source
        self.model = model
        self.client = OpenAI()

        self.index_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "index"
        )
        self.load_resources()

    def load_resources(self):
        """Load FAISS index, metadata, and chunks"""
        index_path = os.path.join(self.index_dir, f"{self.source}_index.faiss")
        metadata_path = os.path.join(self.index_dir, f"{self.source}_metadata.pkl")
        chunks_path = os.path.join(self.index_dir, f"{self.source}_chunks.pkl")

        if not all(os.path.exists(p) for p in [index_path, metadata_path, chunks_path]):
            raise FileNotFoundError(f"Required index files not found for {self.source}")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        # Ensure chunks are strings, not dictionaries
        if self.chunks and isinstance(self.chunks[0], dict):
            print("Warning: Chunks were stored as dictionaries, fixing...")
            # Try to extract text content from metadata
            from embedding.embedding_manager import EmbeddingManager

            emb_manager = EmbeddingManager()

            # Reload the original papers and rebuild the index
            emb_manager.build_faiss_index(source=self.source)

            # Reload the properly formatted chunks
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)

        # Verify that chunks are now strings
        if self.chunks and not isinstance(self.chunks[0], str):
            print(
                f"Warning: Chunks are still not strings, they are {type(self.chunks[0])}"
            )
            # Create default empty chunks as fallback
            self.chunks = ["No text available" for _ in range(len(self.metadata))]

        # Prepare BM25
        self.tokenized_chunks = [self._tokenize(chunk) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def _tokenize(self, text):
        """Simple tokenization for BM25"""
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Split by whitespace
        return text.split()

    def get_query_embedding(self, query):
        """Get embedding for the query"""
        # Clean query of search prefixes for better embedding
        clean_query = query
        for prefix in ["ti:", "au:", "abs:", "all:"]:
            if clean_query.startswith(prefix):
                clean_query = clean_query.replace(prefix, "", 1)

        # Remove quotes that might be part of search syntax
        clean_query = clean_query.replace('"', "")

        response = self.client.embeddings.create(model=self.model, input=clean_query)
        return response.data[0].embedding

    def semantic_search(self, query, top_k=5):
        """Semantic search using FAISS"""
        # Check if this is a title search
        is_title_search = query.lower().startswith("ti:")

        # For title searches, first try to find exact matches
        if is_title_search:
            title_query = query.replace("ti:", "").replace('"', "").strip()
            exact_matches = []

            for idx, metadata in enumerate(self.metadata):
                if title_query.lower() in metadata["title"].lower():
                    exact_matches.append(
                        {
                            "metadata": metadata,
                            "distance": 0.0,  # Give perfect score to title matches
                            "chunk": self.chunks[idx],
                            "score": 2.0,  # Higher than normal scores (normally 0-1)
                        }
                    )

            # If we found exact title matches, return those
            if exact_matches:
                return exact_matches[:top_k]

        # Otherwise do regular semantic search
        query_embedding = self.get_query_embedding(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS returns -1 if there are not enough results
                results.append(
                    {
                        "metadata": self.metadata[idx],
                        "distance": distances[0][i],
                        "chunk": self.chunks[idx],
                        "score": 1.0
                        / (
                            1.0 + distances[0][i]
                        ),  # Convert distance to similarity score
                    }
                )

        return results

    def keyword_search(self, query, top_k=5):
        """Keyword-based search using BM25"""
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "metadata": self.metadata[idx],
                    "score": bm25_scores[idx],
                    "chunk": self.chunks[idx],
                }
            )

        return results

    def hybrid_search(self, query, top_k=5, alpha=0.7):
        """Combine semantic and keyword search with parameter alpha controlling the mix"""
        semantic_results = self.semantic_search(
            query, top_k=top_k * 2
        )  # Get more candidates
        keyword_results = self.keyword_search(query, top_k=top_k * 2)

        # Combine results with a score normalization
        combined_results = {}
        max_semantic_score = (
            max([r["score"] for r in semantic_results]) if semantic_results else 1.0
        )
        max_keyword_score = (
            max([r["score"] for r in keyword_results]) if keyword_results else 1.0
        )

        # Process semantic results
        for result in semantic_results:
            idx = (
                result["metadata"]["paper_id"]
                + "_"
                + str(result["metadata"]["chunk_index"])
            )
            normalized_score = result["score"] / max_semantic_score
            combined_results[idx] = {
                "metadata": result["metadata"],
                "chunk": result["chunk"],
                "semantic_score": normalized_score,
                "keyword_score": 0,
                "combined_score": alpha * normalized_score,
            }

        # Process keyword results
        for result in keyword_results:
            idx = (
                result["metadata"]["paper_id"]
                + "_"
                + str(result["metadata"]["chunk_index"])
            )
            normalized_score = result["score"] / max_keyword_score
            if idx in combined_results:
                combined_results[idx]["keyword_score"] = normalized_score
                combined_results[idx]["combined_score"] += (
                    1 - alpha
                ) * normalized_score
            else:
                combined_results[idx] = {
                    "metadata": result["metadata"],
                    "chunk": result["chunk"],
                    "semantic_score": 0,
                    "keyword_score": normalized_score,
                    "combined_score": (1 - alpha) * normalized_score,
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["combined_score"], reverse=True
        )
        return sorted_results[:top_k]
