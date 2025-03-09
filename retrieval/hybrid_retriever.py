import os
import numpy as np
import faiss
import pickle
from rank_bm25 import BM25Okapi
import re
import string
from openai import OpenAI
import json
import math


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
        """Get embedding for the query with improved handling for concept/topic queries"""
        # Clean query of search prefixes for better embedding
        clean_query = query
        for prefix in ["ti:", "au:", "abs:", "all:", "cat:"]:
            if clean_query.startswith(prefix):
                clean_query = clean_query.replace(prefix, "", 1)

        # Remove quotes that might be part of search syntax
        clean_query = clean_query.replace('"', "")

        # For topic/concept queries, expand them to improve search quality
        if not query.startswith("ti:") and not query.startswith("au:"):
            # If it looks like a concept/topic query, enhance it
            if len(clean_query.split()) <= 5:  # Short queries are likely concepts
                # Append "research" to make it more academic-focused
                augmented_query = f"research papers about {clean_query}"
                print(f"Enhanced topic query: '{clean_query}' → '{augmented_query}'")
                clean_query = augmented_query

        response = self.client.embeddings.create(model=self.model, input=clean_query)
        return response.data[0].embedding

    def semantic_search(self, query, top_k=5):
        """Semantic search using FAISS with improved handling for topic queries"""
        # Handle special search types
        is_title_search = query.lower().startswith("ti:")
        is_author_search = query.lower().startswith("au:")
        is_topic_search = not (is_title_search or is_author_search)

        # Extract search term
        search_term = query
        for prefix in ["au:", "ti:", "abs:", "cat:", "all:"]:
            if query.lower().startswith(prefix):
                search_term = query[len(prefix) :].strip()
                # Remove quotes if present
                if search_term.startswith('"') and search_term.endswith('"'):
                    search_term = search_term[1:-1]
                break

        # Try to load original papers for metadata matching - initialize paper_data here
        paper_data = None
        if is_author_search:  # Only load for author searches
            paper_data_file = os.path.join(
                os.path.dirname(os.path.dirname(self.index_dir)),
                "data",
                f"{self.source.lower()}_spider_papers.json",  # Force lowercase
            )

            try:
                if os.path.exists(paper_data_file):
                    with open(paper_data_file, "r", encoding="utf-8") as f:
                        paper_data = json.load(f)
                    print(f"Loaded {len(paper_data)} papers for author matching")
                else:
                    print(f"Paper data file not found: {paper_data_file}")
            except Exception as e:
                print(f"Could not load paper data for author matching: {e}")
                paper_data = None  # Ensure it's None if there's an error

        # Special handling for topic/concept searches
        if is_topic_search:
            # Get embedding with optimized query
            query_embedding = self.get_query_embedding(query)
            query_embedding = np.array([query_embedding], dtype=np.float32)

            # Use more results for concept search to ensure diversity
            concept_k = min(top_k * 3, 50)
            distances, indices = self.index.search(query_embedding, concept_k)

            # For topic searches, we want to group by paper to avoid duplicates
            # and ensure diverse coverage of the topic
            paper_scores = {}
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 if there are not enough results
                    continue

                paper_id = self.metadata[idx]["paper_id"]
                score = 1.0 / (1.0 + float(distances[0][i]))

                # For each paper, keep track of its best chunk
                if (
                    paper_id not in paper_scores
                    or score > paper_scores[paper_id]["score"]
                ):
                    paper_scores[paper_id] = {
                        "metadata": self.metadata[idx],
                        "chunk": self.chunks[idx],
                        "distance": float(distances[0][i]),
                        "score": score,
                        "chunk_idx": idx,
                    }

            # Sort papers by score and take top_k
            results = list(paper_scores.values())
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        # Improved author search - find exact author matches in metadata
        # Now check if paper_data exists before using it
        if is_author_search and paper_data:
            author_matches = []
            author_name_parts = search_term.lower().split()

            # First build a mapping from paper_id to authors
            paper_authors = {}
            for paper in paper_data:
                paper_authors[paper["paper_id"]] = paper["authors"]

            # Now check all chunks for papers with this author
            for idx, metadata in enumerate(self.metadata):
                paper_id = metadata["paper_id"]

                # Skip if we don't have author info for this paper
                if paper_id not in paper_authors:
                    continue

                # Check if the author matches any author of this paper
                author_match = False
                matched_author = ""

                for author in paper_authors[paper_id]:
                    author_lower = author.lower()
                    # Check if all name parts appear in the author name
                    if all(part in author_lower for part in author_name_parts):
                        author_match = True
                        matched_author = author
                        break

                if author_match:
                    author_matches.append(
                        {
                            "metadata": metadata,
                            "distance": 0.0,
                            "chunk": self.chunks[idx],
                            "score": 2.0,  # High score for exact author matches
                            "matched_author": matched_author,
                        }
                    )

            # Return author matches if any were found
            if author_matches:
                # Remove duplicates by paper_id to avoid showing the same paper multiple times
                seen_papers = set()
                unique_matches = []
                for match in author_matches:
                    paper_id = match["metadata"]["paper_id"]
                    if paper_id not in seen_papers:
                        seen_papers.add(paper_id)
                        unique_matches.append(match)

                print(f"Found {len(unique_matches)} papers by author: {search_term}")
                return unique_matches[:top_k]

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

        # Default semantic search for other query types
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
                        "score": 1.0 / (1.0 + distances[0][i]),
                    }
                )

        return results

    def keyword_search(self, query, top_k=5):
        """Keyword-based search using BM25 with improved topic handling"""
        # For topic searches, remove common research words that add noise
        if not query.startswith("ti:") and not query.startswith("au:"):
            # Clean query for tokenization
            clean_query = query
            for prefix in ["ti:", "au:", "abs:", "all:"]:
                if clean_query.startswith(prefix):
                    clean_query = clean_query.replace(prefix, "", 1)

            # Remove noise words for better topic matching
            clean_query = clean_query.replace('"', "")
            noise_words = [
                "research",
                "paper",
                "study",
                "analysis",
                "about",
                "on",
                "of",
                "in",
            ]
            query_words = clean_query.lower().split()
            filtered_words = [w for w in query_words if w not in noise_words]

            # If we stripped too many words, keep original
            if len(filtered_words) >= 2:
                clean_query = " ".join(filtered_words)
                print(f"Optimized topic search terms: '{clean_query}'")
                tokenized_query = self._tokenize(clean_query)
            else:
                tokenized_query = self._tokenize(clean_query)
        else:
            # For title and author searches, use normal tokenization
            tokenized_query = self._tokenize(query)

        bm25_scores = self.bm25.get_scores(tokenized_query)

        # For topic searches, prioritize paper relevance instead of chunk relevance
        if not query.startswith("ti:") and not query.startswith("au:"):
            # Group scores by paper_id
            paper_scores = {}
            for idx, score in enumerate(bm25_scores):
                paper_id = self.metadata[idx]["paper_id"]
                if (
                    paper_id not in paper_scores
                    or score > paper_scores[paper_id]["score"]
                ):
                    paper_scores[paper_id] = {
                        "metadata": self.metadata[idx],
                        "score": score,
                        "chunk": self.chunks[idx],
                        "chunk_idx": idx,
                    }

            # Sort by score and take top_k
            sorted_results = sorted(
                paper_scores.values(), key=lambda x: x["score"], reverse=True
            )
            return sorted_results[:top_k]
        else:
            # Standard approach for title and author searches
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
        """Combine semantic and keyword search with improved topic relevance"""
        # Adjust alpha based on query type to optimize for topics
        if not query.startswith("ti:") and not query.startswith("au:"):
            # For topic searches, give more weight to semantic search
            topic_alpha = 0.8  # Higher weight on semantic search for concepts
            alpha = topic_alpha
            print(f"Using topic-optimized alpha: {alpha}")

        semantic_results = self.semantic_search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search(query, top_k=top_k * 2)

        # Combine results with a score normalization
        combined_results = {}

        # Get max scores with added safety checks
        try:
            if semantic_results:
                valid_scores = [
                    float(r["score"])
                    for r in semantic_results
                    if isinstance(r["score"], (int, float))
                    and not math.isnan(r["score"])
                    and not math.isinf(r["score"])
                ]
                max_semantic_score = max(valid_scores) if valid_scores else 1.0
            else:
                max_semantic_score = 1.0

            if keyword_results:
                valid_scores = [
                    float(r["score"])
                    for r in keyword_results
                    if isinstance(r["score"], (int, float))
                    and not math.isnan(r["score"])
                    and not math.isinf(r["score"])
                ]
                max_keyword_score = max(valid_scores) if valid_scores else 1.0
            else:
                max_keyword_score = 1.0

            # Ensure we don't have zero division
            if max_semantic_score <= 0:
                max_semantic_score = 1.0
            if max_keyword_score <= 0:
                max_keyword_score = 1.0
        except Exception as e:
            print(f"Error calculating max scores: {e}")
            max_semantic_score = 1.0
            max_keyword_score = 1.0

        # Process semantic results
        for result in semantic_results:
            try:
                idx = (
                    result["metadata"]["paper_id"]
                    + "_"
                    + str(result["metadata"]["chunk_index"])
                )

                # Handle potential NaN or inf in scores
                score = result["score"]
                if (
                    not isinstance(score, (int, float))
                    or math.isnan(score)
                    or math.isinf(score)
                ):
                    score = 0.5  # Default fallback value

                # Safe division with sanity check
                normalized_score = (
                    float(score) / max_semantic_score if max_semantic_score > 0 else 0.5
                )
                # Ensure normalized_score is in [0, 1]
                normalized_score = max(0.0, min(normalized_score, 1.0))

                combined_results[idx] = {
                    "metadata": result["metadata"],
                    "chunk": result["chunk"],
                    "semantic_score": normalized_score,
                    "keyword_score": 0,
                    "combined_score": alpha * normalized_score,
                }
            except Exception as e:
                print(f"Error processing semantic result: {e}")
                continue

        # Process keyword results
        for result in keyword_results:
            try:
                idx = (
                    result["metadata"]["paper_id"]
                    + "_"
                    + str(result["metadata"]["chunk_index"])
                )

                # Handle potential NaN or inf in scores
                score = result["score"]
                if (
                    not isinstance(score, (int, float))
                    or math.isnan(score)
                    or math.isinf(score)
                ):
                    score = 0.5  # Default fallback value

                # Safe division with sanity check
                normalized_score = (
                    float(score) / max_keyword_score if max_keyword_score > 0 else 0.5
                )
                # Ensure normalized_score is in [0, 1]
                normalized_score = max(0.0, min(normalized_score, 1.0))

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
            except Exception as e:
                print(f"Error processing keyword result: {e}")
                continue

        # Final validation of combined scores
        for idx in combined_results:
            score = combined_results[idx]["combined_score"]
            if (
                not isinstance(score, (int, float))
                or math.isnan(score)
                or math.isinf(score)
            ):
                combined_results[idx]["combined_score"] = 0.5
            else:
                # Ensure score is in valid range
                combined_results[idx]["combined_score"] = max(
                    0.0, min(float(score), 1.0)
                )

        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(), key=lambda x: x["combined_score"], reverse=True
        )
        return sorted_results[:top_k]
