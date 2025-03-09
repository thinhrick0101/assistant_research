import os
import json
import numpy as np
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import faiss
import pickle


class EmbeddingManager:
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model
        self.tokenizer = tiktoken.get_encoding(
            "cl100k_base"
        )  # ada uses cl100k_base tokenizer
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.index_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "index"
        )
        os.makedirs(self.index_dir, exist_ok=True)

    def adaptive_chunk_text(
        self, text, min_chunk_size=200, max_chunk_size=1000, chunk_overlap=50
    ):
        """Adaptively chunk text based on content structure"""
        # Use RecursiveCharacterTextSplitter for smart chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: len(self.tokenizer.encode(x)),
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = text_splitter.split_text(text)

        # Combine very small chunks if needed
        result_chunks = []
        current_chunk = ""

        for chunk in chunks:
            if len(self.tokenizer.encode(current_chunk + chunk)) <= max_chunk_size:
                current_chunk += chunk
            else:
                if current_chunk:
                    result_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            result_chunks.append(current_chunk)

        return result_chunks

    def get_embedding(self, text):
        """Get embedding for a text using OpenAI API"""
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_papers(self, source="arxiv"):
        """Load papers from JSON and create embeddings for abstracts"""
        papers_file = os.path.join(self.data_dir, f"{source}_spider_papers.json")
        if not os.path.exists(papers_file):
            print(f"No papers file found at {papers_file}")
            return None, None

        with open(papers_file, "r", encoding="utf-8") as f:
            papers = json.load(f)

        # Store paper chunks and their metadata
        all_chunks = []
        all_metadata = []
        text_chunks = []  # Store the actual text chunks separately

        for paper in papers:
            # First embed the abstract
            abstract_chunks = self.adaptive_chunk_text(paper["abstract"])

            for i, chunk in enumerate(abstract_chunks):
                text_chunks.append(chunk)  # Store the actual text
                all_chunks.append(chunk)
                all_metadata.append(
                    {
                        "paper_id": paper["paper_id"],
                        "title": paper["title"],
                        "source": paper["source"],
                        "chunk_type": "abstract",
                        "chunk_index": i,
                        "url": paper["url"],
                    }
                )

        # Get embeddings for all chunks
        embeddings = [self.get_embedding(chunk) for chunk in all_chunks]

        # Save embeddings and metadata
        return np.array(embeddings, dtype=np.float32), all_metadata, text_chunks

    def build_faiss_index(self, source="arxiv"):
        """Build and save FAISS index for faster similarity search"""
        embeddings, metadata, text_chunks = self.embed_papers(source)

        if embeddings is None:
            return

        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save index, metadata, and text chunks
        faiss.write_index(index, os.path.join(self.index_dir, f"{source}_index.faiss"))
        with open(os.path.join(self.index_dir, f"{source}_metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        with open(os.path.join(self.index_dir, f"{source}_chunks.pkl"), "wb") as f:
            pickle.dump(text_chunks, f)  # Save actual text chunks, not metadata

        print(f"Index built and saved for {source} with {len(metadata)} chunks")
