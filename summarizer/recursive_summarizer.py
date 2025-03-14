from openai import OpenAI
import tiktoken


class RecursiveSummarizer:
    def __init__(self, model="gpt-3.5-turbo"):
        self.client = OpenAI()
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_text_by_tokens(self, text, max_tokens=3000):
        """Chunk text to respect token limits"""
        chunks = []
        tokens = self.tokenizer.encode(text)

        for i in range(0, len(tokens), max_tokens):
            chunk = self.tokenizer.decode(tokens[i : i + max_tokens])
            chunks.append(chunk)

        return chunks

    def summarize_chunk(self, chunk, query=None, max_tokens=600):
        """Summarize a single chunk of text"""
        if query:
            prompt = f"Please summarize this research paper excerpt, focusing on aspects relevant to '{query}':\n\n{chunk}"
        else:
            prompt = f"Please provide a concise summary of this research paper excerpt:\n\n{chunk}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant that provides accurate, concise summaries of academic papers.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def recursive_summarize(self, text, query=None, max_total_tokens=3000):
        """Recursively summarize large text until it fits within token limit"""
        # First check if the text is already small enough
        if len(self.tokenizer.encode(text)) <= max_total_tokens:
            return self.summarize_chunk(text, query)

        # If text is too large, chunk and summarize each part
        chunks = self.chunk_text_by_tokens(text)
        summaries = [self.summarize_chunk(chunk, query) for chunk in chunks]

        # Combine summaries
        combined = "\n\n".join(summaries)

        # If the combined summary is still too large, recursively summarize it
        if len(self.tokenizer.encode(combined)) > max_total_tokens:
            return self.recursive_summarize(combined, query, max_total_tokens)
        else:
            return combined

    def extract_key_points(self, summary, num_points=5):
        """Extract key points from a summary"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the most important points from this research paper summary.",
                },
                {
                    "role": "user",
                    "content": f"Extract {num_points} key points from this summary:\n\n{summary}",
                },
            ],
            max_tokens=500,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    def answer_specific_question(self, text, question):
        """Answer a specific question based on the paper content"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant. Answer questions precisely based on the paper content provided.",
                },
                {
                    "role": "user",
                    "content": f"Based on the following research paper, please answer this question: '{question}'\n\nPaper content:\n{text}",
                },
            ],
            max_tokens=500,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()
