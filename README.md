# Research Paper Assistant

An AI-powered tool for searching, analyzing, and understanding academic papers from arXiv and other academic sources.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage Guide](#usage-guide)


## Overview

The Research Paper Assistant is a powerful tool designed to help researchers discover and understand academic papers. It uses advanced AI techniques including semantic search, keyword matching, and automatic summarization to deliver relevant research papers and extract valuable insights from them.

## Features

- **Multi-mode Paper Search**: Search by topic/concept, paper title, or author
- **Intelligent Retrieval**: Combines semantic and keyword search for optimal results
- **AI Summarization**: Automatically generates summaries and key points
- **Relevance Scoring**: Ranks papers by relevance to your query
- **Topic Overviews**: Creates summaries of entire research topics
- **Author Analysis**: Special handling for author searches with contribution analysis
- **Direct Paper Fetching**: Retrieves papers in real-time from arXiv

## System Architecture

The system consists of the following components:

1. **Data Collection**: Crawls papers from arXiv using the arXiv API
2. **Index Building**: Creates embeddings and search indices for efficient retrieval
3. **Search Engine**: Provides hybrid search combining semantic and keyword matching
4. **Summarization Engine**: Generates summaries and extracts key points
5. **Web UI**: Streamlit-based user interface for interacting with the system

┌───────────────┐     ┌──────────────┐     ┌────────────────┐
│ Paper Crawler │────▶│ Index Builder│────▶│ Search Engine  │
└───────────────┘     └──────────────┘     └────────────────┘
                                                   │
                                                   ▼
┌──────────────┐                          ┌────────────────┐
│    User      │◀─────────────────────────│   Streamlit    │
│  Interface   │                          │      UI        │
└──────────────┘                          └────────────────┘
                                                   ▲
                                                   │
                                           ┌────────────────┐
                                           │ Summarization  │
                                           │    Engine      │
                                           └────────────────


## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- Internet connection for paper crawling

### Installation
Clone the repository:
```sh
git clone [repository-url]
cd assistant_research
pip install -r requirements.txt
```
Create a .env file with your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

Run the application:
```sh
streamlit run app.py
```

