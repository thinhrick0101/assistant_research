import streamlit as st
import os
import subprocess
import sys
from dotenv import load_dotenv
from embedding.embedding_manager import EmbeddingManager
from retrieval.hybrid_retriever import HybridRetriever
from summarizer.recursive_summarizer import RecursiveSummarizer

# Load environment variables
load_dotenv()

# App title
st.set_page_config(page_title="Research Paper Assistant", layout="wide")
st.title("AI Research Assistant")

# Sidebar for search configuration
st.sidebar.title("Settings")

# Select data source
source = st.sidebar.selectbox("Choose Source", ["arxiv"], index=0)

# Set search parameters
search_type = st.sidebar.radio(
    "Search Type", ["Hybrid (Recommended)", "Semantic Only", "Keyword Only"]
)
result_count = st.sidebar.slider(
    "Number of Results", min_value=1, max_value=10, value=3
)

# Tabs for different functions
tab1, tab2, tab3 = st.tabs(["Search Papers", "Crawl New Papers", "Build Index"])

# Tab 1: Search Papers
# (Same as original app)

# Tab 2: Crawl New Papers
with tab2:
    st.subheader("Crawl New Research Papers")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        crawl_query = st.text_input(
            "Search Term (e.g., 'machine learning')", key="crawl_query"
        )
    with col2:
        max_papers = st.number_input(
            "Max Papers", min_value=10, max_value=1000, value=50, step=10
        )
    with col3:
        crawl_button = st.button("Start Crawling", type="primary")

    if crawl_button:
        if not crawl_query:
            st.warning("Please enter a search term")
        else:
            try:
                # Import the direct crawler
                from direct_crawler import crawl_arxiv

                with st.spinner(
                    f"Crawling {source} for papers about '{crawl_query}'..."
                ):
                    # Use the direct crawler
                    output_dir = os.path.join(os.path.dirname(__file__), "data")
                    count = crawl_arxiv(crawl_query, max_papers, output_dir)

                    if count > 0:
                        st.success(f"Successfully crawled {count} papers from {source}")
                        st.info("Next, build the search index in the 'Build Index' tab")
                    else:
                        st.error("No papers found. Try a different search term.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

# Tab 3: Build Index
# (Same as original app)

# Additional info in sidebar
# (Same as original app)
