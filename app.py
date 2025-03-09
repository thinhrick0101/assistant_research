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
source = st.sidebar.selectbox("Choose Source", ["arxiv", "pubmed", "ssrn"], index=0)

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
with tab1:
    # Search box with improved guidance
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your research topic, question, or paper title")
        search_type_hint = st.radio(
            "Search by:",
            ["Topic/Concept", "Paper Title", "Author"],
            horizontal=True,
            help="Select how you want to search",
        )
    with col2:
        search_button = st.button("Search Papers", type="primary")

    # Show search tips
    with st.expander("Search Tips"):
        st.markdown(
            """
        - **For paper titles**: Enter the complete title like "Attention Is All You Need"
        - **For topics**: Use keywords like "transformer architecture" or "deep learning"
        - **For authors**: Prefix with 'au:' like 'au:Hinton' or 'au:"Geoffrey Hinton"'
        """
        )

    if search_button and query:
        try:
            with st.spinner("Searching for relevant papers..."):
                # Modify query based on search type hint
                actual_query = query
                if search_type_hint == "Paper Title" and not query.startswith("ti:"):
                    actual_query = f'ti:"{query}"'  # Wrap in quotes for exact title
                elif search_type_hint == "Author" and not query.startswith("au:"):
                    actual_query = f'au:"{query}"'  # Format as author search

                # First try direct crawling for exact paper title
                if search_type_hint == "Paper Title":
                    try:
                        from direct_crawler import crawl_arxiv

                        output_dir = os.path.join(os.path.dirname(__file__), "data")
                        st.info("Looking for exact paper title match...")
                        count = crawl_arxiv(query, 10, output_dir)
                        if count > 0:
                            # Build index on these results
                            with st.spinner(
                                "Building search index for found papers..."
                            ):
                                embedding_manager = EmbeddingManager()
                                embedding_manager.build_faiss_index(source=source)
                    except Exception as e:
                        print(f"Error in direct paper title search: {str(e)}")

                # Now perform search as normal
                retriever = HybridRetriever(source=source)

                if search_type == "Hybrid (Recommended)":
                    results = retriever.hybrid_search(actual_query, top_k=result_count)
                elif search_type == "Semantic Only":
                    results = retriever.semantic_search(
                        actual_query, top_k=result_count
                    )
                else:  # Keyword Only
                    results = retriever.keyword_search(actual_query, top_k=result_count)

                if results:
                    # Group chunks by paper
                    paper_results = {}
                    for result in results:
                        try:
                            paper_id = result["metadata"]["paper_id"]
                            if paper_id not in paper_results:
                                paper_results[paper_id] = {
                                    "metadata": result["metadata"],
                                    "chunks": [],
                                    "scores": [],
                                }
                            # Ensure chunk is a string
                            chunk = result["chunk"]
                            if not isinstance(chunk, str):
                                chunk = str(chunk)

                            paper_results[paper_id]["chunks"].append(chunk)
                            if "combined_score" in result:
                                paper_results[paper_id]["scores"].append(
                                    result["combined_score"]
                                )
                            else:
                                paper_results[paper_id]["scores"].append(
                                    result["score"]
                                )
                        except Exception as e:
                            st.warning(f"Error processing a search result: {str(e)}")
                            continue

                    # Display each paper
                    for paper_id, paper_data in paper_results.items():
                        st.subheader(paper_data["metadata"]["title"])
                        st.write(
                            f"Source: {paper_data['metadata']['source']} | [View Paper]({paper_data['metadata']['url']})"
                        )

                        # Show relevance score - Fix for NumPy float32 issue
                        # Convert to native Python float to prevent Streamlit error
                        avg_score = float(
                            sum(paper_data["scores"]) / len(paper_data["scores"])
                        )
                        st.progress(
                            min(avg_score, 1.0),
                            text=f"Relevance Score: {avg_score:.2f}",
                        )

                        # Join chunks for this paper
                        full_text = "\n\n".join(paper_data["chunks"])

                        # FIX: Separate expanders for summary and key points
                        summarizer = RecursiveSummarizer()

                        # Summary expander
                        with st.expander("View Summary"):
                            with st.spinner("Generating summary..."):
                                summary = summarizer.recursive_summarize(
                                    full_text, query
                                )
                                st.write(summary)

                        # Key points expander (now at same level, not nested)
                        with st.expander("View Key Points"):
                            with st.spinner("Extracting key points..."):
                                key_points = summarizer.extract_key_points(summary)
                                st.write(key_points)

                        # Original text expander
                        with st.expander("View Original Text"):
                            st.write(full_text)

                        st.divider()
                else:
                    st.info("No relevant papers found. Try modifying your query.")
        except FileNotFoundError:
            st.error(
                "Search index not found. Please build the index first in the 'Build Index' tab."
            )
        except Exception as e:
            st.error(f"An error occurred during search: {str(e)}")
            st.exception(e)
            st.info("Try rebuilding the search index in the 'Build Index' tab")

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

    # Add sample queries for user convenience
    with st.expander("Need help with search queries?"):
        st.markdown(
            """
        ### Sample Queries:
        - Simple: `transformer neural networks`
        - Advanced: `all:"transformer" AND cat:cs.AI`
        - By Author: `au:Bengio`
        - By Date: `submittedDate:[20220101 TO 20221231]`
        
        ### Popular Topic Suggestions:
        - `"large language models"` (in quotes for exact phrase)
        - `"generative adversarial networks"`
        - `"reinforcement learning"`
        - `"computer vision" AND "deep learning"`
        - `"transformer" AND "attention mechanism"`
        
        For advanced searches, you can use boolean operators (AND, OR, NOT) and fields like:
        - `ti:` for title
        - `abs:` for abstract
        - `au:` for author
        - `cat:` for category (e.g., cs.AI, cs.LG)
        """
        )

    if crawl_button:
        if not crawl_query:
            st.warning("Please enter a search term")
        else:
            try:
                # Use direct crawler for simplicity and reliability
                from direct_crawler import crawl_arxiv

                with st.spinner(
                    f"Crawling {source} for papers about '{crawl_query}'..."
                ):
                    # Use direct crawler which doesn't rely on scrapy infrastructure
                    output_dir = os.path.join(os.path.dirname(__file__), "data")
                    count = crawl_arxiv(crawl_query, max_papers, output_dir)

                    if count > 0:
                        st.success(f"Successfully crawled {count} papers from {source}")
                        st.info("Next, build the search index in the 'Build Index' tab")
                    else:
                        st.error("No papers found with this search term.")
                        st.warning(
                            """
                        ### Try these search suggestions:
                        1. Use quotes for exact phrases: `"transformer model"` 
                        2. Add field specifiers: `ti:transformer OR abs:transformer`
                        3. Try related terms: `"attention mechanism"` or `"neural networks"`
                        4. Include categories: `cat:cs.AI AND transformer`
                        5. Check the "Sample Queries" section for more examples
                        """
                        )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)

# Tab 3: Build Index
with tab3:
    st.subheader("Build Search Index")

    col1, col2 = st.columns([3, 1])

    with col2:
        build_button = st.button("Build Index", type="primary")

    with col1:
        st.write(
            "Build the search index for crawled papers. This process will create embeddings and set up the retrieval system."
        )

    if build_button:
        try:
            data_path = os.path.join(
                os.path.dirname(__file__), "data", f"{source}_spider_papers.json"
            )

            if not os.path.exists(data_path):
                st.error(f"No data found for {source}. Please crawl papers first.")
            else:
                with st.spinner(
                    "Building search index... This may take a few minutes."
                ):
                    # Initialize embedding manager and build index
                    embedding_manager = EmbeddingManager()
                    embedding_manager.build_faiss_index(source=source)

                    st.success("Search index built successfully!")
                    st.info("You can now search for papers in the 'Search Papers' tab")
        except Exception as e:
            st.error(f"An error occurred while building the index: {str(e)}")

# Additional info in sidebar
st.sidebar.divider()
st.sidebar.caption("About")
st.sidebar.info(
    """
    This research assistant helps you discover and understand academic papers.
    
    Features:
    - Hybrid search combining semantic and keyword matching
    - AI-powered paper summarization
    - Key points extraction
    """
)
