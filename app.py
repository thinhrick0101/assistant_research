import streamlit as st
import os
import sys
import math
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
source = st.sidebar.selectbox("Choose Source", ["arXiv"], index=0)

# Set search parameters
search_type = st.sidebar.radio(
    "Search Type", ["Hybrid (Recommended)", "Semantic Only", "Keyword Only"]
)
result_count = st.sidebar.slider(
    "Number of Results", min_value=1, max_value=10, value=3
)

# Main UI with just search functionality
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

# Handle search logic
if search_button and query:
    try:
        with st.spinner("Searching for relevant papers..."):
            # Initialize actual_query with the default value
            actual_query = query

            # Enhanced query processing for different search types
            if search_type_hint == "Author":
                # Format author search query properly
                author_name = query
                if not author_name.startswith("au:"):
                    # Format as author search with quotes for exact match
                    actual_query = f'au:"{author_name}"'
                    st.info(f"Searching for papers by author: {author_name}")
                else:
                    actual_query = author_name

                # Try direct crawling for author search
                try:
                    from direct_crawler import crawl_arxiv

                    output_dir = os.path.join(os.path.dirname(__file__), "data")
                    st.info(f"Looking for papers by author: {author_name}")
                    count = crawl_arxiv(actual_query, 20, output_dir)
                    if count > 0:
                        # Build index on these results
                        with st.spinner("Building search index for found papers..."):
                            embedding_manager = EmbeddingManager()
                            embedding_manager.build_faiss_index(source=source)
                except Exception as e:
                    print(f"Error in direct author search: {str(e)}")

            elif search_type_hint == "Paper Title" and not query.startswith("ti:"):
                actual_query = f'ti:"{query}"'  # Wrap in quotes for exact title
            elif search_type_hint == "Author" and not query.startswith("au:"):
                actual_query = f'au:"{query}"'  # Format as author search
            elif search_type_hint == "Topic/Concept":
                # New specialized handling for topic/concept searches
                st.info(f"Searching for papers about: {query}")

                # For concepts, enhance the query if it's short
                if len(query.split()) <= 3:
                    # Try to search in both title and abstract for better topic coverage
                    actual_query = f'ti:{query} OR abs:"{query}"'

                # For topic searches, crawl more papers to improve coverage
                try:
                    from direct_crawler import crawl_arxiv

                    output_dir = os.path.join(os.path.dirname(__file__), "data")
                    with st.status("Finding papers on this topic...") as status:
                        # Use more papers for topic searches
                        topic_count = max(20, result_count * 3)
                        count = crawl_arxiv(actual_query, topic_count, output_dir)

                        if count > 0:
                            status.update(
                                label=f"Found {count} papers on this topic!",
                                state="complete",
                            )
                            # Build index on these results
                            with st.spinner(
                                "Building search index for found papers..."
                            ):
                                embedding_manager = EmbeddingManager()
                                embedding_manager.build_faiss_index(source=source)
                        else:
                            status.update(
                                label="Trying alternative search methods...",
                                state="running",
                            )
                except Exception as e:
                    print(f"Error in topic search: {str(e)}")

            # Automatically handle crawling in the background
            try:
                from direct_crawler import crawl_arxiv

                output_dir = os.path.join(os.path.dirname(__file__), "data")
                with st.status("Finding papers...") as status:
                    # Use appropriate count based on search type
                    if search_type_hint == "Topic/Concept":
                        count = crawl_arxiv(actual_query, 20, output_dir)
                    elif search_type_hint == "Author":
                        count = crawl_arxiv(actual_query, 20, output_dir)
                    else:  # Paper Title
                        count = crawl_arxiv(actual_query, 10, output_dir)

                    if count > 0:
                        status.update(label=f"Found {count} papers!", state="complete")
                        # Automatically build index
                        with st.spinner("Preparing search index..."):
                            embedding_manager = EmbeddingManager()
                            embedding_manager.build_faiss_index(source=source)
                    else:
                        status.update(
                            label="Trying alternative search methods...",
                            state="running",
                        )
            except Exception as e:
                print(f"Error in background search: {str(e)}")

            # Now perform search as normal
            retriever = HybridRetriever(source=source)

            if search_type == "Hybrid (Recommended)":
                results = retriever.hybrid_search(actual_query, top_k=result_count)
            elif search_type == "Semantic Only":
                results = retriever.semantic_search(actual_query, top_k=result_count)
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
                            paper_results[paper_id]["scores"].append(result["score"])
                    except Exception as e:
                        st.warning(f"Error processing a search result: {str(e)}")
                        continue

                # Special handling for topic/concept search results display
                if search_type_hint == "Topic/Concept" and paper_results:
                    # Display a heading for topic results
                    st.subheader(f"📚 Papers about '{query}'")

                    # Add an explanation of the topic relevance
                    st.info(
                        f"Showing {len(paper_results)} papers relevant to this topic. Papers are ranked by relevance to your query."
                    )

                    # Display each paper
                    for paper_id, paper_data in paper_results.items():
                        st.subheader(paper_data["metadata"]["title"])
                        st.write(
                            f"Source: {paper_data['metadata']['source']} | [View Paper]({paper_data['metadata']['url']})"
                        )

                        # Fix for NaN in progress bar - handle empty lists and invalid values
                        try:
                            if paper_data["scores"] and len(paper_data["scores"]) > 0:
                                # Filter out any non-numeric or NaN values
                                valid_scores = []
                                for score in paper_data["scores"]:
                                    try:
                                        score_value = float(score)
                                        if not (
                                            math.isnan(score_value)
                                            or math.isinf(score_value)
                                        ):
                                            valid_scores.append(score_value)
                                    except (ValueError, TypeError):
                                        continue

                                if valid_scores:
                                    avg_score = sum(valid_scores) / len(valid_scores)
                                    # Only show results with meaningful scores (above threshold)
                                    if avg_score < 0.2:  # Very low relevance
                                        continue  # Skip this result
                                else:
                                    # No valid scores means this is likely not relevant
                                    continue  # Skip this result
                            else:
                                # No scores means this is likely not relevant
                                continue  # Skip this result

                            # Final safety check before displaying
                            if (
                                not isinstance(avg_score, (int, float))
                                or math.isnan(avg_score)
                                or math.isinf(avg_score)
                            ):
                                continue  # Skip invalid results

                            # Ensure value is in [0.0, 1.0] range
                            safe_score = max(0.0, min(float(avg_score), 1.0))
                            st.progress(
                                safe_score,
                                text=f"Relevance Score: {safe_score:.2f}",
                            )
                        except Exception as e:
                            # Skip results with errors in score calculation
                            continue

                        # Only proceed with displaying content for valid results
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

                    # Add a topic summary at the end
                    if len(paper_results) >= 2:
                        st.subheader("Topic Summary")
                        with st.spinner("Generating topic overview..."):
                            try:
                                summarizer = RecursiveSummarizer()
                                # Collect titles and abstracts
                                topic_content = []
                                for paper_id, paper_data in list(paper_results.items())[
                                    :5
                                ]:  # Use top 5 papers
                                    title = paper_data["metadata"]["title"]
                                    chunks = paper_data["chunks"]
                                    abstract = (
                                        chunks[0] if chunks else ""
                                    )  # First chunk is typically the abstract
                                    topic_content.append(
                                        f"Title: {title}\nAbstract: {abstract[:500]}..."
                                    )

                                # Generate a topic summary
                                topic_text = "\n\n".join(topic_content)
                                prompt = f"Based on these papers about '{query}', provide a brief overview of this research topic:"
                                topic_summary = summarizer.answer_specific_question(
                                    topic_text, prompt
                                )
                                st.write(topic_summary)
                            except Exception as e:
                                st.error(f"Error generating topic summary: {str(e)}")
                elif search_type_hint == "Author" and paper_results:
                    author_name = query.replace("au:", "").replace('"', "")

                    # Display a heading for author results
                    st.subheader(f"📚 Papers by {author_name}")

                    # Add an info message about the search
                    st.info(
                        f"Showing {len(paper_results)} papers authored by {author_name}"
                    )

                    # Display each paper with author highlighted
                    for paper_id, paper_data in paper_results.items():
                        st.subheader(paper_data["metadata"]["title"])
                        st.write(
                            f"Source: {paper_data['metadata']['source']} | [View Paper]({paper_data['metadata']['url']})"
                        )

                        # Fix for NaN in progress bar - handle empty lists and invalid values
                        try:
                            if paper_data["scores"] and len(paper_data["scores"]) > 0:
                                # Filter out any non-numeric or NaN values
                                valid_scores = []
                                for score in paper_data["scores"]:
                                    try:
                                        score_value = float(score)
                                        if not (
                                            math.isnan(score_value)
                                            or math.isinf(score_value)
                                        ):
                                            valid_scores.append(score_value)
                                    except (ValueError, TypeError):
                                        continue

                                if valid_scores:
                                    avg_score = sum(valid_scores) / len(valid_scores)
                                    # Only show results with meaningful scores (above threshold)
                                    if avg_score < 0.2:  # Very low relevance
                                        continue  # Skip this result
                                else:
                                    # No valid scores means this is likely not relevant
                                    continue  # Skip this result
                            else:
                                # No scores means this is likely not relevant
                                continue  # Skip this result

                            # Final safety check before displaying
                            if (
                                not isinstance(avg_score, (int, float))
                                or math.isnan(avg_score)
                                or math.isinf(avg_score)
                            ):
                                continue  # Skip invalid results

                            # Ensure value is in [0.0, 1.0] range
                            safe_score = max(0.0, min(float(avg_score), 1.0))
                            st.progress(
                                safe_score,
                                text=f"Relevance Score: {safe_score:.2f}",
                            )
                        except Exception as e:
                            # Skip results with errors in score calculation
                            continue

                        # Only proceed with displaying content for valid results
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

                        # Add a special section to highlight author's contribution
                        with st.expander(f"Author's Contributions"):
                            try:
                                with st.spinner(
                                    f"Analyzing {author_name}'s contributions..."
                                ):
                                    # Extract just the portions of the paper that mention this author
                                    author_mentions = []
                                    for chunk in paper_data["chunks"]:
                                        sentences = chunk.split(". ")
                                        for sentence in sentences:
                                            # Check if the author name appears in this sentence
                                            if author_name.lower() in sentence.lower():
                                                author_mentions.append(
                                                    sentence.strip() + "."
                                                )

                                    if author_mentions:
                                        st.write("Sections mentioning this author:")
                                        for mention in author_mentions[
                                            :3
                                        ]:  # Show just a few mentions
                                            st.markdown(f"> {mention}")
                                    else:
                                        st.write(
                                            "No specific mentions of this author found in the paper chunks."
                                        )
                            except Exception as e:
                                st.error(
                                    f"Error analyzing author contributions: {str(e)}"
                                )

                        st.divider()
                else:
                    # Display each paper
                    for paper_id, paper_data in paper_results.items():
                        st.subheader(paper_data["metadata"]["title"])
                        st.write(
                            f"Source: {paper_data['metadata']['source']} | [View Paper]({paper_data['metadata']['url']})"
                        )

                        # Fix for NaN in progress bar - handle empty lists and invalid values
                        try:
                            if paper_data["scores"] and len(paper_data["scores"]) > 0:
                                # Filter out any non-numeric or NaN values
                                valid_scores = []
                                for score in paper_data["scores"]:
                                    try:
                                        score_value = float(score)
                                        if not (
                                            math.isnan(score_value)
                                            or math.isinf(score_value)
                                        ):
                                            valid_scores.append(score_value)
                                    except (ValueError, TypeError):
                                        continue

                                if valid_scores:
                                    avg_score = sum(valid_scores) / len(valid_scores)
                                    # Only show results with meaningful scores (above threshold)
                                    if avg_score < 0.2:  # Very low relevance
                                        continue  # Skip this result
                                else:
                                    # No valid scores means this is likely not relevant
                                    continue  # Skip this result
                            else:
                                # No scores means this is likely not relevant
                                continue  # Skip this result

                            # Final safety check before displaying
                            if (
                                not isinstance(avg_score, (int, float))
                                or math.isnan(avg_score)
                                or math.isinf(avg_score)
                            ):
                                continue  # Skip invalid results

                            # Ensure value is in [0.0, 1.0] range
                            safe_score = max(0.0, min(float(avg_score), 1.0))
                            st.progress(
                                safe_score,
                                text=f"Relevance Score: {safe_score:.2f}",
                            )
                        except Exception as e:
                            # Skip results with errors in score calculation
                            continue

                        # Only proceed with displaying content for valid results
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
