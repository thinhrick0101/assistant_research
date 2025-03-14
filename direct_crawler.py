"""Direct crawler for arXiv papers without using Scrapy."""

import os
import json
import arxiv
from datetime import datetime
import time
import re


def crawl_arxiv(query, max_results=100, output_dir="data"):
    """Crawl arXiv papers using the arXiv API directly."""
    print(f"Crawling arXiv for '{query}', max results: {max_results}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Detect search type
    is_author_search = query.startswith("au:") or any(
        word[0].isupper() and word[1:].islower() for word in query.split()
    )
    is_title_search = query.startswith("ti:")

    # Extract the search term from prefixed queries
    original_query = query
    search_term = query
    for prefix in ["au:", "ti:", "abs:", "cat:", "all:"]:
        if query.startswith(prefix):
            search_term = query[len(prefix) :].strip()
            # Remove quotes if present
            if search_term.startswith('"') and search_term.endswith('"'):
                search_term = search_term[1:-1]
            break

    # Check if this looks like a paper title (capitalized words, 4+ words)
    words = query.split()
    capitalized_words = [w for w in words if w[0].isupper() if len(w) > 1]
    likely_title = len(words) >= 4 and len(capitalized_words) >= 1

    # Try multiple query formats if needed
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=5,
    )

    # Construct query formats based on the search
    query_formats = []

    # If this looks like a paper title, prioritize title search
    if likely_title:
        query_formats = [
            f'ti:"{query}"',  # Exact title match (high priority)
            f"ti:{query}",  # Title contains words
            query,  # Original query
            # Additional formats...
        ]
    else:
        # Standard search formats
        query_formats = [
            query,  # Original query first
            f'"{query}"',  # Exact phrase
            f"ti:{query}",  # Title only
            f"abs:{query}",  # Abstract only
            f"cat:cs.* AND {query}",  # CS categories
            f"all:{query}",  # All fields
        ]

    # Special handling for author searches
    if is_author_search and not query.startswith("au:"):
        # If it looks like an author name but doesn't have the prefix, add it
        query = f'au:"{search_term}"'
        print(f"Detected author name. Using query: {query}")

    if is_author_search:
        # Author search formats - try more precise formats
        query_formats = [
            f'au:"{search_term}"',  # Exact author match with quotes
            f"au:{search_term}",  # Author match without quotes
        ]

        # If this looks like a full name, try searching with last name only as well
        if " " in search_term:
            last_name = search_term.split()[-1]
            query_formats.append(f'au:"{last_name}"')
            query_formats.append(f"au:{last_name}")

    # Detect if this is a topic/concept search (not author, not title)
    is_topic_search = not is_author_search and not is_title_search

    # For topic searches, optimize the query formats
    if is_topic_search:
        # Prepare multiple query formats for better topic coverage
        search_terms = search_term.split()

        # Start with specific query formats for better relevance
        query_formats = [
            query,  # Original query
            f'"{search_term}"',  # Exact phrase
            f"abs:{search_term}",  # Abstract contains terms
        ]

        # If it's a multi-word topic, add combinations
        if len(search_terms) > 1:
            # Add title + abstract combinations for better coverage
            query_formats.append(f'ti:"{search_term}" OR abs:"{search_term}"')

            # Add AND format to require all terms
            and_query = " AND ".join([f"({term})" for term in search_terms])
            query_formats.append(and_query)

            # For topics, use a lower max_retries to avoid API timeouts
            client = arxiv.Client(
                page_size=100,
                delay_seconds=1,
                num_retries=3,
            )

    # Try each query format until we get results
    papers = []
    used_query = ""

    for idx, current_query in enumerate(query_formats):
        try:
            print(f"Trying query format {idx+1}/{len(query_formats)}: {current_query}")

            search = arxiv.Search(
                query=current_query,
                max_results=int(max_results),
                sort_by=arxiv.SortCriterion.Relevance,
            )

            temp_papers = []
            count = 0

            # Process results
            for result in client.results(search):
                try:
                    paper = process_paper(result)
                    temp_papers.append(paper)
                    count += 1
                    if count % 10 == 0:
                        print(f"Found {count} papers so far...")

                    # Break if we have enough results
                    if count >= int(max_results):
                        break
                except Exception as e:
                    print(f"Error processing individual paper: {str(e)}")
                    continue

            # If we got any results, keep them and break the loop
            if temp_papers:
                papers = temp_papers
                used_query = current_query
                print(f"Found {len(papers)} papers with query format: {used_query}")
                break
            else:
                print(f"No results for query format: {current_query}")

        except Exception as e:
            print(f"Error with query format {current_query}: {str(e)}")
            continue

    # If we still don't have papers, try browsing specific categories
    if not papers:
        try:
            print("Attempting direct category browsing...")
            # Try specifically browsing recent papers from relevant categories
            categories = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE", "stat.ML"]

            for category in categories[:2]:  # Try just a couple categories to be faster
                print(f"Browsing recent papers in {category}...")
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=100,  # Search more papers
                    sort_by=arxiv.SortCriterion.SubmittedDate,  # Get newest papers
                )

                # Look for relevant papers
                temp_papers = []
                for result in client.results(search):
                    try:
                        # Check if our query appears in the title or abstract
                        if (
                            query.lower() in result.title.lower()
                            or query.lower() in result.summary.lower()
                        ):
                            paper = process_paper(result)
                            temp_papers.append(paper)
                            if len(temp_papers) >= int(max_results):
                                break
                    except Exception as e:
                        print(f"Error processing paper in category search: {str(e)}")
                        continue

                if temp_papers:
                    papers = temp_papers
                    print(f"Found {len(papers)} relevant papers in category {category}")
                    break
        except Exception as e:
            print(f"Error in category browsing: {str(e)}")

    # If still no papers, try explicit search for famous papers
    if not papers and likely_title:
        try:
            print("Trying direct search for the paper...")
            # Direct search for famous papers by ID (for papers we know should exist)
            known_papers = {
                "attention is all you need": "1706.03762",
                "bert": "1810.04805",
                "gpt": "2005.14165",
                "transformers": "1706.03762",
            }

            # Check if we can find a match for a known paper
            paper_id = None
            for title, pid in known_papers.items():
                if title.lower() in query.lower():
                    paper_id = pid
                    break

            if paper_id:
                try:
                    print(f"Found known paper ID: {paper_id}")
                    search = arxiv.Search(id_list=[paper_id])
                    result = next(client.results(search))
                    papers = [process_paper(result)]
                except Exception as e:
                    print(f"Error retrieving known paper: {str(e)}")

        except Exception as e:
            print(f"Error in direct paper search: {str(e)}")

    # When processing papers in an author search, verify author matches
    if is_author_search and papers:
        # Filter out papers that don't have the author
        verified_papers = []
        author_name_parts = search_term.lower().split()

        for paper in papers:
            author_match = False
            # Check if any author name contains ALL parts of the search name
            for author in paper["authors"]:
                author_lower = author.lower()
                # Check if all name parts appear in the author name
                if all(part in author_lower for part in author_name_parts):
                    author_match = True
                    break

            if author_match:
                verified_papers.append(paper)

        # Use the verified papers if we found any, otherwise keep all results
        if verified_papers:
            papers = verified_papers
            print(f"Filtered to {len(papers)} papers by author: {search_term}")

    # For topic searches, prioritize papers with most recent or most cited
    if is_topic_search and len(papers) > max_results:
        # Sort papers by date (newest first) as a simple proxy for relevance
        papers.sort(key=lambda p: p.get("published_date", ""), reverse=True)

    # Save papers to JSON file if any were found
    if papers:
        output_file = os.path.join(output_dir, "arxiv_spider_papers.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(papers)} papers to {output_file}")

        # Print paper titles for confirmation
        print("\nPaper titles found:")
        for i, paper in enumerate(papers[:5]):  # Print first 5 papers
            print(f"{i+1}. {paper['title']}")
        if len(papers) > 5:
            print(f"...and {len(papers) - 5} more papers")

        return len(papers)
    else:
        print("No papers were found for the query despite multiple attempts.")
        return 0


def process_paper(result):
    """Process a paper from arXiv API result with improved error handling"""
    try:
        # Get categories - handle different formats the API might return
        categories = []
        if hasattr(result, "categories"):
            if isinstance(result.categories, list):
                # If categories is already a list of strings
                for cat in result.categories:
                    if isinstance(cat, str):
                        categories.append(cat)
                    elif hasattr(cat, "term"):
                        categories.append(cat.term)
            elif isinstance(result.categories, str):
                # If categories is a single string
                categories = [result.categories]

        return {
            "paper_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "published_date": (
                result.published.strftime("%Y-%m-%d") if result.published else ""
            ),
            "source": "arxiv",
            "categories": categories,
        }
    except Exception as e:
        print(f"Error in process_paper: {str(e)}")
        # Provide a minimal valid paper structure in case of error
        return {
            "paper_id": "unknown",
            "title": getattr(result, "title", "Unknown Title"),
            "authors": [],
            "abstract": getattr(result, "summary", "No abstract available"),
            "url": getattr(result, "entry_id", ""),
            "pdf_url": getattr(result, "pdf_url", ""),
            "published_date": "",
            "source": "arxiv",
            "categories": [],
        }


if __name__ == "__main__":
    import sys

    # Default parameters
    query = "machine learning"
    max_results = 50

    # Get parameters from command line arguments
    if len(sys.argv) >= 2:
        query = sys.argv[1]
    if len(sys.argv) >= 3:
        max_results = int(sys.argv[2])

    # Execute crawling
    crawl_arxiv(query, max_results)
