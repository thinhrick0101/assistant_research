"""Check and fix the project structure."""

import os
import shutil
import sys


def check_structure():
    """Check and attempt to fix the project structure."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Checking project structure in: {current_dir}")

    # Required directories
    required_dirs = [
        "data",
        "index",
        os.path.join("scraper", "scraper", "spiders"),
    ]

    # Check and create required directories
    for dir_path in required_dirs:
        full_path = os.path.join(current_dir, dir_path)
        if not os.path.exists(full_path):
            print(f"Creating directory: {dir_path}")
            os.makedirs(full_path, exist_ok=True)

    # Check required files
    spider_file = os.path.join(
        current_dir, "scraper", "scraper", "spiders", "arxiv_spider.py"
    )
    if not os.path.exists(spider_file):
        print(f"Spider file doesn't exist at: {spider_file}")
        print("Creating spider file...")

        # Get spider code from direct_crawler
        src_code = """import scrapy
import arxiv
from ..items import ResearchPaperItem
from datetime import datetime, timedelta


class ArxivSpider(scrapy.Spider):
    name = "arxiv_spider"
    allowed_domains = ["arxiv.org"]

    def __init__(self, query=None, max_results=100, *args, **kwargs):
        super(ArxivSpider, self).__init__(*args, **kwargs)
        self.query = query or "artificial intelligence"
        self.max_results = int(max_results)

    def start_requests(self):
        # This is a placeholder - we'll use the arxiv API directly in parse
        yield scrapy.Request("https://arxiv.org/search", self.parse)

    def parse(self, response):
        # Using arxiv API instead of scraping directly
        client = arxiv.Client()
        search = arxiv.Search(
            query=self.query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for result in client.results(search):
            item = ResearchPaperItem()
            item["paper_id"] = result.entry_id.split("/")[-1]
            item["title"] = result.title
            item["authors"] = [author.name for author in result.authors]
            item["abstract"] = result.summary
            item["url"] = result.entry_id
            item["pdf_url"] = result.pdf_url
            item["published_date"] = result.published.strftime("%Y-%m-%d")
            item["source"] = "arxiv"
            item["categories"] = [tag.term for tag in result.categories]

            yield item
"""

        with open(spider_file, "w") as f:
            f.write(src_code)
        print("Spider file created successfully!")

    # Check if __init__.py files exist in all needed directories
    init_dirs = [
        os.path.join("scraper", "scraper"),
        os.path.join("scraper", "scraper", "spiders"),
    ]

    for dir_path in init_dirs:
        init_file = os.path.join(current_dir, dir_path, "__init__.py")
        if not os.path.exists(init_file):
            print(f"Creating __init__.py in {dir_path}")
            with open(init_file, "w") as f:
                f.write(
                    "# This file is required to make Python treat this directory as a package.\n"
                )

    print("Project structure check completed!")


if __name__ == "__main__":
    check_structure()
    print("\nRun one of the following commands to start the application:")
    print("1. python run.py (Standard app)")
    print("2. python run_alt.py (Alternative app)")
