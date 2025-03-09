#!/usr/bin/env python
import os
import sys
import subprocess


def check_scrapy_installed():
    """Check if scrapy is installed, and install it if not."""
    try:
        import scrapy

        return True
    except ImportError:
        print("Scrapy not found. Attempting to install scrapy...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "scrapy==2.8.0"]
            )
            return True
        except subprocess.CalledProcessError:
            print(
                "Failed to install scrapy. Please install it manually with: pip install scrapy==2.8.0"
            )
            return False


def run_spider(spider_name, query, max_results):
    # Check if scrapy is available
    if not check_scrapy_installed():
        return 1

    # List current directory structure to debug import issues
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")

    scrapy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scraper")
    print(f"Scrapy project path: {scrapy_path}")

    if os.path.exists(scrapy_path):
        print("Scrapy directory exists")
        spiders_path = os.path.join(scrapy_path, "scraper", "spiders")
        print(f"Spiders path: {spiders_path}")

        if os.path.exists(spiders_path):
            print("Spiders directory exists")
            spider_file = os.path.join(spiders_path, "arxiv_spider.py")
            print(f"Spider file path: {spider_file}")

            if os.path.exists(spider_file):
                print("Spider file exists")
            else:
                print("Spider file does not exist!")
        else:
            print("Spiders directory does not exist!")
    else:
        print("Scrapy directory does not exist!")

    # Try using direct crawler instead as a fallback
    if spider_name == "arxiv":
        try:
            # Import direct crawler
            print("Attempting to use direct crawler method...")
            from direct_crawler import crawl_arxiv

            # Execute direct crawler
            crawl_arxiv(
                query,
                max_results,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
            )
            print(
                f"Successfully crawled papers for query: {query} using direct crawler"
            )
            return 0

        except ImportError as ie:
            print(f"Failed to import ArxivSpider and direct_crawler: {str(ie)}")
            return 1
    else:
        print(f"Unknown spider: {spider_name}")
        return 1


if __name__ == "__main__":
    # Check arguments
    if len(sys.argv) < 4:
        print("Usage: python run_spider.py <spider_name> <query> <max_results>")
        sys.exit(1)

    spider_name = sys.argv[1]
    query = sys.argv[2]
    max_results = sys.argv[3]

    # Run the spider
    exit_code = run_spider(spider_name, query, max_results)
    sys.exit(exit_code)
