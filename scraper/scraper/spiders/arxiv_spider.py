import scrapy
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
