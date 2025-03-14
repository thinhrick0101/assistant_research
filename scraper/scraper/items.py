import scrapy


class ResearchPaperItem(scrapy.Item):
    paper_id = scrapy.Field()
    title = scrapy.Field()
    authors = scrapy.Field()
    abstract = scrapy.Field()
    url = scrapy.Field()
    pdf_url = scrapy.Field()
    published_date = scrapy.Field()
    source = scrapy.Field()
    categories = scrapy.Field()
    full_text = scrapy.Field()
