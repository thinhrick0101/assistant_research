import json
import os


class ResearchPaperPipeline:
    def __init__(self):
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.items = []

    def process_item(self, item, spider):
        self.items.append(dict(item))
        return item

    def close_spider(self, spider):
        output_file = os.path.join(self.output_dir, f"{spider.name}_papers.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)
        spider.logger.info(f"Saved {len(self.items)} papers to {output_file}")
