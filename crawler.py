from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter
import asyncio
from web_chunker import clean_markdown, markdown_to_sections, detect_dominant_level

async def crawl_url(url: str) -> str:
    config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter()
        )
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)

    answer = result.markdown.fit_markdown or result.markdown.raw_markdown

    return answer


if __name__ == "__main__":
    answer = asyncio.run(crawl_url("https://surimarketing.co.uk/our-work/"))
    print(clean_markdown(answer))