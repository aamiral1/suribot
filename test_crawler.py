from trafilatura.spider import sitemap_search

urls = sitemap_search("https://surimarketing.co.uk")

print(urls)