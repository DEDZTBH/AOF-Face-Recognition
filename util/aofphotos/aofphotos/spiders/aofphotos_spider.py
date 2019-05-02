import scrapy
import json
import requests
from os import path
from scrapy.crawler import CrawlerProcess

save_path = 'known'


class AOFPhotosSpider(scrapy.Spider):
    name = "aofphotos"

    def start_requests(self):
        urls = [
            'https://aof-studios.photoshelter.com/gallery-collection/2018-2019-School-Year/C0000Awpd5DVwCGY'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        letters_url = response.css('.thumbnail').re(r'<a href="([0-9A-Za-z./\-_]+)">')
        for url in letters_url:
            yield response.follow(url, callback=self.parse_letter)

    def parse_letter(self, response):
        pic_objs = response.css('.thumbnail')
        pic_urls = pic_objs.re(r'src="(http.+\.jpg)"')
        pic_names = [
            json.loads(s)['I_FILE_NAME']
            for s in pic_objs.re(r'{.*}')
        ]
        pic_urls = [u.replace('200x200', '500x500') for u in pic_urls]

        for name, url in zip(pic_names, pic_urls):
            r = requests.get(url)
            open(path.join(save_path, name), 'wb').write(r.content)

        next_page_maybe = response.css('.page_next')

        if next_page_maybe.get() is not None:
            next_page_url = next_page_maybe.re(r'<a href="(.+)" ')[0]
            yield scrapy.Request(response.urljoin(next_page_url), callback=self.parse_letter)


process = CrawlerProcess({
    'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
})

process.crawl(AOFPhotosSpider)
process.start()
