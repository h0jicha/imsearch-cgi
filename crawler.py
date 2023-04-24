import os
from PIL import Image

prefix = '/export/space0/tanabe-h/data/kadai3a/collected/'
kw = 'panda'
directory = prefix + 'img'

from icrawler.builtin import GoogleImageCrawler
crawler = GoogleImageCrawler(storage={"root_dir":directory},downloader_threads=5)
crawler.crawl(keyword=kw, max_num=3000)

for filename in os.listdir(directory):
    if filename.endswith('.png'):
        png_file = os.path.join(directory, filename)
        jpg_file = os.path.join(directory, filename[:-4] + '.jpg')
        img = Image.open(png_file)
        img.convert('RGB').save(jpg_file)
        os.remove(png_file)
        print('converted: ' + jpg_file)
