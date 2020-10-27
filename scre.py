# -*- coding: utf-8 -*-

import re
import time
import os
import sys
import datetime

import requests
from bs4 import BeautifulSoup


def download_images(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text.encode(res.encoding), 'html.parser')
    image_items = soup.findAll('a', href=re.compile('(\.jpg|\.png)$'))
    image_urls = [item.get('href') for item in image_items]

    now = datetime.datetime.today()
    dirname = '{}{:0>2}{:0>2}{:0>2}{:0>2}{:0>6}'\
              .format(now.year, now.month, now.day,
                      now.minute, now.second, now.microsecond)
    os.mkdir(dirname)

    first_url = None
    isdup = False
    for i, url in enumerate(image_urls):
        image_data = requests.get(url).content  # バイナリデータ
        _, ext = os.path.splitext(url)
        if first_url is None:
            first_url = url
        elif first_url == url:
            isdup = True
        with open('{0}/{1:0>3}{2}'.format(dirname, i, ext), 'wb') as f:
            filename = '{:0>3}{}'.format(i, ext)
            print('downloading ' + filename)
            f.write(image_data)
        time.sleep(1)
    if isdup:
        print('delete 000' + ext)
        os.remove(dirname + '/000' + ext)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        download_images(sys.argv[1])
    else:
        print('arguments error')
