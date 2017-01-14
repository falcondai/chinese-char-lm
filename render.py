#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image, ImageFont, ImageDraw
import numpy as np

def render(text, font):
    mask = font.getmask(text)
    size = mask.size[::-1]
    a = np.asarray(mask).reshape(size)
    return a

def ascii_print(glyph_array):
    for l in glyph_array:
        for c in l:
            if c != 0:
                print '#',
            else:
                print ' ',
        print

if __name__ == '__main__':
    s = u'你好'
    print 'check utf-8 support:',
    print s.encode('utf-8')
    font = ImageFont.truetype('NotoSansCJKsc-hinted/NotoSansCJKsc-Regular.otf', 24)
    a = render(s, font)
    print a
    ascii_print(a)
