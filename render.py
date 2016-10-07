#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image, ImageFont, ImageDraw
import numpy as np

def render(text, font):
    return font.getmask(text)

def print_glyph_mask(mask):
    size = mask.size[::-1]
    a = np.asarray(mask).reshape(size)
    for l in a:
        for c in l:
            if c != 0:
                print '#',
            else:
                print ' ',
        print

if __name__ == '__main__':
    print 'check utf-8 support:',
    print '你好'
    font = ImageFont.truetype('NotoSansCJKsc-hinted/NotoSansCJKsc-Regular.otf', 24)
    m = font.getmask(u'你好')
    print_glyph_mask(m)
