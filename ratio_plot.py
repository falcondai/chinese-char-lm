 # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.font_manager import FontProperties
import codecs
from vocabulary_builder import Vocabulary_Builder
from render import render_text
ratio_ls = np.loadtxt('./ratio.txt', dtype=float,  delimiter=',')

char_ratio_ls = []
with open('./work/dict.txt', 'r') as fhandle:
    for i,chara in enumerate(fhandle):
        if i > 4000:
            break
        char_ratio_ls.append((chara.strip().decode('utf8'), ratio_ls[i]))

# sort in-place from highest to lowest
char_ratio_ls.sort(key=lambda x: x[1], reverse=True) 

# bar chart print script #
# ===========================

# save the names and their respective ratioss separately
# reverse the tuples to go from most frequent to least frequent 
# characters = zip(*char_ratio_ls)[0][-20:]
# ratios = zip(*char_ratio_ls)[1][-20:]
# x_pos = np.arange(len(characters))[-20:]

# calculate slope and intercept for the linear trend line

# plt.bar(x_pos, ratios, align='center')
# plt.xticks(x_pos, characters, ) 
# plt.ylabel('id/glyph embedding norm ratio')
# plt.show()

# characters = zip(*char_ratio_ls)[0][:20]
# ratios = zip(*char_ratio_ls)[1][:20]
# x_pos = np.arange(len(characters))[:20]

# calculate slope and intercept for the linear trend line

# plt.bar(x_pos, ratios,align='center')
# plt.xticks(x_pos, characters) 
# plt.ylabel('id/glyph embedding norm ratio')
# plt.show()
# ============================

# ============================
# ratio frequency scatter
# print out top and bottom 100 ratio chars
vb = Vocabulary_Builder()
vb.read_in('./segmentation/msr_train_raw')
vb = vb.vocabulary_dict

ratio_freq_ls = []
for i in char_ratio_ls:
	key = i[0].encode('utf8')
	if key in vb:
		if key != '<P>' and key != '</P>':
			ratio_freq_ls.append((i[1], vb[key]))

print ratio_freq_ls
xx, yy = zip(*ratio_freq_ls)
plt.scatter(xx, np.log(yy))
plt.show()
# =============================

# =============================
# grayscale ratio scatter

# grayscale_dict = {}

# for i in char_ratio_ls:
# 	key = i[0].encode('utf8')
# 	glyph = render_text(key)
# 	ave_gray = np.sum(glyph) / (24.0*24.0)
# 	grayscale_dict[key] = ave_gray

# ratio_gray_ls = []
# for i in char_ratio_ls:
# 	key = i[0].encode('utf8')
# 	if key != '<P>' and key != '</P>':
# 		ratio_gray_ls.append((i[1], grayscale_dict[key]))

# print ratio_gray_ls
# xx, yy = zip(*ratio_gray_ls)
# plt.scatter(xx, yy)
# plt.show()
# =============================

# x = zip(*char_ratio_ls)[1]
# # the histogram of the data
# plt.hist(x, bins='auto')  # plt.hist passes it's arguments to np.histogram
# plt.title("Histogram of ratio")
# # add a 'best fit' line
# # y = mlab.normpdf( bins, mu, sigma)
# # l = plt.plot(bins, y, 'r--', linewidth=1)

# plt.show()