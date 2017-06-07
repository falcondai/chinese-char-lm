 # -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.font_manager import FontProperties
import codecs
from vocabulary_builder import Vocabulary_Builder
from render import render_text
import cPickle as pickle
from matplotlib.patches import Rectangle


ratio_ls = np.loadtxt('./ratio.txt', dtype=float,  delimiter=',')

char_ratio_ls = []
with open('./work/dict.txt', 'r') as fhandle:
    for i,chara in enumerate(fhandle):
        if i > 4000:
            break
        char_ratio_ls.append((chara.strip().decode('utf8'), ratio_ls[i]))

# sort in-place from highest to lowest
char_ratio_ls.sort(key=lambda x: x[1], reverse=True) 

for char, ratio in char_ratio_ls:
    print char.encode('utf8'), ratio
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

# # ============================
# # ratio frequency scatter
# # print out top and bottom 100 ratio chars
# vb = Vocabulary_Builder()
# vb.read_in('./segmentation/msr_train_raw')
# vb = vb.vocabulary_dict

# ratio_freq_ls = []
# for i in char_ratio_ls:
# 	key = i[0].encode('utf8')
# 	if key in vb:
# 		if key != '<P>' and key != '</P>':
# 			ratio_freq_ls.append((i[1], vb[key]))

# xx, yy = zip(*ratio_freq_ls)
# plt.scatter(xx, np.log(yy), s=2)
# plt.xlabel('id v.s. glyph embedding average norm ratio', fontsize=18)
# plt.ylabel('log frequency count', fontsize=18)

# plt.show()
# # =============================

# # =============================
# # grayscale ratio scatter

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

# xx, yy = zip(*ratio_gray_ls)
# plt.scatter(xx, yy, s=2)
# plt.xlabel('id v.s. glyph embedding average norm ratio', fontsize=18)
# plt.ylabel('average grayscale of glyph', fontsize=18)
# plt.show()

# # =============================
# # =============================
# # gray glyphnorm scatter

# grayscale_dict = {}

# for i in char_ratio_ls:
# 	key = i[0].encode('utf8')
# 	glyph = render_text(key)
# 	ave_gray = np.sum(glyph) / (24.0*24.0)
# 	grayscale_dict[key] = ave_gray

# with open('./glyph_norm_ls.pkl') as f:
# 	glyph_norm_ls = pickle.load(f)

# gray_norm_ls = []
# for index, i in enumerate(char_ratio_ls):
# 	key = i[0].encode('utf8')

# 	if key != '<P>' and key != '</P>':
# 		gray_norm_ls.append((glyph_norm_ls[index], grayscale_dict[key]))

# xx, yy = zip(*gray_norm_ls)
# plt.scatter(xx, yy, s=2)
# plt.xlabel('glyph norm ', fontsize=18)
# plt.ylabel('gray scale', fontsize=18)
# plt.show()

# # =============================
# # id norm glyph norm scatter

with open('./glyph_norm_ls.pkl') as f:
    glyph_norm_ls = pickle.load(f)

with open('./id_norm_ls.pkl') as f:
    id_norm_ls = pickle.load(f)

xx, yy = glyph_norm_ls, id_norm_ls
plt.scatter(xx, yy, s=2)
plt.xlabel('glyph norm', fontsize=18)
plt.ylabel('id norm', fontsize=18)
plt.show()

# # =============================

# x = zip(*char_ratio_ls)[1]
# # the histogram of the data
# plt.hist(x, bins='auto')  # plt.hist passes it's arguments to np.histogram
# plt.title("Histogram of ratio")
# plt.xlabel('id v.s. glyph embedding norm ratio', fontsize = 14)
# plt.ylabel('character count')
# # add a 'best fit' line
# # y = mlab.normpdf( bins, mu, sigma)
# # l = plt.plot(bins, y, 'r--', linewidth=1)

# plt.show()

# =============================
# glyph norm / id norm histogram plot

cmap = plt.get_cmap('jet')
low = cmap(0.5)
medium =cmap(0.25)

common_params = dict(bins=100, 
                     range=(0, 7))


# the histogram of the data
N_id, bins_id, patches_id = plt.hist(id_norm_ls, **common_params)  # plt.hist passes it's arguments to np.histogram

N_glyph, bins_glyph, patches_glyph = plt.hist(glyph_norm_ls, **common_params)

for patch in patches_id:
    patch.set_facecolor(low)
for patch in patches_glyph:
    patch.set_facecolor(medium)
#create legend
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium]]
labels= ["id norm","glyph norm"]
plt.legend(handles, labels)

plt.legend()
plt.title("Histogram of norms")
plt.xlabel('ratios', fontsize = 14)
plt.ylabel('character count')
# add a 'best fit' line
# y = mlab.normpdf( bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=1)

plt.show()



