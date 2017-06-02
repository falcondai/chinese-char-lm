import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

ratio_ls = np.loadtxt('./ratio.txt', dtype=float,  delimiter=',')

popularity_data = []
with open('./work/dict.txt', 'r') as fhandle:
    for i,chara in enumerate(fhandle):
        if i > 4000:
            break
        popularity_data.append((chara.strip().decode('utf8'), ratio_ls[i]))

# sort in-place from highest to lowest
popularity_data.sort(key=lambda x: x[1], reverse=True) 

# save the names and their respective scores separately
# reverse the tuples to go from most frequent to least frequent 
people = zip(*popularity_data)[0][-20:]
score = zip(*popularity_data)[1][-20:]
x_pos = np.arange(len(people))[-20:]

# calculate slope and intercept for the linear trend line

plt.bar(x_pos, score,align='center')
plt.xticks(x_pos, people) 
plt.ylabel('id/glyph embedding norm ratio')
plt.show()

people = zip(*popularity_data)[0][:20]
score = zip(*popularity_data)[1][:20]
x_pos = np.arange(len(people))[:20]

# calculate slope and intercept for the linear trend line

plt.bar(x_pos, score,align='center')
plt.xticks(x_pos, people) 
plt.ylabel('id/glyph embedding norm ratio')
plt.show()



x = zip(*popularity_data)[1]
# the histogram of the data
plt.hist(x, bins='auto')  # plt.hist passes it's arguments to np.histogram
plt.title("Histogram of ratio")
# add a 'best fit' line
# y = mlab.normpdf( bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=1)

plt.show()