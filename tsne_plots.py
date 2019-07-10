import sys
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.palettes import Category20, Category20b, Category20c
sns.set(rc={'axes.facecolor':"#282828" })
# source_palette = sns.hls_palette(31, l=.25, s=.8)
palette = sns.color_palette("Paired", 62)
yellow = sns.color_palette("Set2")[5]
palette[11::12] = [yellow for _ in palette[5::12]]
source_palette = palette[::2]
target_palette = palette[1::2]


####################
#     Bokeh Color
####################
palette = Category20[20]
palette[14:16] = [i for i in Category20b[20][0:2]]
palette = (palette*4)[:62]
source_palette = palette[1::2]
print(len(palette), len(source_palette))
target_palette = palette[::2]

####################
#     Gruvbox Color
####################
palette = ["#cc241d", "#fb4934", "#98971a", "#b8bb26", "#d79921", "#fabd2f", "#458588", "#83a598", "#b16286", "#d3869b", "#689d6a", "#8ec07c", "#d65d0e", "#fe8019"]
palette = (palette*5)[:62]
source_palette = palette[1::2]
target_palette = palette[::2]


# target_palette = sns.hls_palette(31, l=0.5, s=0.5)
# target_palette = sns.color_palette("bright", 31)
# plt.xticks([])
# plt.yticks([])
pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * .7]
open_circle = matplotlib.path.Path(vert)
source_file = sys.argv[1]
source_name = source_file.split('.')[0]
target_file = sys.argv[2]
target_name = target_file.split('.')[0]
source = np.loadtxt(source_file)
target = np.loadtxt(target_file)

source_labels = pd.read_csv(sys.argv[3], sep=" ", header=None, dtype={0: 'object', 1: 'object'}).ix[:, 1]
target_labels = pd.read_csv(sys.argv[4], sep=" ", header=None, dtype={0:'object', 1: 'object'}).ix[:, 1]
idx = np.random.permutation(target_labels.index)
target_labels = target_labels.reindex(idx)
target = target[idx]

# plt.scatter(source[:, 0], source[:, 1], s=5, c='b')
# plt.scatter(target[:, 0], target[:, 1], s=5, c='r')
sns.scatterplot(x=source[:, 0], y=source[:,1], hue=source_labels, hue_norm=None, palette=source_palette, s=3, edgecolors=None, linewidth=0, label="Source domain", alpha=.5)
sns.scatterplot(x=target[:, 0], y=target[:,1], hue=target_labels, hue_norm=None, marker="v", s=20, linewidth=0.4, edgecolor='#fbf1c7', palette=target_palette, label="Target domain")
# legend_elements = [
#     mlines.Line2D([], [], color=palette[2], marker='.', linestyle='None', markersize=10, label="", alpha=0.8),
#     mlines.Line2D([], [], color=palette[1], marker='d', linestyle='None', markersize=11, markeredgewidth=0.5, markeredgecolor='white', label=""),
#     mlines.Line2D([], [], color=palette[0], marker='.', linestyle='None', markersize=10, label="Source domain", alpha=0.8),
#     mlines.Line2D([], [], color=palette[1], marker='d', linestyle='None', markersize=11, markeredgewidth=0.5, markeredgecolor='white', label="Target domain")]
legend_elements = [(mlines.Line2D([], [], color=palette[s], marker='.', linestyle='None', markersize=10, alpha=0.8, label="Source domain" if s==12 else ""),mlines.Line2D([], [], color=palette[s+1], marker='v', linestyle='None', markeredgewidth=1, markeredgecolor='#fbf1c7', markersize=7, label="Target domain" if s == 12 else "", alpha=1)) for s in range(0, 14, 2)]
legend_elements = [item for sublist in legend_elements for item in sublist]
legend = plt.legend(handles=legend_elements, ncol=7, columnspacing=-2, loc='upper left')
for text in legend.get_texts():
    text.set_color('#fbf1c7')
plt.gca().axes.get_yaxis().set_ticks([])
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticklabels([])
plt.gca().axes.get_xaxis().set_ticklabels([])
plt.savefig(f'{target_name}.png', bbox_inches='tight')