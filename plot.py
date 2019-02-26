#%%
from graspy.plot import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %matplotlib inline
g = nx.MultiDiGraph()
lg = nx.read_graphml("left_edges.graphml")
adj_lg = nx.to_numpy_array(lg)
# heatmap(adj_lg, transform='simple-nonzero')

meta_df = pd.read_csv("left_labels.csv")

# rg = nx.read_graphml('right_edges.graphml')
# adj_rg = nx.to_numpy_array(rg)
# heatmap(adj_rg)
# adj_rg.shape

A = adj_lg
cell_idx = meta_df["type"].values
sorted_idx = np.argsort(cell_idx, kind="stable")

A_ordered = A[sorted_idx, :][:, sorted_idx]

uniques, freq = np.unique(cell_idx, return_counts=True)

meta_df_sorted = meta_df.sort_values(by=["type"], kind="mergesort")

type1 = meta_df_sorted["type"].values

type1_unique, type1_freq = np.unique(type1, return_counts=True)
type1_freq_cumsum = np.hstack([0, type1_freq.cumsum()])

print(type1_unique)

cell_idx = list(meta_df_sorted.index)

magenta = sns.xkcd_rgb["light magenta"]
green = sns.xkcd_rgb[
    "medium green"
]  #'medium green'# sns.color_palette('Set1')[2]#'#00FF00'
black = "#000000"
# sns.palplot(sns.color_palette('Set1'))
orange = sns.xkcd_rgb["orange"]
sky_blue = sns.xkcd_rgb["sky blue"]
my_palette = sns.color_palette([magenta, green, black])
# my_palette = sns.color_palette([orange, sky_blue, black])
# g = heatmap(,
#              labels=['Hermaphrodite', 'Male', 'Both'],
#              font_scale=1,
#              sizes=(10, 150),
#              alpha=1,
#              height=12,
#              palette=my_palette)

ax = heatmap(A, transform="simple-nonzero", cmap="PiYG_r")

# ax = g.ax

# draw lines for separating blocks
# need to do this first
for x in type1_freq_cumsum:
    if x == type1_freq_cumsum[-1]:
        x -= 1
    ax.vlines(x, 0, 426, linestyle="dashed", lw=0.9, alpha=0.25, zorder=3)
    ax.hlines(x, 0, 426, linestyle="dashed", lw=0.9, alpha=0.25, zorder=3)

# horizontal curve
tick_loc = type1_freq.cumsum() - type1_freq / 2
tick_width = type1_freq / 2

divider = make_axes_locatable(ax)
ax_x = divider.new_vertical(size="5%", pad=0.0, pack_start=False)
ax.figure.add_axes(ax_x)

# make curve
lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 50)
tan = np.tan(lx)
curve = np.hstack((tan[::-1], tan))

# top inner curves
tick_loc = type1_freq.cumsum() - type1_freq / 2
tick_width = type1_freq / 2
for x0, width in zip(tick_loc, tick_width):
    x = np.linspace(x0 - width, x0 + width, 100)
    ax_x.plot(x, -curve, c="k")
ax_x.set_yticks([])
ax_x.tick_params(axis="x", which=u"both", length=0, pad=7)
for direction in ["left", "right", "bottom", "top"]:
    ax_x.spines[direction].set_visible(False)
ax_x.set_xticks(tick_loc)
ax_x.set_xticklabels(np.tile(type1_unique, 3), fontsize=15, verticalalignment="center")
ax_x.xaxis.set_label_position("top")
ax_x.xaxis.tick_top()
ax_x.set_xlim(0, A.shape[0])

# inner side curves
ax_y = divider.new_horizontal(size="5%", pad=0.0, pack_start=True)
ax.figure.add_axes(ax_y)
for x0, width in zip(tick_loc, tick_width):
    x = np.linspace(x0 - width, x0 + width, 100)
    ax_y.plot(curve, x, c="k")
ax_y.set_xticks([])
ax_y.tick_params(axis="y", which=u"both", length=0)
for direction in ["left", "right", "bottom", "top"]:
    ax_y.spines[direction].set_visible(False)
ax_y.set_ylim(0, A.shape[0])
ax_y.set_yticks(tick_loc)
ax_y.invert_yaxis()
ax_y.set_yticklabels(np.tile(type1_unique, 3), fontsize=15, verticalalignment="center")
plt.tight_layout()
plt.show()

