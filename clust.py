#%%
import itertools

import graspy
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import linalg
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_covariances_full


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap="viridis", zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis("equal")

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def plot_results(X, Y_, means, covariances, index, title):
    # splot = plt.subplot(2, 1, 1 + index)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    color_iter = sns.color_palette("Set1")
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 1.5, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle, color=color)
        # ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    # plt.xlim(-9.0, 5.0)
    # plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    # for ax in ax.flat:
    #     ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], aspect=1)


# %matplotlib inline
g = nx.MultiDiGraph()
lg = nx.read_graphml("left_edges.graphml")
adj_lg = nx.to_numpy_array(lg)

meta_df = pd.read_csv("left_labels.csv")
cell_labels = meta_df["type"].values


#%% try with agglomerative

n_components = None

A = adj_lg.copy()
A = graspy.utils.pass_to_ranks(A, method="simple-all")
A = graspy.utils.augment_diagonal(A, weight=1)
ase = graspy.embed.AdjacencySpectralEmbed(n_components=n_components)
X, Y = ase.fit_transform(A)
latent = np.concatenate((X, Y), axis=1)
graspy.plot.pairplot(latent, labels=cell_labels, diag_kind="hist")
graspy.plot.pairplot(latent)

#%%
max_components = 11
gmm = graspy.cluster.GaussianCluster(max_components=max_components)
gmm.fit(latent, y=cell_labels)
plt.plot(gmm.bic_)
plt.show()
plt.plot(gmm.ari_)

#%%
max_components = 12
methods = ["ward", "complete", "average", "single"]
for j, method in enumerate(methods):
    bic = []
    ari = []
    for i in range(1, max_components):
        n_clusters = i
        ag = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        ag_labels = ag.fit_predict(latent)

        # initialize GMM with agglomerative
        means = np.zeros((n_clusters, latent.shape[1]))
        weights = []
        for i, l in enumerate(np.unique(ag_labels)):
            points = np.where(ag_labels == l)[0]
            cluster_points = latent[points, :]
            mean = np.mean(cluster_points, axis=0)
            means[i, :] = mean
            weights.append(len(points))
        weights = np.array(weights)
        weights = weights / len(ag_labels)

        resp = np.zeros((latent.shape[0], n_clusters))
        resp[np.arange(latent.shape[0]), ag_labels] = 1
        cov = _estimate_gaussian_covariances_full(
            resp, latent, resp.sum(0), means, 1e-6
        )

        prec = []
        for c in cov:
            p = np.linalg.inv(c)
            prec.append(p)
        prec = np.array(prec)
        gm = GaussianMixture(
            n_components=n_clusters,
            means_init=means,
            weights_init=weights,
            precisions_init=prec,
        )

        # turn this on to see clusters
        # plot_results(latent[:, :2], ag_labels, means[:, :2], cov, 0, "Gaussian Mixture")

        gm.fit(latent)
        b = gm.bic(latent)
        bic.append(b)
        ari.append(adjusted_rand_score(cell_labels, gm.predict(latent)))

    print(method)
    plt.show()
    plt.plot(np.arange(1, max_components), bic)
    plt.title(method + "BIC")
    plt.show()
    plt.plot(np.arange(1, max_components), ari)
    plt.title(method + "ARI")
    plt.show()
#%%
plt.show()
with sns.plotting_context("talk"):
    # plt.scatter(-latent[:, 0], latent[:, 1], s=1)
    plt.figure(figsize=(10, 10))
    df = pd.DataFrame(columns=["Out Dim 1", "Out Dim 2", "Type"])
    df["Out Dim 1"] = -latent[:, 0]
    df["Out Dim 2"] = latent[:, 1]
    df["Type"] = cell_labels
    sns.scatterplot(data=df, x="Out Dim 1", y="Out Dim 2", hue="Type")
    plt.axis("square")
    plt.show()
#%%


# data, target = load_iris(return_X_y=True)

# plot_results(
#     latent[:, :2],
#     gm.predict(latent),
#     gm.means_[:, :2],
#     gm.covariances_,
#     0,
#     "Gaussian Mixture",
# )
