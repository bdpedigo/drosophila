#%%
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
import itertools

# %matplotlib inline
g = nx.MultiDiGraph()
lg = nx.read_graphml("left_edges.graphml")
adj_lg = nx.to_numpy_array(lg)

meta_df = pd.read_csv("left_labels.csv")
cell_labels = meta_df["type"].values

adj_uw = adj_lg.copy()
adj_uw[adj_uw > 0] = 1

#%%
def get_block_probabilities(A, vertex_labels):
    K = len(np.unique(vertex_labels))
    idx = [(np.nonzero(vertex_labels == i)[0], i) for i in np.unique(vertex_labels)]
    P_hat = np.zeros((K, K))
    for i, j in idx:
        P_hat[j, j] = np.mean(A[i, i.min() : i.max()])
    for i, j in itertools.permutations(idx, 2):
        P_hat[i[1], j[1]] = np.mean(A[i[0], j[0].min() : j[0].max()])
    return P_hat, idx


def estimate_sbm_parameters(A, K):
    ase = graspy.embed.AdjacencySpectralEmbed().fit_transform(A)
    if isinstance(ase, tuple):
        ase = np.concatenate((ase[0], ase[1]), axis=1)
    gclust = graspy.cluster.gclust.GaussianCluster(K).fit(ase)
    #     gclust = sklearn.cluster.KMeans(K).fit(ase)
    n_hat = gclust.predict(ase)
    P_hat = get_block_probabilities(A, n_hat)
    return n_hat, P_hat


def cartprod(*arrays):
    N = len(arrays)
    return np.transpose(
        np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)
    ).reshape(-1, N)


def sbm_p(n, p, directed=False, loops=False, wt=1, wtargs=None):
    """
    n: list of int, shape (n_communities)
        the number of vertices in each community. Communities
        are assigned n[0], n[1], ...
    p: array-like, shape (n_communities, n_communities)
        the probability of an edge between each of the communities,
        where p[i, j] indicates the probability of a connection
        between edges in communities [i, j]. 0 < p[i, j] < 1
        for all i, j.
    directed: boolean
        whether or not the graph will be directed.
    loops: boolean
        whether to allow self-loops for vertices.
    wt: object or array-like, shape (n_communities, n_communities)
        if Wt is an object, a weight function to use globally over
        the sbm for assigning weights. 1 indicates to produce a binary
        graph. If Wt is an array-like, a weight function for each of
        the edge communities. Wt[i, j] corresponds to the weight function
        between communities i and j. If the entry is a function, should
        accept an argument for size. An entry of Wt[i, j] = 1 will produce a
        binary subgraph over the i, j community.
    wtargs: dictionary or array-like, shape (n_communities, n_communities)
        if Wt is an object, Wtargs corresponds to the trailing arguments
        to pass to the weight function. If Wt is an array-like, Wtargs[i, j] 
        corresponds to trailing arguments to pass to Wt[i, j].
    """
    # Check n
    if not isinstance(n, (list, np.ndarray)):
        msg = "n must be a list or np.array, not {}.".format(type(n))
        raise TypeError(msg)
    else:
        n = np.array(n)
        if not np.issubdtype(n.dtype, np.integer):
            msg = "There are non-integer elements in n"
            raise ValueError(msg)

    # Check p
    if not isinstance(p, (list, np.ndarray)):
        msg = "p must be a list or np.array, not {}.".format(type(p))
        raise TypeError(msg)
    else:
        p = np.array(p)
        if not np.issubdtype(p.dtype, np.number):
            msg = "There are non-numeric elements in p"
            raise ValueError(msg)
        elif p.shape != (n.size, n.size):
            msg = "p is must have shape len(n) x len(n), not {}".format(p.shape)
            raise ValueError(msg)
        elif np.any(p < 0) or np.any(p > 1):
            msg = "Values in p must be in between 0 and 1."
            raise ValueError(msg)

    # Check wt and wtargs
    if not np.issubdtype(type(wt), np.number) and not callable(wt):
        if not isinstance(wt, (list, np.ndarray)):
            msg = "wt must be a numeric, list, or np.array, not{}".format(type(wt))
            raise TypeError(msg)
        if not isinstance(wtargs, (list, np.ndarray)):
            msg = "wtargs must be a numeric, list, or np.array, not{}".format(
                type(wtargs)
            )
            raise TypeError(msg)

        wt = np.array(wt, dtype=object)
        wtargs = np.array(wtargs, dtype=object)
        # if not number, check dimensions
        if wt.shape != (n.size, n.size):
            msg = "wt must have size len(n) x len(n), not {}".format(wt.shape)
            raise ValueError(msg)
        if wtargs.shape != (n.size, n.size):
            msg = "wtargs must have size len(n) x len(n), not {}".format(wtargs.shape)
            raise ValueError(msg)
        # check if each element is a function
        for element in wt.ravel():
            if not callable(element):
                msg = "{} is not a callable function.".format(element)
                raise TypeError(msg)
    else:
        wt = np.full(p.shape, wt, dtype=object)
        wtargs = np.full(p.shape, wtargs, dtype=object)

    # Check directed
    if not directed:
        if np.any(p != p.T):
            raise ValueError("Specified undirected, but P is directed.")
        if np.any(wt != wt.T):
            raise ValueError("Specified undirected, but Wt is directed.")
        if np.any(wtargs != wtargs.T):
            raise ValueError("Specified undirected, but Wtargs is directed.")

    K = len(n)  # the number of communities
    counter = 0
    # get a list of community indices
    cmties = []
    for i in range(0, K):
        cmties.append(range(counter, counter + n[i]))
        counter += n[i]
    A = np.zeros((sum(n), sum(n)))

    for i in range(0, K):
        if directed:
            jrange = range(0, K)
        else:
            jrange = range(i, K)
        for j in jrange:
            block_wt = wt[i, j]
            block_wtargs = wtargs[i, j]
            block_p = p[i, j]
            # identify submatrix for community i, j
            # cartesian product to identify edges for community i,j pair
            cprod = cartprod(cmties[i], cmties[j])
            # get idx in 1d coordinates by ravelling
            triu = np.ravel_multi_index((cprod[:, 0], cprod[:, 1]), A.shape)
            pchoice = np.random.uniform(size=len(triu))
            # connected with probability p
            # triu = triu[pchoice < block_p]
            if type(block_wt) is not int:
                block_wt = block_wt(size=len(triu), **block_wtargs)
            triu = np.unravel_index(triu, A.shape)
            A[triu] = block_p

    if not loops:
        A = A - np.diag(np.diag(A))
    if not directed:
        A = symmetrize(A)
    return A


def sbm(n, p, directed=False, loops=False, wt=1, wtargs=None, dc=None, dcargs=None):
    """
    n: list of int, shape (n_communities)
        the number of vertices in each community. Communities
        are assigned n[0], n[1], ...
    p: array-like, shape (n_communities, n_communities)
        the probability of an edge between each of the communities,
        where p[i, j] indicates the probability of a connection
        between edges in communities [i, j]. 0 < p[i, j] < 1
        for all i, j.
    directed: boolean
        whether or not the graph will be directed.
    loops: boolean
        whether to allow self-loops for vertices.
    wt: object or array-like, shape (n_communities, n_communities)
        if Wt is an object, a weight function to use globally over
        the sbm for assigning weights. 1 indicates to produce a binary
        graph. If Wt is an array-like, a weight function for each of
        the edge communities. Wt[i, j] corresponds to the weight function
        between communities i and j. If the entry is a function, should
        accept an argument for size. An entry of Wt[i, j] = 1 will produce a
        binary subgraph over the i, j community.
    wtargs: dictionary or array-like, shape (n_communities, n_communities)
        if Wt is an object, Wtargs corresponds to the trailing arguments
        to pass to the weight function. If Wt is an array-like, Wtargs[i, j] 
        corresponds to trailing arguments to pass to Wt[i, j].
    dc: function or array-like, shape (n_vertices)
        if dc is a function, it should generate a random number to be used
        as a weight to create a heterogenous degree distribution. A weight 
        will be generated for each vertex, normalized so that the sum of weights 
        in each block is 1. If dc is array-like, it should be of length sum(n)
        and the elements in each block should sum to 1. Tt will be directly used 
        as the weightings for each vertex.
    dcargs: dictionary
        if dc is a function, dcargs corresponds to its named arguments.
    References
    ----------
    .. [1] Tai Qin and Karl Rohe. "Regularized spectral clustering under the 
        Degree-Corrected Stochastic Blockmodel," Advances in Neural Information 
        Processing Systems 26, 2013
    """
    # Check n
    if not isinstance(n, (list, np.ndarray)):
        msg = "n must be a list or np.array, not {}.".format(type(n))
        raise TypeError(msg)
    else:
        n = np.array(n)
        if not np.issubdtype(n.dtype, np.integer):
            msg = "There are non-integer elements in n"
            raise ValueError(msg)

    # Check p
    if not isinstance(p, (list, np.ndarray)):
        msg = "p must be a list or np.array, not {}.".format(type(p))
        raise TypeError(msg)
    else:
        p = np.array(p)
        if not np.issubdtype(p.dtype, np.number):
            msg = "There are non-numeric elements in p"
            raise ValueError(msg)
        elif p.shape != (n.size, n.size):
            msg = "p is must have shape len(n) x len(n), not {}".format(p.shape)
            raise ValueError(msg)
        elif np.any(p < 0) or np.any(p > 1):
            msg = "Values in p must be in between 0 and 1."
            raise ValueError(msg)

    # Check wt and wtargs
    if not np.issubdtype(type(wt), np.number) and not callable(wt):
        if not isinstance(wt, (list, np.ndarray)):
            msg = "wt must be a numeric, list, or np.array, not {}".format(type(wt))
            raise TypeError(msg)
        if not isinstance(wtargs, (list, np.ndarray)):
            msg = "wtargs must be a numeric, list, or np.array, not {}".format(
                type(wtargs)
            )
            raise TypeError(msg)

        wt = np.array(wt, dtype=object)
        wtargs = np.array(wtargs, dtype=object)
        # if not number, check dimensions
        if wt.shape != (n.size, n.size):
            msg = "wt must have size len(n) x len(n), not {}".format(wt.shape)
            raise ValueError(msg)
        if wtargs.shape != (n.size, n.size):
            msg = "wtargs must have size len(n) x len(n), not {}".format(wtargs.shape)
            raise ValueError(msg)
        # check if each element is a function
        for element in wt.ravel():
            if not callable(element):
                msg = "{} is not a callable function.".format(element)
                raise TypeError(msg)
    else:
        wt = np.full(p.shape, wt, dtype=object)
        wtargs = np.full(p.shape, wtargs, dtype=object)

    # Check directed
    if not directed:
        if np.any(p != p.T):
            raise ValueError("Specified undirected, but P is directed.")
        if np.any(wt != wt.T):
            raise ValueError("Specified undirected, but Wt is directed.")
        if np.any(wtargs != wtargs.T):
            raise ValueError("Specified undirected, but Wtargs is directed.")

    K = len(n)  # the number of communities
    counter = 0
    # get a list of community indices
    cmties = []
    for i in range(0, K):
        cmties.append(range(counter, counter + n[i]))
        counter += n[i]

    # Check degree-corrected input parameters
    if callable(dc):
        # Check that the paramters are a dict
        if not isinstance(dcargs, dict):
            msg = "dcargs must be of type dict not{}".format(type(dcargs))
            raise TypeError(msg)
        # Create the probability matrix for each vertex
        dcProbs = np.array([dc(**dcargs) for _ in range(0, sum(n))], dtype="float")
        for indices in cmties:
            dcProbs[indices] /= sum(dcProbs[indices])
    elif isinstance(dc, (list, np.ndarray)):
        dcProbs = np.array(dc)
        # Check size and element types
        if not np.issubdtype(dcProbs.dtype, np.float_) or not np.issubdtype(
            dcProbs.dtype, np.number
        ):
            msg = "There are non-numeric elements in dc, {}".format(dcProbs.dtype)
            raise ValueError(msg)
        elif dcProbs.shape != (sum(n),):
            msg = "dc must have size equal to number vertices {0} not {1}".format(
                sum(n), dcProbs.shape
            )
            raise ValueError(msg)
        elif np.any(dcProbs < 0) or np.any(dcProbs > 1):
            msg = "Values in dc must be in between 0 and 1."
            raise ValueError(msg)
        # Check that probabilities sum to 1 in each block
        for i in range(0, K):
            if not np.isclose(sum(dcProbs[cmties[i]]), 1, atol=1.0e-8):
                msg = "Block {0} probabilities must sum to 1 not {1}.".format(
                    i, sum(dcProbs[cmties[i]])
                )
                raise ValueError(msg)
    elif dc is not None:
        msg = "dc must be a function, list, or np.array, not {}".format(type(dc))
        raise ValueError(msg)

    # End Checks, begin simulation
    A = np.zeros((sum(n), sum(n)))
    P = np.zeros_like(A)
    for i in range(0, K):
        if directed:
            jrange = range(0, K)
        else:
            jrange = range(i, K)
        for j in jrange:
            block_wt = wt[i, j]
            block_wtargs = wtargs[i, j]
            block_p = p[i, j]
            # identify submatrix for community i, j
            # cartesian product to identify edges for community i,j pair
            cprod = cartprod(cmties[i], cmties[j])
            # get idx in 1d coordinates by ravelling
            triu = np.ravel_multi_index((cprod[:, 0], cprod[:, 1]), dims=A.shape)
            pchoice = np.random.uniform(size=len(triu))
            dcP = dcProbs[cprod[:, 0]] * dcProbs[cprod[:, 1]]
            if dc is not None:
                # (v1,v2) connected with probability p*k_i*k_j*dcP[v1]*dcP[v2]
                triu2 = triu.copy()
                triu = np.random.choice(
                    triu, size=sum(pchoice < block_p), replace=False, p=dcP
                )
            else:
                # connected with probability p
                triu = triu[pchoice < block_p]
            if type(block_wt) is not int:
                block_wt = block_wt(size=len(triu), **block_wtargs)
            triu = np.unravel_index(triu, dims=A.shape)
            triu2 = np.unravel_index(triu2, shape=A.shape)
            A[triu] = block_wt
            print(dcP.shape)
            P[triu2] = dcP
    if not loops:
        A = A - np.diag(np.diag(A))
    if not directed:
        A = symmetrize(A)
    return A, P


#%% unsupervised

# label, block = estimate_sbm_parameters(A, 5)
# ls, counts = np.unique(label, return_counts=True)
# sim_sbm = graspy.simulations.sbm(counts, block, directed=True)
# graspy.plot.heatmap(sim_sbm)
def calculate_node_degrees(A):
    out_degree = A.sum(axis=1)
    in_degree = A.sum(axis=0)
    degree = out_degree + in_degree
    return degree


def argsort_by_degree(A):
    degree = calculate_node_degrees(A)
    sorted_inds = np.argsort(degree, kind="stable")
    sorted_inds = sorted_inds[::-1]
    return sorted_inds


def degree_sorted_heatmap(A, labels, title=None):
    inds = argsort_by_degree(A)
    labels_sorted = labels[inds]
    A_sorted = A[inds, :][:, inds]
    if title == "Drosophila left MB":
        cmap = "PiYG"
    else:
        cmap = "PiYG_r"
    ax = graspy.plot.heatmap(
        A_sorted,
        cmap=cmap,
        inner_hier_labels=labels_sorted,
        cbar=False,
        title=title,
        context="notebook",
    )
    plt.suptitle(title, y=0.98, x=0.06, fontsize=40, horizontalalignment="left")
    # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig(title)
    return ax


for i, l in enumerate(cell_labels):
    if l == "MBON":
        cell_labels[i] = "MO"
    elif l == "MBIN":
        cell_labels[i] = "MI"

# inds = argsort_by_degree(adj_uw)
# adj_uw = adj_uw[inds, :][:, inds]
# cell_labels = cell_labels[inds]
#%% wrangle cell labels
cell_types, inds, inv, counts = np.unique(
    cell_labels, return_index=True, return_inverse=True, return_counts=True
)
swap = np.argsort(inds)
cell_types = cell_types[swap]
inds = inds[swap]
counts = counts[swap]
int_label_map = dict(zip(cell_types, np.arange(0, len(cell_types))))
cell_types_int = [int_label_map[c] for c in cell_labels]

#%%
# calculate SBM parameters
label_to, label_from = np.meshgrid(cell_types_int, cell_types_int)
l_tuples = zip(label_from.ravel(), label_to.ravel())
p = np.zeros((len(cell_types), len(cell_types)))
a = adj_uw.ravel()
n = np.zeros_like(p)
for ind, tup in enumerate(l_tuples):
    i, j = tup
    p[i, j] += a[ind]
    n[i, j] += 1
p = p / n

# calculate DCSBM parameters
degree = calculate_node_degrees(adj_uw)
dc = degree.copy()
inds = np.append(inds, [degree.size])
degree_sum_vec = np.zeros_like(degree)
for i, com in enumerate(cell_types):
    start = inds[i]
    end = inds[i + 1]
    degree_sum = np.sum(degree[start:end])
    dc[start:end] /= degree_sum
    degree_sum_vec[start:end] = degree_sum

#%%
# original

name = "Drosophila left MB"
fig = plt.figure()
degree_sorted_heatmap(adj_uw, cell_labels, name)

# ER
total_p = np.sum(adj_uw) / adj_uw.size
sim_er = graspy.simulations.er_np(adj_uw.shape[0], total_p)
degree_sorted_heatmap(sim_er, cell_labels, "ER")

# SBM
sim_sbm_supervised = graspy.simulations.sbm(counts, p, directed=True)
degree_sorted_heatmap(sim_sbm_supervised, cell_labels, "SBM")

# DCSBM
sim_dcsbm, dcsbm_P = sbm(counts, p, dc=dc, directed=True)
degree_sorted_heatmap(sim_dcsbm, cell_labels, "DCSBM")

# RDPG
X, Y = graspy.embed.AdjacencySpectralEmbed().fit_transform(adj_uw)
sim_rdpg = graspy.simulations.rdpg(X, Y, rescale=False, directed=True, loops=False)
degree_sorted_heatmap(sim_rdpg, cell_labels, "RDPG")


# #%% show the actual model fits
# p_sbm_supervised = sbm_p(counts, p, directed=True)

# ax = graspy.plot.heatmap(p_sbm_supervised, cmap="PiYG_r", inner_hier_labels=cell_labels)
# plt.show()
# p_rdpg = X @ Y.T
# p_rdpg[p_rdpg > 1] = 1
# p_rdpg[p_rdpg < 0] = 0
# graspy.plot.heatmap(p_rdpg, cmap="PiYG_r", inner_hier_labels=cell_labels)
# plt.show()
# #%%

# get_block_probabilities(sim_dcsbm, cell_types_int)

# #%%
# a, b = np.meshgrid(dc, dc)
# dc_mat = a * b
# f = dc_mat * p_sbm_supervised
# # graph = graspy.simulations.sample_edges(f)
# graspy.plot.heatmap(f)
# dc_inv = cartprod(degree_sum_vec, degree_sum_vec)

# np.sum(f[:101, :101])
# #%%
# np.outer()

#%%
plt.figure(figsize=(10, 10))
f = sns.color_palette("PiYG_r", n_colors=100)
purp = f[-1]
# f = sns.color_palette("PRGn_r", n_colors=100)
green = f[0]
n_verts = adj_uw.shape[0]
params = [n_verts ** 2, 2 * X.size, 2 * n_verts + p.size, p.shape[0] + p.size, 1]
params = np.array(params)
log_params = np.log10(params)
labels = ["Truth", "RDPG", "DCSBM", "SBM", "ER"]
col = ["Real", "N", "N", "N", "N"]
data = pd.DataFrame(columns=["Model", "# Parameters", "Type"])
data["Model"] = labels
data["# Parameters"] = params
data["Type"] = col
# ax = plt.subplot(111, aspect="equal")

with sns.plotting_context("talk", font_scale=1.6):
    ax = sns.pointplot(
        data=data,
        x="Model",
        y="# Parameters",
        hue="Type",
        palette=sns.color_palette([green, purp]),
        saturation=1,
        join=False,
        scale=1.5,
        markers="s",
    )
ax.set_xlabel("Model", fontsize=35)
ax.set_ylabel("# Parameters", fontsize=35)
ax.set_yscale("log")
# ax.axvline(0.5, linestyle="--")
ax.get_legend().remove()
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
plt.savefig("bar")


#%%
n_verts * n_verts

