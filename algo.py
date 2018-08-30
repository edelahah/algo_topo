"""Module containing the algorithm to retrieve interesting point according to \
        the topological method

Example
-------
>>> # define X, Y and a classiffier here ...
>>> from algo import algo
>>> index_best_points = algo(X=X, n=10, s=5, min_size=2, metric='euclidean')
>>> print(X[index_best_points], Y[index_best_points])
>>>
"""


import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def algo(X, n, s, min_size=0, metric='euclidean', start=0, debug=False):
    """
    Parameters
    ---------
    X : 2D array or sparse matrix
        The dataset to analyse
    n : int
        The number of pairs you'd like to get
    s : int
        The number of pairs the algo will retrieve at each levels of the dendrogram
    min_size : int
        The minimum size of a cluster in order to consider it as relevant. If len(cluster) < min_size, it is skipped
    metric : str
        The metric to use to compute the distance. See sklearn.metrics.pairwise_distances for more information. Common ones are "euclidean", "cityblock" and "cosine".
    start : int
        Where to start in the dendrogram. If start != 0, we'll skip start levels of the dendrogram.
    debug : boolean
        If True, print some debug informations

    For implementation details, see comments in the code

    Returns
    -------
    res : 1D array
        An array of length n*2 containing indexes of the points of the dataset that are interesting to be labelled.
    """
    #  How does the algo work (in pratice ?)
    #  The basic idea is that we run through the dendrogram from top to bottom.
    #  At every split of the dendrogram, we have two clusters. If one of the cluster
    #  has less points than min_size, we go to the next split. If there are enough points
    #  in the 2 clusters, we add the indexes of the s-closest pairs to the result.
    #  In order to get the i-th split of the dendrogram, we make an Agglomerative Clustering
    #  with i+1 cluster. and one with i+2 clusters. Then we look at the difference between
    #  them both and deduce which points belongs the new branch of the dendrogram.
    #  We stop either when we reach the end of the dendrogram or when we have taken n pairs of points

    res = [] # we'll store indexes to return in here
    i = start
    D = pairwise_distances(X, metric=metric) # distance matrix of shape n_samples*n_samples
    old = AgglomerativeClustering(i+1, linkage="single", affinity='precomputed')
    old.fit(D)
    while len(res)<n*2:
        if debug:
            print("Split N°", i+1, end="\t")
        try:
            new = AgglomerativeClustering(i+2, linkage="single", affinity='precomputed')
            new.fit(D)
        except:
            print("\nEnd of the dendrogram, returning with len(res) < n*2")
            return res
        # we look at the difference between the 2 clusterings
        idx1, idx2 = get_subclusters(old.labels_, new.labels_)
        old = new # in order to NOT recompute the same clustering twice (i+2) become the (i+1)+1
        if len(idx1) < min_size or len(idx2) < min_size:
            if debug:
                print("cluster too small {}".format((len(idx1), len(idx2))))
            i += 1
            continue

        # get the s closest pairs of points belonging to different clusters
        closest_points = get_k_closest_pairs(idx1, idx2, s, D[idx1][:, idx2])
        if debug:
            print("len(closest_points) = ", len(closest_points), end="\t")
        # add them to the final result
        for x in closest_points:
            if x in res and debug:
                print(x, end="...")
            else:
                res.append(x)
        if debug:
            print("len(res) = ", len(res), "/", n*2)
        # loop
        i += 1
    return res[:2*n]


def get_k_closest_pairs(i1, i2, k, dist):
    """
    Return k-pairs of closest points between i1 and 2 according to the distance matrix

    Parameters
    ----------
    i1 : 1D array of integers
        Represent corresponding indexes between the bigger matrix and the subset of it
    i2 : 1D array of integers
        Represent corresponding indexes between the bigger matrix and the subset of it
    k : int
        The number of closest-pairs we want
    dist : 2-D array
        The subset of the full distance matrix

    Returns
    -------
    res_ : 1D array
        Array of indexes of closest points between the 2 clusters.
    """ 
    if k > i1.shape[0]:
        k = i1.shape[0]
    if k > i2.shape[0]:
        k = i2.shape[0]
    res_ = [] # container of indexes that will be returned
    for _ in range(k):
        argmin = dist.argmin() # argmin is an int
        i, j = np.unravel_index(argmin, dist.shape) # convert argmin to "real" couple of indexes
        res_.append(i1[i])
        res_.append(i2[j])
        #  Now we "forgot" the 2 points already retrieved to NOT take them again
        dist[i, :] = np.inf
        dist[:, j] = np.inf
    return res_

def get_subclusters(old, new):
    """
    Given a n-clustering and a n+1-clustering, returns the indexes of the \
            points belonging to the 2 new clusters

    Parameters
    ----------
    old : 1D array of integers
        Index/labels returned by a n-clustering
    new : 1D array of integers
        Index/labels returned by a n+1-clustering

    Returns
    -------
    c1 : 1D array
        Array of indexes of the points belonging to the first part of the divided cluster
    c2 : 1D array
        Array of indexes of the points belonging to the second part of the divided cluster

    Example
    -------
    >>> old = np.array([0,0,0,0,1,1,1,1]) # points 0 to 3 (included) are a cluster, points 4 fo 7 are a different cluster
    >>> new = np.array([1,1,1,1,0,0,2,2]) # above was a 2-clustering, here is a 3 clustering
    >>> get_subclusters(old, new)
    (np.array([4, 5]), np.array([6, 7])
    """ 
    for i in np.unique(old):
        idx = np.where(old==i)[0]
        points = new[idx]
        if np.all(points==points[0]):
            continue  # it's not the cluster that has been divided
        else:
            foo1 = np.argwhere(points==points[0]).flatten()
            foo2 = np.argwhere(points!=points[0]).flatten()
            c1 = idx[foo1]
            c2 = idx[foo2]
            return c1, c2

