"""Functions here are mostly data generator or enhanced plot function"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.semi_supervised.label_propagation import LabelPropagation


def plot_interesting_points(X, idxs, alpha=0.5, alpha_main=0.8):
    bleu = [i for i in np.arange(X.shape[0]) if i not in idxs]
    plt.scatter(X[bleu, 0], X[bleu,1], alpha=alpha_main)  # plot all points
    plt.scatter(X[idxs,0], X[idxs,1], c='r', s=40, alpha=alpha)  # plot interesting points in red again

def plot(a, b, c=None):
    plt.scatter(a[:, 0], a[:, 1], c="r")
    plt.scatter(b[:, 0], b[:, 1], c="b")
    if c is not None:
        plt.scatter(c[:, 0], c[:, 1], c="g")
    plt.show()

def denoise(d, k, eps):
    """
    Denoise a dataset
    
    Return point that do NOT satisfy the condition: distance(i, k-th nearest neighbor of i) < eps
    
    Parameters :
    ------------
    d : n*n matrix
        distance matrix
    k : int
    eps: float
    """
    if eps == np.inf: # just in case
        return []
    to_del = []
    for c, i in enumerate(d):
        k_neighbour = np.argsort(i)[k]
        if d[c, k_neighbour] > eps:
            to_del.append(c)
    return to_del

def gen_random(size=100):
    a = np.random.multivariate_normal([10, 7], [[1, 0], [0, 2]], size=[size,])
    b = np.random.multivariate_normal([4, 15], [[1, 0], [0, 4]], size=[size,]) 
    c = np.random.multivariate_normal([3, -2], [[2, 0], [0, 2]], size=[size,])
    d = np.random.multivariate_normal([-6, -6], [[2, 0], [0, 1]], size=[size,])
    return a, b, c, d

def square(n, center=(0, 0), size=1):
    points = (np.random.rand(n, 2)*2-1)*size
    points[:, 0] += center[0]
    points[:, 1] += center[1]
    return points

def disk(n, center=(0,0), r=1):
    points = np.zeros((n, 2))
    c = 0
    while c < n:
        point = np.random.rand(2)*2-1
        if point[0]**2+point[1]**2 <= 1:
            points[c] = point*r
            c +=1
    points[:, 0] += center[0]
    points[:, 1] += center[1]
    return points

def square_without_disk(n, center=(0,0), size=1, circle_center=(0,0), r_circle=0.1):
    points = np.zeros((n, 2))
    c = 0
    while c < n:
        point = (np.random.rand(2)*2-1)*size
        point[0] += center[0]
        point[1] += center[1]
        if (point[0]-circle_center[0])**2+(point[1]-circle_center[1])**2 <= r_circle**2:
            continue
        else:
            points[c] = point
            c +=1
    return points

def compare(X, Y, n, index, kernel='rbf', tol=0.01, max_iter=100, gamma=20):
    """
    Compare Propagation between our method and random method

    Parameters
    ----------
    X : 2D array
        The dataset
    Y : 1d array
        True label of every points in X
    n : int
        number of point to label first
    index :1d array of int
        array of index of the point to label
    kernel : str
        rbf
    tol : float
        tol for labelpropagation
    max_iter : int
        max_iter for labelpropagation
    gamma : int
        gamma for labelpropagation
    """
    plt.figure(figsize=(17, 8))
    # start with randomness
    ok = False
    c = 0
    # make sure we have a point of each class (at least)
    while not ok:
        label = np.random.randint(Y.shape[0], size=n)
        c += 1
        if len(np.unique(Y[label])) == len(np.unique(Y)):
            ok = True
    Y_unlabel = -np.ones(Y.shape[0])
    Y_unlabel[label] = Y[label]
    
    label_spread = LabelPropagation(kernel=kernel, tol=tol, max_iter=max_iter, n_jobs=-1, gamma=gamma)
    label_spread.fit(X, Y_unlabel)
    Y_predict = label_spread.transduction_
    iter_random = label_spread.n_iter_

    plt.subplot(1, 2, 1)
    oui = np.where(Y_predict == 0)[0]
    non = np.where(Y_predict == 1)[0]
    plt.scatter(X[oui, 0], X[oui, 1], color='b',
                marker='s', lw=0, s=10)
    plt.scatter(X[non, 0], X[non, 1], color='g',
                marker='s', lw=0, s=10)
    plot_interesting_points(X, label, 1, 0)
    
    # now active part
    label = index[:n]
    if len(np.unique(Y[label])) < len(np.unique(Y)):
        raise Exception("we have less points than classes")
    Y_unlabel = -np.ones(Y.shape[0])
    Y_unlabel[label] = Y[label]
    
    label_spread = LabelPropagation(kernel=kernel, tol=tol, max_iter=max_iter, n_jobs=-1, gamma=gamma)
    label_spread.fit(X, Y_unlabel)
    Y_predict = label_spread.transduction_
    iter_active = label_spread.n_iter_
    # plotting
    plt.subplot(1, 2, 2)
    oui = np.where(Y_predict == 0)[0]
    non = np.where(Y_predict == 1)[0]
    plt.scatter(X[oui, 0], X[oui, 1], color='b',
                marker='s', lw=0, s=10)
    plt.scatter(X[non, 0], X[non, 1], color='g',
                marker='s', lw=0, s=10)
    plot_interesting_points(X, label, 0.5, 0)        
    return (iter_random, iter_active)

def plot_accuracy(x, y, label0, x1, y1, label1, step, factor=1, nb_iter=0, filename="", size=100):
        """Plot the accuracy previously computed.
        Parameters
        ----------
        x, y, x1, y1 : array
            Curves to print
        label0, label1 :
            name of the curve
        step, nb_iter:
            should be same step and nb_iter used for the algo
        factor : int
            Because annotating every points isn't readable, you can provide \
                    a number such that we annonate every *factor* point.
        size :
            size of the dataset, to print percentage at the bottom
        """
        x_ticks = x[::int(factor)]
        y_ticks = y[::int(factor)]
        x1_ticks = x1[::int(factor)]
        y1_ticks = y1[::int(factor)]
        percentage = [str(i)+"\n"+str(np.rint(i/size*100).astype(int))+"%" for i in x1_ticks]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'b-', label=label0)
        ax.plot(x1, y1, 'r-', label=label1)
        plt.xticks(x1_ticks, percentage)
        plt.ylim(-5, 102)
        for i, j in zip(x_ticks, y_ticks):
            ax.annotate(str(j)[:5], xy=(i, j))
        for i, j in zip(x1_ticks, y1_ticks):
            ax.annotate(str(j)[:5], xy=(i, j))
            ax.plot([i, i], [0, j], 'C7', linestyle='--', linewidth=2)
        ax.annotate(str(y[-1])[:5], xy=(x[-1], y[-1]))
        ax.annotate(str(y1[-1])[:5], xy=(x[-1], y1[-1]))
        ax.plot([x[-1], x[-1]], [0, y1[-1]], 'C7', linestyle='--', linewidth=2)

        ax.set_xlabel("Number of points used for the training")
        ax.set_ylabel("Score")
        plt.legend(loc=4)
        plt.title("Accuracy of the prediciton\nFile : {} | step = {} | \
nb_iter = {}".format(filename, step, nb_iter))
