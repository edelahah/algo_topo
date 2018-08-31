import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

def new_sets(X, Y, n):
    subclass = np.random.randint(X.shape[0], size=n)
    X_t = X[subclass]
    Y_t = Y[subclass]
    U = X
    T = Y
    #  print(X, Y)
    return X_t, Y_t, U, T

def single_svm(X, Y, n, C):
    error = True
    while error:
        X_t, Y_t, U, T = new_sets(X, Y, n)
        try:
            svm = SVC(kernel='linear', C=C, class_weight="balanced")
            svm.fit(X_t, Y_t)
            error = False
            return svm.score(U, T)
        except:
            pass

def median_svm(X, Y, n, C, nb_iter=1):
    res = np.zeros(nb_iter)
    for i in range(nb_iter):
        res[i] = single_svm(X, Y, n, C)
    return np.median(res)

def test_svm_sparse(X, Y, n, step, nb_iter, C):
    x = list(range(2, n, step))
    y = np.zeros(len(x))
    for i, val in enumerate(x):
        y[i] = median_svm(X, Y, val, C, nb_iter)
        print("%.2f" % (i/(n/step)*100), end="\t")
    return x, y
