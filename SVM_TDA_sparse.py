import numpy as np
from sklearn.svm import SVC

def single(X, Y, n, best, C):
    X_train = X[best[:n]]
    Y_train = Y[best[:n]]
    #  U = full_data[best[n:], :-1]
    #  T = full_data[best[n:], -1]
    U = X
    T = Y
    try:
        svm = SVC(kernel='linear', C=C, class_weight="balanced")
        svm.fit(X_train, Y_train)
        return svm.score(U, T)
    except:
        return 0

def test_tda_svm_sparse(X, Y, n, step, best, C):
    x = list(range(2, n, step))
    y = np.zeros(len(x))
    for i, val in enumerate(x):
        y[i] = single(X, Y, val, best, C)
        print("%.2f" % (i/len(y)*100), end="\t")
    return x, y

#########
# clean
#########

def single_clean(Xc, Yc, X, Y, n, best, C):
    X_train = Xc[best[:n]]
    Y_train = Yc[best[:n]]
    U = X
    T = Y
    try:
        svm = SVC(kernel='linear', C=C, class_weight="balanced")
        svm.fit(X_train, Y_train)
        return svm.score(U, T)
    except:
        return 0

def test_tda_svm_sparse_clean(Xc, Yc, X, Y, n, step, best, C):
    x = list(range(2, n, step))
    y = np.zeros(len(x))
    for i, val in enumerate(x):
        y[i] = single_clean(Xc, Yc, X, Y, val, best, C)
        print("%.2f" % (i/len(y)*100), end="\t")
    return x, y
