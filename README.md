This repo contains the algo we developped and some examples of its applications.

Main tools used :
* [NumPy](https://docs.scipy.org/doc/numpy-1.13.0/reference/)
* [AgglomerativeClustering](http://scikit-learn.org/dev/modules/generated/sklearn.cluster.AgglomerativeClustering.html) with "single" linkage
* [linkage](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.linkage.html) and [dendrogram](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html) for plotting purpose
* [pairwise_distances](http://scikit-learn.org/dev/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html) used to compute distance matrix with choosen metric
* [LabelPropagation](http://scikit-learn.org/dev/modules/generated/sklearn.semi_supervised.LabelPropagation.html)
* [mlxtend](https://github.com/rasbt/mlxtend) to plot decision boundaries
* [PCA](http://scikit-learn.org/dev/modules/generated/sklearn.decomposition.PCA.html) to plot data in 2D

To get a single linkage, we need scikit-learn dev version (the 31/08/2018).

# Installing
Assuming you are using Linux (Debian, Ubuntu).
```bash
sudo su
apt-get install python3 python3-pip git
pip3 install numpy scipy matplotlib mlxtend jupyter
pip3 install git+git://github.com/scikit-learn/scikit-learn.git
exit
git clone https://github.com/edelahah/algo_topo.git
cd algo_topo
jupyter notebook
```
