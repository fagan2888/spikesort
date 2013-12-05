Cluster
*******

This module is used to cluster tetrode waveforms found using the process module.  The main class for this is Sorter.  Sorter takes the tetrode waveforms, finds the most important features using `PCA (or ICA) <http://scikit-learn.org/stable/modules/decomposition.html#decompositions>`_, then sorts the data into clusters in feature space using a `Gaussian Mixture Model <http://scikit-learn.org/stable/modules/mixture.html>`_.  After sorting, the data is stored in a Clusters object, which can be viewed using a Viewer or with a Sorter, which inherits from Sorter.

**Examples** ::

    tetrodes = spikesort.process.form_tetrodes(data, times)
    sorter = spikesort.cluster.Sorter(K=9, dims=10, assign_prob=0.5)
    sorter.sort(tetrodes)

A good way to pick the best number of clusters is by finding the smallest BIC between many models.  Using IPython's parallel processing system: ::

    min_K, max_K = 5, 40
    def f(data, K):
        import spikesort as ss
        sorter = ss.cluster.Sorter(K=K, dims=10, cov_type='full', assign_prob=0.5)
        sorter.sort(data)
        return sorter

    Ks = range(min_K, max_K+1)
    models = c[:].map(f, (tetrodes for _ in Ks), Ks)
    bics = [ model.bic() for model in models ]
    sorter = models[argmin(bics)]

Functions
---------
.. automodule:: cluster
    :members: cluster, fit_ica, fit_pca, load_clusters

Classes
-------
.. automodule:: cluster
    :members: Clusters, Viewer, Sorter, ClusterIdError