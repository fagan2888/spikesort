Cluster
*******

This module is used to cluster tetrode waveforms found using the process module.  The main class for this is Sorter.  Sorter takes the tetrode waveforms, finds the most important features using `PCA (or ICA) <http://scikit-learn.org/stable/modules/decomposition.html#decompositions>`_, then sorts the data into clusters in feature space using a `Gaussian Mixture Model <http://scikit-learn.org/stable/modules/mixture.html>`_.  After sorting, the data is stored in a Clusters object, which can be viewed using a Viewer or with a Sorter, which inherits from Viewer.

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

Once the automated sorting is done, you'll most likely have to combine clusters.  For instance, you'll have clusters of noise or artifacts which you can combine into one "noise" cluster.  Also, there can be multiple clusters of spikes from the same neuron.  You can identify good clusters by looking at the waveforms with ``sorter.spikes()``.  Neurons have a refractory period limiting the time between spikes, showing up as a dip in the autocorrelation (``sorter.autocorrs()``) at low inter-spike intervals.  You can look at the crosscorrelations with ``sorter.crosscorrs()`` to identify clusters of spikes belonging to the same neuron.

Functions
---------
.. automodule:: cluster
    :members: cluster, fit_ica, fit_pca, load_clusters

Classes
-------
.. automodule:: cluster
    :members: Clusters, Viewer, Sorter

Exceptions
----------
.. automodule:: cluster
    :members: ClusterIdError