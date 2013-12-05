""" 
.. module:: cluster
    :synopsis: This module is intended for unsupervised clustering using a Gaussian
        Mixture Model (GMM).

.. moduleauthor:: Mat Leonard <leonard.mat@gmail.com>
"""

import numpy as np

from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GMM
import plots as plt

class ClusterIdError(Exception):
    pass

def fit_pca(data, dims):
    """ Fit PCA decomposition to the data with dims components 

        **Arguments**:
            *data*:
             Data for PCA fitting
            *dims*:
             Number of PCA components to use

        **Returns**:
            PCA object (from sklearn.decomposition),
            N x dims NumPy array of data projected onto PCA components

    """
    pca = PCA(n_components=dims)
    x = data - np.mean(data, axis=0)
    return pca, pca.fit_transform(x)

def fit_ica(data, dims):
    """ Fit ICA decomposition to the data with dims components 

        **Arguments**:
            *data*:
             Data for ICA fitting
            *dims*:
             Number of ICA components to use

        **Returns**:
            ICA object (from sklearn.decomposition),
            N x dims NumPy array of data projected onto ICA components

        I haven't found this to work well, but your results might be better.
    """
    ica = FastICA(n_components=dims)
    x = data - np.mean(data, axis=0)
    return ica, ica.fit_transform(x)

def cluster(data, K=10, cov_type='full', assign_prob=None):
    """ Fit a GMM (from sklearn.mixture) to the data, forming clusters of 
        similar data points.
        
        **Keywords**:
            *K*: 
             number of clusters to fit to the data
            *cov_type*: 
             Type of covariance matrix to use in the GMM. 
             ('tied', 'diag', default is 'full')
            *assign_prob*: 
             The lower probability limit that a data point belongs to a 
             cluster to assign it to that cluster.  If not set, assigns 
             a point to the cluster with the highest probability.  Don't
             set this to less than 0.5 since it is possible to have one data 
             point with p>0.49 of belonging in two different clusters.

        **Returns**:
            *gmm*:
             GMM object (from sklearn.mixture)
            *clusters*:
             A dictionary with keys as cluster ids and values are arrays of
             indices of the data belonging to the cluster.  So, calling 
             clusters[1] will return the indices of the data rows belonging to
             cluster 1.

    """
    
    gmm = GMM(n_components = kwargs.get('K', 10),
              covariance_type = kwargs.get('cov_type', 'full'))
    
    gmm.fit(data)

    if assign_prob is None:
        # If assign_prob isn't given...
        predicted = gmm.predict(data)
        clusters = { int(cl):np.where(predicted==cl)[0] 
                     for cl in np.unique(predicted)}
    else:
        # If assign_prob is given
        probs = gmm.predict_proba(data)
            
        # Assign to clusters where p > assign_prob
        inds, cls = np.where(probs>assign_prob)
        clusters = {cl+1:inds[cls==cl] for cl in np.unique(cls)}
        
        # 0th cluster for data points that don't make it into any clusters.
        clusters[0] = np.where((probs>assign_prob).any(axis=1) == False)[0]
        
    return gmm, clusters

def load_clusters(filepath):
    """ Loads clusters from pickled file at filepath. """
    import cPickle as pkl
    with open(filepath, 'r') as f:
        clusters = pkl.load(f)
    return Clusters(clusters)

class Clusters(dict):
    """ A dictionary that contains clustered spike data.

        The methods for this class assumes the values are numpy recarrays
        with fields 'spikes', 'times', and 'feats' (features).

        **Attributes**:
            *ids*:
             Cluster ids

        Inherits from :func:`dict`.

    """
    
    def __init__(self, *args, **kwargs):
        super(Clusters, self).__init__(*args, **kwargs)
            
    def select(self, clusters):
        """ Selects multiple clusters which you want returned for further 
            analysis.

            **Arguments**:
                *clusters*:
                 A list of the clusters you want returned.
        """
        if clusters is None:
            return self
        
        return Clusters({cl:self[cl] for cl in clusters})
        
    def sizes(self):
        """ Returns the size of each cluster. """
        return {cl:len(self[cl]) for cl in self}
    
    def features(self):
        """ Returns only the feature arrays from the clusters. """
        return Clusters({cl:self[cl]['feats'] for cl in self})
    
    def times(self):
        """ Returns only the timestamp arrays from the clusters. """
        return Clusters({cl:self[cl]['times'] for cl in self})
    
    def spikes(self):
        """ Returns only the spike waveform arrays from the clusters. """
        return Clusters({cl:self[cl]['spikes'] for cl in self})
    
    def flatten(self):
        """ Flattens the clusters into a single array """
        return np.concatenate([self[cl] for cl in self])

    def combine(self, source, destination):
        """ Combine source cluster into destination cluster. """

        # Get data from source clusters
        try:
            clusters = [ self[cl] for cl in source ]
        except TypeError: # If source isn't iterable
            source = [source]
            clusters = [ self[cl] for cl in source ]
        except KeyError as e: # If one of the source clusters doesn't exist
            raise ClusterIdError("Cluster id {} does not exist".format(e))

        # Remove source clusters
        _ = [ self.pop(cl) for cl in source ]
        
        # Add source clusters to destination cluster
        clusters.append(self[destination])
        combined = np.concatenate(clusters)
        combined.sort(order='times')
        self.update({destination:combined})
        
        return self

    def change_id(self, current_id, new_id):
        """ Change the id number of a cluster. """

        # First, make sure new id doesn't exist
        if new_id in self:
            raise ClusterIdError('Cluster {} already exists.'.format(new_id))
        else:
            self.update({destination:self.pop(source)})
        return self

    def copy(self):
        """ Make a copy. """
        return Clusters(super(Clusters, self).copy())

    @property
    def ids(self):
        return self.keys()

    def save(self, filepath):
        """ Save the clusters as a pickled dictionary to filepath. """
        import cPickle as pkl
        from os import path

        filepath = path.expanduser(filepath)

        with open(filepath, 'w') as f:
            output = {key: value for key, value in self.iteritems()}
            pkl.dump(output, f)
        print("{} clusters saved at {}".format(len(self), filepath))

    def __repr__(self):
        return str("Clusters: {}".format(self.sizes()))

class Viewer(object):
    """ A class used to view clustered data. 

        Takes a Clusters object and supplies various methods for viewing the
        clustered data.

        **Attributes**:
            *clusters*:
             Clusters object containing the spike data
            *cm*:
             Matplotlib color map used for coloring the clusters.  Feel free 
             to change this, I like this color map though.
    """
    
    def __init__(self, clusters):
        """ Takes either a Clusters object, or the path to a pickled Clusters
            object.
        """
        if isinstance(clusters, basestring):
            self.clusters = load_clusters(clusters)
        else:
            self.clusters = clusters
        self.cm = plt.plt.cm.Paired
        
    def scatter(self, clusters=None, components=[1,2,3], limit=500):
        """ Generates a scatter plot in feature space of the clustered data.
        """
        from scipy.misc import comb
        from itertools import combinations
        
        components = [ c-1 for c in components ]
        feats, col_array = self._scatter_helper(clusters, limit)
        N_plots = int(comb(len(components), 2, exact=True))
        
        fig, axes = plt.generate_axes(N_plots, ncols=3, num=1, figsize=(16,5))
        for ax, (x,y) in zip(axes,combinations(components,2)):
            ax.clear() # Clear the axes before replotting
            plt.scatter(feats[:,x], feats[:,y], colors=col_array, ax=ax)
            ax.set_xlabel("Component {}".format(x))
            ax.set_ylabel("Component {}".format(y))
        
        fig.tight_layout()
        
        return self
    
    def scatter3D(self, clusters=None, components=[1,2,3], limit=500):
        """ Generates a 3D scatter plot for viewing clusters. 
        """
        cx, cy, cz = [ c-1 for c in components ]
        feats, col_array = self._scatter_helper(clusters, limit)
        x, y, z = feats[:,cx], feats[:,cy], feats[:,cz]
        ax = plt.scatter3D(x, y, z, colors = col_array)
        
        ax.figure.tight_layout()
        
        return self
        
    def spikes(self, clusters=None, limit=50, figsize=None):
        """ Generates plots of clustered spike waveforms.
        """
        
        cls = self.clusters.select(clusters)
        cl_spikes = cls.spikes()
        colors = plt.get_colors(max(self.clusters.keys()) + 1, self.cm)
        
        fig, axes = plt.generate_axes(len(cls), 4, num=2, sharex=True,
                                      figsize=figsize)
        for ax, cl in zip(axes, cl_spikes):
            ax.clear()
            spks = cl_spikes[cl]
            plt.spikes(spks, ax=ax, color=colors[cl], patch_size=spks.shape[1]/4)
            ax.set_title('Cluster {}'.format(cl))
            ax.set_ylabel('Voltage (mv)')
            ax.set_xticklabels('')
        
        fig.tight_layout()
        
        return self
        
    def autocorrs(self, clusters=None, bin_width=0.0015, limit=0.03, 
                        figsize=None):
        """ Plots of autocorrelations of clustered spike times.

            **Keywords**:
                *clusters*: list or iterable
                 List of clusters to plot
                *bin_width*: float
                 Width of bins in the autocorrelation calculation
                *limit*: float
                 Time limit over which to calculate the autocorrelation
        """
        
        cls = self.clusters.select(clusters)
        cl_times = cls.times()
        colors = plt.get_colors(max(self.clusters.keys()) + 1, self.cm)

        fig, axes = plt.generate_axes(len(cls), 4, num=3, figsize=figsize,
                                      sharex=True)
        for ax, cl in zip(axes, cl_times):
            ax.clear()
            tstamps = cl_times[cl]
            plt.autocorr(tstamps, ax=ax, color=colors[cl], 
                         bin_width=bin_width, limit=limit)
            ax.set_title('Cluster {}'.format(cl))
            ax.set_xlabel('Time (ms)')
        
        fig.tight_layout()
        
        return self

    def crosscorrs(self, clusters=None, bin_width=0.0015, limit=0.03, 
                         figsize=(9,5)):
        """ Plots of cross-correlations of clustered spike times. """

        times = self.clusters.select(clusters).times()
        colors = plt.get_colors(max(self.clusters.keys()) + 1, self.cm)
        
        # Set the number of rows and columns to plot
        n_rows, n_cols = [len(times)]*2
        fig, axes = plt.generate_crosscorr_axes(n_rows, n_cols, num=4,
                                                figsize=figsize)
        
        for (ii, jj) in axes:
            ax = axes[(ii,jj)]
            cl1, cl2 = times.keys()[ii], times.keys()[jj]
            t1, t2 = times[cl1], times[cl2]
            # Get the cross-correlation for different clusters
            if ii != jj:
                plt.crosscorr(t1, t2, ax=ax, bin_width=bin_width, limit=limit)
                ax.set_xticklabels('')
                ax.set_yticklabels('')
            
            # If cluster 1 is the same as cluster 2, get the autocorrelation
            else:
                plt.autocorr(t1, ax=ax, color=colors[cl1],
                                 bin_width=bin_width, limit=limit)
                ax.set_ylabel('{}'.format(cl1))
                ax.set_xticklabels('')
                ax.set_yticklabels('')
        
        return self

    def timestamps(self, clusters=None, color='k', xlims=(0,4000), figsize=(9,6)):
        """ Plot the timestamps for clusters. """
        clusters = self.clusters.select(clusters)
        times = clusters.times()
        fig, axes = plt.generate_axes(len(clusters), 2, figsize=figsize)
        for cl, ax in zip(times, axes):
            plt.timestamps(times[cl], ax=ax, color=color, xlims=xlims)
            ax.set_title('Cluster {}'.format(cl))
        fig.tight_layout()

    def feature_trace(self, dimension, clusters=None, marker='o', color='k', xlims=(0,4000), figsize=(9,6)):
        clusters = self.clusters.select(clusters)
        fig, axes = plt.generate_axes(len(clusters), 2, figsize=figsize)
        for cl, ax in zip(clusters, axes):
            plt.feature_trace(clusters[cl]['feats'][:, dimension],
                              clusters[cl]['times'], 
                              ax=ax, color=color, marker=marker, xlims=xlims)
            ax.set_title('Cluster {}'.format(cl))
        fig.tight_layout()

    def _scatter_helper(self, clusters=None, limit=500):
        """ A helper method to generate the data for the scatter plots. """
        
        cls = self.clusters.select(clusters)
        # Here, limit the data so that we don't plot everything
        cls = Clusters({cl:plt.limit_data(cls[cl], limit) for cl in cls})
        
        # Get the color and feature arrays for fast plotting
        colors = plt.get_colors(max(self.clusters.keys()) + 1, self.cm)
        col_array = plt.color_array(cls, colors)
        feats = cls.features().flatten()
        
        return feats, col_array

    def __len__(self):
        return len(self.clusters)

class Sorter(Viewer):
    """ This class performs the clustering using a Gaussian Mixture Model.
        It uses Viewer for plotting the results of the clustering.

        **Arguments**:
            *K*: (int)
             The number of clusters to fit to the data
            *dims*: (int)
             The number of dimensions to reduce the data into
            *cov_type*: ('diag', 'tied', or 'full')
             Type of covariance matrix to use in the GMM, 'tied' is faster,
             but less accurate than 'full'. 
            *decomp*: ('pca' or 'ica')
             use PCA or ICA to reduce the dimensions of the data

        **Attributes**:
            *clusters*:
             Clusters object containing the spike data
            *cm*:
             Matplotlib color map used for coloring the clusters.  Feel free 
             to change this, I like this color map though.
            *params*:
             Dictionary containing parameters for the Sorter

        
        Inherits from :class:`Viewer`.
    """
    
    decomps = {'pca':fit_pca, 'ica':fit_ica}
    
    def __init__(self, K=10, dims=10, cov_type='full', decomp='pca', assign_prob=0.5):
        """ 
            **Arguments**:
                *K*: (int)
                 The number of clusters to fit to the data
                *dims*: (int)
                 The number of dimensions to reduce the data into
                *cov_type*: ('diag', 'tied', or 'full')
                 Type of covariance matrix to use in the GMM, 'tied' is faster,
                 but less accurate than 'full'. 
                *decomp*: ('pca' or 'ica')
                 use PCA or ICA to reduce the dimensions of the data
        """
        
        self.params = {'K':K, 'dims':dims, 'cov_type':cov_type, 'decomp':decomp, 
                       'assign_prob':assign_prob}
        self.params.update(kwargs)
        self.data = None
        self.gmm, self.clusters = None, None
        self.cm = plt.plt.cm.Paired

    def sort(self, data):
        """ Sort the data into clusters based on spike waveform features. """
        self.data = data
        data = data['spikes']
        _, spike_size = data.shape
        
        decomp_func = self.decomps[self.params['decomp']]
        decomped, reduced = decomp_func(data, dims=self.params['dims'])
        self.gmm, clustered = cluster(reduced, **self.params)
        
        self.clusters = Clusters()
        for c_id in clustered:
            cl = clustered[c_id]
            N = len(cl)
            recs = np.zeros(N, dtype=[('spikes', 'f8', spike_size),
                                  ('times', 'f8', 1),
                                  ('feats', 'f8', self.params['dims'])
                                  ])
            recs['spikes']= self.data['spikes'][cl]
            recs['times'] = self.data['times'][cl]
            recs['feats'] = reduced[cl]
            recs.sort(order='times')
            self.clusters[c_id] = recs
        return self
    
    def bic(self):
        """ Get the Baysian Information Criterion (BIC) for the model.
        """
        return self.gmm.bic(self.clusters.features().flatten())

    def split(self, source, K=2):
        """ Resort a cluster into K new clusters. """
        source_cluster = self.clusters.pop(source)
        _, clustered = cluster(source_cluster['feats'], K=K)

        # Find unused cluster numbers and assign new clusters
        unused = [ i for i in range(len(self)+K) if i not in self.clusters]
        for i, cl in zip(unused, clustered.itervalues()):
            self.clusters[i] = source_cluster[cl]
        return self

    def combine(self, source, destination):
        """ Combine two clusters, from source into destination. """
        self.clusters.combine(source, destination)
        return self

    def save(self, filepath):
        """ Save the clusters as a pickled dictionary to filepath. """
        self.clusters.save(filepath)

    def __len__(self):
        return len(self.clusters)

    def __repr__(self):
        self.params['K'] = len(self)
        return 'Sorter({})'.format(str(self.params))
    