""" This module is intended for unsupervised clustering using a Gaussian
    Mixture Model (GMM).
    
    Author: Mat Leonard
    First version: June 6, 2013
"""

import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GMM
import plots as plt

def fit_pca(data, dims):
    """ Fit PCA decomposition to the data with dims components """
    pca = PCA(n_components=dims)
    x = data - np.mean(data, axis=0)
    return pca, pca.fit_transform(x)

def fit_ica(data, dims):
    """ Fit ICA decomposition to the data with dims components """
    ica = FastICA(n_components=dims)
    x = data - np.mean(data, axis=0)
    return ica, ica.fit_transform(x)

def cluster(data, **kwargs):
    """ Fit a GMM to the data, forming clusters of similar data points.
    """
    
    X = data
    gmm = GMM(n_components = kwargs['K'],
              covariance_type = kwargs['cov_type'],
              )
    
    while not gmm.converged_:
        gmm.init_params = ''
        gmm.fit(X)
    
    predicted = gmm.predict(X)
    clusters = { cl:np.where(predicted==cl)[0] for cl in np.unique(predicted)}
    return gmm, clusters
    
def load_clusters(filepath):
    import cPickle as pkl
    with open(filepath, 'r') as f:
        clusters = pkl.load(f)
    return clusters

class Clusters(dict):
    
    def __init__(self, other=None):
        super(dict, self).__init__()
        
        if other:
            try:
                self.update(other)
            except TypeError:
                raise ValueError("Initialize as you would when calling dict()")
            
    def select(self, clusters):
        """ Selects multiple clusters which you want returned for further 
            analysis.
        """
        if clusters == None:
            return self
        
        try:
            cls = Clusters({cl:self[cl] for cl in clusters})
        except TypeError as e:
            if 'not iterable' in e[0]:
                clusters = [clusters]
        
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
    
    def __repr__(self):
        return str("Clusters: {}".format(self.sizes()))

class Viewer(object):
    """ A class used to view clustered data. """
    
    def __init__(self, clusters):
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
        
    def spikes(self, clusters=None, limit=50):
        """ Generates plots of clustered spike waveforms.
        """
        
        cls = self.clusters.select(clusters)
        
        colors = plt.get_colors(len(self), self.cm)
        cl_spikes = cls.spikes()
        
        fig, axes = plt.generate_axes(len(cls), 4, num=2, sharex=True)
        for ax, cl in zip(axes, cl_spikes):
            ax.clear()
            spks = cl_spikes[cl]
            plt.spikes(spks, ax=ax, color=colors[cl])
            ax.set_title('Cluster {}'.format(cl))
            ax.set_ylabel('Voltage (mv)')
            ax.set_xticklabels('')
        
        fig.tight_layout()
        
        return self
        
    def autocorrs(self, clusters=None, bin_width=0.0015, limit=0.03):
        """ Creates plots of autocorrelations for clustered spike times.
        """
        
        cls = self.clusters.select(clusters)
        
        colors = plt.get_colors(len(self), self.cm)
        times = cls.times()
        fig, axes = plt.generate_axes(len(cls), 4, num=3)
        for ax, cl in zip(axes, times):
            ax.clear()
            tstamps = times[cl]
            plt.autocorr(tstamps, ax=ax, color=colors[cl], 
                         bin_width=bin_width, limit=limit)
            ax.set_title('Cluster {}'.format(cl))
        
        fig.tight_layout()
        
        return self
        
    def _scatter_helper(self, clusters=None, limit=500):
        """ A helper method to generate the data for the scatter plots. """
        
        cls = self.clusters.select(clusters)
        # Here, limit the data so that we don't plot everything
        cls = Clusters({cl:plt.limit_data(cls[cl], limit) for cl in cls})
        
        # Get the color and feature arrays for fast plotting
        colors = plt.get_colors(len(self), self.cm)
        col_array = plt.color_array(cls, colors)
        feats = cls.features().flatten()
        
        return feats, col_array

    def __len__(self):
        return len(self.clusters)

class Sorter(Viewer):
    """ This class performs the clustering using a Gaussian Mixture Model.
        It inherits from Viewer for plotting the results of the clustering.
    """
    
    decomps = {'pca':fit_pca, 'ica':fit_ica}
    params = {'K':9, 'dims':10, 'cov_type':'tied', 'decomp':'pca'}
    
    def __init__(self, **kwargs):
        self.params.update(kwargs)
        self.data = None
        self.gmm, self.clusters = None, None
        self.cm = plt.plt.cm.Paired

    def sort(self, data):
        self.data = data
        data = data['waveforms']
        _, spike_size = data.shape
        
        decomp_func = self.decomps[self.params['decomp']]
        decomped, reduced = decomp_func(data, dims=self.params['dims'])
        self.gmm, clusters = cluster(reduced, **self.params)
        
        self.clusters = Clusters()
        for c_id in clusters:
            cl = clusters[c_id]
            N = len(cl)
            recs = np.zeros(N, dtype=[('spikes', 'f8', spike_size),
                                  ('times', 'f8', 1),
                                  ('feats', 'f8', self.params['dims'])
                                  ])
            recs['spikes']= self.data['waveforms'][cl]
            recs['times'] = self.data['times'][cl]
            recs['feats'] = reduced[cl]
            recs.sort(order='times')
            self.clusters[c_id] = recs
            
    def __repr__(self):
        return str(self.params)
    