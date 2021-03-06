"""
.. module:: plots
    :synopsis: A module for plotting various data of sorted units.

.. moduleauthor:: Mat Leonard <leonard.mat@gmail.com>
"""
from functools import wraps

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def color_array(clusters, color_dict):
    """ Returns a 1D array with the color for each data point """
    
    colors = [ color_dict[cl] for cl in clusters ]
    col_sizes = zip(colors, clusters.sizes().values())
    col_array = np.concatenate([[col]*size for col, size in col_sizes])
    
    return col_array
    
def get_colors(K, colormap=plt.cm.Paired):
    """ Method for setting the colors used when plotting clusters. """
    colors = colormap(np.arange(0.0,1.0,1.0/K))
    color_dict = { cl:color for cl, color in zip(range(K), colors) }
    return color_dict

def trim_data(*args, **kwargs):
    """ Returns a smaller number of data points, reduced by the value of trim.
        Useful for plotting fewer points for speed purposes.
        
        **Arguments**
        
        Any number of position arguments.  They should all be numpy arrays of
        the same dimension.
        
        **Keywords**
        trim : float between 0 and 1, inclusive. 
            This is the factor the data points are trimmed by.
        
    """
    
    if 'trim' in kwargs:
        trim = kwargs['trim']
    else:
        trim = 1
    
    trimmed=[]
    if 0 <= trim <= 1:
        N = len(args[0])
        chosen = np.random.choice(N, size=int(trim*N), replace=False)
        for i, arg in enumerate(args[:]):
            trimmed.append(arg[chosen])
        
        if len(trimmed)>1:
            return trimmed
        else:
            return trimmed[0]
            
    else:
        raise ValueError("trim must be between 0 and 1.")

def passed_or_new_ax(func):
    """ This is a decorator for the plots in this module.  Most plots can be
        passed an existing axis to plot on.  If it isn't passed an axis, then
        a new figure and new axis should be generated and passed to the plot.
        This decorator ensures this happens.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        if 'ax' in kwargs:
            return func(*args, **kwargs)
        else:
            fig, ax = plt.subplots()
            kwargs.update({'ax':ax})
        return func(*args, **kwargs)
    return inner

@passed_or_new_ax
def scatter(x, y, ax=None, trim=1, colors='k', s=5):
    """ Generates a scatter plot.  The number of points is reduced by the
        value of trim, so if you are plotting a bunch of points, it won't
        show them all.  Set trim=0.5 to plot half the points, for example.
        
        **Arguments**
        
            *x, y*: 
             array-like, must be the same length, data values to plot
            *ax*:
             matplotlib axes
            *trim*:
             float between 0 and 1, inclusive, trims data points
            *colors*:
             colors of the data points, can be an array of the same length
             as x and y
    """
    
    if len(colors)==1:
        x, y = trim_data(x, y, trim=trim)
    else:
        x, y, colors = trim_data(x, y, colors, trim=trim)
    ax.scatter(x, y, c=colors, marker='.', s=s, edgecolor = 'face')
    return ax

@passed_or_new_ax
def spikes(data, ax=None, color='r', limit=50, patch_size=30):
    """ Plots the spike waveforms assuming the data is from a tetrode.
        
        **Arguments**
            *data*:
             Array of spike waveforms
            *ax*: 
             Matplotlib axis
            *color*:
             Color of the waveforms
            *limit*: 
             Number of waveforms to plot
    """
    patch = patch_size
    gap = 10
    spikes = limit_data(data, limit).T
    spike_mean = spikes.mean(axis=1)
    
    # I'm going to plot the waveforms in patches to separate them visually.
    xslices = [ ((patch+gap)*i, (patch+gap)*i+patch) for i in range(4) ]
    pslices = [ (patch*i, patch*i+patch) for i in range(4) ]
    xs = [ np.arange(left, right) for left, right in xslices ]
    patches = [ spikes[left:right,:] for left, right in pslices ]
    means = [ spike_mean[left:right] for left, right in pslices ]
    
    for x, p, m in zip(xs, patches, means):
        ax.plot(x, p, color=color, alpha=0.2)
        #ax.plot(x, m, color='k')
        ax.set_xlim((0,4*patch_size+30))
    
    return ax

def scatter3D(x, y, z, ax=None, trim=1, colors='k'):
    """ Generates a scatter plot in 3D.  The number of points is reduced by 
        the value of trim, so if you are plotting a bunch of points, it won't
        show them all.  Set trim=0.5 to plot half the points, for example. 
    """
    
    if not ax:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')     
    
    if len(colors)==1:
        x, y, z = trim_data(x, y, z, trim=trim)
    else:
        x, y, z, colors = trim_data(x, y, z, colors, trim=trim)
    
    ax.scatter(x, y, z, c=colors, s=5, marker='.', edgecolor = 'face')
    return ax

@passed_or_new_ax
def autocorr(times, ax=None, color='k', bin_width=0.0015, limit=0.03):
    """ Plots the autocorrelation of times. """
    
    counts, bins = correlogram(times, bin_width = bin_width, 
                                limit = limit, auto=True)
    ax.bar(bins[:-1]*1000, counts, width = bin_width*1000, 
           color = color, edgecolor = 'none')
    ax.set_xlim((-limit-bin_width)*1000, (limit+bin_width)*1000)
    
    return ax

@passed_or_new_ax
def crosscorr(t1, t2, ax=None, color='k', bin_width=0.0015, limit=0.03):
    """ Plots the cross-correlation of t1 and t2. """
    
    counts, bins = correlogram(t1, t2, bin_width = bin_width, 
                                limit = limit, auto=False)
    ax.bar(bins[:-1]*1000, counts, width = bin_width*1000, 
           color = color, edgecolor = 'none')
    ax.set_xlim((-limit-bin_width)*1000, (limit+bin_width)*1000)
    
    return ax

@passed_or_new_ax
def timestamps(times, ax=None, color='k', xlims=(0,4000)):
    """ Plot time stamps, randomly scattered along the y-axis.  """
    ax.scatter(times, np.random.rand(*times.shape), marker='|', color=color)
    ax.set_yticklabels('')
    ax.set_xlim(*xlims)
    ax.set_xlabel('Time (s)')
    return ax

@passed_or_new_ax
def feature_trace(features, times, ax=None, color='k', marker='.', xlims=(0,4000)):
    """ Plot feature values over time. """
    ax.scatter(times, features, marker=marker, edgecolor='none', 
               facecolor=color, alpha=0.5)
    ax.set_xlim(*xlims)
    ax.set_xlabel('Time (s)')
    return ax

def generate_axes(N_plots, ncols, **kwargs):
    """ Generate multiple axes on a figure, given the number of desired 
        plots and columns.
    """
    nrows = (N_plots-1)/ncols+1
    kwargs.update({'nrows':nrows, 'ncols':ncols})
    figsize = kwargs.get('figsize', None)
    if figsize is None:
        figsize = (10, 1+2*nrows)
        kwargs.update({'figsize':figsize})
    fig, axes = plt.subplots(**kwargs)
    axes = axes.flatten()
    
    return fig, axes

def generate_crosscorr_axes(n_rows, n_cols, **kwargs):
    """ Return a dict of indices and axes for the figure used in 
        cross-correlations. """
    
    # Build a list used to iterate through all the correlograms
    l_rows = np.arange(n_rows+1)
    l_cols = np.arange(n_cols)
    ind = [(x, y) for x in l_rows for y in l_cols if y >= x]

    fig = plt.figure(**kwargs)
    axes = { (ii,jj): fig.add_subplot(n_rows, n_cols, ii*n_cols + jj + 1) 
             for ii, jj in ind }
    return fig, axes

def limit_data(data, max_limit):
    """ Returns data points limited in number by max_limit """
    
    N = len(data)
    if N <= max_limit:
        return data
    else:
        chosen = np.random.choice(N, size=int(max_limit), replace=False)
    return data[chosen]
    
def correlogram(t1, t2=None, bin_width=.001, limit=.02, auto=False):
    
    """Return crosscorrelogram of two spike trains.
    
    Essentially, this algorithm subtracts each spike time in `t1` 
    from all of `t2` and bins the results with numpy.histogram, though
    several tweaks were made for efficiency.
    
    **Arguments**
        t1 : first spiketrain, raw spike times in seconds.
        t2 : second spiketrain, raw spike times in seconds.
        bin_width : width of each bar in histogram in sec
        limit : positive and negative extent of histogram, in seconds
        auto : if True, then returns autocorrelogram of `t1` and in
            this case `t2` can be None.
    
    **Returns**
        (count, bins) : a tuple containing the bin edges (in seconds) and the
        count of spikes in each bin.

        `bins` is relative to `t1`. That is, if `t1` leads `t2`, then
        `count` will peak in a positive time bin.
    """
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if auto: t2 = t1

    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)
    bins = np.linspace(-limit, limit, num=(2 * limit/bin_width + 1))

    # This is the old way to calculate bin edges. I think it is more
    # sensitive to numerical error. The new way may slightly change the
    # way that spikes near the bin edges are assigned.
    #bins = np.arange(-limit, limit + bin_width, bin_width)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to numpy.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] -= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins

def pub_format(ax, legend=None, spine_width=1.3, spine_color='#111111', 
               hide_spines=['top', 'right'], 
               yticks=3, xticks=3, yminorticks=1, xminorticks=1):
    """ Format plots for publication. """
    if hide_spines is None:
        pass
    else:
        for each in hide_spines:
            ax.spines[each].set_color('w')
    ax.set_axis_bgcolor('w')
    
    shown_spines = [each for each in ['top', 'bottom', 'left', 'right']
                    if each not in hide_spines]
    for each in shown_spines:
        ax.spines[each].set_linewidth(spine_width)
        ax.spines[each].set_color(spine_color)
    
    ax.yaxis.set_major_locator(MaxNLocator(yticks))
    ax.xaxis.set_major_locator(MaxNLocator(xticks))
    ax.yaxis.set_minor_locator(MaxNLocator(yticks*yminorticks))
    ax.xaxis.set_minor_locator(MaxNLocator(xticks*xminorticks))
    
    ax.tick_params(axis='both', which='minor', length=3, width=1.3, 
                   direction='out', colors='#111111')
    ax.tick_params(axis='both', which='major', length=4, width=1.3, 
                   direction='out', colors='#111111')
    ax.tick_params(which='both', **dict(zip(hide_spines, ['off']*len(hide_spines))))
    ax.tick_params(direction='out', width=spine_width, color=spine_color)
    ax.grid(which='both', b=False)

    if legend:
        leg_frame = legend.get_frame()
        leg_frame.set_facecolor('w')
        leg_frame.set_edgecolor('none')
    
    return ax
    