""" This module is intended for processing raw electrical signals.  For
    example, to detect and extract spikes or other threshold crossings from
    an electrophysiology recording.
    
    Author: Mat Leonard
    First version: June 6, 2013
"""

import numpy as np
from functools import wraps

class ProcessingError(Exception):
    pass

def map_parallel(func, data, processes=2):
    """ This maps the data to func in parallel using multiple processes. 
        This works fine in the IPython terminal, but not in IPython notebook.
    """

    from multiprocessing import Pool
    pool = Pool(processes=processes)
    output = pool.map(func, data)
    pool.close()
    pool.join()
    
    return output

def tetrode_chans(tet_num):
    """ Returns the channel numbers for the requested tetrode.  These channels
        are only valid for the H04 adapter used in our lab.
    """
    
    tetrodes = {1:[16,18,17,20], 2:[19,22,21,24], 3:[23,26,25,28],
                4:[27,30,29,32]}
        
    return tetrodes[tet_num]
    
def load_data(filename, channels=None, tetrode=None):
    ''' Loads data from an ns5 file.  This returns a generator, so you only get
        one channel at a time, but you don't have to load in all the channels
        at once, saving memory since it is a LOT of data.
    '''
    import ns5
    
    loader = ns5.Loader(filename)
    loader.load_file()
    bit_to_V = 4096.0 / 2.**15 # mV/bit
    
    if tetrode:
        channels = tetrode_chans(tetrode)

    for chan in channels:
        yield loader.get_channel_as_array(chan)*bit_to_V

def common_ref(data, n=None):
    """ Calculates the common average reference from the data.  If the length
        of data can't be found with len(), set n to the length of data.
    """
    try:
        n = len(data)
    except:
        n = n
    return sum(data)/float(n)

def process_data(filename, channels, low=300, high=6000, rate=30000, 
                 common_ref=None):
    """ This function processes the data from filename and returns an array
        of tetrode spike waveforms and their spike timestamps.
    """

    # Detect the spike times first
    if common_ref != None:
        data = ( dat - common_ref for dat in load_data(filename, channels) )
    else:
        data = load_data(filename, channels)
    filtered = bfilter(data, low=low, high=high, rate=rate)
    spikes = detect_spikes(filtered)
    
    # Then use those spike times to form the tetrode waveforms.  We want to
    # extract from the data without the common reference removed.
    data = load_data(filename, channels)
    filtered = bfilter(data, low=low, high=high, rate=rate)
    spikes = map_parallel(detect_spikes, filtered)
    extracted = form_tetrode(filtered, spikes)
    return extracted

def save_spikes(filename, spikes):
    """ Saves spikes record array to file. """

    with open(filename, 'w') as f:
        spikes.tofile(f)
    print('Saved to {}'.format(filename))

def load_spikes(filename, ncols=120):
    """ Loads recarray saved with save_spikes.  The keyword ncols should be
        set to the length of the spike waveform.
    """

    with open(filename, 'r') as f:
        loaded = np.fromfile(filename)

    spikes = loaded.reshape(len(loaded)/(ncols+1), ncols+1)
    records = [('spikes', 'f8', ncols), ('times', 'f8', 1)]
    recarray = np.zeros(len(spikes), dtype = records)
    recarray['spikes'] = spikes[:,:120]
    recarray['times'] = spikes[:,-1]

    return recarray

def detect_spikes(data, threshold=4, patch_size=30):
    """ Detect spikes in data.  Returns spike waveform patches and peak 
        samples.

        Arguments
        ---------
        data : np.array : data to extract spikes from

        Keyword Arguments
        -----------------
        threshold : int, float : threshold*sigma for detection
        patch_size : int : number of samples for extracted spike
    """
    import time
    start = time.time()
 
    threshold = medthresh(data, threshold=threshold)
    peaks = crossings(data, threshold, polarity='neg')
    peaks = censor(peaks, 30)
    
    spikes, times = extract(data, peaks, patch_size=patch_size)
    
    records = [('spikes', 'f8', patch_size), ('times', 'f8', 1)]
    detected = np.zeros(len(times), dtype=records)
    detected['spikes'] = spikes
    detected['times'] = times
    
    elapsed = time.time() - start
    print("Detected {} spikes in {} seconds".format(len(times), elapsed))

    return detected

def form_tetrode(data, times, patch_size=30, offset=0, samp_rate=30000):
    """ Build tetrode waveforms from voltage data and detected spike times.

        Arguments
        ---------
        data : the data from which to extract the spike waveforms
        spikes : recarray from detect_spikes.  Should have a field 'times'.
            The times are used to extract waveforms from data.
    """

    extracted = [ extract(chan, times, 
                          patch_size=patch_size,
                          offset=offset) 
                  for chan in data]
    waveforms = np.concatenate([ wv for wv, time in extracted ], axis=1)
    records = [('spikes', 'f8', patch_size*4), ('times', 'f8', 1)]
    tetrodes = np.zeros(len(times), dtype=records)
    tetrodes['spikes'] = waveforms
    tetrodes['times'] = times/float(samp_rate)

    return tetrodes

def trim_data(*args, **kwargs):
    """ Returns a smaller number of data points, reduced by the value of trim.
        Useful for plotting fewer points for speed purposes.
        
        Arguments
        ---------
        Any number of position arguments.  They should all be numpy arrays of
        the same dimension.
        
        Keyword Arguments
        -----------------
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

def medthresh(data, threshold=4):
    """ A function that calculates the spike crossing threshold 
        based off the median value of the data.
    
    Arguments
    ---------
    data : your data
    threshold : the threshold multiplier
    """
    return threshold*np.median(np.abs(data)/0.6745)

def bfilter(data, low=300, high=6000, rate=30000):
    """ Filters the data with a 3-pole Butterworth bandpass filter.
    
        Arguments
        ---------
        data : numpy array : data you want filtered
        low : int, float : low frequency rolloff
        high : int, float : high frequency rolloff
        rate : int, float : sample rate
    
    """
    import scipy.signal as sig
    
    if high > rate/2.:
        high = rate/2.-1
        print("High rolloff frequency can't be greater than the Nyquist \
               frequency.  Setting high to {}").format(high)

    filter_lo = low #Hz
    filter_hi = high #Hz
    samp_rate = float(rate)
    
    #Take the frequency, and divide by the Nyquist freq
    norm_lo = filter_lo/(samp_rate/2)
    norm_hi = filter_hi/(samp_rate/2)
    
    # Generate a 3-pole Butterworth filter
    b, a = sig.butter(3, [norm_lo,norm_hi], btype="bandpass");
    return sig.filtfilt(b, a, data)

def censor(data, width=30):
    """ This is used to insert a censored period in found threshold crossings.
        For instance, when you find a crossing in the signal, you don't
        want the next 0.5-1 ms, you just want the first crossing.
        
        Arguments
        ---------
        data : numpy array : data you want censored
        width : int : number of samples censored after a first crossing
    """
    try:
        edges = [data[0]]
    except IndexError:
        raise ValueError("data is empty")
    
    for sample in data:
        if sample > edges[-1] + width:
            edges.append(sample)
    return np.array(edges)
    
def crossings(data, threshold, polarity='pos'):
    """ Finds threshold crossings in data 
    
        Arguments:
        data : numpy array
        threshold : int, float : crossing threshold, always positive
        polarity : 'pos', 'neg', 'both' :
            'pos' : detects crossings for +threshold
            'neg' : detects crossings for -threshold 
            'both' : both + and - threshold
    """
    
    if type(data) != list:
        data = [data]
    peaks = []
    for chan in data:
        if polarity == 'neg' or polarity == 'both':
            below = np.where(chan<-threshold)[0]
            peaks.append(below[np.where(np.diff(below)==1)])
        elif polarity == 'pos' or polarity == 'both':
            above = np.where(chan>threshold)[0]
            peaks.append(above[np.where(np.diff(above)==1)])

    return np.concatenate(peaks)
    
def extract(data, peaks, patch_size=30, offset=0, polarity='neg'):
    """ Extract peaks from data based on sample values in peaks. 
    
        Arguments
        ---------
        patch_size : int : 
            number of samples to extract centered on peak + offset
        offset : int : 
            number of samples to offset the extracted patch from peak
        polarity : 'pos' or 'neg' :
            Set to 'pos' if your spikes have positive polarity
        
        Returns
        -------
        spikes : numpy array : N x patch_size array of extracted spikes
        peaks : numpy array : sample values for the peak of each spike
    """
    
    spikes, peak_samples = [], []
    size = patch_size/2
    for peak in peaks:
        patch = data[peak-size:peak+size]
        if polarity == 'pos':
            peak_sample = patch.argmax()
        elif polarity == 'neg':
            peak_sample = patch.argmin()
        centered = peak-size+peak_sample+offset
        peak_sample = peak-size+peak_sample
        final_patch = data[centered-size:centered+size]
        peak_samples.append(peak_sample)
        spikes.append(final_patch)
        
    return np.array(spikes), np.array(peak_samples)