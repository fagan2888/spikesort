""" This module is intended for processing raw electrical signals.  For
    example, to detect and extract spikes or other threshold crossings from
    an electrophysiology recording.
    
    Author: Mat Leonard
    First version: June 6, 2013
"""

import numpy as np
from functools import wraps
from Queue import Empty

class ProcessingError(Exception):
    pass
    
def batch_process(func):
    """ This is a decorator for running functions in multiple processes.
        It should be applied only to functions with one positional argument
        and an arbitrary number of keyword arguments.  The first argument
        must be iterable.  Also, as it is written now, it will generate a
        process for each item in the iterable first argument, so you will want
        to be careful.  I'll change this to set a limit on the number of
        processes it will create. #TODO
    """
    from multiprocessing import Process, Queue
    
    @wraps(func)
    def batched(*args, **kwargs):
        
        # This function calls func and puts it in the queue
        def f(q, dat, **kwargs):
            q.put(func(dat, **kwargs))
        
        # Create the queue and processes
        que = Queue()
        jobs = [ Process(target=f, args=(que, dat), kwargs=kwargs) 
                 for dat in args[0] ]
        
        # Run jobs and get the results
        for job in jobs: job.start()
        output = [ que.get() for i in range(len(jobs)) ]
        
        # Stop processes and such
        for job in jobs: job.join(timeout=30)
        for job in jobs: job.terminate()
        for job in jobs: job.join(timeout=30)
        que.close()
        
        if len(output) == 1:
            return output[0]
        else:
            return output
    
    return batched
    
def tetrode_chans(tet_num):
    """ Returns the channel numbers for the requested tetrode.  These channels
        are only valid for the H04 adapter used in our lab.
    """
    
    tetrodes = {1:[16,18,17,20], 2:[19,22,21,24], 3:[23,26,25,28],
                4:[27,30,29,32]}
        
    return tetrodes[tet_num]
    
def load_data(filename, channels):
    ''' Loads data from an ns5 file.  This returns a generator, so you only get
        one channel at a time, but you don't have to load in all the channels
        at once, saving memory since it is a LOT of data.
    '''
    import ns5
    
    loader = ns5.Loader(filename)
    loader.load_file()
    
    for chan in channels:
        yield loader.get_channel_as_array(chan)

def common_reference(filename, channels):
    """ Calculates the common average refernce from the data stored in a file,
        from the chosen channels.
    """
    n = len(channels)
    data = load_data(filename, channels)
    return sum(data)/float(n)

def process_data(filename, channels, common_reference=True):
    """ This function processes the data from filename and returns an array
        of tetrode spike waveforms and their spike timestamps.
    """
    
    if common_reference:
        car = common_reference(filename, range(16))
    else:
        car = 0
    
    data = load_data(filename, channels)
    spikes = detect_spikes(data, common_reference=car)
    
    form_tetrode(load_data(filename, channels), spikes) 

def form_tetrode(data, spikes):
    """ Build tetrode waveforms from voltage data and detected spike times.
    """
    # Need to get spike times from every channel in the tetrode
    times = np.concatenate([ spks['times'] for spks in spikes])
    times = censor(times)
    return
    
@batch_process
def detect_spikes(data, threshold=4, patch_size=30, common_reference=None):
    """ Detect spikes in data.  Returns spike waveform patches and peak samples. 
    """
    if common_reference:
        data = data - common_reference
    filtered = butter_filter(data, low=300, high=6000, rate=30000) 
    threshold = medthresh(filtered, threshold)
    peaks = crossings(filtered, threshold, polarity='neg')
    peaks = censor(peaks, 30)
    
    spikes, times = extract(data, peaks, patch_size=patch_size)
    
    records = [('spikes', 'f8', patch_size), ('times', 'f8', 1)]
    detected = np.zeros(len(times), dtype=records)
    detected['spikes'] = spikes
    detected['times'] = times
    
    return detected

@batch_process
def medthresh(data, threshold=4):
    """ A function that calculates the spike crossing threshold 
        based off the median value of the data.
    
    Arguments
    ---------
    data : your data
    threshold : the threshold multiplier
    """
    return threshold*np.median(np.abs(data)/0.6745)

@batch_process
def butter_filter(data, low=300, high=6000, rate=30000):
    """ Uses a 3-pole Butterworth filter to reduce the noise of data.
    
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