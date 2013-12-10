Process
*******

.. note::

    This code has been written to load and analyze data from ns5 binary files.  However, the functions as a whole accept data as numpy arrays.  So, as long as you can get your data into numpy arrays, it doesn't matter what the initial data format is.

Typical usage
-------------
You will probably want to do this in parallel because you end up doing things like filtering four data channels at once.  Each channel voltage signal is a bunch of data, so this ends up taking a lot of time, but since the filtering is done with scipy, there really isn't anyway to speed this up. It's slow due only to the amount of data.  The solution is to do this stuff in parallel.

There are multiple options here.  I prefer doing this in an IPython notebook to take advantage of the easy parallel processing.  Typically, you'll start up four engines (hopefully your CPU has four or more cores), then: ::

    from spikesort import process
    from IPython.parallel import Client
    c = Client()

If your data is noisy, a method to improve the signal to noise ratio is subtracting the common average reference (CAR) from the data, before filtering and extracting spikes. ::

    data = process.load_ns5(filename, channels=[16,17,18,20])
    all_channels = process.load_ns5(datafile, channels=range(16,33))
    car = process.common_ref(all_channels, n=16)
    data_minus_car = ( d - car for d in data )
    filtered = c[:].map(process.filter, data_minus_car)
    extracted = c[:].map(process.detect_spikes, filtered.get())
    spikes = extracted.get()

If you want to skip that step, ::
    
    data = process.load_ns5(filename, channels=[16,17,18,20])
    filtered = c[:].map(process.filter, data)
    extracted = c[:].map(process.detect_spikes, filtered.get())
    spikes = extracted.get()

Alternatively, you can use ``process.map`` which uses Python's multiprocessing module, although this doesn't work in an IPython notebook.

Then use the extracted spikes to get the tetrode waveforms. ::

    timestamps = np.concatenate([ s['times'] for s in spikes ])
    times = process.censor(timestamps)
    tetrodes = process.form_tetrode(filtered, times)

Functions
---------

.. automodule:: process
    :members: