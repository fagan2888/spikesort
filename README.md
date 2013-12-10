Spikesort
=========

Spikesort is a Python package for [spike sorting](http://scholarpedia.org/article/Spike_sorting).  The code extracts spike waveforms from raw voltage recordings, then using a Gaussian Mixture Model, sorts the spikes into clusters of similar waveform shapes.  Code for creating a catalog to store cluster metadata is also included.

Dependencies and Python version
-------------------------------

The code requires at least numpy, scipy, scikits-learn, and matplotlib. It also requires sqlalchemy if you want to use the catalog module.  All the code was written with Python 2.7.3.  It would probably work for Python 3.x if converted using 2to3, but I haven't tested it.

I suggest using IPython to do the sorting, it works really well in an IPython Notebook with inline matplotlib figures.

Documentation
-------------

Documentation is located in the docs folder, with instructions to make the files in the format of your choosing.