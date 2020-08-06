#!/usr/bin/env python3
import numpy
import h5py
import sys
from datetime import datetime

def show_shape(name, arr, fill=''):
    if name.endswith('time'):
        extra = ("; interval %0.3f ms; start %s" %
                 (numpy.median(arr[1:] - arr[:-1])*1e3,
                  datetime.utcfromtimestamp(arr[0])))
    elif name.endswith('freq'):
        extra = ("; interval %.3f kHz; median %.1f MHz" %
                 ((arr[1]-arr[0])/1e3,
                  numpy.median(arr)/1e6))
    else:
        extra = ""
    print("%s%s: %s %s%s" % (fill, name, arr.shape, arr.dtype, extra))

def show_h5_recur(f, fill='  '):
    for k in f.keys():
        if hasattr(f[k], 'shape'):
            show_shape(k, f[k], fill)
        else:
            print("%s%s:" % (fill, k))
            show_h5_recur(f[k], fill+'  ')

def show_h5(name):
    print("%s:" % name)
    with h5py.File(name, 'r') as f:
        show_h5_recur(f)
            
def show_npy(name):
    show_shape(name, numpy.load(name, mmap_mode='r'))

def dispatch_arg(arg):
    if arg.endswith(".npy"):
        show_npy(arg)
    elif arg.endswith(".hdf5"):
        show_h5(arg)
    else:
        print("%s: ??" % arg)
            
if __name__ == "__main__":
    for arg in sys.argv[1:]:
        dispatch_arg(arg)
