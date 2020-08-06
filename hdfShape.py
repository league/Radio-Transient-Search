#!/usr/bin/env python3
import numpy
import h5py
import sys
from datetime import datetime
import argparse

args = None

parser = argparse.ArgumentParser(
    description = 'Display the hierarchy of keys and shapes in an HDF5 file'
)
parser.add_argument('files', nargs='*')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='More output, especially attribute values')

def pluralize(num, noun1, noun2):
    if num == 1:
        return '1 ' + noun1
    else:
        return str(num) + ' ' + noun2

def show_shape(name, arr, fill=''):
    extra = '; ' + pluralize(len(arr.attrs), 'attr', 'attrs')
    if name.endswith('time'):
        extra += ('; interval %0.3f ms; start %s' %
                  (numpy.median(arr[1:] - arr[:-1])*1e3,
                   datetime.utcfromtimestamp(arr[0])))
    elif name.endswith('freq'):
        extra += ('; interval %.3f kHz; median %.1f MHz' %
                  ((arr[1]-arr[0])/1e3,
                   numpy.median(arr)/1e6))
    print('%s%s: %s %s%s' % (fill, name, arr.shape, arr.dtype, extra))

def show_h5_recur(f, fill='  '):
    for k in f.keys():
        if hasattr(f[k], 'shape'):
            show_shape(k, f[k], fill)
        else:
            print('%s%s: %s' % (fill, k, pluralize(len(f[k].attrs), 'attr', 'attrs')))
            if args.verbose:
                for a in f[k].attrs:
                    print('%s  %s = %s' % (fill, a, f[k].attrs[a]))
            show_h5_recur(f[k], fill+'  ')

def show_h5(name):
    print("%s:" % name)
    with h5py.File(name, 'r') as f:
        show_h5_recur(f)
            
def show_npy(name):
    show_shape(name, numpy.load(name, mmap_mode='r'))

def dispatch_arg(f):
    if f.endswith(".npy"):
        show_npy(f)
    elif f.endswith(".hdf5"):
        show_h5(f)
    else:
        print("%s: ??" % f)
            
if __name__ == "__main__":
    args = parser.parse_args()
    for f in args.files:
        dispatch_arg(f)
