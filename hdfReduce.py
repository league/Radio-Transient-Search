#!/usr/bin/env python3
import argparse
import os.path
import sys
import h5py
import numpy
from skimage.measure import block_reduce

DEFAULT_TIME_FACTOR = 32
DEFAULT_FREQ_FACTOR = 4
DEFAULT_OUTPUT_SUFFIX = '-reduced'

outfile = None
args = None

parser = argparse.ArgumentParser(
    description = 'Reduce the number of time and/or frequency bins in an HDF5 waterfall image'
)
parser.add_argument('filename', type=str, help='File to process')
parser.add_argument('-t', '--time', metavar='FACTOR', type=int, default=DEFAULT_TIME_FACTOR,
                    help='Reduction factor for time bins (default %d)' % DEFAULT_TIME_FACTOR)
parser.add_argument('-f', '--frequency', metavar='FACTOR', type=int, default=DEFAULT_FREQ_FACTOR,
                    help='Reduction factor for frequency bins (default %d)' % DEFAULT_FREQ_FACTOR)
parser.add_argument('--force', action='store_true',
                    help='Overwrite output file if it already exists')
parser.add_argument('-o', '--output', metavar='FILENAME',
                    help='Output filename (default adds "%s")' % DEFAULT_OUTPUT_SUFFIX)

def reduce_dataset(k, obj):
    if isinstance(obj, h5py.Dataset):
        if k.endswith('time'):
            block_size = (args.time,)
        elif k.endswith('freq'):
            block_size = (args.frequency,)
        elif k.endswith('Saturation'):
            block_size = (args.time, 1)
        elif k.endswith('XX') or k.endswith('YY'):
            block_size = (args.time, args.frequency)
        else:
            print("%s: unanticipated object; preserving %s" % (k, obj.shape))
            out = outfile.create_dataset(k, obj.shape, dtype=obj.dtype, compression='gzip')
            out[...] = obj[...]
            return

        print("%s: %s..." % (k, obj.shape))
        r = block_reduce(obj, block_size=block_size, func=numpy.mean)
        print("  -> %s" % (r.shape,))
        out = outfile.create_dataset(k, r.shape, dtype=obj.dtype, compression='gzip')
        out[...] = r[...]
        
def main():
    if args.output is None:
        (base, ext) = os.path.splitext(args.filename)
        args.output = base + DEFAULT_OUTPUT_SUFFIX + ext
    if os.path.exists(args.output) and not args.force:
        print("Error: %s: already exists" % args.output)
        sys.exit(1)
    with h5py.File(args.filename, 'r') as infile:
        global outfile
        try:
            outfile = h5py.File(args.output, 'w')
            infile.visititems(reduce_dataset)
        finally:
            outfile.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main()
