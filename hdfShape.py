#!/usr/bin/env python3
import numpy
import h5py
import sys
from datetime import datetime
import argparse

verbose = False

parser = argparse.ArgumentParser(
    description="Display the hierarchy of keys and shapes in HDF5 or Numpy files"
)
parser.add_argument("files", nargs="*")
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="More output, especially attribute values",
)


def pluralize(num, noun1, noun2):
    if num == 1:
        return "1 " + noun1
    else:
        return str(num) + " " + noun2


def describe_attrs(obj):
    return pluralize(len(obj.attrs), "attr", "attrs")


def describe_shape(name, arr, fill=""):
    extra = ""
    if hasattr(arr, "attrs"):
        extra += "; " + describe_attrs(arr)
    if name.endswith("time"):
        interval_ms = numpy.median(arr[1:] - arr[:-1]) * 1e3
        extra += "; interval %0.2f ms; start %s; duration %.02f min" % (
            interval_ms,
            datetime.utcfromtimestamp(arr[0]),
            (interval_ms * arr.shape[0]) / 60e3,
        )
    elif name.endswith("freq"):
        extra += "; interval %.3f kHz; median %.1f MHz" % (
            (arr[1] - arr[0]) / 1e3,
            numpy.median(arr) / 1e6,
        )
    print("%s%s: %s %s%s" % (fill, name, arr.shape, arr.dtype, extra))


def describe_hdf5_recursively(f, fill="  "):
    for k in f.keys():
        if hasattr(f[k], "shape"):
            describe_shape(k, f[k], fill)
        else:
            print("%s%s: %s" % (fill, k, describe_attrs(f[k])))
            if verbose:
                for a in f[k].attrs:
                    print("%s  %s = %s" % (fill, a, f[k].attrs[a]))
            describe_hdf5_recursively(f[k], fill + "  ")


def describe_hdf5_file(name):
    print("%s:" % name)
    with h5py.File(name, "r") as f:
        describe_hdf5_recursively(f)


def describe_npy_file(name):
    describe_shape(name, numpy.load(name, mmap_mode="r"))


def describe_files(files):
    for f in files:
        if f.endswith(".npy"):
            describe_npy_file(f)
        elif f.endswith(".hdf5"):
            describe_hdf5_file(f)
        else:
            print("%s: ??" % f)


if __name__ == "__main__":
    args = parser.parse_args()
    verbose = args.verbose
    describe_files(args.files)
