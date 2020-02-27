#! /usr/bin/env/python
# coding: utf-8

"""
dproc.py
Copyright (C) 2020 antcc

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see
https://www.gnu.org/licenses/gpl-3.0.html.
"""

import numpy as np
import sys

L = 1        # Normalize to interval [-L, L]
DELIM = ' '  # Dimension delimiter in input file

def print_data(data):
    """Print data in the same format as the input data."""

    nrows, ncols = data.shape
    for i in range(nrows):
        for j in range(ncols):
            print(str(data[i, j]) + ("," if j != ncols - 1 else ""), end = "")
        print("")

def compute_range(data, ncols):
    """ Compute min and max values in every dimension."""

    min_cols = [data[:, j].min() for j in range(ncols)]
    max_cols = [data[:, j].max() for j in range(ncols)]

    return np.array([min_cols, max_cols])

def norm(data, nrows, ncols, range_cols):
    """Normalize data to [-L, L] from original values."""

    for j in range(ncols):
        min = range_cols[0, j]
        max = range_cols[1, j]

        if max == min:
            for i in range(nrows):
                data[i, j] = 0.0
        else:
            for i in range(nrows):
                data[i, j] = - L + ((data[i, j] - min) * (2 * L)) / (max - min)

def denorm(data, nrows, ncols, range_cols):
    """Denormalize data from [-L, L] to original values."""

    for j in range(ncols):
        min = range_cols[0, j]
        max = range_cols[1, j]

        for i in range(nrows):
    	    data[i, j] = min + ((data[i, j] + L) * (max - min)) / (2 * L)

def main():
    if (len(sys.argv) < 3):
        print("use: ./dproc.py [--norm, --denorm, --range] DATA [DATA_RANGE]")
        sys.exit(1)

    data = np.genfromtxt(sys.argv[2], delimiter = DELIM)
    nrows, ncols = data.shape

    if (sys.argv[1] == "--norm"):
        if (len(sys.argv) < 4):
            print("Data range file is needed to normalize. It can be obtained with --range.")
            sys.exit(1)

        range_cols = np.genfromtxt(sys.argv[3], delimiter=',')
        norm(data, nrows, ncols, range_cols)
        print_data(data)

    elif (sys.argv[1] == "--denorm"):
        if (len(sys.argv) < 4):
            print("Data range file is needed to denormalize. It can be obtained with --range.")
            sys.exit(1)

        range_cols = np.genfromtxt(sys.argv[3], delimiter=',')
        denorm(data, nrows, ncols, range_cols)
        print_data(data)

    elif (sys.argv[1] == "--range"):
        print_data(compute_range(data, ncols))

    else:
        print("use: ./dproc.py [--norm, --denorm, --range] DATA [DATA_RANGE]")

if __name__ == "__main__":
    main()
