#!/usr/bin/env/python
# coding: utf-8

"""
plot.py
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
from matplotlib import pyplot as plt
import sys

DELIM = ','  # Dimension delimiter in input file

def plot2D(*args):
    """Plot the given 2D data, specified as a numpy array."""

    plt.figure(figsize = (8, 8))
    for data in args:
        plt.scatter(data[:, 0], data[:, 1], alpha = 0.5, marker = '.')
    plt.show()

def main():
    """Main function."""

    if (len(sys.argv) < 2):
        print("use: ./plot.py FILE")
        sys.exit(1)

    data_lst = [np.genfromtxt(arg, delimiter = DELIM) for arg in sys.argv[1:]]
    plot2D(*data_lst)

if __name__ == "__main__":
    main()
