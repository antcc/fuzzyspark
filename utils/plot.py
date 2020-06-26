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
    plt.xlabel("x")
    plt.ylabel("y")
    x=np.linspace(-1, 1, 100)
    y=np.sin(x)/x
    plt.plot(x, y, linestyle = "--", label = "sin(x)/x")
    colors = ['r', 'lime']
    labels = ["Chiu", "WM"]
    for i, data in enumerate(args):
        plt.plot(data[:, 0], data[:, 1], c = colors[i], label = labels[i])
    
    plt.legend()
    plt.show()

def main():
    """Main function."""

    if (len(sys.argv) < 2):
        print("use: ./plot.py FILE")
        sys.exit(1)

    data_lst = [np.genfromtxt(arg, delimiter = DELIM) for arg in sys.argv[1:]]
    indices_lst = [np.argsort(data[:, 0]) for data in data_lst]
    x1 = data_lst[0][:,0]
    x1= x1[indices_lst[0]]
    y1 = data_lst[0][:,1]
    y1 = y1[indices_lst[0]]
    a = np.array([[x, y] for x,y in zip(x1, y1)])

    x2 = data_lst[1][:,0]
    x2= x2[indices_lst[1]]
    y2 = data_lst[1][:,1]
    y2 = y2[indices_lst[1]]
    b = np.array([[x, y] for x,y in zip(x2, y2)])

    plot2D(a, b)

if __name__ == "__main__":
    main()
