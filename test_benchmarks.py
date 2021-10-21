from benchmarks.utils import loadFunction
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
	Little tool to visualize 2D benchmark functions
	example usage:
	>python test_benchmarks.py katsuura -x -10 10 -y -10 10 -res 200 
"""

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs='?', type=str, default="discus")
parser.add_argument("-x", nargs=2, type=int, default=[-100, 100], help="range on the x axis")
parser.add_argument("-y", nargs=2, type=int, default=[-100, 100], help="range on the y axis")
parser.add_argument("-res", dest="resolution", type=int, default=100, help="resolution of the plot. high resolution: ~200, medium: ~100")
args = parser.parse_args()

function=loadFunction(args.name)

fig = plt.figure(args.name)
ax = plt.axes(projection='3d')

x = np.linspace(args.x[0], args.x[1], args.resolution)
y = np.linspace(args.y[0], args.y[1], args.resolution)

X, Y = np.meshgrid(x, y)
Z = [[function([i, j]) for j in y] for i in x]

ax.scatter3D(X, Y, Z, s=1, cmap="summer", c=Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()


