
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from shapely.geometry import MultiLineString
from matplotlib.path import Path
import numpy as np
# from mayavi.mlab import *
import math

import matplotlib.animation as animation
from matplotlib import collections as mc
from scipy.spatial import Delaunay


def plot_delaunay_mesh(NodesUsedByLevel):

    for lev in NodesUsedByLevel:
        points = []
        for n in lev:
            points.append(n.point[:2])

    points = np.array(points)
    tri = Delaunay(points)
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()

def plot_bases(vo, v, l, indx, c="red"):
    def _update_plot(i, fig, scat, x, y):
        v_new = vo+ v*math.sin(math.sqrt(abs(l))*i)
        p = []
        for i in range(len(v_new)/2):
            p.append([v_new[2*i], v_new[2*i+1]])
        scat.set_offsets(p)
        return scat,

    fig =  plt.figure()

    x = v[0:][::2]
    y = v[1:][::2]

    ax = fig.add_subplot(111)
    ax.grid(True, linestyle = '-', color = '0.75')
    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 5])
    ax.add_collection(mc.LineCollection([]))

    scat = plt.scatter(x, y, c = x)
    scat.set_alpha(1)

    anim = animation.FuncAnimation(fig, _update_plot, fargs = (fig, scat, x, y),
                               frames = 200, interval = 100)

    anim.save(str(indx)+'Level3Basis.mp4', fps=30)

    plt.show()


def plot(X, Y, Z,c = "red"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf = ax.plot_wireframe(X, Y, Z, color =c)
    ax.scatter(X, Y, Z, color = "blue")

    # Customize the z axis.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a color bar which maps values to colors.
    plt.show()
