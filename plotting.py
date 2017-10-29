
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from shapely.geometry import MultiLineString
from matplotlib.path import Path
import numpy as np

def plot_path(tris, dom):
    fig, ax = plt.subplots()
    poly = plt.Polygon(tris, ec = "k")
    x,y = zip(*tris)

    ax.scatter(x,y, color="r", alpha = 0.6, zorder = 3, s = 10*10*10)

    plt.axis([dom[0][0], dom[1][0] -1, dom[0][1], dom[1][1] -1])

    major = np.arange(0, dom[1][0]-1, 1)
    ax.set_xticks(major)
    ax.set_yticks(major)
    ax.grid(which='major', alpha=1.0)
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


def plot_tri(tris):
    COLOR = {
        True:  '#6699cc',
        False: '#ffcc33'
        }

    def v_color(ob):
        return COLOR[ob.is_simple]
    def plot_coords(ax, ob):
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)

    def plot_bounds(ax, ob):
        x, y = zip(*list((p.x, p.y) for p in ob.boundary))
        ax.plot(x, y, 'o', color='#000000', zorder=1)

    def plot_lines(ax, ob):
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, color=v_color(ob), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

    fig = plt.figure()
    ax = fig.add_subplot(122)
    mline2 = MultiLineString(tris)

    plot_coords(ax, mline2)
    plot_bounds(ax, mline2)
    plot_lines(ax, mline2)

    plt.show()
