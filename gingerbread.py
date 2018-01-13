import dd as ref
import sim as sim
import solver as solver
import utilities as utils
import global_variables as GV

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import scipy
import sys, os
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.2f}'.format(x)})


def load_obj(name = "woody.obj"):
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()

    igl.readOBJ(name, V, F);
    return V, F

def create_gingerbread_layers(fineLevel):
    layers = []
    V, F = load_obj()
    l1 = ref.Level(d = None, V=V, F=F)
    layers.append(l1)
    for i in range(fineLevel):
        layers.append(layers[i].split())
    return layers


def display_mesh(meshH, meshL=None):
    viewer = igl.viewer.Viewer()

    def key_down(viewer):
        meshH.step(h=1e-3)

        Emesh = meshH.get_embedded_mesh()
        viewer.data.clear()
        V1 = igl.eigen.MatrixXd(Emesh)
        F1 = igl.eigen.MatrixXi(meshH.EmbeddedTri)
        viewer.data.set_mesh(V1, F1)
        meshH.simSteps +=1
        return True

    key_down(viewer)
    viewer.core.is_animating = True
    viewer.callback_post_draw = key_down
    viewer.launch()


def setup_gingerbread(fineLevel = 0):
    hMesh = create_gingerbread_layers(fineLevel)
    u_f_L = [1 for n in hMesh[fineLevel].nodes]
    actNodes_L = sim.get_active_nodes([hMesh[fineLevel]], u_f=u_f_L)
    mesh = solver.get_mesh_from_displacement(actNodes_L, [n for n in hMesh[fineLevel].nodes])
    utils.serialize_mesh("testlayerserialize.pkl", mesh)
    # solver.display_mesh(mesh)

setup_gingerbread()
