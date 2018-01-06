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

def time_warp(meshF, meshC, E_0=None, K_0=None, seconds=1, timestep=1e-3):
    time_in_secs = 0
    E_i = solver.solve(meshF, meshC, E_0=E_0)
    some_epsilon = 2
    Vf = igl.eigen.MatrixXd()
    Ff = igl.eigen.MatrixXi()
    for i in range(int(seconds/timestep)):
        meshC.step()
        time_in_secs= timestep*i
        if(i%100):
            #Check meshes and re-solve
            name = "checkmeshes/mesh@"+str(GV.Global_Youngs)+"E@"+str(time_in_secs)+"s.obj"
            igl.readOBJ(name, Vf, Ff)
            n = np.linalg.norm(meshC.get_embedded_mesh() - np.delete(Vf, -1, 1))
            print("n", n, "DOF", meshC.nonDupSize, "elem", len(meshC.activeElems))
            exit()
            if(n>some_epsilon):
                E_i = solver.solve(meshF, meshC, E_0=E_i)

def run_fine_mesh(mesh, Ek=None, seconds=1, timestep=1e-3):
    time_in_secs = 0
    for i in range(int(seconds/timestep)):
        mesh.step()
        time_in_secs = timestep*i
        if(i%100):
            name = "checkmeshes/mesh@"+str(GV.Global_Youngs)+"E@"+str(time_in_secs)+"s.obj"
            utils.serialize_mesh(name, mesh.get_embedded_mesh(), mesh.EmbeddedTri)


def setup_timewarp():
    seconds = 1
    timestep = 1e-3

    meshF, meshC = solver.set_up_solver()
    E_0 = np.empty(len(meshC.activeElems))
    E_0.fill(GV.Global_Youngs)
    K_0 = np.zeros((2*meshC.nonDupSize, 2*meshC.nonDupSize))

    # run_fine_mesh(meshF, seconds=seconds, timestep=timestep)
    time_warp(meshF, meshC, E_0=E_0, K_0=K_0, seconds=seconds, timestep=timestep)



setup_timewarp()
