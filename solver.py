import dd as ref
import sim as sim
import utilities as utils
import global_variables as GV

import numpy as np
import copy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import sys, os
sys.path.insert(0, os.getcwd()+"/../libigl/python/")
import pyigl as igl


dom = ((0,0), (5, 5))

class Mesh:

    def __init__(self, x, v, f, M, K, activeElems, sortedFlatB, map_nodes):
        self.x = x
        self.p = copy.copy(x)
        self.v = v
        self.f = f
        self.M = M
        self.invM = np.linalg.inv(M)
        self.K = K
        self.V = np.zeros((len(x)/2, 2))
        self.activeElems = activeElems
        self.sortedFlatB = sortedFlatB
        self.map_nodes = map_nodes

        self.X_to_V(self.V, self.x)
        self.tri = Delaunay(self.V).simplices

    def reset(self, Knew=None):
        if(Knew is not None):
            self.K = Knew
            self.p = copy.copy(self.x)
            self.v = np.zeros(len(self.p))
            self.X_to_V(self.V, self.p)
        else:
            self.p = copy.copy(self.x)
            self.v = np.zeros(len(self.p))
            self.X_to_V(self.V, self.p)

    def X_to_V(self, V, x):
        for i in range(V.shape[0]):
            V[i, 0] = x[2*i]
            V[i, 1] = x[2*i+1]

    def step(self, h=1e-3):
        for i in range(1):
            print("p", self.p, len(self.p))
            print("x", self.x, len(self.x))
            print("v", self.v, len(self.v))
            print("invM", self.invM.shape)
            print("K", self.K.shape)
            self.p = self.p + h*self.v
            self.v = self.v + h*self.invM.dot(self.K.dot(self.x - self.p) + self.f)
            # print("p", self.x)
            # print("v", self.v)
        self.X_to_V(self.V, self.p)

    def get_grid_displacement_norms(self):
        Vx = np.zeros((len(self.x)/2, 2))
        self.X_to_V(Vx, self.x)
        u = [[0 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
        for i in range(len(Vx)):
            u[int(Vx[i][0])][int(Vx[i][1])] = np.linalg.norm(Vx[i] - self.V[i])

        return u

    def get_grid_displacement(self):
        Vx = np.zeros((len(self.x)/2, 2))
        self.X_to_V(Vx, self.x)
        d = [[0 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
        for i in range(len(Vx)):
            d[int(Vx[i][0])][int(Vx[i][1])] = self.V[i]

        return d

def get_reference_points(meshRef, meshH):
    disp_grid = meshRef.get_grid_displacement()

    u = []

    Vh = np.zeros((len(meshH.x)/2, 2))
    meshH.X_to_V(Vh, meshH.x)
    for i in range(len(Vh)):
        p = disp_grid[int(Vh[i][0])][int(Vh[i][1])]
        u.append(p[0])
        u.append(p[1])

    return u

def get_mesh_from_displacement(actNodes):
    sortedFlatB = sorted([i for sublist in actNodes for i in sublist], key=lambda x:x.id)
    map_nodes_old = sim.create_active_nodes_index_map(sortedFlatB)
    nonDuplicateSize, map_nodes, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes)

    print(nonDuplicateSize)
    print("duplicates accounted for")
    print(map_points_to_bases)
    print(map_nodes)
    # print("map nodes old")
    # print(mesh_H.map_nodes)

    M_L = np.zeros((2*nonDuplicateSize, 2*nonDuplicateSize))
    K_L = np.zeros((2*nonDuplicateSize, 2*nonDuplicateSize))
    f_L = np.zeros(2*nonDuplicateSize)
    x_L = np.zeros(2*nonDuplicateSize)
    v_L = np.zeros(2*nonDuplicateSize)

    sim.compute_mass(M_L, sortedFlatB, map_nodes)
    sim.compute_stiffness(K_L, sortedFlatB, map_nodes)
    sim.compute_gravity(f_L, M_L)
    sim.set_x_initially(x_L, sortedFlatB, map_nodes)

    M_L += 1e-6*np.identity(2*nonDuplicateSize)
    print("M is PD ", utils.is_pos_def(M_L))

    E = set()#set of active cells
    for n in sortedFlatB:
        E |= n.in_elements

    mesh = Mesh(x_L, v_L, f_L, M_L, K_L, E, sortedFlatB, map_nodes)


    sim.fix_left_end(mesh.V, mesh.invM)

    return mesh

def display_mesh(mesh, Ek=None):
    viewer = igl.viewer.Viewer()
    time = 0
    K_k = np.zeros((2*len(mesh.sortedFlatB), 2*len(mesh.sortedFlatB)))
    sim.compute_stiffness(K_k, mesh.sortedFlatB, mesh.map_nodes, Youngs=Ek)
    mesh.reset(Knew=K_k)
    def key_down(viewer, key, modifier):
        mesh.step()
        viewer.data.clear()

        V1 = igl.eigen.MatrixXd(mesh.V)
        # print(V1)
        viewer.data.add_points(V1, igl.eigen.MatrixXd([[0,0,0]]))
        for e in mesh.tri:
            viewer.data.add_edges(V1.row(e[0]), V1.row(e[1]),igl.eigen.MatrixXd([[1, 1, 1]]))
            viewer.data.add_edges(V1.row(e[1]), V1.row(e[2]),igl.eigen.MatrixXd([[1, 1, 1]]))
            viewer.data.add_edges(V1.row(e[0]), V1.row(e[2]),igl.eigen.MatrixXd([[1, 1, 1]]))

        return True

    key_down(viewer, ord('5'), 0)
    viewer.core.is_animating = True
    viewer.callback_key_down = key_down
    viewer.launch()

def solve(meshL, meshH):
    print("Youngs Solve")
    #initially youngs guess
    E_0 = np.empty(len(meshH.activeElems))
    E_0.fill(GV.Global_Youngs)
    bnds = ((0, None) for i in range(len(E_0)))
    meshL.step()
    meshL.step()
    uL = get_reference_points(meshL, meshH)
    def func(E_k):
        K_k = np.zeros((2*len(meshH.sortedFlatB), 2*len(meshH.sortedFlatB)))
        sim.compute_stiffness(K_k, meshH.sortedFlatB, meshH.map_nodes, Youngs=E_k)
        meshH.reset(Knew=K_k)
        meshH.step()
        meshH.step()
        no = np.linalg.norm(uL - meshH.p)
        print(no)
        return no

    res = minimize(func, E_0, method='Nelder-Mead', bounds=bnds, tol=0.01, options={"disp": True})
    print(res)
    return res.x


def set_up_solver():

    hMesh = sim.get_hierarchical_mesh(dom)

    # FOR L3 MESH
    # print("L Mesh")
    # u_f_L = [[1 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # actNodes_L = sim.get_active_nodes([hMesh[2]], dom, u_f=u_f_L)
    # mesh_L = get_mesh_from_displacement(actNodes_L)
    # display_mesh(mesh_L)


    #FOR H MESH
    print("H Mesh")
    l1_e = sorted(list(hMesh[0].nodes), key=lambda x:x.id)
    n1 = l1_e[0]
    n2 = l1_e[1]
    n3 = l1_e[2]
    n4 = l1_e[3]
    # u_f_H = [[n1.basis[x][y]+n2.basis[x][y]+n3.basis[x][y]+n4.basis[x][y] for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    u_f_H = [[x+y**2 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]

    actNodes_H = sim.get_active_nodes(hMesh, dom, u_f=u_f_H)
    nonDuplicateSize, map_duplicate_nodes_to_ind, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes_H)
    mesh_H = get_mesh_from_displacement(actNodes_H)

    display_mesh(mesh_H)
    #
    # Ek = solve(mesh_L, mesh_H)
    # print("New Ek", Ek)
    # display_mesh(mesh_H, Ek)




set_up_solver()
