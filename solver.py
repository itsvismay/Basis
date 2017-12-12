import dd as ref
import sim as sim
import utilities as utils
import numpy as np
import copy

from scipy.spatial import Delaunay

dom = ((0,0), (5, 5))

class Mesh:

    def __init__(self, x, v, f, M, K):
        self.x = x
        self.p = copy.copy(x)
        self.v = v
        self.f = f
        self.M = M
        self.invM = np.linalg.inv(M)
        self.K = K
        self.V = np.zeros((len(x)/2, 2))

        self.X_to_V(self.V, self.x)
        self.tri = Delaunay(self.V).simplices

    def X_to_V(self, V, x):
        for i in range(V.shape[0]):
            V[i, 0] = x[2*i]
            V[i, 1] = x[2*i+1]

    def step(self, h=1e-3):
        for i in range(100):
            self.p = self.p + h*self.v
            self.v = self.v + h*self.invM.dot(self.K.dot(self.x - self.p) + self.f)

        self.X_to_V(self.V, self.p)


def set_up_solver():

    hMesh = sim.get_hierarchical_mesh(dom)

    # FOR L3 MESH
    u_f_L = [[1 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]

    actNodes_L = sim.get_active_nodes([hMesh[2]], dom, u_f=u_f_L)

    sortedFlatB_L = sorted([i for sublist in actNodes_L for i in sublist], key=lambda x:x.id)
    map_L = sim.create_active_nodes_index_map(sortedFlatB_L)
    dupSize_L = len(sortedFlatB_L)

    K_L = np.zeros((2*dupSize_L, 2*dupSize_L))
    M_L = np.zeros((2*dupSize_L, 2*dupSize_L))
    f_L = np.zeros(2*dupSize_L)
    x_L = np.zeros(2*dupSize_L)
    v_L = np.zeros(2*dupSize_L)

    sim.compute_stiffness(K_L, sortedFlatB_L, [hMesh[2]], map_L)
    sim.compute_mass(M_L, sortedFlatB_L, map_L)
    sim.compute_gravity(f_L, M_L)
    sim.set_x_initially(x_L, sortedFlatB_L, map_L)

    mesh_L = Mesh(x_L, v_L, f_L, M_L, K_L)

    print("ML is SPD ", utils.is_pos_def(mesh_L.M))

    sim.fix_left_end(mesh_L.V, mesh_L.invM)

    mesh_L.step()
    

    #FOR H MESH
    # l1_e = sorted(list(hMesh[0].nodes), key=lambda x:x.id)
    # n1 = l1_e[0]
    # n2 = l1_e[1]
    # n3 = l1_e[2]
    # n4 = l1_e[3]
    # u_f_H = [[n1.basis[x][y]+n2.basis[x][y]+n3.basis[x][y]+n4.basis[x][y] for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # actNodes_H = sim.get_active_nodes(hMesh, dom, u_f_H)
    #
    # sortedFlatB_H = sorted([i for sublist in actNodes_H for i in sublist], key=lambda x:x.id)
    # map_H = sim.create_active_nodes_index_map(sortedFlatB_H)
    # dupSize_H = len(sortedFlatB_H)
    # print(dupSize_H)


set_up_solver()
