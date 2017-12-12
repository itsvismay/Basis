import dd as ref
import sim as sim
import utilities as utils
import numpy as np
import copy

from scipy.spatial import Delaunay
import sys, os
sys.path.insert(0, os.getcwd()+"/../libigl/python/")
import pyigl as igl


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

    def get_grid_displacements(self):
        Vx = np.zeros((len(self.x)/2, 2))
        self.X_to_V(Vx, self.x)
        u = [[0 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
        for i in range(len(Vx)):
            u[int(Vx[i][0])][int(Vx[i][1])] = np.linalg.norm(Vx[i] - self.V[i])

        return u




def get_mesh_from_displacement(actNodes_L):
    sortedFlatB_L = sorted([i for sublist in actNodes_L for i in sublist], key=lambda x:x.id)
    map_L = sim.create_active_nodes_index_map(sortedFlatB_L)
    dupSize_L = len(sortedFlatB_L)

    K_L = np.zeros((2*dupSize_L, 2*dupSize_L))
    M_L = np.zeros((2*dupSize_L, 2*dupSize_L))
    f_L = np.zeros(2*dupSize_L)
    x_L = np.zeros(2*dupSize_L)
    v_L = np.zeros(2*dupSize_L)

    sim.compute_stiffness(K_L, sortedFlatB_L, map_L)
    sim.compute_mass(M_L, sortedFlatB_L, map_L)
    sim.compute_gravity(f_L, M_L)
    sim.set_x_initially(x_L, sortedFlatB_L, map_L)

    mesh_L = Mesh(x_L, v_L, f_L, M_L, K_L)

    print("M is SPD ", utils.is_pos_def(mesh_L.M))

    sim.fix_left_end(mesh_L.V, mesh_L.invM)

    return mesh_L

def display_mesh(mesh):
    viewer = igl.viewer.Viewer()
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


def set_up_solver():

    hMesh = sim.get_hierarchical_mesh(dom)

    # FOR L3 MESH
    u_f_L = [[1 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    actNodes_L = sim.get_active_nodes([hMesh[2]], dom, u_f=u_f_L)
    mesh_L = get_mesh_from_displacement(actNodes_L)
    mesh_L.step()

    #FOR H MESH
    print("H Mesh")
    l1_e = sorted(list(hMesh[0].nodes), key=lambda x:x.id)
    n1 = l1_e[0]
    n2 = l1_e[1]
    n3 = l1_e[2]
    n4 = l1_e[3]
    u_f_H = [[n1.basis[x][y]+n2.basis[x][y]+n3.basis[x][y]+n4.basis[x][y] for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # u = mesh_L.get_grid_displacements()
    actNodes_H = sim.get_active_nodes(hMesh, dom, u_f=u_f_H)
    nonDuplicateSize, map_duplicate_nodes_to_ind, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes_H)
    print("duplicates ", map_duplicate_nodes_to_ind, map_points_to_bases)
    mesh_H = get_mesh_from_displacement(actNodes_H)

    display_mesh(mesh_H)





set_up_solver()
