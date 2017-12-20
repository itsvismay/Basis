import dd as ref
import sim as sim
import utilities as utils
import global_variables as GV

import numpy as np
import copy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import scipy
import sys, os
sys.path.insert(0, os.getcwd()+"/../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.2f}'.format(x)})



dom = ((0,0), (5, 5))

class Mesh:

    def __init__(self, x, v, f, M, K, activeElems, sortedFlatB, map_nodes, nonDupSize, EmbeddingNodes):
        self.x = x
        self.p = copy.copy(x)
        self.v = v
        self.f = f
        self.M = M
        self.W = sim.get_weighting_matrix(nonDupSize, sortedFlatB, map_nodes)
        self.invM = np.linalg.inv(M)
        self.K = K
        self.V = np.zeros((len(x)/2, 2))
        self.activeElems = activeElems
        self.sortedFlatB = sortedFlatB
        self.map_nodes = map_nodes
        self.nonDupSize = nonDupSize

        self.EmbeddingNodes = EmbeddingNodes
        self.EmbeddedMesh, self.EmbeddedTri = self.create_embedded_mesh()


        self.X_to_V(self.V, self.x)

        self.tri = Delaunay(self.V).simplices


    def create_embedded_mesh(self):
        Emesh = np.zeros((len(self.EmbeddingNodes), 2))
        i = 0
        for a in sorted(self.EmbeddingNodes, key=lambda x:x.id):
            Emesh[i] = np.array([a.point[0], a.point[1]])
            i+=1

        return Emesh, Delaunay(Emesh).simplices

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
        # invMhhK = np.linalg.inv(self.M - h*h*self.K)
        P = sim.fix_left_end(self.V)

        for i in range (100):
            self.p = self.p + h*np.matmul(P, P.T).dot(self.v)
            forces = self.f + self.K.dot(self.x - self.p)
            self.v = self.v + h*P.dot(np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces)))
            # newv = np.copy(self.v)
            # func = lambda x: 0.5*np.dot(x.T, self.W.dot(x))
            # def constr(x):
            #     return x - (self.v + h*P.dot(np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces))))
            # cons = ({'type': 'eq', 'fun': constr })
            #

            # res = scipy.optimize.minimize(func, newv, method="SLSQP", constraints=cons)
            # print(res)
            # self.v = res.x

        self.X_to_V(self.V, self.p)

    def NMstep(self, h=1e-2):
        P = sim.fix_left_end(self.V)

        for its in range(10):
            p_g = np.copy(self.p)
            NewtonMax = 100
            for i in range(NewtonMax):
                forces = self.f + self.K.dot(self.x - p_g)
                g_block = p_g - self.p - h*(self.v + h*self.invM.dot(forces))
                grad_g_block = np.identity(2*self.nonDupSize) - h*h*np.matmul(self.invM, self.K)
                # g_block = P.T.dot(p_g) - P.T.dot(self.p) - h*P.T.dot(self.v) - h*h*np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces))
                # grad_g_block =  np.matmul(np.matmul(P.T, np.identity(2*(self.nonDupSize))), P) - h*h*np.matmul(np.matmul(np.matmul(P.T, self.invM), P), np.matmul(np.matmul(P.T, self.K), P))
                Q,R = np.linalg.qr(grad_g_block)
                Qg = Q.T.dot(-1*g_block)
                dp = np.linalg.solve(R, Qg)
                p_g += dp

                print("gblock norm")
                print(np.linalg.norm(g_block))
                print("")
                if np.linalg.norm(g_block)/len(g_block) < 1e-4:
                    print("solved in ", i)
                    break
                if i == 10:
                    print("Error: not converging")
                    exit()
            self.v = (p_g - self.p)/h
            self.p = np.copy(p_g)
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

    def get_embedded_mesh(self):
        update_disp = np.zeros((len(self.EmbeddingNodes), 2))
        i = 0
        for a in sorted(self.EmbeddingNodes, key=lambda x:x.id):
            u_a_x = 0
            u_a_y = 0
            for b in self.sortedFlatB:
                u_b_x = (self.p[2*self.map_nodes[b.id]] - self.x[2*self.map_nodes[b.id]])
                u_b_y = (self.p[2*self.map_nodes[b.id]+1] - self.x[2*self.map_nodes[b.id]+1])
                N_b_at_a = b.basis[a.point[0]][a.point[1]]

                u_a_x += N_b_at_a*u_b_x
                u_a_y += N_b_at_a*u_b_y

            update_disp[i,0] = u_a_x
            update_disp[i,1] = u_a_y
            i+=1

        return self.EmbeddedMesh + update_disp

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

def get_mesh_from_displacement(actNodes, EmbeddingNodes):
    sortedFlatB = sorted([i for sublist in actNodes for i in sublist], key=lambda x:x.id)
    map_nodes_old = sim.create_active_nodes_index_map(sortedFlatB)
    nonDuplicateSize, map_nodes, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes)

    print(nonDuplicateSize)
    print("duplicates accounted for")
    print(map_points_to_bases)
    print(map_nodes)


    M_L = np.zeros((2*nonDuplicateSize, 2*nonDuplicateSize))
    K_L = np.zeros((2*nonDuplicateSize, 2*nonDuplicateSize))
    f_L = np.zeros(2*nonDuplicateSize)
    x_L = np.zeros(2*nonDuplicateSize)
    v_L = np.zeros(2*nonDuplicateSize)

    sim.compute_mass(M_L, sortedFlatB, map_nodes)
    sim.compute_stiffness(K_L, sortedFlatB, map_nodes)
    sim.compute_gravity(f_L, M_L)
    sim.set_x_initially(x_L, sortedFlatB, map_nodes)

    # M_L += 1*np.identity(2*nonDuplicateSize)
    print("M is SPD ", utils.is_sym_pos_def(M_L))

    # exit()
    E = set()#set of active cells
    for n in sortedFlatB:
        E |= n.in_elements


    mesh = Mesh(x_L, v_L, f_L, M_L, K_L, E, sortedFlatB, map_nodes, nonDuplicateSize, EmbeddingNodes)


    return mesh

def display_mesh(mesh, Ek=None):
    viewer = igl.viewer.Viewer()
    time = 0
    # K_k = np.zeros((2*mesh.nonDupSize, 2*mesh.nonDupSize))
    # sim.compute_stiffness(K_k, mesh.sortedFlatB, mesh.map_nodes, Youngs=Ek)
    # mesh.reset(Knew=K_k)
    def key_down(viewer, key, modifier):
        mesh.step()
        Emesh = mesh.get_embedded_mesh()

        viewer.data.clear()
        V1 = igl.eigen.MatrixXd(Emesh)
        F1 = igl.eigen.MatrixXi(mesh.EmbeddedTri)
        viewer.data.set_mesh(V1, F1)
        # print(V1)
        # viewer.data.add_points(V1, igl.eigen.MatrixXd([[0,1,0]]))
        # for e in mesh.EmbeddedTri:
        #     viewer.data.add_edges(V1.row(e[0]), V1.row(e[1]),igl.eigen.MatrixXd([[1, 1, 1]]))
        #     viewer.data.add_edges(V1.row(e[1]), V1.row(e[2]),igl.eigen.MatrixXd([[1, 1, 1]]))
        #     viewer.data.add_edges(V1.row(e[0]), V1.row(e[2]),igl.eigen.MatrixXd([[1, 1, 1]]))

        return True

    key_down(viewer, ord('5'), 0)
    viewer.core.is_animating = True
    viewer.callback_key_down = key_down
    viewer.launch()

def solve(meshL, meshH):
    print("Youngs Solve")
    #initially youngs guess
    E_0 = np.empty(len(meshH.activeElems))
    E_0.fill(GV.Global_Youngs/1.0)
    bnds = ((0, None) for i in range(len(E_0)))
    meshL.step()
    meshL.step()
    uL = get_reference_points(meshL, meshH)
    def func(E_k):
        K_k = np.zeros((2*meshH.nonDupSize, 2*meshH.nonDupSize))
        sim.compute_stiffness(K_k, meshH.sortedFlatB, meshH.map_nodes, Youngs=E_k)
        meshH.reset(Knew=K_k)
        meshH.step()
        meshH.step()
        no = np.linalg.norm(uL - meshH.p)
        print(no)
        return no

    res = minimize(func, E_0, method='Nelder-Mead', bounds=bnds, options={"disp": True})
    print(res)
    return res.x


def set_up_solver():

    hMesh = sim.get_hierarchical_mesh(dom)

    # FOR L3 MESH
    print("L Mesh")
    u_f_L = [[1 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    actNodes_L = sim.get_active_nodes([hMesh[2]], dom, u_f=u_f_L)
    mesh_L = get_mesh_from_displacement(actNodes_L, [n for n in hMesh[2].nodes])

    # display_mesh(mesh_L)


    #FOR H MESH
    print("H Mesh")
    l1_e = sorted(list(hMesh[0].nodes), key=lambda x:x.id)
    n1 = l1_e[0]
    n2 = l1_e[1]
    n3 = l1_e[2]
    n4 = l1_e[3]
    u_f_H = [[n1.basis[x][y]+n2.basis[x][y]+n3.basis[x][y]+n4.basis[x][y] for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # u_f_H = [[x**2 + y**2 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # u_f_H = [[np.sqrt(x**2 + y**2) for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]

    actNodes_H = sim.get_active_nodes(hMesh, dom, u_f=u_f_H)
    nonDuplicateSize, map_duplicate_nodes_to_ind, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes_H)
    mesh_H = get_mesh_from_displacement(actNodes_H, [n for n in hMesh[2].nodes])
    display_mesh(mesh_H)


    # Ek = solve(mesh_L, mesh_H)
    # print("New Ek", Ek)
    # display_mesh(mesh_H, Ek)




set_up_solver()
