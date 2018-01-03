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
        self.Nc = self.create_Nc_matrix()

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

    def NM_static(self):
        # print(self.K)
        P = sim.fix_left_end(self.V)
        p_g = np.copy(self.p)
        f_ext = np.zeros(len(self.f))
        sim.compute_gravity(f_ext, self.M, self.sortedFlatB, self.map_nodes, axis=0, mult=-10)
        NewtonMax = 100
        for i in range(NewtonMax):
            forces = f_ext + self.K.dot(p_g - self.x)

            g_block = P.T.dot(forces)
            grad_g_block = np.matmul(np.matmul(P.T, self.K), P)

            Q,R = np.linalg.qr(grad_g_block)
            Qg = Q.T.dot(-1*g_block)
            dp = np.linalg.solve(R, Qg)
            # print(dp)
            p_g += P.dot(dp)

            # print("gblock norm")
            # print(np.linalg.norm(g_block))
            # print("")
            if (np.linalg.norm(g_block)/len(g_block)) < 1e-2:
                # print("solved in ", i)
                break
            if i == 10:
                print("Error: not converging")
                exit()
        self.p = copy.copy(p_g)
        self.X_to_V(self.V, self.p)

    def step(self, h=1e-2):
        # invMhhK = np.linalg.inv(self.M - h*h*self.K)
        P = sim.fix_left_end(self.V)
        print("Mass")
        print(self.M)
        for i in range (1):
            self.p = self.p + h*np.matmul(P, P.T).dot(self.v)
            forces = self.f + self.K.dot(self.p - self.x)
            self.v = self.v + h*P.dot(np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces)))
            print("f", forces)
            print("p", self.p)
            print(self.K.dot(self.x - self.p))
            # newv = np.copy(self.v)
            # func = lambda x: 0.5*np.dot(x.T, self.W.dot(x))
            # def constr(x):
            #     return x - (self.v + h*P.dot(np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces))))
            # cons = ({'type': 'eq', 'fun': constr })
            #
            # res = scipy.optimize.minimize(func, newv, method="SLSQP", constraints=cons)
            # self.v = copy.copy(res.x)



        self.X_to_V(self.V, self.p)

    def new_verlet_step(self, h=1e-2):
        P = sim.fix_left_end(self.V)

        p_g = self.p + h*np.matmul(P, P.T).dot(self.v)
        forces = self.f + self.K.dot(p_g - self.x)

        newv = np.copy(self.v)
        func = lambda x: 0.5*np.dot(x.T, self.W.dot(x))
        def constr(x):
            return x - (self.v + h*P.dot(np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces))))
        cons = ({'type': 'eq', 'fun': constr })

        res = scipy.optimize.minimize(func, newv, method="SLSQP", constraints=cons)
        return p_g, res.x

    def NMstep(self, h=1e-2):
        P = sim.fix_left_end(self.V)
        p_g = np.copy(self.p)

        for its in range(1):
            NewtonMax = 100
            for i in range(NewtonMax):
                forces = self.f + self.K.dot(p_g - self.x)

                # f_block = forces
                # f_grad_block = np.matmul(np.matmul(P.T, self.K), P)
                g_block = P.T.dot(p_g) - P.T.dot(self.p) - h*P.T.dot(self.v) - h*h*np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces))
                grad_g_block =  np.matmul(np.matmul(P.T, np.identity(2*(self.nonDupSize))), P) - h*h*np.matmul(np.matmul(np.matmul(P.T, self.invM), P), np.matmul(np.matmul(P.T, self.K), P))

                Q,R = np.linalg.qr(grad_g_block)
                Qg = Q.T.dot(g_block)
                dp = -1*np.linalg.solve(R, Qg)
                p_g += P.dot(dp)

                print("gblock norm")
                print(np.linalg.norm(g_block))
                print("")
                if (np.linalg.norm(g_block)/len(g_block)) < 1e-2:
                    print("solved in ", i)
                    break
                if i == 10:
                    print("Error: not converging")
                    exit()

            self.v = (p_g - self.p)/h
            self.p = copy.copy(p_g)
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

    def create_Nc_matrix(self):
        Nc = np.zeros((2*len(self.EmbeddingNodes), 2*self.nonDupSize)) # embedded x hierarch
        print("NC shape")
        print(Nc.shape)
        i = 0
        for a in sorted(self.EmbeddingNodes, key=lambda x:x.id):
            n_b_at_a = 0
            for b in self.sortedFlatB:
                n_b_at_a = b.basis[a.point[0]][a.point[1]]
                Nc[2*i, 2*self.map_nodes[b.id]] += n_b_at_a
                Nc[2*i+1, 2*self.map_nodes[b.id]+1] += n_b_at_a
            i+=1

        return Nc

    def get_embedded_mesh(self):
        update_disp = np.zeros((len(self.EmbeddingNodes), 2))

        self.X_to_V(update_disp, self.Nc.dot(self.p - self.x))

        return self.EmbeddedMesh + update_disp

def get_reference_points(meshRef, meshH):
    EmbRef = meshRef.get_embedded_mesh()
    EmbH = meshH.get_embedded_mesh()

    n = np.linalg.norm(EmbRef - EmbH)
    return n

def get_mesh_from_displacement(actNodes, EmbeddingNodes):
    sortedFlatB = sorted([i for sublist in actNodes for i in sublist], key=lambda x:x.id)
    map_nodes_old = sim.create_active_nodes_index_map(sortedFlatB)
    nonDuplicateSize, map_nodes, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes)

    print("non duplicate size")
    print(nonDuplicateSize)

    M_L = np.zeros((2*nonDuplicateSize, 2*nonDuplicateSize))
    K_L = np.zeros((2*nonDuplicateSize, 2*nonDuplicateSize))
    f_L = np.zeros(2*nonDuplicateSize)
    x_L = np.zeros(2*nonDuplicateSize)
    v_L = np.zeros(2*nonDuplicateSize)

    sim.compute_mass(M_L, sortedFlatB, map_nodes)
    sim.compute_stiffness(K_L, sortedFlatB, map_nodes)
    sim.compute_gravity(f_L, M_L, sortedFlatB, map_nodes, axis=1)
    sim.set_x_initially(x_L, sortedFlatB, map_nodes)

    # exit()
    # M_L += 1*np.identity(2*nonDuplicateSize)
    print("M is SPD ", utils.is_sym_pos_def(M_L))

    # exit()
    print("Active Elements")
    E = set()#set of active cells
    for n in sortedFlatB:
        E |= n.in_elements
        # print(n)

    print(E)
    mesh = Mesh(x_L, v_L, f_L, M_L, K_L, E, sortedFlatB, map_nodes, nonDuplicateSize, EmbeddingNodes)


    return mesh

def display_mesh(mesh, Ek=None):
    viewer = igl.viewer.Viewer()
    time = 0

    K_k = np.zeros((2*mesh.nonDupSize, 2*mesh.nonDupSize))
    sim.compute_stiffness(K_k, mesh.sortedFlatB, mesh.map_nodes, Youngs=Ek)
    mesh.reset(Knew=K_k)
    # mesh.NM_static()
    def key_down(viewer, key, modifier):
        mesh.step()
        # mesh.NMstep()
        # print(mesh.p)
        # print(mesh.v)
        Emesh = mesh.get_embedded_mesh()

        viewer.data.clear()
        V1 = igl.eigen.MatrixXd(Emesh)
        F1 = igl.eigen.MatrixXi(mesh.EmbeddedTri)
        viewer.data.set_mesh(V1, F1)
        return True

    key_down(viewer, ord('5'), 0)
    viewer.core.is_animating = True
    viewer.callback_key_down = key_down
    viewer.launch()



def solve(meshL, meshH, K_E=None, E_0=None):
    print("Youngs Solve")

    bnds = ((0, None) for i in range(len(E_0)))
    timestep = 1e-2

    print("Ek")
    print(E_0)
    print(meshH.activeElems)
    def func(E_k):
        #v_squiggle(E_squiggle, F_squiggle)
        print("why is this running")
        print(E_k)
        sim.compute_stiffness(K_E, meshH.sortedFlatB, meshH.map_nodes, Youngs=E_k)
        meshH.reset(Knew=K_E)
        p_squiggle, v_squiggle = meshH.new_verlet_step(h=timestep)

        #Term 1: 0.5*v~^T* N^T*M*N *v~
        t1 = 0.5*np.dot(v_squiggle,np.matmul(np.matmul(meshH.Nc.T, meshL.M), meshH.Nc).dot(v_squiggle))

        #Term 2: v~^T * N^T*M*N * v~old
        t2 = np.dot(v_squiggle,np.matmul(np.matmul(meshH.Nc.T, meshL.M), meshH.Nc).dot(meshH.v))

        #Term 3: 0.5 u~^T * N^T*K*N * u~ #Energy
        u_squiggle = p_squiggle - meshH.x
        t3 = 0.5*np.dot(u_squiggle, np.matmul(np.matmul(meshH.Nc.T, meshL.K), meshH.Nc).dot(u_squiggle))

        #Term 4: -h* N^T*Fext*v~
        t4 = -1*timestep*np.dot(meshH.Nc.T.dot(meshL.f), v_squiggle)

        no = t1+t2+t3+t4
        print("     ",no, t1, t2, t3, t4)
        return 0

    # res = minimize(func, E_0, method='Nelder-Mead', bounds=bnds, options={"disp": True, "fatol":1e-2})
    # sim.compute_stiffness(K_k, meshH.sortedFlatB, meshH.map_nodes, Youngs=res.x)
    # meshH.reset(Knew = K_k)
    print("hi")
    return 0
    # return res.x

def new_display_mesh(meshL, meshH, Kk=None, Ek=None):
    viewer = igl.viewer.Viewer()
    time = 0

    def key_down(viewer, key, modifier):
        solve(meshL, meshH,K_E = Kk, E_0=Ek)
        meshH.step()
        print(meshH.p)
        print(meshH.v)
        Emesh = meshH.get_embedded_mesh()
        viewer.data.clear()
        V1 = igl.eigen.MatrixXd(Emesh)
        F1 = igl.eigen.MatrixXi(meshH.EmbeddedTri)
        viewer.data.set_mesh(V1, F1)
        return True

    key_down(viewer, ord('5'), 0)
    viewer.core.is_animating = True
    viewer.callback_key_down = key_down
    viewer.launch()

def set_up_solver():

    # hMesh = sim.get_hierarchical_mesh(dom)
    #
    # # FOR L3 MESH
    # print("L Mesh")
    # u_f_L = [[1 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # actNodes_L = sim.get_active_nodes([hMesh[2]], dom, u_f=u_f_L)
    # mesh_L = get_mesh_from_displacement(actNodes_L, EmbeddingNodes=[n for n in hMesh[2].nodes])
    # eigvals, eigvecs = utils.general_eig_solve(mesh_L.K, mesh_L.M)
    # ## tot = 0
    # ## for i in range(mesh_L.M.shape[0]):
    # ##     tot +=sum(mesh_L.M[i])
    # ## print(tot)
    # ## print(eigvecs[:,3])
    # # display_mesh(mesh_L)
    #
    #
    # #FOR H MESH
    # print("H Mesh")
    # l1_e = sorted(list(hMesh[0].nodes), key=lambda x:x.id)
    # n1 = l1_e[0]
    # n2 = l1_e[1]
    # n3 = l1_e[2]
    # n4 = l1_e[3]
    # # u = [[0 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # u = [[n1.basis[x][y]+n2.basis[x][y]+n3.basis[x][y]+n4.basis[x][y] for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # # u[1][1] = 2
    # # u = [[x**2 + y**2 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # # u = [[np.sqrt(x**2 + y**2) for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    #
    # # sim.set_desired_config(u, mesh_L.sortedFlatB, eigvecs[:,4], mesh_L.map_nodes)
    # # sim.set_desired_config(u, mesh_L.sortedFlatB, eigvecs[:,5], mesh_L.map_nodes)
    # actNodes_H = sim.get_active_nodes(hMesh, dom, tolerance=1e-3, u_f=u)
    # nonDuplicateSize, map_duplicate_nodes_to_ind, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes_H)
    # mesh_H = get_mesh_from_displacement(actNodes_H, [n for n in hMesh[2].nodes])
    #
    # E_0 = np.empty(len(mesh_H.activeElems))
    # E_0.fill(GV.Global_Youngs*1)
    # K_0 = np.zeros((2*mesh_H.nonDupSize, 2*mesh_H.nonDupSize))
    # # new_display_mesh(mesh_L, mesh_H, K_0, E_0)
    #
    # # Ek = solve(mesh_L, mesh_H)
    # # print("New Ek", Ek)
    # display_mesh(mesh_H, E_0)

    hMesh = sim.get_hierarchical_mesh(dom)

    # FOR L3 MESH
    # print("L Mesh")
    u_f_L = [[1 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    actNodes_L = sim.get_active_nodes([hMesh[0]], dom, u_f=u_f_L)
    mesh_L = get_mesh_from_displacement(actNodes_L, [n for n in hMesh[0].nodes])
    eigvals, eigvecs = utils.general_eig_solve(mesh_L.K, mesh_L.M)
    # # print(eigvecs[:,3])
    # display_mesh(mesh_L)


    #FOR H MESH
    print("H Mesh")
    l1_e = sorted(list(hMesh[0].nodes), key=lambda x:x.id)
    n1 = l1_e[0]
    n2 = l1_e[1]
    n3 = l1_e[2]
    n4 = l1_e[3]
    # u = [[0 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    u = [[n1.basis[x][y]+n2.basis[x][y]+n3.basis[x][y]+n4.basis[x][y] for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # u_f_H = [[x**2 + y**2 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # u = [[np.sqrt(x**2 + y**2) for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]

    # sim.set_desired_config(u, mesh_L.sortedFlatB, eigvecs[:,9], mesh_L.map_nodes)
    # sim.set_desired_config(u, mesh_L.sortedFlatB, eigvecs[:,5], mesh_L.map_nodes)
    print(u)
    actNodes_H = sim.get_active_nodes(hMesh, dom, tolerance=1e-3, u_f=u)
    nonDuplicateSize, map_duplicate_nodes_to_ind, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes_H)
    mesh_H = get_mesh_from_displacement(actNodes_H, [n for n in hMesh[2].nodes])
    E_0 = np.empty(len(mesh_H.activeElems))
    E_0.fill(GV.Global_Youngs*1.0)
    # M_f = np.matmul(np.matmul(mesh_H.Nc.T, mesh_L.M), mesh_H.Nc)
    # M_c = mesh_H.M
    # arb_v = np.zeros(2*mesh_H.nonDupSize)
    display_mesh(mesh_H, E_0)


    # Ek = solve(mesh_L, mesh_H)
    # print("New Ek", Ek)
    # display_mesh(mesh_H, Ek)




set_up_solver()
