import dd as ref
import sim as sim
import utilities as utils
import global_variables as GV

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import numdifftools as nd
import scipy
import sys, os
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.2f}'.format(x)})



dom = ((0,0), (2, 2))

class Mesh:

    def __init__(self, x, v, f, M, K, activeElems, sortedFlatB, map_nodes, nonDupSize, EmbeddingNodes):
        self.x = x
        self.p = np.copy(x)
        self.v = v
        self.f = f
        self.M = M
        self.W = sim.get_weighting_matrix(nonDupSize, sortedFlatB, map_nodes)
        self.invM = np.linalg.inv(M)
        print("CHECK F=MA for grav")
        print(self.invM.dot(self.f))
        self.K = K
        self.YM = None
        self.V = np.zeros((len(x)/2, 2))
        self.activeElems = activeElems


        self.sortedFlatB = sortedFlatB
        self.map_nodes = map_nodes
        self.nonDupSize = nonDupSize
        self.simSteps = 0

        E_0 = np.empty(len(activeElems))
        E_0.fill(1)
        self.K_no_E = np.zeros((2*nonDupSize, 2*nonDupSize))
        sim.compute_stiffness(self.K_no_E, self.sortedFlatB, self.map_nodes, Youngs=E_0)

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

    def resetYM(self, Y):
        self.YM= Y
        sim.compute_stiffness(self.K, self.sortedFlatB, self.map_nodes, Youngs=self.YM)

    def resetMesh(self):
        self.p = np.copy(self.x)
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
                print("Newton Method Error: not converging NM_static")
                exit()
        self.p = np.copy(p_g)
        self.X_to_V(self.V, self.p)

    def step(self, h=None):
        # invMhhK = np.linalg.inv(self.M - h*h*self.K)
        P = sim.fix_left_end(self.V)
        # print("Mass")
        # print(self.M)
        for i in range (100):
            self.p = self.p + h*P.dot(P.T.dot(self.v))
            forces = self.f + self.K.dot(self.p - self.x)
            self.v = self.v + h*P.dot(P.T.dot(np.matmul(self.invM, P).dot(P.T.dot(forces))))
            # print("")
            # print("o", h*self.invM.dot(forces))
            # print("f", h*P.dot(P.T.dot(self.invM.dot(forces))))
            # print("p", self.p)
            # print("v", self.v)
            # newv = np.copy(self.v)
            # func = lambda x: 0.5*np.dot(x.T, self.W.dot(x))
            # def constr(x):
            #     return x - (self.v + h*P.dot(np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces))))
            # cons = ({'type': 'eq', 'fun': constr })
            #
            # res = scipy.optimize.minimize(func, newv, method="SLSQP", constraints=cons)
            # self.v = np.copy(res.x)

        self.X_to_V(self.V, self.p)

    def new_verlet_step(self, h=None):
        P = sim.fix_left_end(self.V)

        p_g = self.p + h*np.matmul(P, P.T).dot(self.v)
        forces = self.f + self.K.dot(p_g - self.x)

        v_g = self.v + h*P.dot(P.T.dot(self.invM.dot(forces)))

        # newv = np.copy(self.v)
        # func = lambda x: 0.5*np.dot(x.T, self.W.dot(x))
        # def constr(x):
        #     return x - (self.v + h*P.dot(np.matmul(np.matmul(P.T, self.invM), P).dot(P.T.dot(forces))))
        # cons = ({'type': 'eq', 'fun': constr })
        #
        # res = scipy.optimize.minimize(func, newv, method="SLSQP", constraints=cons)
        return p_g, v_g

    def new_nm_step(self, h=None):
        P = sim.fix_left_end(self.V)
        p_g = np.copy(self.p)
        rayleigh = -0

        NewtonMax = 100
        for i in range(NewtonMax):
            forces = self.f + self.K.dot(p_g - self.x) + (rayleigh/h)*self.K.dot(p_g - self.p)

            g_block = P.T.dot(p_g - self.p - h*self.v - h*h*self.invM.dot(forces))
            grad_g_block =  np.matmul(P.T, np.matmul(np.identity(2*(self.nonDupSize)) - h*h*np.matmul(self.invM, self.K), P))

            Q,R = np.linalg.qr(grad_g_block)
            Qg = Q.T.dot(g_block)
            dp = -1*np.linalg.solve(R, Qg)
            p_g += P.dot(dp)

            # print("gblock norm")
            # print(np.linalg.norm(g_block))
            # print("")
            if (np.linalg.norm(g_block)/len(g_block)) < 1e-2:
                # print("solved in ", i)
                break
            if i == 10:
                print("Newton Error: not converging, new_nm_step")
                exit()

        v_g = (p_g - self.p)/h
        return p_g, v_g

    def NMstep(self, h=None):
        P = sim.fix_left_end(self.V)
        p_g = np.copy(self.p)
        rayleigh = -0

        NewtonMax = 100
        for i in range(NewtonMax):
            forces = self.f + self.K.dot(p_g - self.x) + (rayleigh/h)*self.K.dot(p_g - self.p)

            # f_block = forces
            # f_grad_block = np.matmul(np.matmul(P.T, self.K), P)
            g_block = P.T.dot(p_g - self.p - h*self.v - h*h*self.invM.dot(forces))
            grad_g_block =  np.matmul(P.T, np.matmul(np.identity(2*(self.nonDupSize)) - h*h*np.matmul(self.invM, self.K), P))

            Q,R = np.linalg.qr(grad_g_block)
            Qg = Q.T.dot(g_block)
            dp = -1*np.linalg.solve(R, Qg)
            p_g += P.dot(dp)

            # print("gblock norm")
            # print(np.linalg.norm(g_block))
            # print("")
            if (np.linalg.norm(g_block)/len(g_block)) < 1e-2:
                # print("solved in ", i)
                break
            if i == 10:
                print("Error: not converging, NMstep")
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
    # # # map_nodes_old = sim.create_active_nodes_index_map(sortedFlatB)
    nonDuplicateSize, map_nodes, map_points_to_bases = sim.remove_duplicate_nodes_map(actNodes)

    # print("non duplicate size")
    # print(nonDuplicateSize)
    # print("Mesh Elements")
    # print(sortedFlatB)

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
    E = set()#set of active cells
    for n in sortedFlatB:
        E |= n.in_elements

    mesh = Mesh(x_L, v_L, f_L, M_L, K_L, E, sortedFlatB, map_nodes, nonDuplicateSize, EmbeddingNodes)

    YM = np.empty(len(mesh.activeElems))
    YM.fill(GV.Global_Youngs)
    mesh.YM = YM

    return mesh

def solve(meshL, meshH):
    print("Old Solve")
    timestep = 1e-1

    P = sim.fix_left_end(meshH.V)
    meshL.NMstep(h=1e-1)

    v_squiggle = P.dot(P.T.dot(meshH.Nc.T.dot(meshL.v)))
    p_squiggle = meshH.p + timestep*v_squiggle
    u_squiggle = p_squiggle - meshH.x

    def func(E_k):
        #v_squiggle(E_squiggle, F_squiggle)
        meshH.resetYM(E_k)

        #Term 1: 0.5*v~^T* N^T*M*N *v~
        t1 = 0.5*np.dot(v_squiggle,np.matmul(np.matmul(meshH.Nc.T, meshL.M), meshH.Nc).dot(v_squiggle))

        #Term 2: v~^T * N^T*M*N * v~old
        t2 = -1*np.dot(v_squiggle,np.matmul(np.matmul(meshH.Nc.T, meshL.M), meshH.Nc).dot(meshH.v))

        #Term 3: 0.5 u~^T * N^T*K*N * u~ #Energy
        # print(v_squiggle)
        t3 = 0.5*np.dot(u_squiggle, meshH.K.dot(u_squiggle))

        #Term 4: -h* N^T*Fext*v~
        t4 = -1*timestep*np.dot(meshH.Nc.T.dot(meshL.f), v_squiggle)

        no = t1+t2+t3+t4
        # print("     no",no, "t1 ",t1, "t2 ",t2, "t3 ",t3, "t4 ",t4)
        # print(E_k)
        return np.fabs(no)

    def func_der(x):
        J = nd.Gradient(func)(x)
        # print(">>>>>grad", J, x)
        return J.ravel()

    res = minimize(func, meshH.YM, method='Nelder-Mead', options={'disp': True})
    meshH.resetYM(res.x)
    meshH.v = v_squiggle
    meshH.p = p_squiggle
    meshH.X_to_V(meshH.V, meshH.p)
    print("RESULT")
    print(res)
    return res.x

    # def func(Ek):
    #     meshH.resetYM(Ek)
    #     p_squiggle, v_squiggle = meshH.new_nm_step(h=timestep)
    #     v_proj = meshH.Nc.T.dot(meshL.v)
    #
    #     #norm projected velocities
    #     no = np.linalg.norm(v_squiggle - v_proj)
    #     print("     no:", no)
    #     return no
    #
    # def func_der(x):
    #     J = nd.Gradient(func)(x)
    #     # print(">>>>>grad", J, x)
    #     return J.ravel()
    #
    # res = minimize(func, meshH.YM, method='CG', jac=func_der, options={'disp': True})
    #
    # meshH.resetYM(res.x)
    # print("RESULT")
    # print(res)


def display_mesh(meshH, meshL=None):
    viewer = igl.viewer.Viewer()

    # if(meshL!=None):
    #     meshL.resetMesh()
    #     solve(meshL, meshH)


    def key_down(viewer):
        # meshH.NMstep(h=1e-1)
        # if(meshL!=None and meshH.simSteps<=10000):
        #     solve(meshL, meshH)

        # P = sim.fix_left_end(meshH.V)
        meshH.NMstep(h=1e-1)
        #
        # meshH.v = P.dot(P.T.dot(meshH.Nc.T.dot(meshL.v)))
        # meshH.p = meshH.p + 1e-1*meshH.v
        # meshH.X_to_V(meshH.V, meshH.p)
        # print(meshH.v)
        # print(meshH.p)

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


def new_solve(meshL, meshH):
    print("New Solve")

    bnds = ((0, None) for i in range(len(meshH.YM)))
    timestep = 1e-3

    def func(E_k):
        #v_squiggle(E_squiggle, F_squiggle)
        meshH.resetYM(E_k)

        # p_squiggle, v_squiggle = meshH.new_verlet_step(h=timestep)
        p_squiggle, v_squiggle = meshH.new_nm_step(h=timestep)

        #Term 1: 0.5*v~^T* N^T*M*N *v~
        t1 = 0.5*np.dot(v_squiggle,np.matmul(np.matmul(meshH.Nc.T, meshL.M), meshH.Nc).dot(v_squiggle))

        #Term 2: v~^T * N^T*M*N * v~old
        t2 = np.dot(v_squiggle,np.matmul(np.matmul(meshH.Nc.T, meshL.M), meshH.Nc).dot(meshH.v))

        #Term 3: 0.5 u~^T * N^T*K*N * u~ #Energy
        u_squiggle = p_squiggle - meshH.x
        # print(v_squiggle)
        t3 = 0.5*np.dot(u_squiggle, np.matmul(np.matmul(meshH.Nc.T, meshL.K), meshH.Nc).dot(u_squiggle))

        #Term 4: -h* N^T*Fext*v~
        t4 = -1*timestep*np.dot(meshH.Nc.T.dot(meshL.f), v_squiggle)

        no = t1+t2+t3+t4
        print("     no",no, "t1 ",t1, "t2 ",t2, "t3 ",t3, "t4 ",t4)
        # print(E_k)
        return np.fabs(no)

    def func_der(x):
        J = nd.Gradient(func)(x)
        print(">>>>>grad", J, x)
        return J.ravel()

    res = minimize(func, meshH.YM, method='BFGS', jac=func_der, options={'disp': True})
    meshH.resetYM(res.x)
    print("RESULT")
    print(res)
    print(meshH.YM)
    # print("after solve", id(meshH.K))
    # print(meshH.K)
    return res.x

    # sim.compute_stiffness(meshH.K, meshH.sortedFlatB, meshH.map_nodes, Youngs=E_0)
    # return 0

def new_display_mesh(meshH, meshL):
    viewer = igl.viewer.Viewer()
    if(meshH.YM != None):
        new_solve(meshL, meshH)
    def key_down(viewer):
        # if(meshH.simSteps<=5):
        # meshH.step(h=1e-2)
        # if(meshH.YM != None):
        #     new_solve(meshL, meshH)
        meshH.NMstep(h=1e-1)
        # print(meshH.K)
        # print("u", meshH.p - meshH.x)
        # print("v",meshH.v)
        # print("f",meshH.f)
        # Emesh = meshH.get_embedded_mesh()
        viewer.data.clear()
        V1 = igl.eigen.MatrixXd(meshH.V)
        F1 = igl.eigen.MatrixXi(meshH.tri)
        viewer.data.set_mesh(V1, F1)
        meshH.simSteps+=1

        return True

    key_down(viewer)
    viewer.core.is_animating = True
    viewer.callback_post_draw = key_down
    viewer.launch()

def set_up_solver(fineLevel=1):
    hMesh = sim.get_hierarchical_mesh(dom)

    # FOR L3 MESH
    print("L Mesh")
    u_f_L = [[1 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    actNodes_L = sim.get_active_nodes([hMesh[fineLevel]], dom, u_f=u_f_L)
    mesh_L = get_mesh_from_displacement(actNodes_L, EmbeddingNodes=[n for n in hMesh[fineLevel].nodes])
    eigvals, eigvecs = utils.general_eig_solve(mesh_L.K, mesh_L.M)
    display_mesh(mesh_L)
    # mesh_L.v[0] =1
    # for i in range(100):
    #     mesh_L.NMstep()
    # print(mesh_L.p)
    # print(mesh_L.v)


    #FOR H MESH
    print("H Mesh")
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
    # actNodes_H = sim.get_active_nodes([hMesh[1]], dom, tolerance=5e-3, u_f=u)
    # mesh_H = get_mesh_from_displacement(actNodes_H, [n for n in hMesh[fineLevel].nodes])
    # print(mesh_H.sortedFlatB)

    # exit()
    # E_0 = np.empty(len(mesh_H.activeElems))
    # E_0.fill(GV.Global_Youngs)
    # E_0 =  np.array([133976.09, 142732.94, 83519.35, 94675.98, 90789.00, 120582.21, 40378.85, 50402.73, 86073.33, 105846.86, 120927.38, 78347.65, 147223.67, 80322.28, 84096.08, 106055.73, 109612.69,
    #    92483.46, 144294.45, 75641.41, 55946.07, 155050.56, 133797.53, 112925.10, 93251.21, 77559.87, 111344.65])
    # E_0.fill(3e5)

    # print("Afterwards")
    # print("Original", id(mesh_H.K))



    # display_mesh(mesh_H)
    # display_mesh(mesh_H, mesh_L) #dont run because it screws up the velocities used in the E_t+1 solver
    # new_display_mesh(mesh_H, mesh_L)
    # print(mesh_H.K)
    # t = 0
    # for i in range(mesh_H.M.shape[0]):
    #     print(sum(mesh_H.M[i]))
    # # print("id", id(mesh_H.K))
    # print("grav acceleration")
    # print(np.linalg.inv(mesh_H.M).dot(mesh_H.f))

    # return mesh_L, mesh_H



set_up_solver()
