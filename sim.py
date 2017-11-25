import numpy as np

import dd as ref
import utilities as utils
import plotting as plt
import math

from scipy.spatial import Delaunay
np.set_printoptions(threshold="nan")
import sys, os
sys.path.insert(0, os.getcwd()+"/../libigl/python/")
import pyigl as igl
# import render as renderer

import matplotlib.pyplot as plt

def Bs_(e):
    phi = set()
    if e.n1.active == True:
        phi.add(e.n1)
    if e.n2.active == True:
        phi.add(e.n2)
    if e.n3.active == True:
        phi.add(e.n3)
    return phi

def Ba_(e):
    e_a = e.ancestor
    if e_a == None:
        return set()
    return Bs_(e_a)

def basis_supports_cell(b, cell):
    # print(b)
    # print(cell)
    n1_under_b = b.support_points[cell.n1.point[0]][cell.n1.point[1]]
    n2_under_b = b.support_points[cell.n2.point[0]][cell.n2.point[1]]
    n3_under_b = b.support_points[cell.n3.point[0]][cell.n3.point[1]]
    if(n1_under_b == 1 and n2_under_b == 1 and n3_under_b == 1):
        return True
    else:
        return False

def slope_over_cell(b, e):
    # print(b)
    # print(e)
    if not basis_supports_cell(b, e):
        return 0, 0

    #from here http://www.math.lsa.umich.edu/~glarose/classes/calcIII/web/13_5/planeeqn.html
    n1 = np.array([e.n1.point[0], e.n1.point[1], b.basis[e.n1.point[0]][e.n1.point[1]]])
    n2 = np.array([e.n2.point[0], e.n2.point[1], b.basis[e.n2.point[0]][e.n2.point[1]]])
    n3 = np.array([e.n3.point[0], e.n3.point[1], b.basis[e.n3.point[0]][e.n3.point[1]]])
    z_axis = np.array([0,0,1])
    normal = np.cross((n1 - n2), (n1 - n3))
    #Equations for plane
    #a(x - x0) + b(y - y0) + c(z - z0) = 0,
    #a x + b y + c z = d    and
    #z = d + m x + n y
    x_slope = -1.0*normal[0]/normal[2]
    y_slope = -1.0*normal[1]/normal[2]
    return x_slope, y_slope

#basis equation for b1 over element e
def basis_value_over_e_at_xy(b, e, x, y):
    n1 = np.array([e.n1.point[0], e.n1.point[1], b.basis[e.n1.point[0]][e.n1.point[1]]])
    n2 = np.array([e.n2.point[0], e.n2.point[1], b.basis[e.n2.point[0]][e.n2.point[1]]])
    n3 = np.array([e.n3.point[0], e.n3.point[1], b.basis[e.n3.point[0]][e.n3.point[1]]])
    normal = np.cross((n1 - n2), (n1 - n3))
    z = -1.0*(normal[0]*(x-n1[0]) + normal[1]*(y-n1[1]))/normal[2] + n1[2]
    return z

def Integrate_K(K, map_node_id_to_index, b1, b2, e):
    #Using the stiffness matrix formula Kij from here:
    #https://en.wikiversity.org/wiki/Introduction_to_finite_elements/Axial_bar_finite_element_solution
    #except for 2 dimensions dx, dy
    A = e.get_area()
    E = 1e-1 #Youngs mod


    #b1 slope over cell e
    dB1_dx, dB1_dy = slope_over_cell(b1, e)
    dB2_dx, dB2_dy = slope_over_cell(b2, e)


    K[2*map_node_id_to_index[b1.id], 2*map_node_id_to_index[b2.id]] = A*E*dB1_dx*dB2_dx
    K[2*map_node_id_to_index[b1.id]+1, 2*map_node_id_to_index[b2.id]+1] = A*E*dB1_dy*dB2_dy


def Integrate_M(M, map_node_id_to_index, b1, b2, e):
    A = e.get_area()

    #1 point gauss quadrature over centroid of triangle
    centroid_x = (e.n1.point[0] + e.n2.point[0] + e.n3.point[0])/3.0
    centroid_y = (e.n1.point[1] + e.n2.point[1] + e.n3.point[1])/3.0

    mass = e.get_area()*basis_value_over_e_at_xy(b1, e, centroid_x, centroid_y)\
                        *basis_value_over_e_at_xy(b2, e, centroid_x, centroid_y)

    M[2*map_node_id_to_index[b1.id], 2*map_node_id_to_index[b2.id]] += mass
    M[2*map_node_id_to_index[b1.id]+1, 2*map_node_id_to_index[b2.id]+1] += mass


def Integrate_f(f, map_node_id_to_index, b, e, x = None):
    if(x == None):
        return

    if not basis_supports_cell(b, e):
        return

    p0 = np.array([b.point[0], b.point[1], 1])
    p1 = np.array([e.n1.point[0], e.n1.point[1], b.basis[e.n1.point[0]][e.n1.point[1]]])
    p2 = np.array([e.n2.point[0], e.n2.point[1], b.basis[e.n2.point[0]][e.n2.point[1]]])
    p3 = np.array([e.n3.point[0], e.n3.point[1], b.basis[e.n3.point[0]][e.n3.point[1]]])
    a = p1 - p0
    b = p2 - p0
    c = p3 - p0
    tet_vol = (1.0/6)*np.linalg.det(np.dot(a, np.cross(b, c)))
    rec_vol = 0.0
    if(p1[2] < p2[2]):
        rec_vol += e.get_area()*p1[2]
    else:
        rec_vol += e.get_area()*p2[2]

    vol = tet_vol + rec_vol

    t = 1.0
    a = 30
    force_x = (t/(2*e.get_area()))*vol*a*x[map_node_id_to_index[b.id]]
    force_y = (t/(2*e.get_area()))*vol*a*x[map_node_id_to_index[b.id]+1]

    f[2*map_node_id_to_index[b.id]] = force_x
    f[2*map_node_id_to_index[b.id]+1] = force_y



#(section 3.3 CHARMS)
def compute_stiffness(K, B, hMesh, map_node_id_to_index, x = None):

    E = set()#set of active cells
    for n in B:
        E |= n.in_elements


    for e in E:
        Bs_e = Bs_(e)
        Ba_e = Ba_(e)

        for b in Bs_e:
            Integrate_K(K, map_node_id_to_index, b, b, e)

            Bs_eNotb = Bs_e - set([b])
            for phi in Bs_eNotb:
                Integrate_K(K, map_node_id_to_index, b, phi, e)
                Integrate_K(K, map_node_id_to_index, phi, b, e)

            for phi in Ba_e:
                Integrate_K(K, map_node_id_to_index, b, phi, e)
                Integrate_K(K, map_node_id_to_index, phi, b, e)


def compute_mass(M, B, map_node_id_to_index):
    E = set()#set of cells with active nodes
    for n in B:
        E |= n.in_elements


    for e in E:
        Bs_e = Bs_(e)
        Ba_e = Ba_(e)

        for b in Bs_e:
            Integrate_M(M, map_node_id_to_index, b, b, e)

            Bs_eNotb = Bs_e - set([b])
            for phi in Bs_eNotb:
                Integrate_M(M, map_node_id_to_index, b, phi, e)
                Integrate_M(M, map_node_id_to_index, phi, b, e)

            for phi in Ba_e:
                Integrate_M(M, map_node_id_to_index, b, phi, e)
                Integrate_M(M, map_node_id_to_index, phi, b, e)


def compute_force(f, B, map_node_id_to_index, x = None):
    E = set()#set of active cells
    for n in B:
        E |= n.in_elements


    for e in E:
        Bs_e = Bs_(e)
        Ba_e = Ba_(e)

        for b in Bs_e:
            Integrate_f(f, map_node_id_to_index, b, e, x)

            Bs_eNotb = Bs_e - set([b])
            for phi in Bs_eNotb:
                Integrate_f(f, map_node_id_to_index, phi, e, x)

            for phi in Ba_e:
                Integrate_f(f, map_node_id_to_index, phi, e, x)


def get_hierarchical_mesh(dom):
    l1 = ref.Level(dom)
    l1.create_bases()
    l2 = l1.split()
    l2.create_bases()
    l3 = l2.split()
    l3.create_bases()
    return [l1, l2, l3]

def get_active_nodes(hMesh, dom, tolerance = 0.0001):
    u_f = [[(x-2)**4 + (y-2)**4 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    aN = ref.solve(hMesh, u_f, tolerance)

    return aN


def create_active_nodes_index_map(N):
    #remove remove_redundant_nodes
    #always select the nodes in finer meshes first
    d_p_b = {} #dictionary from point to basis
    d_b_i = {} #dictionary from basis id to index for simulation: many -> 1
    ind = 0
    for l in range(len(N)-1, -1, -1):
        # print([(b.id, b.point) for b in N[l]])
        for b in N[l]:
            p = (int(b.point[0]), int(b.point[1]))
            if p not in d_p_b:
                d_p_b[p] = b
                d_b_i[b.id] = ind
                ind+=1
            else:
                existing_ind = d_b_i[ d_p_b[p].id ]
                d_b_i[b.id] = existing_ind


    return ind, d_b_i

def start():
    dom = ((0,0),(5,5))
    hMesh = get_hierarchical_mesh(dom)
    actNodes = get_active_nodes(hMesh, dom)

    vsize, map_node_to_ind = create_active_nodes_index_map(actNodes)
    sortedflatB = [i for sublist in actNodes for i in sublist]

    K = np.zeros((2*vsize, 2*vsize))
    f = np.zeros(2*vsize)
    M = np.zeros((2*vsize, 2*vsize))


    compute_stiffness( K, sortedflatB, hMesh, map_node_to_ind)
    compute_mass(M, sortedflatB, map_node_to_ind)
    print(utils.is_invertible(M-1e-3*K))
    print(utils.is_pos_def(M))
    x = np.zeros(2*vsize)
    v = np.zeros(2*vsize)
    v[0] = 1

    V = np.zeros((vsize, 2))


    points = []
    #SET X initially
    for b in sortedflatB:
        x[2*map_node_to_ind[b.id]] = b.point[0]
        x[2*map_node_to_ind[b.id]+1] = b.point[1]
        # print(b.id, "- ", map_node_to_ind[b.id], "- ", b.point)

    def X_to_V(V, x):
        for i in range(V.shape[0]):
            V[i, 0] = x[2*i]
            V[i, 1] = x[2*i+1]

    X_to_V(V, x)

    tri = Delaunay(V)
    h = 1e-1
    invMdtK = np.linalg.inv(M - h*h*K)
    # for t in range(0, 200):
    #     v = np.matmul(invMdtK, M).dot(v) + h*np.matmul(invMdtK, K).dot(x)
    #     x = x + h*v
    #     X_to_V(V, x)
    #     plt.triplot(V[:,0], V[:,1], tri.simplices.copy())
    #     plt.plot(V[:,0], V[:,1], 'o')
    #     plt.show()
    #
    #
    #
    # # def plot_sim():
    # #     while True:
    # #         renderer.render(V, [2, 9])
    #
    # # plot_sim()

start()
