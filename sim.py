import numpy as np

import dd as ref
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


def compute_mass(M, B, map_node_id_to_index):
    E = set()#set of cells with active nodes
    for n in B:
        E |= n.in_elements


    for e in E:
        Bs_e = Bs_(e)
        p = 1.0 #density
        m = p*e.get_area()/len(Bs_e)#evenly spead mass over all nodes
        for b in Bs_e:
            M[2*map_node_id_to_index[b.id], 2*map_node_id_to_index[b.id]] += m
            M[2*map_node_id_to_index[b.id]+1, 2*map_node_id_to_index[b.id]+1] += m


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

def create_active_nodes_index_map(B):
    d = {}
    for i in range(len(B)):
        d[B[i].id] = i

    return d

def start():
    dom = ((0,0),(5,5))
    hMesh = get_hierarchical_mesh(dom)
    actNodes = get_active_nodes(hMesh, dom)

    flatB = [i for sublist in actNodes for i in sublist]
    sortedflatB = sorted(flatB, key=lambda x:x.id)
    map_node_to_ind = create_active_nodes_index_map(sortedflatB)

    K = np.zeros((2*len(sortedflatB), 2*len(sortedflatB)))
    f = np.zeros(2*len(sortedflatB))
    M = np.zeros((2*len(sortedflatB), 2*len(sortedflatB)))


    compute_stiffness( K, sortedflatB, hMesh, map_node_to_ind)
    compute_mass(M, sortedflatB, map_node_to_ind)
    # print(K - K.T)
    # print(M)

    x = np.zeros(2*len(sortedflatB))
    v = np.zeros(2*len(sortedflatB))
    V = np.zeros((len(sortedflatB), 2))

    points = []
    #SET X initially
    for i in range(len(sortedflatB)):
        x[2*i] = sortedflatB[i].point[0]
        x[2*i+1] = sortedflatB[i].point[1]
        points.append(sortedflatB[i].point[:2])

    points = np.array(points)
    def X_to_V(V, x):
        for i in range(V.shape[0]):
            V[i, 0] = x[2*i]
            V[i, 1] = x[2*i+1]

    X_to_V(V, x)
    tri = Delaunay(V)
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()



    # def plot_sim():
    #     while True:
    #         renderer.render(V, [2, 9])

    # plot_sim()

start()
