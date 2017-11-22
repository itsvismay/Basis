import numpy as np

import dd as ref
import plotting as plt
import math

from scipy.spatial import Delaunay
np.set_printoptions(threshold="nan")

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




def Integrate(b1, b2, e):
    #Using the stiffness matrix formula Kij from here:
    #https://en.wikiversity.org/wiki/Introduction_to_finite_elements/Axial_bar_finite_element_solution
    #except for 2 dimensions dx, dy
    A = e.get_area()
    E = 1e-3 #Youngs mod


    #b1 slope over cell e
    dB1_dx, dB1_dy = slope_over_cell(b1, e)
    dB2_dx, dB2_dy = slope_over_cell(b2, e)


    return 1

#(section 3.3 CHARMS)
def compute_stiffness_matrix_simple(K, B, hMesh):

    E = set()#set of active cells
    for n in B:
        E |= n.in_elements

    for e in E:
        Bs_e = Bs_(e)
        Ba_e = Ba_(e)

        for b in Bs_e:

            K[b.id, b.id] = Integrate(b, b, e)
            Bs_eNotb = Bs_e - set([b])
            Ba_e = Ba_(e)
            for phi in Bs_eNotb:
                K[b.id, phi.id] = Integrate(b, phi, e)
                K[phi.id, b.id] = Integrate(phi, b, e)

            for phi in Ba_e:
                K[b.id, phi.id] = Integrate(b, phi, e)
                K[phi.id, b.id] = Integrate(phi, b, e)




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


def start():
    dom = ((0,0),(5,5))
    hMesh = get_hierarchical_mesh(dom)
    actNodes = get_active_nodes(hMesh, dom)

    flatB = [i for sublist in actNodes for i in sublist]
    sortedflatB = sorted(flatB, key=lambda x:x.id)
    K = np.zeros((2*ref.Node.number, 2*ref.Node.number))

    compute_stiffness_matrix_simple(K, sortedflatB, hMesh)
    # print(K)

# start()
