import numpy as np
import copy
import dd as ref
import utilities as utils
import plotting as plot
import math

from scipy.spatial import Delaunay
np.set_printoptions(threshold="nan", linewidth=190, precision=3, formatter={'all': lambda x:'{:2.2f}'.format(x)})
import sys, os
sys.path.insert(0, os.getcwd()+"/../libigl/python/")
import pyigl as igl
import global_variables as GV
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

def Ba_(e_a):
    #recursively go through all ancestor
    #to find the ancestral support set of bases
    if e_a == None:
        return set()

    b = Bs_(e_a) | Ba_(e_a.ancestor)
    return b

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
    x_slope = -1.0*normal[0]/(normal[2])/(2*e.get_area())
    y_slope = -1.0*normal[1]/(normal[2])/(2*e.get_area())
    # print("-----------", b.id, x_slope, y_slope)
    return x_slope, y_slope

#basis equation for b1 over element e
def basis_value_over_e_at_xy(b, e, x, y):
    n1 = np.array([e.n1.point[0], e.n1.point[1], b.basis[e.n1.point[0]][e.n1.point[1]]])
    n2 = np.array([e.n2.point[0], e.n2.point[1], b.basis[e.n2.point[0]][e.n2.point[1]]])
    n3 = np.array([e.n3.point[0], e.n3.point[1], b.basis[e.n3.point[0]][e.n3.point[1]]])
    normal = np.cross((n1 - n2), (n1 - n3))
    # print(e)
    # print(normal)
    z = -1.0*(normal[0]*(x-n1[0]) + normal[1]*(y-n1[1]))/normal[2] + n1[2]

    # if(z<0):
    #     print(e.id)
    #     print(z)
    #     print(b.basis)
    #     print(normal)
    #     print(x, y)
    #     print(n1)
    #     print(n2)
    #     print(n3)

    return z


def AnotherQuadratureMethod(b1, b2, e):
    DEPTH = 5
    #Mentioned by Dave: Divide up the triangles into pieces,
    #and integrate using 1 point quadrature, multiply by area
    def break_up_triangle(p1, p2, p3, depth):
        if(depth == 0):
            area = np.linalg.norm(np.cross((p1 - p2), (p1 - p3)))*0.5
            centroid_x = (p1[0] + p2[0] + p3[0])/3.0
            centroid_y = (p1[1] + p2[1] + p3[1])/3.0
            # print("AREA ",area)
            # print("CENTROID ", centroid_x, centroid_y)
            mass = basis_value_over_e_at_xy(b1, e, centroid_x, centroid_y)*basis_value_over_e_at_xy(b2, e, centroid_x, centroid_y)
            # print("MASS", mass)
            return mass*area
        else:
            centroid = np.array([(p1[0] + p2[0] + p3[0])/3.0, (p1[1] + p2[1] + p3[1])/3.0])
            #tri 1
            t1 = break_up_triangle(centroid, p1, p2, depth-1)
            #tri 2
            t2 = break_up_triangle(centroid, p1, p3, depth-1)
            #tri 3
            t3 = break_up_triangle(centroid, p3, p2, depth-1)
            return t1+t2+t3

    total_mass = break_up_triangle(e.n1.point[:2], e.n2.point[:2], e.n3.point[:2], DEPTH)

    return total_mass

def GaussQuadrature(b1, b2, e):
    #As defined here http://people.maths.ox.ac.uk/parsons/Specification.pdf
    weights = [5.0/9.0, 8.0/9.0, 5.0/9.0]#, [8.0/9.0, 8.0/9.0, 8.0/9.0], [5.0/9.0, 8.0/9.0, 5.0/9.0]]
    # weights = [1.0, 1.0, 1.0]
    #hard coded x, y points for the standard triangle
    x_standard = [0.11270166537, 0.5, 0.88729833462]
    y_standard = [[0.1, 0.44364916731, 0.78729833462], \
            [0.05635083268 , 0.25, 0.44364916731],\
            [0.01270166537 , 0.05635083268, 0.1]]

    points_on_ref_tri = []

    #Ref = F*Std
    F = e.reference_shape_matrix()*np.linalg.inv(e.standardized_shape_matrix())
    tot = 0.0
    for i in range(len(x_standard)):
        for j in range(len(y_standard[i])):
            p = np.array(F.dot(np.array([x_standard[i], y_standard[i][j]])))[0]
            p = p + e.n1.point[:2]

            m1 = basis_value_over_e_at_xy(b1, e, p[0], p[1])
            m2 = basis_value_over_e_at_xy(b2, e, p[0], p[1])
            if(not utils.PointInTriangle(p, e.n1.point, e.n2.point, e.n3.point)):
                print("OH SHIT! Sim.py Integrate_M error, transformed pont not in triangle")
                exit()
            tot += weights[i]*weights[j]*m1*m2#*basis_value_over_e_at_xy(b1, e, p[0], p[1])*basis_value_over_e_at_xy(b2, e, p[0], p[1])

    mass = abs(np.linalg.det(F))*tot*(1.0/8) # multiply by det(F) = new area/ old area
    return mass

def Integrate_M(M, map_node_id_to_index, b1, b2, e):
    density = 100
    mass = AnotherQuadratureMethod(b1, b2, e)*density

    # print(map_node_id_to_index[b1.id], map_node_id_to_index[b2.id], mass)
    M[2*map_node_id_to_index[b1.id], 2*map_node_id_to_index[b2.id]] += mass
    M[2*map_node_id_to_index[b1.id]+1, 2*map_node_id_to_index[b2.id]+1] += mass
    return mass

def Integrate_f(f, map_node_id_to_index, b, e, x = None):
    if(x is None):
        return

    if not basis_supports_cell(b, e):
        return

    p0 = np.array([b.point[0], b.point[1], 1])
    p1 = np.array([e.n1.point[0], e.n1.point[1], b.basis[e.n1.point[0]][e.n1.point[1]]])
    p2 = np.array([e.n2.point[0], e.n2.point[1], b.basis[e.n2.point[0]][e.n2.point[1]]])
    p3 = np.array([e.n3.point[0], e.n3.point[1], b.basis[e.n3.point[0]][e.n3.point[1]]])

    tet_vol = (1.0/6)*utils.volume_of_tet(p0, p1, p2, p3)
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

def get_local_B(b, e):
    dB_dx, dB_dy = slope_over_cell(b, e)
    return np.matrix([[dB_dx, 0],
                    [0, dB_dy],
                    [dB_dy, dB_dx]])

#(section 3.3 CHARMS)
def compute_stiffness(K, B, map_node_id_to_index, x = None, Youngs=None):
    E = set()#set of active cells
    for n in B:
        E |= n.in_elements

    if(Youngs==None):
        Youngs = np.empty(len(E))
        Youngs.fill(GV.Global_Youngs)

    elem = 0
    for e in E:
        Bs_e = sorted(list(Bs_(e)), key = lambda x: x.id)
        Ba_e = sorted(list(Ba_(e.ancestor)), key = lambda x: x.id)

        Be = np.matrix([]).reshape(3, 0)
        # print(Be)
        for b in Bs_e+Ba_e:
            Be = np.concatenate((Be, get_local_B(b, e)), axis=1)


        t = 1 #thickness of element
        D = np.matrix([[1-GV.Global_Poissons, GV.Global_Poissons, 0],
                        [ GV.Global_Poissons, 1-GV.Global_Poissons, 0],
                        [ 0, 0, 0.5-GV.Global_Poissons]])*(abs(Youngs[elem])/((1+GV.Global_Poissons)*(1-2*GV.Global_Poissons)))

        local_K = (np.transpose(Be)*D*Be)*t*e.get_area()
        indices = [map_node_id_to_index[b.id] for b in Bs_e+Ba_e]

        j = 0
        for r in local_K:
            kj = j%2
            for s in range(r.shape[1]/2):
                dfxrdxs = r.item(2*s)
                dfxrdys = r.item(2*s+1)

                K[2*indices[j/2]+kj, 2*indices[s]] += dfxrdxs
                K[2*indices[j/2]+kj, 2*indices[s]+1] += dfxrdys

            j+=1
        elem+=1



def compute_mass(M, B, map_node_id_to_index):
    E = set()#set of cells with active nodes
    for n in B:
        E |= n.in_elements

    for e in E:
        Bs_e = Bs_(e)
        Ba_e = Ba_(e.ancestor)
        for b in Bs_e:
            Integrate_M(M, map_node_id_to_index, b, b, e)

            Bs_eNotb = Bs_e - set([b])
            for phi in Bs_eNotb:
                Integrate_M(M, map_node_id_to_index, b, phi, e)

            for phi in Ba_e:
                m1 = Integrate_M(M, map_node_id_to_index, b, phi, e)


def compute_force(f, B, map_node_id_to_index, x = None):
    E = set()#set of active cells
    for n in B:
        E |= n.in_elements


    for e in E:
        Bs_e = Bs_(e)
        Ba_e = Ba_(e.ancestor)

        for b in Bs_e:
            Integrate_f(f, map_node_id_to_index, b, e, x)

            Bs_eNotb = Bs_e - set([b])
            for phi in Bs_eNotb:
                Integrate_f(f, map_node_id_to_index, phi, e, x)

            for phi in Ba_e:
                Integrate_f(f, map_node_id_to_index, phi, e, x)

def compute_gravity(f, M):
    for i in range(f.shape[0]):
        if(i%2 == 1):
            f[i] = sum(M[i])*-9.8

def get_hierarchical_mesh(dom):
    l1 = ref.Level(dom)
    l2 = l1.split()
    l3 = l2.split()
    return [l1, l2, l3]
    # return [l3]

def get_active_nodes(hMesh, dom, tolerance = 0.0001, u_f=None):
    if(u_f == None):
        l1_e = sorted(list(hMesh[0].nodes), key=lambda x:x.id)
        n1 = l1_e[0]
        n2 = l1_e[1]
        n3 = l1_e[2]
        n4 = l1_e[3]
        u_f = [[n1.basis[x][y]+n2.basis[x][y]+n3.basis[x][y]+n4.basis[x][y] for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
        # u_f = [[2 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]

    aN = ref.solve(hMesh, u_f, tolerance)
    return aN

def fix_vertex(v, invM):
    # print("FIX", v)
    invM[2*v] =0
    invM[2*v+1] =0

    invM[:, 2*v] =0
    invM[:, 2*v+1] =0


def fix_left_end(V, invM):
    vert_ind = 0
    for p in V:
        if(p[0] == 0):
            fix_vertex(vert_ind, invM)
        vert_ind +=1


def create_active_nodes_index_map(B):
    d = {}
    for i in range(len(B)):
        d[B[i].id] = i

    return d

def remove_duplicate_nodes_map(N):
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

    return ind, d_b_i, d_p_b

def set_x_initially(x, B, map_node_to_ind):
    #SET X initially
    for b in B:
        x[2*map_node_to_ind[b.id]] = b.point[0]
        x[2*map_node_to_ind[b.id]+1] = b.point[1]
        # print(b.id, "- ", map_node_to_ind[b.id], "- ", b.point)


def start():
    dom = ((0,0),(5,5))
    hMesh = get_hierarchical_mesh(dom)
    actNodes = get_active_nodes(hMesh, dom)

    nonDuplicateSize, map_duplicate_nodes_to_ind, map_points_to_bases = remove_duplicate_nodes_map(actNodes)

    flatB = [i for sublist in actNodes for i in sublist]
    sortedflatB = sorted(flatB, key=lambda x:x.id)
    map_node_to_ind = create_active_nodes_index_map(sortedflatB)
    dupSize = len(sortedflatB)

    K = np.zeros((2*dupSize, 2*dupSize))
    global f
    f = np.zeros(2*dupSize)
    M = np.zeros((2*dupSize, 2*dupSize))

    compute_stiffness( K, sortedflatB, map_node_to_ind)
    compute_mass(M, sortedflatB, map_node_to_ind)
    compute_gravity(f, M)
    print("M - hK inverts", utils.is_invertible(M-1e-3*K))
    # print(M)
    print("Mass is spd", utils.is_pos_def(M))
    x = np.zeros(2*dupSize)
    global v
    v = np.zeros(2*dupSize)
    # v[2] = 5
    # v[4] = 5
    V = np.zeros((dupSize, 2))


    points = []
    set_x_initially(x, sortedflatB, map_node_to_ind)

    def X_to_V(V, x):
        for i in range(V.shape[0]):
            V[i, 0] = x[2*i]
            V[i, 1] = x[2*i+1]


    X_to_V(V, x)
    # print(V)

    global tri
    tri = Delaunay(V)
    h = 1e-3

    invMdtK = np.linalg.inv(M - h*h*K)
    invM = np.linalg.inv(M)

    fix_left_end(V, invM)

    #
    # print(K)
    # print(M)
    print(f)

    # exit()

    global p
    p = copy.copy(x)

    def draw():
        global p
        global v
        for i in range(100):
            p = p + h*v
            v = v + h*invM.dot(K.dot(x-p) + f)
        X_to_V(V, p)

    viewer = igl.viewer.Viewer()

    def key_down(viewer, key, modifier):
        draw()
        viewer.data.clear()

        V1 = igl.eigen.MatrixXd(V)
        # print(V1)
        viewer.data.add_points(V1, igl.eigen.MatrixXd([[0,0,0]]))
        for e in tri.simplices:
            viewer.data.add_edges(V1.row(e[0]), V1.row(e[1]),igl.eigen.MatrixXd([[1, 1, 1]]))
            viewer.data.add_edges(V1.row(e[1]), V1.row(e[2]),igl.eigen.MatrixXd([[1, 1, 1]]))
            viewer.data.add_edges(V1.row(e[0]), V1.row(e[2]),igl.eigen.MatrixXd([[1, 1, 1]]))

        return True

    key_down(viewer, ord('5'), 0)
    # F1 = igl.eigen.MatrixXi()
    viewer.core.is_animating = True
    viewer.callback_key_down = key_down
    viewer.launch()

# start()
