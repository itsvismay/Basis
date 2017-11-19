import numpy as np

import dd as ref
import plotting as plt

from scipy.spatial import Delaunay
np.set_printoptions(threshold="nan")

def compute_stiffness_matrix_simple(K, B, hMesh):
    #(section 3.3 CHARMS)
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

    def Integrate(b1, b2, e):
        return 1

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
    print(K)

start()
