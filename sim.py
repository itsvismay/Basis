

import dd as ref

def get_stiffness_matrix():


def get_hierarchical_mesh(dom):
    l1 = ref.Level(dom)
    l1.create_bases()
    l2 = l1.split()
    return [l1, l2]

def get_active_nodes(hMesh, dom, tolerance = 0.0001):
    u_f = [[(x-2)**4 + (y-2)**4 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    aN = ref.solve(hMesh, u_f, tolerance)


def start():
    dom = ((0,0),(5,5))
    hMesh = get_hierarchical_mesh(dom)
    actNodes = get_active_nodes(hMesh, dom)
