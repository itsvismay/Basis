
import dd as ref
import sim as sim

import numpy as np
import utilities as utils

def test_creation_of_levels():
    dom = ((0,0), (5,5))
    l1 = ref.Level(dom)
    l1.create_bases()
    assert(ref.Node.getNumberOfNodes() == 4)
    assert(ref.Element.getNumberOfElements() == 2)

    l2 = l1.split()
    l2.create_bases()
    assert(ref.Node.getNumberOfNodes() == 13)
    assert(ref.Element.getNumberOfElements() == 10)

    l3 = l2.split()
    l3.create_bases()
    assert(ref.Node.getNumberOfNodes() == 38)
    print('test_creation_of_levels OK')

def test_Bs_Ba_():
    dom = ((0,0), (5,5))
    hMesh = sim.get_hierarchical_mesh(dom)
    list(hMesh[0].elements)[0].active = True
    list(hMesh[0].elements)[0].n1.active = True
    list(hMesh[0].elements)[0].n2.active = True

    list_l2_e = list(hMesh[1].elements)
    list_l2_e[0].n1.active = True
    list_l2_e[1].n2.active = True

    active_samelevel_set = sim.Bs_(list_l2_e[1])
    active_ancestorlevel_set = sim.Ba_(list_l2_e[0])
    assert(len(active_samelevel_set) == 2)
    assert(len(active_ancestorlevel_set) == 2)


def test_basis_supports_cell():
    dom = ((0,0), (5,5))
    hMesh = sim.get_hierarchical_mesh(dom)
    l1_e = sorted(list(hMesh[0].elements), key=lambda x:x.id)
    l2_e = sorted(list(hMesh[1].elements), key=lambda x:x.id)

    b = l1_e[0].n2
    assert(sim.basis_supports_cell(b, l2_e[0]) == True)
    assert(sim.basis_supports_cell(b, l2_e[4]) == False)


def test_slope_over_cell():
    dom = ((0,0), (5,5))
    hMesh = sim.get_hierarchical_mesh(dom)
    l1_e = sorted(list(hMesh[0].elements), key=lambda x:x.id)
    l2_e = sorted(list(hMesh[1].elements), key=lambda x:x.id)

    b = l1_e[0].n2
    assert(np.linalg.norm(np.array(sim.slope_over_cell(b, l2_e[0]))- np.array([0.25, -0.25])) <0.0001 )
    assert(np.linalg.norm(np.array(sim.slope_over_cell(b, l2_e[4]))- np.array([0, 0])) <0.0001 )



def test_create_stiffness_matrix():
    dom = ((0,0), (2,2))
    useMesh = 0
    hMesh = sim.get_hierarchical_mesh(dom)
    # print(hMesh[0].get_stiffness_matrix())
    for n in hMesh[useMesh].nodes:
        n.active = True

    print("Stiffness matrix")
    # K_1 = hMesh[0].get_stiffness_matrix();
    print(hMesh[useMesh].K)

    sortedflatB = sorted(list(hMesh[useMesh].nodes), key=lambda x: x.id )
    map_k = sim.create_active_nodes_index_map(sortedflatB)
    K = np.zeros((2*len(sortedflatB), 2*len(sortedflatB)))
    f = np.zeros(2*len(sortedflatB))
    M = np.zeros((2*len(sortedflatB), 2*len(sortedflatB)))
    print("Hierarchical Stiffness")
    sim.compute_stiffness(K, sortedflatB, hMesh, map_k)
    print(map_k)
    print(K - hMesh[useMesh].K)
    x = np.zeros(2*len(sortedflatB))
    for b in sortedflatB:
        x[2*map_k[b.id]] = b.point[0]
        x[2*map_k[b.id]+1] = b.point[1]
    print(K, x)
    print(K.dot(x))
    print(utils.is_pos_def(K))
    print("Strain E")
    print(np.dot(x, K.dot(x)))

def test_create_mass_matrix():
    dom = ((0,0), (5,5))
    hMesh = sim.get_hierarchical_mesh(dom)
    # print(hMesh[0].get_mass_matrix())
    for n in hMesh[0].nodes:
        n.active = True


    sortedflatB = sorted(list(hMesh[0].nodes), key=lambda x: x.id)
    map_k = sim.create_active_nodes_index_map(sortedflatB)
    M = np.zeros((2*len(sortedflatB), 2*len(sortedflatB)))
    sim.compute_mass(M, sortedflatB, map_k)

    print(hMesh[0].get_mass_matrix())
    print("Lumped Mass SPD",utils.is_pos_def(hMesh[0].get_mass_matrix()))
    print("")
    print(M)
    print("Sim Mass SPD", utils.is_pos_def(M))
    # for r in M:
    #     print(sum(r))


def test_Gaussian_Quadrature():
    l_trash = ref.Level()

    n1 = ref.Node(0,0, 0)
    n2 = ref.Node(1,0, 0)
    n3 = ref.Node(0,1, 0)

    print(utils.volume_of_tet([0,0,0], [1,0,0], [0,1,0], [0,0,0]))
    e = ref.Element(n1, n2, n3, 0)

    n1.update_basis()
    n2.update_basis()
    n3.update_basis()

    print("QUAD CODE")
    l_trash.GaussQuadrature_2d_3point(n1, e)

def test_Another_Quadrature_Method():
    l_trash = ref.Level()

    n1 = ref.Node(0,0, 0)
    n2 = ref.Node(4,0, 0)
    n3 = ref.Node(0,4, 0)
    e = ref.Element(n1, n2, n3, 0)

    n1.update_basis()
    n2.update_basis()
    n3.update_basis()

    # n1.basis = [[1 for i in range(5)] for j in range(5)]
    # n2.basis = [[1 for i in range(5)] for j in range(5)]
    # n3.basis = [[1 for i in range(5)] for j in range(5)]
    print("MAKE SURE TO REMOVE THE SECOND BASIS TERM IN THE SUM w*b1*b2")

    print("VOL ", utils.volume_of_tet([0,0,0], [4,0,0], [0,4,0], [0,0,1]))
    m1 = sim.AnotherQuadratureMethod(n2, n3, e)
    print("ANOTHER QUAD METHOD: ",m1)
    # m2 = sim.GaussQuadrature(n3, n3, e)
    # print("GAUSS QUAD: ", m2)

# test_creation_of_levels()
# test_Bs_Ba_()
# test_basis_supports_cell()
# test_slope_over_cell()
test_create_stiffness_matrix()
# test_Gaussian_Quadrature()
# test_create_mass_matrix()
# test_Another_Quadrature_Method()
