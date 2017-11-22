
import dd as ref
import sim as sim

import numpy as np

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



def test2():
    dom = ((0,0), (5,5))
    hMesh = sim.get_hierarchical_mesh(dom)


# test_creation_of_levels()
# test_Bs_Ba_()
# test_basis_supports_cell()
test_slope_over_cell()
