import numpy as np


def check_mode_symmetry(mode_vec, lev):
    minNodeNum = Node.number - len(lev.nodes)
    u_f = [[0 for x in range(len(X))] for y in range(len(Y))]
    for n in lev.nodes:
        p1 = np.array([ mode_vec[2*(n.id-minNodeNum)], mode_vec[2*(n.id-minNodeNum)+1], 0 ])
        u_f[n.point[0]][n.point[1]] = np.linalg.norm(p1)

    U = np.matrix(u_f)
    print(np.allclose(U, U.T, atol=1e-2))

def is_invertible(H):
    return H.shape[0]==H.shape[1] and np.linalg.matrix_rank(H) == H.shape[0]

def is_pos_def(x):
    # print(sorted(np.linalg.eigvals(x)))
    return np.all(np.linalg.eigvals(x) >= 0)

def volume_of_tet(n1, n2, n3, n4):
    x = np.array([n1[0] - n4[0], n2[0] - n4[0], n3[0] - n4[0]])
    y = np.array([n1[1] - n4[1], n2[1] - n4[1], n3[1] - n4[1]])
    z = np.array([n1[2] - n4[2], n2[2] - n4[2], n3[2] - n4[2]])

    Dm = np.matrix([x, y, z])
    v = (1.0/6)*np.linalg.det(Dm)
    return v
