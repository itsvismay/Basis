import numpy as np


def check_mode_symmetry(mode_vec, lev):
    minNodeNum = Node.number - len(lev.nodes)
    u_f = [[0 for x in range(len(X))] for y in range(len(Y))]
    for n in lev.nodes:
        p1 = np.array([ mode_vec[2*(n.id-minNodeNum)], mode_vec[2*(n.id-minNodeNum)+1], 0 ])
        u_f[n.point[0]][n.point[1]] = np.linalg.norm(p1)

    U = np.matrix(u_f)
    print(np.allclose(U, U.T, atol=1e-2))
