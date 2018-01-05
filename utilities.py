import numpy as np
import scipy
import sys, os
sys.path.insert(0, os.getcwd()+"/../libigl/python/")
import pyigl as igl

def dist(p1, p2):
    return np.linalg.norm(p1-p2)

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

def is_sym_pos_def(x):
    # print(sorted(np.linalg.eigvals(x)))
    pd = np.all(np.linalg.eigvals(x) >= -1e-8)
    sym = np.allclose(x, x.T, atol=1e-10)
    return pd and sym

def volume_of_tet(n1, n2, n3, n4):
    x = np.array([n1[0] - n4[0], n2[0] - n4[0], n3[0] - n4[0]])
    y = np.array([n1[1] - n4[1], n2[1] - n4[1], n3[1] - n4[1]])
    z = np.array([n1[2] - n4[2], n2[2] - n4[2], n3[2] - n4[2]])
    # print(n1, n2, n3, n4)
    Dm = np.matrix([x, y, z])
    v = (1.0/6)*np.linalg.det(Dm)
    # print(np.linalg.det(Dm))
    return v

def PointInTriangle(pt, v1, v2, v3):
    #Code from stack overflow here:
    #https://stackoverflow.com/questions/20248076/how-do-i-check-if-a-point-is-inside-a-triangle-on-the-line-is-ok-too
    def sign(p1, p2, p3):
        # print(p1, p2, p3)
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


    def PointInAABB(pt, c1, c2):
        return c2[0] <= pt[0] <= c1[0] and \
                c2[1] <= pt[1] <= c1[1]

    b1 = sign(pt, v1, v2) <= 0
    b2 = sign(pt, v2, v3) <= 0
    b3 = sign(pt, v3, v1) <= 0

    return ((b1 == b2) and (b2 == b3)) and \
        PointInAABB(pt, map(max, v1, v2, v3), map(min, v1, v2, v3))


def general_eig_solve(A, B):
    #pass in A = K matrix, and B = M matrix

    # eigvals, eigvecs = scipy.sparse.linalg.eigs(np.matrix(A),k = 4, M = np.matrix(B))
    eigvals, eigvecs = scipy.linalg.eigh(np.matrix(A), np.matrix(B), overwrite_a = False, overwrite_b = False)
    #eigvals, eigvecs = np.linalg.eig(np.linalg.inv(B)*A)

    # print("Check general eigen problem solution")
    # print(B)
    # print((B.dot(eigvecs)).T.dot(eigvecs))
    # print(eigvals)

    return eigvals, eigvecs

def serialize_mesh(name, V, F):
    igl.writeOBJ(name, igl.eigen.MatrixXd(V), igl.eigen.MatrixXi(F))
