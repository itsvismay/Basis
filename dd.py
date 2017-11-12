#DRAFT 4: Oct 16 2017

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sets import Set
import numpy as np
import random

import plotting

dom = ((0,0),(5,5))

X=np.arange(dom[0][0], dom[1][0], 1)#X matrix is 1 bigger than domain
Y=np.arange(dom[0][1], dom[1][1], 1)#Y matrix is 1 bigger than domain
X, Y = np.meshgrid(X, Y)

class Node:
    #class variables
    number = 0
    level_node_number = 0 #the ith node in this level

    def __init__(self, x, y, l):
        #Instance Variables
        self.point = np.array([x, y, 0]) #location
        self.level = l #hierarchy level
        self.in_elements = set()  #this node is contained in elements
        self.basis = [[0 for i in Y] for j in X]
        self.id = Node.number
        self.mass = 0
        Node.number+=1

    def __str__(self):
        return str(self.id)
    def __repr__(self):
        return str(self.id)

    #Functions
    @staticmethod
    def getNumberOfNodes():
        return Node.number

    def update_basis(self):
        def point_in_element(q, n1, n2, n3):
            p = q - n1
            b = n2 - n1
            c = n3 - n1
            d = b[0]*c[1] - c[0]*b[1]
            wa = 1.0*(p[0]*(b[1] - c[1]) + p[1]*(c[0] - b[0]) + d)/d
            wb = 1.0*(p[0]*c[1] - p[1]*c[0])/d
            wc = 1.0*(p[1]*b[0] - p[0]*b[1])/d
            if(0<= wa and 0<=wb and 0<=wc and wa<=1 and wb<=1 and wc<=1):
                return True
            else:
                return False

        def shape_with(n2, n3):
            n1 = np.array([self.point[0], self.point[1], 1]) #Set the z coord of self to be 1
            plane = np.cross((n1 - n2), (n1 - n3))
            # print("in shape_with")
            # print(self.point)
            # print(n1)
            # print(n2)
            # print(n3)
            for x in range(dom[1][0]):
                for y in range(dom[1][1]):
                    if(point_in_element(np.array([x,y, 0]), self.point, n2, n3)):
                        self.basis[x][y] = -1.0*(plane[0]*(x-n1[0]) + plane[1]*(y-n1[1]))/plane[2] + n1[2]



        for e in self.in_elements:
            if self.id == e.n1.id:
                shape_with(e.n2.point, e.n3.point)
            elif self.id == e.n2.id:
                shape_with(e.n1.point, e.n3.point)
            else:
                shape_with(e.n1.point, e.n2.point)

class Element:
    #class vars
    number = 0

    def __init__(self, n1, n2, n3, l):
        #InstanceVariables
        self.level = l#hierarchy level
        self.n1 = n1#node1
        self.n2 = n2#node2
        self.n3 = n3#node3
        self.splitted = False
        self.id = Element.number
        Element.number +=1

        self.n1.in_elements.add(self)
        self.n2.in_elements.add(self)
        self.n3.in_elements.add(self)

    def __str__(self):
        return str(self.id)
    def __repr__(self):
        return str(self.id) + ", " + str(self.n1.point) + ", " + str(self.n2.point) + ", "+ str(self.n3.point)

    #Functions
    @staticmethod
    def getNumberOfElements():
        return Element.number

    def get_area(self):
        #Input: nothing
        #Output: double
        #Use: area of triangle
        return np.linalg.norm(np.cross((self.n1.point - self.n2.point), (self.n1.point - self.n3.point)))*0.5

    def compute_stiffness(self):
        #https://femmatlab.wordpress.com/2015/04/24/triangle-elements-stiffness-matrix/
        #http://www.civil.egmu.net/wonsiri/fe6.pdf
        J11 = 1.1*self.n1.point[0] - self.n3.point[0]
        J12 =self.n1.point[1] - self.n3.point[1]
        J21 =self.n2.point[0] - self.n3.point[0]
        J22 =self.n2.point[1] - self.n3.point[1]
        J = np.matrix([[J11, J12],[J21, J22]])
        Be = np.matrix([[J22, 0, -1*J12, 0, -1*J22+J12, 0],
                        [0, -1*J21, 0, J11, 0, J21-J11],
                        [-1*J21, J22, J11, -1*J12, J21-J11, -1*J22+J12]])*(1.0/np.linalg.det(J))

        E = 1e+1
        v = 0.35
        D = np.matrix([[1-v, v, 0],
                    [ v, 1-v, 0],
                    [ 0, 0, 0.5 -v]])*(E/(1-v))

        K = (np.transpose(Be)*D*Be)*1*self.get_area()
        # print(self.n1.id, "-", self.n1.point)
        # print(self.n2.id, "-", self.n2.point)
        # print(self.n3.id, "-", self.n3.point)
        # print(K)
        return K



    def split(self, nodedict):
        #Input: dictionary of nodek and nodej and their mid point node
        #Output: returns set of elements
        #Use: splits element into 4 new triangles

        if(self.splitted):
            return set()

        refined_elements = set()

        #Check if midpoint node exists between any edge
        #(prevents re-creation of node with same location at given hierarchy level)

        #Get the three nodes for the outside triangle first
        try:
            pn1 = nodedict[self.n1.id]
        except:
            pn1 = Node(self.n1.point[0], self.n1.point[1], self.level+1)
            nodedict[self.n1.id] = pn1
        try:
            pn2 = nodedict[self.n2.id]
        except:
            pn2 = Node(self.n2.point[0], self.n2.point[1], self.level+1)
            nodedict[self.n2.id] = pn2
        try:
            pn3 = nodedict[self.n3.id]
        except:
            pn3 = Node(self.n3.point[0], self.n3.point[1], self.level+1)
            nodedict[self.n3.id] = pn3

        #Get the midpoint nodes
        try:
            pn1n2 = nodedict[(self.n1.id, self.n2.id)]
        except:
            n1n2 = self.n1.point + (self.n2.point - self.n1.point)/2
            pn1n2 = Node(n1n2[0], n1n2[1], self.level+1)
            nodedict[(self.n1.id, self.n2.id)] = pn1n2
            nodedict[(self.n2.id, self.n1.id)] = pn1n2

        try:
            pn2n3 = nodedict[(self.n2.id, self.n3.id)]
        except:
            n2n3 = self.n2.point + (self.n3.point - self.n2.point)/2
            pn2n3 = Node(n2n3[0], n2n3[1], self.level+1)
            nodedict[(self.n2.id, self.n3.id)] = pn2n3
            nodedict[(self.n3.id, self.n2.id)] = pn2n3

        try:
            pn1n3 = nodedict[(self.n1.id, self.n3.id)]
        except:
            n1n3 = self.n1.point + (self.n3.point - self.n1.point)/2
            pn1n3 = Node(n1n3[0], n1n3[1], self.level+1)
            nodedict[(self.n1.id, self.n3.id)] = pn1n3
            nodedict[(self.n3.id, self.n1.id)] = pn1n3

        refined_elements.add(Element(pn1, pn1n3, pn1n2, self.level+1))
        refined_elements.add(Element(pn2, pn1n2, pn2n3, self.level+1))
        refined_elements.add(Element(pn3, pn1n3, pn2n3, self.level+1))
        refined_elements.add(Element(pn1n3, pn2n3, pn1n2, self.level+1))
        self.splitted = True

        # print("Refined")
        # print(pn1n2.point)
        # print(pn2n3.point)
        # print(pn1n3.point)
        return refined_elements


    # def quadrisect():
    #     set intersection between nodes nx, ny
    #     if another element exists, create quadrisection

    # def create_basis(self):

    #     if((self.n1.point - self.n2.point).dot(self.n1.point - self.n3.point) == 0):
    #         #n1 is right anglehttps://mail.google.com/mail/u/0/#label/DGP/15f6ebcb8bcc860d
    #         self.n1.basis[self.n1.point[0]][self.n1.point[1]] = 1

class Level:
    #class vars
    number = 0

    def __init__(self, domain):
        #Instance Vars
        self.domain = domain
        self.elements = set()
        self.nodes = set()
        self.splitnodedict = {}
        self.K = []
        self.M = []
        self.depth = Level.number
        if(Level.number == 0):
            self.create_level_one()

        Level.number +=1

    #Functions
    def create_level_one(self):
        #Input: nothing
        #Output: Nothing
        #Use: Creates Level 0, only level 0
        n1 = Node(dom[0][0], dom[0][1], Level.number)
        n2 = Node(dom[1][0] -1, dom[0][1], Level.number)
        n3 = Node(dom[1][0] -1, dom[1][1] -1, Level.number)
        n4 = Node(dom[0][0], dom[1][1] -1, Level.number)
        assert(Node.getNumberOfNodes() == 4)
        assert(n3.id == 2)
        print("OK Node creation")

        #Element creation ok
        e1 = Element(n1, n2, n3, Level.number)
        e2 = Element(n1, n4, n3, Level.number)
        assert(Element.getNumberOfElements() == 2)
        assert(e2.id == 1)
        assert(not (e1 in n4.in_elements) and (e2 in n4.in_elements))
        assert((e1 in n1.in_elements) and (e2 in n1.in_elements))
        print("OK Element creation")

        self.add_elements(set([e1]))
        self.add_elements(set([e2]))
        assert(e1 in self.elements)
        assert(e2 in self.elements)

        self.get_mass_matrix()
        self.get_stiffness_matrix()

    def add_elements(self, elemts):
        #Input: a set of unique elements
        #Output: Nothing
        #Use: Adds elements to level, only during level construction

        for e in elemts:
            self.nodes.add(e.n1)
            self.nodes.add(e.n2)
            self.nodes.add(e.n3)

        self.elements|=elemts

    def create_bases(self):
        #Input: Nothing
        #Output: Nothing
        #Use: creates basis to span elements in this level
            #for each node in level
                #for each element in node.in_elements:
                    #update node.basis

        for n in self.nodes:
            n.update_basis()

        # for n in nodes:
        #     print(np.array(n.basis))

    def get_mass_matrix(self):
        if(len(self.M) != 0):
            return self.M

        if(len(self.nodes) == 0):
            print("You haven't created a basis yet")
            return self.M

        for e in self.elements:
            element_mass = 1*(e.get_area()*1)/3.0
            e.n1.mass += element_mass
            e.n3.mass += element_mass
            e.n2.mass += element_mass

        massVec = []
        for n in self.nodes:
            massVec.append(n.mass)
            massVec.append(n.mass)

        self.M = np.diag(massVec)
        # print(self.M)
        return self.M




    def get_stiffness_matrix(self):
        #
        #
        #Use: Indexing the K matrix only works if its created before the level has been split!!
        if(len(self.K) != 0):
            return self.K

        if(len(self.nodes) == 0):
            print("You haven't created a basis yet")
            return

        self.K = np.zeros((2*len(self.nodes), 2*len(self.nodes))) #2*n because 2 dimensions
        offset = Node.number - len(self.nodes)
        for e in self.elements:
            minNodeNum = Node.number - len(self.nodes) #re-indexes nodes to be min = 0 max = len(# nodes_in_level)
            indices = [e.n1.id - minNodeNum,
                        e.n2.id - minNodeNum,
                        e.n3.id - minNodeNum]

            local_k = e.compute_stiffness()
            j = 0
            for r in local_k:
                dfxrdx1 = r.item(0)
                dfxrdy1 = r.item(1)
                dfxrdx2 = r.item(2)
                dfxrdy2 = r.item(3)
                dfxrdx3 = r.item(4)
                dfxrdy3 = r.item(5)

                kj = j%2
                self.K[2*indices[j/2]+kj][2*indices[0]] += dfxrdx1
                self.K[2*indices[j/2]+kj][2*indices[0]+1] += dfxrdy1

                self.K[2*indices[j/2]+kj][2*indices[1]] += dfxrdx2
                self.K[2*indices[j/2]+kj][2*indices[1]+1] += dfxrdy2

                self.K[2*indices[j/2]+kj][2*indices[2]] += dfxrdx3
                self.K[2*indices[j/2]+kj][2*indices[2]+1] += dfxrdy3
                j+=1

        return self.K


    def split(self):
        #Output: a new level subdivided from old level
        #Input : nothing
        #Use : creates a new, more refined level of bases

        lk = Level(self.domain)

        #starts iterative split using dict (key = node1, key = node2) => val = in-between node
        for e in self.elements:
            lk.add_elements(e.split(self.splitnodedict))

        lk.get_mass_matrix()
        lk.get_stiffness_matrix()

        return lk


import scipy
def general_eig_solve(l, A, B):

    v_old = np.empty(2*len(l.nodes))
    minNodeNum = Node.number - len(l.nodes)
    for n in l.nodes:
        # print(n.id,"-", n.point)
        v_old[2*(n.id-minNodeNum)] = n.point[0]
        v_old[2*(n.id-minNodeNum)+1] = n.point[1]
    # print(v_old)
    # eigvals, eigvecs = scipy.sparse.linalg.eigs(np.matrix(A),k = 4, M = np.matrix(B))
    eigvals, eigvecs = scipy.linalg.eigh(np.matrix(A), np.matrix(B), overwrite_a = False, overwrite_b = False)
    #eigvals, eigvecs = np.linalg.eig(np.linalg.inv(B)*A)

    # print("check general eigen problem solution")
    # print(B)
    # print((B.dot(eigvecs)).T.dot(eigvecs))
    # print(eigvals)
    indx = 0
    plotting.plot_bases(v_old, eigvecs[19], eigvals[19], 19)
    indx+=1
    return eigvals, eigvecs

from scipy.optimize import linprog
from numpy.linalg import solve
def solve(levels, u_f, toll):
    bases = []
    nodes = []
    for l in levels:
        for n in l.nodes:
            bases.append(np.ravel(n.basis))
            nodes.append(n)

    N = np.transpose(np.matrix(bases)) # domain^2 x # of total nodes
    u = np.ravel(u_f) #domain^2 x 1
    c = np.array([1 for k in range(N.shape[1])])
    bds = np.array([(-20, None) for k in range(N.shape[1])])
    epsilon = np.array([toll for k in range(2*N.shape[0])])
    bigU = epsilon + np.concatenate((u, -1*u), axis=0)
    bigN = np.concatenate((N, -1*N), axis=0)
    print(epsilon.shape)
    print(bigU.shape)
    print(bigN.shape)

    res = linprog(c, A_ub = bigN, b_ub = bigU, options={"disp": True})
    # print(res)

    NodesUsedByLevel = [[] for l in levels]
    for i in range(0,len(res.x)):
        if (res.x[i]>0):
            res.x[i] = 1
            # print("used node id: ", nodes[i])
            NodesUsedByLevel[nodes[i].level].append(nodes[i])
            # plot(np.reshape(bases[i], (dom[1][0], dom[1][1])))
        else:
            res.x[i] = 0

    count = 0
    for lev  in NodesUsedByLevel:
        tri = []
        for n in lev:
            # print(n.point)
            tri += [n.point[:2]]
            count +=1
                #tri.append((e.n2.point[:2], e.n1.point[:2], e.n3.point[:2], e.n2.point[:2]))
        # plotting.plot_path(tri, dom)
    print("tol, nodes ",toll,", ",count)
    return count
    # print(u)
    # print(N)

import matplotlib.pyplot as plt
def test():
    hierarchyLevel = 0
    #Level creation ok
    numLevels = 2
    l1 = Level(dom)
    assert(l1.depth == 0)

    l1.create_bases()

    l2 = l1.split()
    l2.create_bases()
    #
    l3 = l2.split()
    l3.create_bases()

    print("OK Level creation")

    # #TEST THE SOLVER
    # #create the function u(x)
    # u_f = [[(0.25*(x-4) - 0.25*y + 1) if x>=y else 0 for y in range(len(X))] for x in range(len(Y))]
    # # u_f = [[(-0.25*x+2) if x>=y else 0 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # # u_f = [[random.randint(0, 5) for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # plotting.plot(X, Y, u_f)
    # solve([l1, l2, l3], u_f)
    # print("OK Solve")

    D, V = general_eig_solve(l3, l3.get_stiffness_matrix(), l3.get_mass_matrix())
    minNodeNum = Node.number - len(l3.nodes)
    #ignore the first 3 eigvecs, use the 5th, just because
    u_f = [[0 for x in range(len(X))] for y in range(len(Y))]
    for i in range(19, 20):
        ev = V[:,i]

        for n in l3.nodes:
            p1 = np.array([ ev[2*(n.id-minNodeNum)], ev[2*(n.id-minNodeNum)+1], 0 ])
            u_f[n.point[0]][n.point[1]] = np.linalg.norm(p1)

        #TRASH THIS CODE WHEN DONE WITH Tol vs # Nodes plot
        plotting.plot(X,Y, u_f)
        tolerances = [0.0001, 0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
        y =[]
        for t in tolerances:
            y.append(solve([l1, l2, l3], u_f, t))

        plt.plot(tolerances, y)
        plt.title("Level3-"+str(i))

    plt.show()





test()
