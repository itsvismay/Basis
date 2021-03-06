#DRAFT 4: Oct 16 2017

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sets import Set
import numpy as np
import random

import plotting
import utilities as util
import global_variables as GV


class Node:
    #class variables
    number = 0
    level_node_number = 0 #the ith node in this level

    def __init__(self, x, y, l):
        #Instance Variables
        self.point = np.array([x, y, 0]) #location
        self.level = l #hierarchy level
        self.in_elements = set()  #this node is contained in elements
        self.basis_functions = []
        self.id = Node.number
        self.mass = 0
        self.active = False
        self.slope_over_element={}
        self.basis_over_points={}
        Node.number+=1

    def __str__(self):
        return str(self.id)+" "+str(self.point)
    def __repr__(self):
        return str(self.id)+", "+str(self.point)

    #Functions
    @staticmethod
    def getNumberOfNodes():
        return Node.number

    def point_in_element(self, q, n1, n2, n3):
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


    def basis_embedded_onto_fine_mesh_domain(self, fine_level):
        b = []
        for n in sorted(list(fine_level.nodes), key=lambda x:x.id):
            # print(self)
            # print(n)
            r = self.basis(n.point[0], n.point[1])[0]
            # print("     ", r)
            b.append(r)
            # print("")
        return b



    def basis(self, x, y):
        if((round(x,2),round(y,2)) in self.basis_over_points):
            return self.basis_over_points[(round(x, 2), round(y,2))]

        supports_point = False
        for e in self.in_elements:
            if(self.point_in_element(np.array([x, y, 0]), e.n1.point, e.n2.point, e.n3.point)):
                supports_point = True

        if not supports_point:
            self.basis_over_points[(round(x, 2), round(y,2))] = (0, False)
            return 0, False

        minvalue = 123456
        for i in range(len(self.basis_functions)):
            v, n1, normal = self.basis_functions[i](x, y)
            # print("     ",v, normal, n1)
            if(v<=minvalue):
                minvalue = v

        if(minvalue==123456):
            self.basis_over_points[(round(x, 2), round(y,2))] = (0, True)
            return 0, True
        else:
            self.basis_over_points[(round(x, 2), round(y,2))] = (minvalue, True)
            return minvalue, True


    def update_basis(self):
        def shape_with(n2, n3, eid):
            #Creates basis function and adds it to basis_functions list
            n1 = np.array([self.point[0], self.point[1], 1]) #Set the z coord of self to be 1
            normal = np.cross((n1 - n2), (n1 - n3))
            x_slope = -1.0*normal[0]/(normal[2])#/(2.0*e.get_area())
            y_slope = -1.0*normal[1]/(normal[2])#/(2.0*e.get_area())
            self.slope_over_element[eid] = (x_slope, y_slope)
            self.basis_functions.append(lambda x,y,normal=normal,n1=n1: (-1.0*(normal[0]*(x-n1[0]) + normal[1]*(y-n1[1]))/normal[2] + n1[2], n1, normal))

            self.basis_over_points[(round(n1[0],2), round(n1[1],2))] = (1, True)
            self.basis_over_points[(round(n2[0],2), round(n2[1],2))] = (0, True)
            self.basis_over_points[(round(n3[0],2), round(n3[1],2))] = (0, True)


        for e in self.in_elements:
            if self.id == e.n1.id:
                shape_with(e.n2.point, e.n3.point, e.id)
            elif self.id == e.n2.id:
                shape_with(e.n1.point, e.n3.point, e.id)
            else:
                shape_with(e.n1.point, e.n2.point, e.id)

class Element:
    #class vars
    number = 0

    def __init__(self, n1, n2, n3, l, ancestorElement = None):
        #InstanceVariables
        self.level = l#hierarchy level
        node_list = sorted([n1, n2, n3], key=lambda x:x.id)
        self.n1 = node_list[0]#node1
        self.n2 = node_list[1]#node2
        self.n3 = node_list[2]#node3
        self.splitted = False
        self.active = False
        self.id = Element.number
        self.ancestor = ancestorElement
        Element.number +=1

        self.n1.in_elements.add(self)
        self.n2.in_elements.add(self)
        self.n3.in_elements.add(self)

    def __str__(self):
        return "str elem: "+str(self.id)+", "+str(self.n1)+" "+str(self.n2)+"  "+str(self.n3)
    def __repr__(self):
            return "repr: "+str(self.id)+",\n    nodes "+str(self.n1)+"\n   "+str(self.n2)+"\n      "+str(self.n3)+"\n"

    #Functions
    @staticmethod
    def getNumberOfElements():
        return Element.number

    def standardized_shape_matrix(self):
        #Used for Change of Coordinates to the standard triangle
        return np.matrix([[1, 0], [0, 1]])

    def reference_shape_matrix(self):
        return np.matrix([[self.n3.point[0] - self.n1.point[0], self.n2.point[0] - self.n1.point[0]], \
                            [self.n3.point[1] - self.n1.point[1], self.n2.point[1] - self.n1.point[1]]])

    def get_centroid(self):
        centroid_x = (self.n1.point[0] + self.n2.point[0] + self.n3.point[0])/3.0
        centroid_y = (self.n1.point[1] + self.n2.point[1] + self.n3.point[1])/3.0
        return np.array([centroid_x, centroid_y])

    def get_area(self):
        #Input: nothing
        #Output: double
        #Use: area of triangle
        return np.linalg.norm(np.cross((self.n1.point - self.n2.point), (self.n1.point - self.n3.point)))*0.5

    def compute_stiffness(self):
        #Reference Links
        #https://femmatlab.wordpress.com/2015/04/24/triangle-elements-stiffness-matrix/
        #http://www.civil.egmu.net/wonsiri/fe6.pdf
        J11 = self.n1.point[0] - self.n3.point[0]
        J12 =self.n1.point[1] - self.n3.point[1]
        J21 =self.n2.point[0] - self.n3.point[0]
        J22 =self.n2.point[1] - self.n3.point[1]
        J = np.matrix([[J11, J12],[J21, J22]])
        Be = np.matrix([[J22, 0, -1*J12, 0, -1*J22+J12, 0],
                        [0, -1*J21, 0, J11, 0, J21-J11],
                        [-1*J21, J22, J11, -1*J12, J21-J11, -1*J22+J12]])*(1.0/np.linalg.det(J))


        D = np.matrix([[1-GV.Global_Poissons, GV.Global_Poissons, 0],
                        [ GV.Global_Poissons, 1-GV.Global_Poissons, 0],
                        [ 0, 0, 0.5 -GV.Global_Poissons]])*(GV.Global_Youngs/((1+GV.Global_Poissons)*(1-2*GV.Global_Poissons)))

        t = 1 #thickness of element
        K = (np.transpose(Be)*D*Be)*t*self.get_area()
        # print(self.n1.id, "-", self.n1.point)
        # print(self.n2.id, "-", self.n2.point)
        # print(self.n3.id, "-", self.n3.point)
        # print("E: ", self.id, "Nodes ", self.n1.id, self.n2.id, self.n3.id)
        # print(Be)
        # print(K)
        # [[0.05 0.00 -0.05 0.05 0.00 -0.05]
        #  [0.00 0.00 0.00 0.00 0.00 0.00]
        #  [-0.05 0.00 0.05 -0.05 0.00 0.05]
        #  [0.05 0.00 -0.05 0.05 0.00 -0.05]
        #  [0.00 0.00 0.00 0.00 0.00 0.00]
        #  [-0.05 0.00 0.05 -0.05 0.00 0.05]]

        return K



    def split(self, nodedict):
        #Input: dictionary of nodek and nodej and their mid point node
        #Output: returns set of elements
        #Use: splits element into 4 new triangles

        if(self.splitted):
            return set()

        refined_elements = set()

        #(try-excepts prevents re-creation of node with same location at given hierarchy level)

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

        #Check if midpoint node exists between any edge
        #Get the midpoint nodes
        try:
            pn1n2 = nodedict[(self.n1.id, self.n2.id)]
        except:
            n1n2 = self.n1.point + (self.n2.point - self.n1.point)/2.0
            pn1n2 = Node(n1n2[0], n1n2[1], self.level+1)
            nodedict[(self.n1.id, self.n2.id)] = pn1n2
            nodedict[(self.n2.id, self.n1.id)] = pn1n2

        try:
            pn2n3 = nodedict[(self.n2.id, self.n3.id)]
        except:
            n2n3 = self.n2.point + (self.n3.point - self.n2.point)/2.0
            pn2n3 = Node(n2n3[0], n2n3[1], self.level+1)
            nodedict[(self.n2.id, self.n3.id)] = pn2n3
            nodedict[(self.n3.id, self.n2.id)] = pn2n3

        try:
            pn1n3 = nodedict[(self.n1.id, self.n3.id)]
        except:
            n1n3 = self.n1.point + (self.n3.point - self.n1.point)/2.0
            pn1n3 = Node(n1n3[0], n1n3[1], self.level+1)
            nodedict[(self.n1.id, self.n3.id)] = pn1n3
            nodedict[(self.n3.id, self.n1.id)] = pn1n3

        refined_elements.add(Element(pn1, pn1n3, pn1n2, self.level+1, self))
        refined_elements.add(Element(pn2, pn1n2, pn2n3, self.level+1, self))
        refined_elements.add(Element(pn3, pn1n3, pn2n3, self.level+1, self))
        refined_elements.add(Element(pn1n3, pn2n3, pn1n2, self.level+1, self))
        self.splitted = True

        # print("Refined")
        # print(pn1n2.point)
        # print(pn2n3.point)
        # print(pn1n3.point)
        return refined_elements



class Level:
    #class vars
    number = 0
    domain = ((0, 0), (5, 5)) #default domain

    def __init__(self, d=None, V=None, F=None):
        #Instance Vars
        self.elements = set()
        self.nodes = set()
        self.splitnodedict = {}
        self.K = []
        self.M = []
        self.depth = Level.number

        if(Level.number == 0):
            if d != None:
                Level.domain = d
                self.create_default_level_one()
            else:
                self.create_gingerbread_level_one(V, F)



        Level.number +=1


    #Functions
    def create_gingerbread_level_one(self, V, F):
        if(V is None or F is None):
            print("ERROR: Mesh is non - gingerbread_level_one")
            exit()

        n = []
        for i in range(V.rows()):
            n.append(Node(float(V.row(i)[0]), float(V.row(i)[1]), Level.number))

        e = set()
        for i in range(F.rows()):
            e.add(Element(n[int(F.row(i)[0])], n[int(F.row(i)[1])], n[int(F.row(i)[2])], Level.number))

        self.add_elements(e)
        self.create_bases()

    def create_default_level_one(self):
        #Input: nothing
        #Output: Nothing
        #Use: Creates Level 0, only level 0
        n1 = Node(Level.domain[0][0], Level.domain[0][1], Level.number)
        n2 = Node(Level.domain[1][0] -1, Level.domain[0][1], Level.number)
        n3 = Node(Level.domain[1][0] -1, Level.domain[1][1] -1, Level.number)
        n4 = Node(Level.domain[0][0], Level.domain[1][1] -1, Level.number)
        assert(Node.getNumberOfNodes() == 4)
        assert(n3.id == 2)

        #Element creation ok
        e1 = Element(n1, n2, n3, Level.number)
        e2 = Element(n1, n4, n3, Level.number)
        assert(Element.getNumberOfElements() == 2)
        assert(e2.id == 1)
        assert(not (e1 in n4.in_elements) and (e2 in n4.in_elements))
        assert((e1 in n1.in_elements) and (e2 in n1.in_elements))

        self.add_elements(set([e1]))
        self.add_elements(set([e2]))
        assert(e1 in self.elements)
        assert(e2 in self.elements)

        self.create_bases()
        # self.get_mass_matrix()
        # self.get_stiffness_matrix()

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

        # self.elements

    #basis equation for b1 over element e
    # def basis_value_over_e_at_xy(self, b, e, x, y):
    #     n1 = np.array([e.n1.point[0], e.n1.point[1], b.basis[e.n1.point[0]][e.n1.point[1]]])
    #     n2 = np.array([e.n2.point[0], e.n2.point[1], b.basis[e.n2.point[0]][e.n2.point[1]]])
    #     n3 = np.array([e.n3.point[0], e.n3.point[1], b.basis[e.n3.point[0]][e.n3.point[1]]])
    #     # n1 = np.array([e.n1.point[0], e.n1.point[1], 0])
    #     # n2 = np.array([e.n2.point[0], e.n2.point[1], 0])
    #     # n3 = np.array([e.n3.point[0], e.n3.point[1], 0])
    #     normal = np.cross((n1 - n2), (n1 - n3))
    #
    #     # print(x,y)
    #     z = -1.0*(normal[0]*(x-n1[0]) + normal[1]*(y-n1[1]))/normal[2] + n1[2]
    #     # print(z)
    #     return z
    # def get_mass_matrix(self):
    #     if(len(self.M) != 0):
    #         return self.M
    #
    #     if(len(self.nodes) == 0):
    #         print("You haven't created a basis yet")
    #         return self.M
    #
    #     for e in self.elements:
    #         centroid_x = (e.n1.point[0] + e.n2.point[0] + e.n3.point[0])/3.0
    #         centroid_y = (e.n1.point[1] + e.n2.point[1] + e.n3.point[1])/3.0
    #
    #         # element_mass = 1*(e.get_area()*1)/3.0
    #         e.n1.mass += e.get_area()*self.basis_value_over_e_at_xy(e.n1, e, centroid_x, centroid_y)*self.basis_value_over_e_at_xy(e.n1, e, centroid_x, centroid_y)
    #         e.n2.mass += e.get_area()*self.basis_value_over_e_at_xy(e.n2, e, centroid_x, centroid_y)*self.basis_value_over_e_at_xy(e.n2, e, centroid_x, centroid_y)
    #         e.n3.mass += e.get_area()*self.basis_value_over_e_at_xy(e.n3, e, centroid_x, centroid_y)*self.basis_value_over_e_at_xy(e.n3, e, centroid_x, centroid_y)
    #
    #     massVec = []
    #     for n in sorted(list(self.nodes), key=lambda x:x.id):
    #         # print(n.id)
    #         massVec.append(n.mass)
    #         massVec.append(n.mass)
    #
    #     self.M = np.diag(massVec)
    #     return self.M
    # def get_stiffness_matrix(self):
    #     #
    #     #
    #     #Use: Indexing the K matrix only works if its created before the level has been split!!
    #     if(len(self.K) != 0):
    #         return self.K
    #
    #     if(len(self.nodes) == 0):
    #         print("You haven't created a basis yet")
    #         return
    #
    #     self.K = np.zeros((2*len(self.nodes), 2*len(self.nodes))) #2*n because 2 dimensions
    #     offset = Node.number - len(self.nodes)
    #     for e in self.elements:
    #         minNodeNum = Node.number - len(self.nodes) #re-indexes nodes to be min = 0 max = len(# nodes_in_level)
    #         indices = [e.n1.id - minNodeNum,
    #                     e.n2.id - minNodeNum,
    #                     e.n3.id - minNodeNum]
    #
    #         local_k = e.compute_stiffness()
    #         j = 0
    #         for r in local_k:
    #             dfxrdx1 = r.item(0)
    #             dfxrdy1 = r.item(1)
    #             dfxrdx2 = r.item(2)
    #             dfxrdy2 = r.item(3)
    #             dfxrdx3 = r.item(4)
    #             dfxrdy3 = r.item(5)
    #
    #             kj = j%2
    #             self.K[2*indices[j/2]+kj][2*indices[0]] += dfxrdx1
    #             self.K[2*indices[j/2]+kj][2*indices[0]+1] += dfxrdy1
    #
    #             self.K[2*indices[j/2]+kj][2*indices[1]] += dfxrdx2
    #             self.K[2*indices[j/2]+kj][2*indices[1]+1] += dfxrdy2
    #
    #             self.K[2*indices[j/2]+kj][2*indices[2]] += dfxrdx3
    #             self.K[2*indices[j/2]+kj][2*indices[2]+1] += dfxrdy3
    #             j+=1
    #
    #     return self.K


    def split(self):
        #Output: a new level subdivided from old level
        #Input : nothing
        #Use : creates a new, more refined level of bases

        lk = Level(self.domain)

        #starts iterative split using dict (key = node1, key = node2) => val = in-between node
        for e in sorted(list(self.elements), key=lambda e: e.id):
            lk.add_elements(e.split(self.splitnodedict))

        # print(self.elements)
        # print("")
        # print(lk.elements)
        lk.create_bases()

        # lk.get_mass_matrix()
        # lk.get_stiffness_matrix()

        return lk


from scipy.optimize import linprog
from numpy.linalg import solve
def solve(levels, u_f, toll):
    print("gingerbread")
    bases = []
    nodes = []
    for l in levels:
        for n in sorted(list(l.nodes), key=lambda x:x.id):
            print("gingerbread done", n, len(l.nodes), Node.number)
            n.active = False
            r = n.basis_embedded_onto_fine_mesh_domain(levels[-1])
            bases.append(r)
            nodes.append(n)

    N = np.transpose(np.matrix(bases)) # fine nodes x # of total nodes
    print("Nshape", N.shape)
    u = np.ravel(u_f) #fine nodes x 1
    print("u",len(u))
    c = np.array([1 for k in range(N.shape[1])])
    bds = np.array([(-20, None) for k in range(N.shape[1])])
    epsilon = np.array([toll for k in range(2*N.shape[0])])
    bigU = epsilon + np.concatenate((u, -1*u), axis=0)
    bigN = np.concatenate((N, -1*N), axis=0)


    res = linprog(c, A_ub = bigN, b_ub = bigU, options={"disp": True})

    NodesUsedByLevel = [[] for l in range(Level.number)]
    print(len(res.x), len(nodes))
    for i in range(0,len(res.x)):
        if (res.x[i]>1e-10):
            res.x[i] = 1
            # print("used node id: ", nodes[i])
            nodes[i].active = True
            NodesUsedByLevel[nodes[i].level].append(nodes[i])
            # plot(np.reshape(bases[i], (dom[1][0], dom[1][1])))
        else:
            res.x[i] = 0


    return NodesUsedByLevel


import matplotlib.pyplot as plt
def test():
    dom = ((0,0),(5,5))
    X=np.arange(dom[0][0], dom[1][0], 1)#X matrix is 1 bigger than domain
    Y=np.arange(dom[0][1], dom[1][1], 1)#Y matrix is 1 bigger than domain
    X, Y = np.meshgrid(X, Y)

    l1 = Level(dom)
    l1.create_bases()
    l2 = l1.split()
    l2.create_bases()
    l3 = l2.split()
    l3.create_bases()

    # #TEST THE SOLVER
    # #create the function u(x)
    # u_f = [[(0.25*(x-4) - 0.25*y + 1) if x>=y else 0 for y in range(len(X))] for x in range(len(Y))]
    # u_f = [[(-0.25*x+2) if x>=y else 0 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    #u_f = [[random.randint(0, 5) for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    u_f = [[(x-2)**4 + (y-2)**4 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    plotting.plot(X, Y, u_f)
    NodesUsedByLevel = solve([l1, l2, l3], u_f, 0.0001)

    U = np.matrix(u_f)
    print(np.allclose(U, U.T, atol=1e-2))
    plotting.plot_delaunay_mesh(NodesUsedByLevel)

def run_tolerance_vs_nodes_test():
    dom = ((0,0),(5,5))
    l1 = Level(dom)
    l1.create_bases()
    l2 = l1.split()
    l2.create_bases()
    l3 = l2.split()
    l3.create_bases()
    D, V = general_eig_solve(l3, l3.get_stiffness_matrix(), l3.get_mass_matrix())

    minNodeNum = Node.number - len(l3.nodes)
    #ignore the first 3 eigvecs, use the 5th, just because
    u_f = [[0 for x in range(dom[1][0])] for y in range(dom[1][1])]

    for i in range(2, len(V)):
        ev = V[:,i]
        util.check_mode_symmetry(ev, l3)
        for n in l3.nodes:
            p1 = np.array([ ev[2*(n.id-minNodeNum)], ev[2*(n.id-minNodeNum)+1], 0 ])
            u_f[n.point[0]][n.point[1]] = np.linalg.norm(p1)

        #TRASH THIS CODE WHEN DONE WITH Tol vs # Nodes plot
        tolerances = [0.0001, 0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
        y =[]
        for t in tolerances:
            NodesUsedByLevel = solve([l1, l2, l3], u_f, t)
            count = 0
            for lev  in NodesUsedByLevel:
                count += len(lev)
            plotting.plot_delaunay_mesh(NodesUsedByLevel)
            break

            print("tol, nodes ",t,", ",count)
            y.append(count)

    return


# test()
# run_tolerance_vs_nodes_test()
