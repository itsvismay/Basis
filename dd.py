#DRAFT 4: Oct 16 2017

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sets import Set
import numpy as np
import random

dom = ((0,0),(5,5))

X=np.arange(dom[0][0], dom[1][0], 1)#X matrix is 1 bigger than domain
Y=np.arange(dom[0][1], dom[1][1], 1)#Y matrix is 1 bigger than domain
X, Y = np.meshgrid(X, Y)

class Node:
    #class variables
    number = 0
    
    def __init__(self, x, y, l):
        #Instance Variables
        self.point = np.array([x, y, 0]) #location
        self.level = l #hierarchy level
        self.in_elements = set()  #this node is contained in elements
        self.basis = [[0 for i in Y] for j in X]
        self.id = Node.number
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
    
    def area(self):
        #Input: nothing
        #Output: double
        #Use: area of triangle
        return np.linalg.norm(np.cross((self.n1.point - self.n2.point), (self.n1.point - self.n3.point)))*0.5
    
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
    #         #n1 is right angle
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
        
        self.depth = Level.number
        Level.number +=1
    
    #Functions
    def add_elements(self, elemts):
        #Input: a set of unique elements
        #Output: Nothing
        #Use: Adds elements to level
        self.elements|=elemts      
    
    def create_bases(self):
        #Input: Nothing
        #Output: Nothing 
        #Use: creates basis to span elements in this level
        
        #for each node in level
            #for each element in node.in_elements:
                #update node.basis
                
        #for now, do this, later make it not allocate new mem  
        #make it so that as new nodes are created, they get saved into the level
    
        for e in self.elements:
            self.nodes.add(e.n1)
            self.nodes.add(e.n2)
            self.nodes.add(e.n3)
        
        for n in self.nodes:
            n.update_basis()
        
        # for n in nodes:
        #     print(np.array(n.basis))
       
    def split(self):
        #Output: a new level subdivided from old level
        #Input : nothing
        #Use : creates a new, more refined level of bases
        
        lk = Level(self.domain)
        
        #starts iterative split using dict (key = node1, key = node2) => val = in-between node
        for e in self.elements:
            lk.add_elements(e.split(self.splitnodedict))

        return lk
            
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from shapely.geometry import MultiLineString
from matplotlib.path import Path

def plot_path(tris):
    fig, ax = plt.subplots()
    poly = plt.Polygon(tris, ec = "k")
    x,y = zip(*tris)

    ax.scatter(x,y, color="r", alpha = 0.6, zorder = 3, s = 10*10*10)

    plt.axis([dom[0][0], dom[1][0] -1, dom[0][1], dom[1][1] -1])

    major = np.arange(0, dom[1][0]-1, 1)
    ax.set_xticks(major)
    ax.set_yticks(major)
    ax.grid(which='major', alpha=1.0) 
    plt.show()

def plot(Z,c = "red"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf = ax.plot_wireframe(X, Y, Z, color =c)
    ax.scatter(X, Y, Z, color = "blue")

    # Customize the z axis.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a color bar which maps values to colors.
    plt.show()


def plot_tri(tris):
    COLOR = {
        True:  '#6699cc',
        False: '#ffcc33'
        }

    def v_color(ob):
        return COLOR[ob.is_simple]
    def plot_coords(ax, ob):
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, 'o', color='#999999', zorder=1)

    def plot_bounds(ax, ob):
        x, y = zip(*list((p.x, p.y) for p in ob.boundary))
        ax.plot(x, y, 'o', color='#000000', zorder=1)

    def plot_lines(ax, ob):
        for line in ob:
            x, y = line.xy
            ax.plot(x, y, color=v_color(ob), alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

    fig = plt.figure()
    ax = fig.add_subplot(122)
    mline2 = MultiLineString(tris)

    plot_coords(ax, mline2)
    plot_bounds(ax, mline2)
    plot_lines(ax, mline2)

    plt.show()


from scipy.optimize import linprog
from numpy.linalg import solve   
def solve(levels, u_f):
    bases = []
    nodes = []
    for l in levels:
        for n in l.nodes:
            bases.append(np.ravel(n.basis))
            nodes.append(n)

    N = np.transpose(np.matrix(bases)) # domain^2 x # of total nodes
    u = np.ravel(u_f) #domain^2 x 1
    c = np.array([1 for k in range(N.shape[1])])

    res = linprog(c, A_eq = N, b_eq =u, options={"disp": True})
    print(res)

    NodesUsedByLevel = [[] for l in levels]
    for i in range(0,len(res.x)):
        if (res.x[i]>0):
            res.x[i] = 1
            print("used node id: ", nodes[i])
            NodesUsedByLevel[nodes[i].level].append(nodes[i])
            # plot(np.reshape(bases[i], (dom[1][0], dom[1][1])))
        else:
            res.x[i] = 0

    for lev  in NodesUsedByLevel:
        tri = []
        for n in lev:
            print(n.point)  
            tri += [n.point[:2]]
                #tri.append((e.n2.point[:2], e.n1.point[:2], e.n3.point[:2], e.n2.point[:2]))
        plot_path(tri)
    
    # print(u)
    # print(N)

def test():
    hierarchyLevel = 0
    
    #Nodes are created properly
    n1 = Node(dom[0][0], dom[0][1], hierarchyLevel)
    assert(Node.getNumberOfNodes() == 1)
    n2 = Node(dom[1][0] -1, dom[0][1], hierarchyLevel)
    n3 = Node(dom[1][0] -1, dom[1][1] -1, hierarchyLevel)
    assert(Node.getNumberOfNodes() == 3)
    assert(n3.id == 2)
    n4 = Node(dom[0][0], dom[1][1] -1, hierarchyLevel)
    print("OK Node creation")
    
    #Element creation ok
    e1 = Element(n1, n2, n3, hierarchyLevel)
    e2 = Element(n1, n4, n3, hierarchyLevel)
    assert(Element.getNumberOfElements() == 2)
    assert(e2.id == 1)
    assert(not (e1 in n4.in_elements) and (e2 in n4.in_elements))
    assert((e1 in n1.in_elements) and (e2 in n1.in_elements))
    assert(abs(e1.area()-8) < 0.000001)
    print("OK Element creation")
    
    #Level creation ok
    numLevels = 2
    l1 = Level(dom)
    assert(l1.depth == 0)
    l1.add_elements(set([e1]))
    l1.add_elements(set([e2]))
    assert(e1 in l1.elements)
    assert(e2 in l1.elements)
    
    
    l1.create_bases()
    
    l2 = l1.split()
    l2.create_bases() 

    l3 = l2.split()
    l3.create_bases()

    print("OK Level creation")

    #TEST THE SOLVER
    #create the function u(x)
    u_f = [[(0.25*(x-4) - 0.25*y + 1) if x>=y else 0 for y in range(len(X))] for x in range(len(Y))]
    # u_f = [[(-0.25*x+2) if x>=y else 0 for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    # u_f = [[random.randint(0, 5) for y in range(dom[0][1], dom[1][1])] for x in range(dom[0][0], dom[1][0])]
    plot(u_f)

    solve([l1, l2, l3], u_f)

    print("OK Solve")

test()
