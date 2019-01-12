# local python tests
from pprint import pprint
from igraph import *
import igraph.test
import pickle
import os
import numpy as np
from scipy.sparse import csr_matrix
import collections
import copy




"""

    %%%%%%%%%%%%%%%%%%%%%%%%%%% gcn tests %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x  feature vectors
    tx feature vectors for test (unlabeled)
    allx feature vectors for all

    y  label vectors       numpy.ndarray
    ty test label vectors  numpy.ndarray
    ally  test label of all

    graph : adjacency matrix (dict of lists)
    test.index: node id list for tests?


    steps
    1) inspect ind.citeseer.x/tx/y/ty/allx/ally/test.index

    2) build those files from code

    3) call learning algorithm
        cd src/notebooks
        python gcn_tests_input.py&&  cp ind.* ../gcn/gcn/gcn/data
        cd src/gcn/gcn/gcn
        python train.py --dataset t1


    4) look at embedding result

"""


def pickleToDisk(myobj, filepath):
    pickle.dump(myobj, open(filepath, "wb"))


def printGraphObject(filepath):


    myobject = os.path.basename(filepath)
    print("\n\n*************** "+myobject+" ********************")
    y = pickle.load(open(filepath,'rb'))
    print(type(y))
    if type(y) ==  type([]) or type(y) == type(np.ndarray([])):
        
        print(len(y))
        pprint(y[:min(3,len(y)-1)])
    elif type(y) == type({}) or type(y)==type(collections.defaultdict()):
        pprint(y.keys()[:min(3,len(y)-1)])
        pprint(y[y.keys()[0]])
        pprint(y[y.keys()[1]])
        pprint(y[y.keys()[20]])
        pprint(y[y.keys()[3000]])

    else:
        
        pprint(y.shape)
        pprint(y[0,:min(3,y.shape[1]-1)].toarray())
        pprint(y[0,0].shape)
        pprint(type(y[0,0]))
        pprint(y[0,0])

def readListFile(filepath):
    myobject = os.path.basename(filepath)
    print("\n\n*************** "+myobject+" ********************")
    

    testInstances =[]
    with open(filepath, 'rb') as f:
        for line in f.readlines():
            testInstances.append(int(line))

    pprint(testInstances[:min(3,len(testInstances)-1)])


def inspectData():

    printGraphObject('../gcn/gcn/gcn/data/ind.citeseer.y')
    printGraphObject('../gcn/gcn/gcn/data/ind.citeseer.ty')
    printGraphObject('../gcn/gcn/gcn/data/ind.citeseer.x')
    printGraphObject('../gcn/gcn/gcn/data/ind.citeseer.tx')
    printGraphObject('../gcn/gcn/gcn/data/ind.citeseer.graph')
    readListFile('../gcn/gcn/gcn/data/ind.citeseer.test.index')
    printGraphObject('../gcn/gcn/gcn/data/ind.citeseer.allx')
    printGraphObject('../gcn/gcn/gcn/data/ind.citeseer.ally')





def generateData():

    #igraph.test.run_tests()

    # generation
    #g = Graph.Tree(127, 2)
    #g = Graph.GRG(100, 0.2)
    #g = Graph.Erdos_Renyi(100, 0.2)
    #g = Graph.Watts_Strogatz(1, 100, 4, 0.5 )
    g = Graph.Barabasi(1700)
    summary(g)

    # graph metrics
    # pprint(g.degree([2,3,6,99]))
    # pprint(g.edge_betweenness())
    # pprint(g.pagerank())
    #pprint(g.get_adjacency())
    #pprint(dir(g))
    


    # test.index
    # list first 5 nodes -> to test.index
    testList = range(30)
    with open('ind.t1.test.index','wb') as f:
        for i in testList:
            f.write(str(i)+"\n")

    # adjacency dict (graph)
    i = 0
    A = g.get_adjlist()
    graphAdj = {}
    for node in A:
        graphAdj[i] = node
        i+=1
    pickleToDisk(graphAdj, "ind.t1.graph")

    # features (allx, x, tx)
    #  no features -> a 0 for each node 
    #   or node degree
    #  into a sparce matrix
    xFeatures = g.degree()
    print(xFeatures[:10])
    # see if any degree is negative
    degs = np.array(xFeatures)
    degs2 = degs < 0
    degs3 = degs2.sum(axis=0)
    print(degs3)
    x = copy.deepcopy(xFeatures)
    tx = [ xFeatures[i] for i in testList]


    # remove test List
    # remove also validation
    for elem in tx:
        x.pop(x.index(elem))

    validationList = range(testList[-1], testList[-1]+500)
    for elindex in validationList:
        x.pop(elindex)

    
    allx = csr_matrix(xFeatures)
    allx = np.transpose(allx)
    pickleToDisk(allx, "ind.t1.allx")
    x = np.transpose(x)
    x = csr_matrix(x)
    pickleToDisk(x, "ind.t1.x")
    tx = csr_matrix(tx)
    tx = np.transpose(tx)
    pickleToDisk(tx, "ind.t1.tx")

    print(x.shape)
    print(allx.shape)



    # labels (ally,y , ty)
    prs = g.pagerank()
    print(prs[:10])
    prs = np.array(prs)
    threshold = 0.01
    prs = prs >= threshold
    prs2 = prs < threshold

    prs = np.vstack((prs,prs2)).T
    pprint(prs)

    prs_train = copy.deepcopy(prs)
    prs_test = [ prs[i] for i in testList]
    testList.extend(validationList)
    finalList = testList
    prs_train = [ prs[i] for i in range(prs.shape[0]) if i not in finalList ]

    ally = np.array(prs)
    y = np.array(prs_train)  
    ty = np.array(prs_test)


    pickleToDisk(ally, "ind.t1.ally")
    pickleToDisk(y, "ind.t1.y")
    pickleToDisk(ty, "ind.t1.ty")

    print(ally.shape)
    print(ty.shape)
    print(y.shape)

    # run training on those files






if __name__ == "__main__":
    generateData()