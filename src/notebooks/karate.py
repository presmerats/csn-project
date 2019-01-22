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
    g = igraph.Graph.Famous('Zachary')
    n_node = g.vcount()
    summary(g)

    
    n_train = 10
    n_test = n_node
    testList = range(n_node-n_test, n_node)
    with open('ind.karate.test.index','wb') as f:
        for i in testList:
            f.write(str(i)+"\n")

    # adjacency dict (graph)
    i = 0
    A = g.get_adjlist()
    graphAdj = {}
    for node in A:
        graphAdj[i] = node
        i+=1
    pickleToDisk(graphAdj, "ind.karate.graph")

    # features (allx, x, tx)
    #  no features -> a 0 for each node 
    #   or node degree
    #  into a sparce matrix
    xFeatures = np.identity(n_node)
    
    allx = copy.deepcopy(xFeatures[range(n_node-n_test)])
    x = copy.deepcopy(xFeatures[range(n_train)])
    #tx = copy.deepcopy(xFeatures[range(n_node-n_test, n_node)])
    tx = [xFeatures[i] for i in testList]
    
    allx = csr_matrix(allx)
    #allx = np.transpose(allx)
    pickleToDisk(allx, "ind.karate.allx")
    x = csr_matrix(x)
    #x = np.transpose(x)
    pickleToDisk(x, "ind.karate.x")
    tx = csr_matrix(tx)
    #tx = np.transpose(tx)
    pickleToDisk(tx, "ind.karate.tx")

    print(x.shape)
    print(tx.shape)
    print(allx.shape)
    


    # labels (ally,y , ty)
    prs = list()
    with open('karate_clusters.txt', 'r') as f:
        for t in f:
            ind_clust = int(t.strip())
            aux = np.zeros(4, dtype=int)
            aux[ind_clust] = 1
            prs.append(aux)
    prs = np.array(prs)
    #pprint(prs)

    prs_all = copy.deepcopy(prs[range(n_node-n_test)])
    prs_train = copy.deepcopy(prs[range(n_train)])
    prs_test = [prs[i] for i in testList]
    #testList.extend(validationList)
    #finalList = testList
    #prs_train = [ prs[i] for i in range(prs.shape[0]) if i not in finalList ]

    ally = np.array(prs_all)
    y = np.array(prs_train)  
    ty = np.array(prs_test)


    pickleToDisk(ally, "ind.karate.ally")
    pickleToDisk(y, "ind.karate.y")
    pickleToDisk(ty, "ind.karate.ty")
    
    print(y.shape)
    print(ty.shape)
    print(ally.shape)
    
    

    # run training on those files






if __name__ == "__main__":
    generateData()