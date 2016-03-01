#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
import math

#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float)
    # Coursework 1 task 1 should be inserted here
    bc = bincount(theData[:, root])
    prior = bc / float(sum(bc))
    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserted here
    for row in theData[:, [varC, varP]]:
        cPT[row[0], row[1]] += 1

    cPT /= sum(cPT, axis=0)
    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here
    for row in theData[:, [varRow, varCol]]:
        jPT[row[0], row[1]] += 1

    jPT /= sum(jPT)
    # end of coursework 1 task 3
    return jPT


#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    aJPT /= sum(aJPT, axis=0)

    # coursework 1 taks 4 ends here
    return aJPT


#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    # Finally complete the function Query which calculates the probability distribution over the root node of a naive
    # Bayesian network.
    # The returned value is a list (or vector) giving the posterior probability distribution over the states of the root
    # node, for example [0.1,0.3,0.4,0.2].
    rootPdf += naiveBayes[0]
    for i in range(len(theQuery)):
        rootPdf *= naiveBayes[i + 1][theQuery[i]]

    rootPdf /= sum(rootPdf)
    # end of coursework 1 task 5
    return rootPdf


#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi = 0.0
    # Coursework 2 task 1 should be inserted here
    # calculate the marginalised distributions
    pX = sum(jP, axis=0)
    pX /= sum(pX)
    pY = sum(jP, axis=1)
    pY /= sum(pY)
       
    # compute the mutual information
    for x in range(size(jP, axis=0)):
        for y in range(size(jP, axis=1)):
            # Condition to catch for a log(0) error
            if jP[x][y] * pX[y] * pY[x] != 0.:
                i = jP[x][y] * math.log(jP[x][y] / (pX[y] * pY[x]), 2)
                mi += i
           
    return mi


#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables, noVariables))
    # Coursework 2 task 2 should be inserted here
    for x in range(noVariables):
        for y in range(x, noVariables):
            jP = JPT(theData, x, y, noStates)
            i = MutualInformation(jP)
            MIMatrix[x][y] = i
            MIMatrix[y][x] = i

        # end of coursework 2 task 2
    return MIMatrix


# Function to compute an ordered list of dependencies
def DependencyList(depMatrix):
    depList = []
    # Coursework 2 task 3 should be inserted here
    for x in range(len(depMatrix)):
        for y in range(x, len(depMatrix)):
            depList.append([depMatrix[x][y], x, y])
    
    depList2 = array(depList)
    depList2 = sort(depList2.view('f8,int,int'), order=['f0'], axis=0).view(float)
    depList2 = depList2[::-1]
    
    # end of coursework 2 task 3
    return array(depList2)


#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []

    # Add arcs in order given, provided resulting structure has no loops
    # A loop is given when we can connect L1 with L2 through the spanning tree
    for arc in depList:
        if(not CausesLoops(spanningTree, arc[1:])):
            spanningTree.append(list(arc[1:]))            

    return array(spanningTree)
    
    
def CausesLoops(spanningTree, proposedArc):
    # If start and end nodes are the same then this is immediately false
    if proposedArc[0] == proposedArc[1]: return True
    
    leadingEdge = [proposedArc[0]]
    visited = list(proposedArc)
    
    tempSTree = list(spanningTree)

    # Find all instances of variables in the leading edge, and expand any connected arcs
    # If there are no more instances of the leading edge in the remaining unexpanded tree, remove from leading edge
    while len(leadingEdge) != 0:
        node = leadingEdge[0]

        iArc = len(tempSTree)
        while iArc > 0:
            iArc -= 1
            arc = tempSTree[iArc]
            if (node in arc):
                newNode = arc[0]
                if (arc[0] == node):
                    newNode = arc[1]

                if (newNode in visited):
                    # We have found an alternative route to our objective, return true
                    return True
                else:
                    # This is a new location, and we should add it to the visited and leading edge lists
                    visited.append(newNode)
                    leadingEdge.append(newNode)
                    tempSTree.remove(arc)
                    

        # Since this is a breadth first search, we are now finished with this node and can remove it from the leading
        # edge

        leadingEdge.remove(node)


    # If we've got here then every path has been explored and we have not found a loop
    return False

#
# End of coursework 2
#


# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child], noStates[parent1], noStates[parent2]], float)
    
    # Coursework 3 task 1 should be inserted here
    
    # The conditional probability of (C | A & B) is given by the counts of C when A and B are in    
    # a particular state
    
    for a, b, c in theData[:, (parent1, parent2, child)]:
        cPT[c, a, b] += 1
        
    cPT /= sum(cPT, axis=0)
    cPT = nan_to_num(cPT)
     
    # End of Coursework 3 task 1
    return cPT


#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0], [1], [2, 0], [3, 2, 1], [4, 3], [5, 3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    #print(cpt3)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList


# Coursework 3 task 2 begins here
def HepCNetwork(theData, noStates):
    arcList = [[0], [1], [7, 1, 0], [6, 1], [4, 1],  [5, 4], [3, 4], [8, 7], [2, 0]]
    
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT_2(theData, 7, 1, 0, noStates)
    cpt3 = CPT(theData, 6, 1, noStates)
    cpt4 = CPT(theData, 4, 1, noStates)
    cpt5 = CPT(theData, 5, 4, noStates)
    cpt6 = CPT(theData, 3, 4, noStates)
    cpt7 = CPT(theData, 8, 7, noStates)
    cpt8 = CPT(theData, 2, 0, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5, cpt6, cpt7, cpt8]
    
    return arcList, cptList
# end of coursework 3 task 2
#


        

# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
    # Coursework 3 task 3 begins here    
    
    #Model size = |Bn| * log(N, 2) / 2
    #N = number of data points
    mdlSize += math.log(noDataPoints, 2) / 2.0
    
    #|Bn| is the sum of parameters needed to represent the data
    #i.e. m-1 for priors and n * (m-1) for link matricies
    B = 0
    for arc in arcList:
        tB = noStates[arc[0]] - 1
        if(len(arc) > 1):
            for node in arc[1:]:
                tB *= noStates[node]
        
        B += tB
        
    mdlSize *= B
    # Coursework 3 task 3 ends here
    return mdlSize


#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
    # Coursework 3 task 4 begins here
    
    # The JPT is the probability of the data given the network
    for arc, cpt in zip(arcList, cptList):    
        t = cpt
        for node in arc:
              t = t[dataPoint[node]]                
        #print(t)
        jP *= t

    # Coursework 3 task 4 ends here
    return jP

#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy = 0
    # Coursework 3 task 5 begins here
    
    for dataPoint in theData:
        mdlAccuracy += math.log(JointProbability(dataPoint, arcList, cptList), 2)

    # Coursework 3 task 5 ends here
    return mdlAccuracy

def MDLScore(theData, arcList, cptList, noStates):
    return MDLSize(arcList, cptList, len(theData), noStates) - MDLAccuracy(theData, arcList, cptList)

# Write a function to find the best scoring network formed by deleting one arc from the spanning tree.
def BestNetwork(theData, arcList, cptList, noStates):
    
    bestScore = MDLScore(theData, arcList, cptList, noStates)
    bestNetwork = list(arcList) # Ensures we make a copy of the arcList
    
    for i in range(len(arcList)):
        arc = arcList[i]
        cpt = cptList[i]
        arcListPrime = list(arcList)
        cptListPrime = list(cptList)  
        # The first element of the list is the child
        for j in range(1, len(arc)):   
                    
                
                arcListPrime[i] = [arc[0]]
                
                if j > 1 :
                    arcListPrime[i] = arcListPrime[i] + arc[1:j]
                
                if j < len(arc) - 1:
                    arcListPrime[i] = arcListPrime[i] + arc[j + 1:]
                   
                cptListPrime[i] = cpt.sum(axis=j)
                cptListPrime[i] /= cptListPrime[i].sum(axis=0)
                
                #if len(arc) == 3:
                    #print('************{0}**************'.format(i))
                    #print(cpt, cpt.sum(axis=j), cptListPrime[i].sum(axis=0))
          
                BIC = MDLScore(theData, arcListPrime, cptListPrime, noStates)

                #print(arcListPrime[i], BIC)
                if BIC < bestScore:
                    bestScore = BIC
                    bestNetwork = list(arcListPrime) # Ensures we make a copy of the arcList
    
#    return bestNetwork
    return bestScore   
   
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables = theData.shape[1]
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables = theData.shape[1]
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar


def CreateEigenfaceFiles(theBasis):
    adummystatement = 0  # delete this when you do the coursework


# Coursework 4 task 3 begins here

# Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)


def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  # delete this when you do the coursework


# Coursework 4 task 5 begins here

# Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending
    # order of their eignevalues magnitudes


    # Coursework 4 task 6 ends here
    return array(orthoPhi)

# main program part for Coursework 2
#
resultsFile ="IDAPIResults03.txt"
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
# arcList, cptList = ExampleBayesianNetwork(theData, noStates)
arcList, cptList = HepCNetwork(theData, noStates)

# Parameters to match examples in slides
# noVariables = 2
# noRoots = 2
# noStates = [2, 2]
# arcList = [[0], [0, 1]]
# arcList = [[0], [1]]

# noDataPoints = 4
# theData = [[0, 0], [1, 0], [1, 1], [1, 1]]
# cptList = [[1/4.0, 3/4.0],[[1, 0], [1/3.0, 2/3.0]]]

# noDataPoints = 8
# theData = [[0, 0], [1, 0], [1, 1], [1, 1], [1, 1], [0, 0], [0, 0], [1, 1]]
# cptList = [[3/8.0, 5/8.0],[[1, 0], [1/5.0, 4/5.0]]]
# cptList = [[3/8.0, 5/8.0],[1/2.0, 1/2.0]]

AppendString(resultsFile,"Coursework Three Results by Peter Efstathiou (ple15)")
AppendString(resultsFile,"") #blank line
AppendString(resultsFile,"The MDLSize of the your network for Hepatitis C data set")
modelSize = MDLSize(arcList, cptList, noDataPoints, noStates)
AppendString(resultsFile, modelSize)
AppendString(resultsFile, "")

AppendString(resultsFile,"The MDLAccuracy of the your network for Hepatitis C data set")
modelAccuracy = MDLAccuracy(theData, arcList, cptList)
AppendString(resultsFile, modelAccuracy)
AppendString(resultsFile, "")

AppendString(resultsFile,"The MDLScore of the your network for Hepatitis C data set")
modelScore = MDLScore(theData, arcList, cptList, noStates)
AppendString(resultsFile, modelScore)
AppendString(resultsFile, "")

AppendString(resultsFile,"The score of your best network with one arc removed")
best = BestNetwork(theData, arcList, cptList, noStates)
AppendString(resultsFile, best)
AppendString(resultsFile, "")
