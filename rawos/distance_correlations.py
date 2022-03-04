

import numpy as np

import os
import csv

import gtda

import matplotlib.pyplot as plt

import metrics
from sklearn.metrics import pairwise_distances
import scipy
from scipy import special
from scipy.stats import pearsonr
import skbio


def jensenshannon_distance(filename1, filename2):
    """Computes the Jensen-Shanon distance of two embeddings"""
    
    X= np.loadtxt(filename1, delimiter="\t")
    diamx = metrics.diameter(X)
    X= X/diamx
    
    Y= np.loadtxt(filename2, delimiter="\t")
    diamy = metrics.diameter(Y)
    Y= Y/diamy
    
    dX = pairwise_distances(X, metric='euclidean')
    dY = pairwise_distances(Y, metric='euclidean')

    DX = []
    DY = []
    for i in range(X.shape[0]): 
        for j in range(i+1, X.shape[0]): 
            DX.append(dX[i,j])


    for i in range(Y.shape[0]): 
        for j in range(i+1, Y.shape[0]): 
            DY.append(dY[i,j])

    DistX, binsX, patches = plt.hist(DX, bins = 100)
    #plt.show() 

    DistY, binsY, patches = plt.hist(DY, bins = 100)
    #plt.show()
    ## uses base e, upper bound of JSD is ln(2)
    ## in base b upper boud of JSD is log_b(2)
    return  scipy.spatial.distance.jensenshannon(DistY, DistX, base=None) 


def distance_correlation(filename1, filename2):
	""" computes  the correlation between the distance matrices of two point clouds
		by only considering the upper triangle part of the matrices in order to avoid redundant values and the zeros on the diagonal
    /!\ for this to make sense we need the point clouds to come from the same graphs and be ordered the same way!!
        --> otherwise use mantel or dcov test
	"""

    X= np.loadtxt(filename1, delimiter="\t")
    diamx = metrics.diameter(X)
    X= X/diamx

    Y= np.loadtxt(filename2, delimiter="\t")
    diamy = metrics.diameter(Y)
    Y= Y/diamy

    dX = pairwise_distances(X, metric='euclidean')
    dY = pairwise_distances(Y, metric='euclidean')

    DX = []
    DY = []
    for i in range(X.shape[0]): 
        for j in range(i+1, X.shape[0]): 
            DX.append(dX[i,j])
            DY.append(dY[i,j])

    #plt.scatter(DX, DY)
   	#plt.show()
    
    corr, _ = pearsonr(DX, DY)
    print('Pearsons correlation: %.3f' % corr)
    
    return corr 



def Mantel_test(filename1, filename2, method = 'pearson', permutations = 100): 
    """Computes the correlation of two distance matrices of same size via the Mantel test
        permutations = number of permutations performed for the test 
        method = 'pearson' or 'spearman' corrrelation coefficient
    """
    X= np.loadtxt(filename1, delimiter="\t")
    diamx = metrics.diameter(X)
    X= X/diamx

    Y= np.loadtxt(filename2, delimiter="\t")
    diamy = metrics.diameter(Y)
    Y= Y/diamy

    dX = pairwise_distances(X, metric='euclidean')
    dY = pairwise_distances(Y, metric='euclidean')
    
    ## make sure they are symmetric because package seems sensitive to slight assymetries.. 
    dX = (dX+dX.T)/2
    dY = (dY+dY.T)/2
    ## fit the package fomrmat of Distance 
    DDX= skbio.stats.distance.DissimilarityMatrix(dX)
    DDY= skbio.stats.distance.DissimilarityMatrix(dY)
    
    Mantel = skbio.stats.distance.mantel(DDX,DDY, method=method, permutations=permutations, alternative='two-sided', strict=True, lookup=None)
    corr = Mantel[0]
    pval = Mantel[1]
    
    return corr, pval
    

