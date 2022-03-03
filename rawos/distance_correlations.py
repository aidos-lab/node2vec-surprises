

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


def jensenshannon_distance(filename1, filename2):
    """Computes the Jensen-Shanon distance of two embeddings of the same number of points"""
    
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
    

