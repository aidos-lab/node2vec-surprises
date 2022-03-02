
import numpy as np

import os
import csv

import gtda
from gtda import diagrams
from gtda import homology
import metrics



def get_dimension(diagrams, dim=0):
    """Get specific dimension of persistence diagram."""
    # This reads a little bit weird because of the idiosyncratic way of
    # handling masks. We first extract all triples in all time steps
    # that match the specified dimension. *Then* we remove the dimension
    # information (since it is spurious) and reshape the array to
    # account for the original number of time steps again. Whew!
    mask = diagrams[..., 2] == dim
    diagrams = diagrams[mask][:, :2]
    return diagrams



def compute_total_persistence(filename, max_dim): 
    """takes as input an embedding in the file called filename 
    and returns a vector of the total persistence in each dim"""
    
    emb = np.loadtxt(filename, delimiter="\t")
    diam = metrics.diameter(emb)
    emb= emb/diam 
    ph = gtda.homology.VietorisRipsPersistence(metric='euclidean', max_edge_length=np.inf, homology_dimensions=tuple(range(max_dim+1)), coeff=2, infinity_values=None, n_jobs=None).fit_transform([emb])
    
    total_pers=np.zeros((max_dim+1,1))
    for i in range(max_dim+1): 
        dgm_i= get_dimension(ph[0,:,:],i)
        total_pers[i]= total_persistence(dgm_i)
        
    return total_pers
    
    