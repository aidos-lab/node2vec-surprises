import argparse

import torch
import uuid

import numpy as np

from torch_geometric.nn import Node2Vec
from torch_geometric.utils import erdos_renyi_graph

import os
import csv

import gph
import gtda
from gtda import diagrams
from gtda import homology

import matplotlib.pyplot as plt


def main(args):
	## import files
	d = args.dimension
	n = args.num_walks
	l = args.length
	c = args.context
	max_dim =args.hom_dim

	filename = 'er'
	filename += '-c'+str(c)
	filename += '-d'+ str(d)
	filename += '-l'+str(l)
	filename += '-n'+ str(n)

	## get filenames according to filter
	path = '../rawos'
	files = []
	for i in os.listdir(path):
	    if os.path.isfile(os.path.join(path,i)) and filename in i:
	        files.append(i)
	        
		## load embeddings
	Emb_list= []
	dim = 0
	for i,file in zip(range(len(files)),files): 
	    emb = np.loadtxt(path+'/'+file, delimiter="\t")
	    Emb_list.append(emb)
	    if emb.shape[0] > dim:
	        dim = emb.shape[0]


	Embs = np.zeros((len(Emb_list),dim,5)) 
	for i in range(len(Emb_list)):
	    Embs[i,:,:] = Emb_list[i] 
	##compute PH
	
	ph = gtda.homology.VietorisRipsPersistence(metric='euclidean', max_edge_length=np.inf, homology_dimensions=tuple(range(max_dim+1)), coeff=2, infinity_values=None, n_jobs=None).fit_transform(Embs)

	## compute pairwise distances
	Dist  =gtda.diagrams.PairwiseDistance(metric='wasserstein', metric_params=None, order=None, n_jobs=None).fit_transform(ph, None)

	for i in range(Dist.shape[2]):
	    plt.imshow(Dist[:,:,i])
	    plt.colorbar()
	    plt.title('Wasserstein distance in dimension {}'.format(i))
	    plt.savefig('blabla{}'.format(i))
	    #plt.show()

	for i in range(max_dim+1):
	    dist_filename = filename + '_Wasserstein_dim' + str(i) +'.tsv'

	    np.savetxt(
	            dist_filename,
	            Dist[:,:,0],
	            delimiter='\t',
	            fmt='%.4f'
	        )
		    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--context', type=int, default=5)
    parser.add_argument('-d', '--dimension', type=int, default=5)
    parser.add_argument('-l', '--length', type=int, default=10)
    parser.add_argument('-n', '--num-walks', type=int, default=10)
    parser.add_argument('-q', '--hom-dim', type=int, default=1)

    args = parser.parse_args()


main(args)

