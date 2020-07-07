import pandas as pd
import os
import glob
import pickle
import phate
import scprep
import meld
import graphtools as gt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import datetime
import scanpy as sc
from sklearn.decomposition import PCA
from py_pcha import PCHA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import sparse
import random
import sys
sys.path.append('')
import utils # add utils fpath

pdfp = '/home/*/project/*/data/processed'
pfp = '/home/*/project/*/results/'

adata = utils.load_adata(os.path.join(pdfp,'hbec.h5ad'))


# cluster genes
random_genes = False

if random_genes:
    genes = adata.var_names.to_list()
    genes = random.sample(genes, 10)
else:
    genes = adata.var_names.to_list()

# cluster genes

print('Aggregating data')
gdata = pd.DataFrame()
tic = time.time()
X = pd.DataFrame()
X['Condition'] = adata.obs['Condition'].to_list()
X['Infected'] = adata.obs['scv2_10+'].map({1:'Infected',0:'Bystander'}).to_list()
X.loc[X['Condition']=='Mock','Infected'] = 'Mock'
X['Inferred time'] = adata.obs['ees_t'].to_list()
for j,gene in enumerate(genes):
    if j % 100 == 0:
        iter_left = len(genes) - (j+1)
        p_left=100*(j+1)/len(genes)
        toc = time.time()-tic
        print('  data aggregated for {:.1f}-% genes\tin {:.2f}-s\t~{:.2f}-min remain'.format(p_left,toc,((toc/(j+1))*iter_left)/60))

    X[gene] = np.asarray(adata[:,gene].layers['imputed_bbknn']).flatten()


    # DREMI-plots
    nbins = 20
    norm = True

    x=X.loc[(X['Infected']=='Infected') | (X['Infected']=='Mock'), 'Inferred time']
    y=X.loc[(X['Infected']=='Infected') | (X['Infected']=='Mock'), gene]
    H, x_edges, y_edges = np.histogram2d(x, y,
                                     bins=nbins, density=False,
                                     range=[[np.quantile(x, q=0.0275),
                                             np.quantile(x, q=0.975)],
                                            [0,
                                             np.quantile(y, q=0.99)]])
    if norm:
        H = H / H.sum(axis=0)
        H[np.isnan(H)] = 0

    inf = np.reshape(H,-1)

    x=X.loc[(X['Infected']=='Bystander') | (X['Infected']=='Mock'), 'Inferred time']
    y=X.loc[(X['Infected']=='Bystander') | (X['Infected']=='Mock'), gene]
    H, x_edges, y_edges = np.histogram2d(x, y,
                                     bins=nbins, density=False,
                                     range=[[np.quantile(x, q=0.0275),
                                             np.quantile(x, q=0.975)],
                                            [0,
                                             np.quantile(y, q=0.99)]])
    if norm:
        H = H / H.sum(axis=0)
        H[np.isnan(H)] = 0

    uninf = np.reshape(H,-1)

    dt = pd.DataFrame(np.append(inf,uninf))
    dt = dt.T
    dt['gene'] = gene

    gdata = gdata.append(dt, ignore_index=False)

    X = X.drop(columns=gene)

if False:
    # save
    gdata.to_csv(os.path.join(pdfp,'gdynamics_concat.csv'))
