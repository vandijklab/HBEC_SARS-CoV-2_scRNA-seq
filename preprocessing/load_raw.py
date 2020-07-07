'''
data loader

ngr.200406
'''

import os, sys, glob, re, math, pickle
import phate,scprep,magic,meld
import graphtools as gt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time,random,datetime
from sklearn import metrics
from sklearn import model_selection
from scipy import sparse
from scipy.stats import mannwhitneyu, tiecorrect, rankdata
from statsmodels.stats.multitest import multipletests
import scanpy as sc
import scvelo as scv
from adjustText import adjust_text
import warnings



# settings
plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['text.usetex']=False
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=0.5
sc.set_figure_params(dpi=300,dpi_save=600,
                     frameon=False,
                     fontsize=9)
plt.rcParams['savefig.dpi']=600
sc.settings.verbosity=2
sc._settings.ScanpyConfig.n_jobs=-1
sns.set_style("ticks")


# reproducibility
rs = np.random.seed(42)

# fps
dfp = '/home/ngr4/project/sccovid/data/'
pfp = '/home/ngr4/project/sccovid/results/'
pdfp = '/home/ngr4/project/sccovid/shared/data/processed/'
sc.settings.figdir = pfp

# loader
data_folders = ['/gpfs/ycga/sequencers/pacbio/gw92/10x/Single_Cell/20200405_cw824/MOCK_HBEC_HCR/outs/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/Single_Cell/20200405_cw824/1dpi_CoV2_HCR/outs/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/Single_Cell/20200405_cw824/2dpi_CoV2_HCR/outs/',
                '/gpfs/ycga/sequencers/pacbio/gw92/10x/Single_Cell/20200405_cw824/3dpi_CoV2_orf1/outs/']
data_folders = [os.path.join(i,'filtered_feature_bc_matrix') for i in data_folders]

files_not_found = []
for i in data_folders :
    if not os.path.exists(i) :
        files_not_found.append(i)
    if not files_not_found == [] :
        print('Folders not found...')
        for j in files_not_found :
            print(j)
        raise IOError('Change path to data')

total = time.time()

if True :
    # first load
    running_cellcount=0
    start = time.time()
    adatas = {}
    for i,folder in enumerate(data_folders) :
        sample_id = os.path.split(folder.split('/outs/')[0])[1]
        print('... storing %s into dict (%d/%d)' % (sample_id,i+1,len(data_folders)))
        adatas[sample_id] = sc.read_10x_mtx(folder)
        running_cellcount+=adatas[sample_id].shape[0]
        print('...     read {} cells; total: {} in {:.2f}-s'.format(adatas[sample_id].shape[0],running_cellcount,time.time()-start))
    batch_names = list(adatas.keys())
    print('\n... concatenating of {}-samples'.format(len(data_folders)))
    adata = adatas[batch_names[0]].concatenate(adatas[batch_names[1]],adatas[batch_names[2]],
                                               adatas[batch_names[3]],
                                               batch_categories = batch_names)
    print('Raw load in {:.2f}-min'.format((time.time() - start)/60))

    if True :
        # save
        adata.write(os.path.join(pdfp,'scv2.h5ad'))
        print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))

    if True :
        print(adata)

    # filter cells/genes, transform
    sc.pp.calculate_qc_metrics(adata,inplace=True)
    mito_genes = adata.var_names.str.startswith('MT-')
    adata.obs['pmito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    print('Ncells=%d have >10percent mt expression' % np.sum(adata.obs['pmito']>0.1))
    print('Ncells=%d have <200 genes expressed' % np.sum(adata.obs['n_genes_by_counts']<200))
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3) # filtering cells gets rid of some genes of interest
    if False:
        adata = adata[adata.obs.pmito <= 0.1, :]
    adata.raw = adata
    sc.pp.normalize_total(adata)
    sc.pp.sqrt(adata)


    if True :
        # save
        adata.write(os.path.join(pdfp,'scv2.h5ad'))
        print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))

    if True :
        # batch effect plot
        tdata = adata[:,[i for i in adata.var_names if 'scv2_orf1-10' not in i]]
        tdata.obs['value']=0
        sc.tl.pca(tdata,n_comps=100)
        sc.pp.neighbors(tdata,n_neighbors=30,n_pcs=100)
        sc.tl.umap(tdata)
        fig,axarr=plt.subplots(1,2,figsize=(8,4))
        pal18=['#ee5264','#565656','#75a3b7','#ffe79e','#fac18a','#f1815f','#ac5861','#62354f','#2d284b','#f4b9b9','#c4bbaf',
           '#f9ebae','#aecef9','#aeb7f9','#f9aeae','#9c9583','#88bb92','#bde4a7','#d6e5e3']
        cmap_sample = {v:pal18[i] for i,v in enumerate(adata.obs['batch'].unique())}

        scprep.plot.scatter2d(tdata.obsm['X_pca'],
                              c=adata.obs['batch'],
                              cmap=cmap_sample,
                              ticks=None,
                              label_prefix='PCA',
                              legend=False,
                              ax=axarr[0],
                              s = 0.1,
                              alpha=0.6,
                              rasterized=True,
                              title='Before batch correction')
        scprep.plot.scatter2d(tdata.obsm['X_umap'],
                              c=adata.obs['batch'],
                              cmap=cmap_sample,
                              ticks=None,
                              label_prefix='UMAP',
                              legend=True,
                              legend_loc=(1.01,0.0),
                              ax=axarr[1],
                              s = 0.1,
                              alpha=0.6,
                              rasterized=True,
                              title='Before batch correction')

        fig.savefig(os.path.join(pfp,'batchEffect_scv2.pdf'),dpi=300,bbox_inches='tight')
        del tdata


print('Done with loading and storing data and plotting batch effect crct\n  (w/o SCV2 gene)')
