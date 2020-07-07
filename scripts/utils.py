import os
import time
import datetime
import glob
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scprep
import graphtools as gt
import phate
from scipy import sparse
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import mannwhitneyu, tiecorrect, rankdata
from statsmodels.stats.multitest import multipletests
import warnings
from adjustText import adjust_text
import sys

# manifold learning
def adata_phate(adata):
    # compute PHATE
    G = gt.Graph(data=adata.obsp['connectivities']+sparse.diags([1]*adata.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True)
    G.knn_max = None

    phate_op = phate.PHATE(knn_dist='precomputed',
                           gamma=0,
                           n_jobs=-1,
                           random_state=42)
    adata.obsm['X_phate']=phate_op.fit_transform(G.K)

    return adata


# if True :
#     # MELD
#     adata.obs['res_sca1']=[1 if i=='SCA1' else -1 for i in adata.obs['genotype']]
#     adata.obs['ees_sca1']=meld.MELD().fit_transform(G=G,RES=adata.obs['res_sca1'])
#     adata.obs['ees_sca1']=adata.obs['ees_sca1']-adata.obs['ees_sca1'].mean() # mean center
#     if True :
#         # save adata obj with batch correction
#         adata.write(os.path.join(pdfp,'mouse_200614.h5ad'))
#         print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))

# if False :
#     # MAGIC
#     magic_op=magic.MAGIC().fit(X=adata.X,graph=G) # running fit_transform produces wrong shape
#     adata.layers['imputed_bbknn']=magic_op.transform(adata.X,genes='all_genes')
# #         adata.layers['imputed_bbknn']=sparse.csr_matrix(magic_op.transform(adata.X,genes='all_genes')) # causes memory spike

#     if True :
#         # save adata obj with batch correction & imputation
#         adata.write(os.path.join(pdfp,'mouse_200614.h5ad'))
#         print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))


# data loading
def load_adata(fname,backed=None) :
    """Load adata object.

    Arguments:
        fname (str): full filename with filepath

    Returns:
        (AnnData)

    """
    start = time.time()
    adata = sc.read_h5ad(filename=fname,backed=backed)
    print('loaded @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    print('took {:.2f}-s to load data'.format(time.time()-start))
    return adata

def write_adata(fname,AnnData) :
    """Write existing AnnData to h5ad

    Arguments:
        fname (str): fpath + fname to write to. INCLUDE .h5ad
        AnnData (AnnData): adata object to save
    """
    start = time.time()
    AnnData.write(fname)
    print('saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    print('took {:.2f}-s to save data'.format(time.time()-start))

def loadpkl(fname):
    """Load pickle from file.

    Arguments:
        fname (str): fpath + fname of pickle to load
    """
    with open(filename, 'rb') as f:
        datapkl = pickle.load(f)
    f.close()
    return datapkl

# # load looms
# if False:
#     # first load,
#     ## read in looms
#     ### point to directory where ./sample1 ./sample2, etc. exists and ./sample/*loom exists
#     loom_files = glob.glob(os.path.join(dfp,'*/*.loom'))
#     sample_names = [os.path.split(os.path.split(loom_files[i])[0])[1] for i in range(len(loom_files))]

#     adata_looms = {}
#     for i in range(len(loom_files)):
#         start = time.time()
#         if i == 0:
#             adata_loom = scv.utils.merge(scv.read_loom(loom_files[i],sparse=True,cleanup=True), adata)
#         else:
#             adata_looms[sample_names[i]] = scv.utils.merge(scv.read_loom(loom_files[i],sparse=True,cleanup=True), adata)
#     try:
#         adata_loom = adata_loom.concatenate(*adata_looms.values(), batch_categories=sample_names)
#     except ValueError:
#         adata_loom = adata_loom.concatenate(*adata_looms.values(), batch_categories=sample_names)

# plotting
def phateumap(X,plot=None,recalculate=False,save=None,title=None,bbknn=True,cluster='batch',cmap=None) :
    """Plot or recalculate then plot

    Args:
        X (AnnData): subsetted AnnData object
        plot (ax object): optional. give ax object to plot in multiple for loop
        save (str): optional. Save the plot with the full filepath indicated, otherwise return ax
    """
    if recalculate :
        # umap/louvain based off batch-balanced graph
        sc.tl.pca(X,n_comps=100)
        if bbknn:
            sc.external.pp.bbknn(X,batch_key='batch')
        else :
            sc.pp.neighbors(X)
        sc.tl.umap(X)
        if False:
            # louvain not working
            sc.tl.louvain(X,resolution=0.5)
        else:
            sc.tl.leiden(X)

        # compute PHATE
        G = gt.Graph(data=X.uns['neighbors']['connectivities']+sparse.diags([1]*X.shape[0],format='csr'),
                     precomputed='adjacency',
                     use_pygsp=True)
        G.knn_max = None

        phate_op = phate.PHATE(knn_dist='precomputed',
                               gamma=0,
                               n_jobs=-1)
        X.obsm['X_phate']=phate_op.fit_transform(G.K)

    if plot is not None :
        if not isinstance(plot,plt.Axes) :
            fig,ax=plt.subplots(1,2,figsize=(8,3))
        else :
            ax = plot

        if cluster=='louvain' :
            color = X.obs[cluster].astype(int)
        else :
            color = X.obs[cluster]

        if cmap is None :
            if cluster=='louvain':
                cmap = sns.color_palette('colorblind',len(X.obs[cluster].unique()))
                cmap = {v:cmap[i] for i,v in enumerate(np.unique(X.obs[cluster].astype(int)))}
            cmap = sns.color_palette('colorblind',len(X.obs['batch'].unique())) # problematic if n_batch > 12

        # pt_size
#         s = 10*X.shape[0]*4.8602673147023086e-06 # based on s=0.2 for N=hdata.shape[0]


        scprep.plot.muttter2d(X.obsm['X_umap'],
                      c=color,
                      cmap=cmap,
                      ticks=None,
                      label_prefix='UMAP',
                      legend=False,
                      ax=ax[0],
#                       s = s,
                      alpha=0.6,
                      title=title,
                      rasterized=True)
        scprep.plot.muttter2d(X.obsm['X_phate'],
                              c=color,
                              cmap=cmap,
                              ticks=None,
                              label_prefix='PHATE',
                              legend=True,
                              legend_loc=(1.01,0.1),
                              ax=ax[1],
#                               s = s,
                              alpha=0.6,
                              title=title,
                              rasterized=True)
        if save is not None :
            if '.pdf' in save :
                fig.savefig(save,dpi=300, bbox_inches='tight')
            else :
                fig.savefig(save+'.pdf',dpi=300, bbox_inches='tight')

    return X


# DGE
def marker_check(markers, AnnData, verbose=True):
    """Check that the maker annotation is in the dataset

    TODO (before distribution): use ensembl ids instead of names

    Arguments:
        markers (dict): keys=cell type, values are markers associated
    """
    missing = {}
    for k,v in markers.items():
        missing_genes = []
        for g in v:
            if g not in AnnData.var_names:
                missing_genes.append(g)
        if len(missing_genes) != 0:
            missing[k] = missing_genes

    if verbose:
        for ctype in missing.keys():
            print('{} has missing genes:'.format(ctype))
            for g in missing[ctype]:
                print('  {}'.format(g))
    if len(missing) == 0:
        print('All markers in data.')
    return missing

def similarity_chk(marker, AnnData):
    print([g for g in AnnData.var_names if marker.lower() in g.lower()])

def get_name(ensembleid, AnnData):
    adata.var.loc[adata.var['gene_ids']==ensembleid,:].index.to_list()[0] # assumes unique

def chk_overlap(markers, verbose=True):
    overlap = {}
    genes = [g for k,v in markers.items() for g in v]
    g = pd.Series(genes)
    g = g[g.duplicated()].to_list()
    ctype = []
    for i in g:
        temp = []
        for k,v in markers.items():
            for gene in v:
                if gene==i:
                    temp.append(k)
        ctype.append(temp)

    if verbose:
        print('\nFollowing genes are markers in more than one cell type:')
        for gene,c in zip(g,ctype):
            print('  {} ({})'.format(gene, c))
    return g

def remove_overlap(ctypes, markers, verbose=True):
    """Get set difference between overlapping markers.

    Once you have these, can go back and remove them in
    marker list.

    Arguments:
        ctypes (list)

    Returns:
        (dict): new markers for those
    """
    new_markers = {}
    for ctype in ctypes:
        comparison_set = set()
        other = [i for i in ctypes if i!=ctype]
        for i in other:
            comparison_set = comparison_set | set(markers[i]) # set(cell_markers['Neutrophil']) - set(cell_markers['Macrophage'])  - set(cell_markers['Monocyte'])
        new_markers[ctype] = list(set(markers[ctype]) - comparison_set)
    if verbose:
        print('\nNew markers removing overlap between {}'.format(ctypes))
        for k in new_markers.keys():
            print('  {}: {}'.format(k, new_markers[k]))
    return new_markers


def mwu(X,Y,gene_names,correction=None,debug=False,verbose=False) :
    '''
    Benjamini-Hochberg correction implemented. Can change to Bonferonni

    gene_names (list)
    if X,Y single gene expression array, input x.reshape(-1,1), y.reshape(-1,1)

    NOTE: get zeros sometimes because difference (p-value is so small)
    '''
    p=pd.DataFrame()
    print('starting Mann-Whitney U w/Benjamini/Hochberg correction...\n')
    start = time.time()
    for i,g in enumerate(gene_names) :
        if i==np.round(np.quantile(np.arange(len(gene_names)),0.25)) :
            print('... 25% completed in {:.2f}-s'.format(time.time()-start))
        elif i==np.round(np.quantile(np.arange(len(gene_names)),0.5)) :
            print('... 50% completed in {:.2f}-s'.format(time.time()-start))
        elif i==np.round(np.quantile(np.arange(len(gene_names)),0.75)) :
            print('... 75% completed in {:.2f}-s'.format(time.time()-start))
        p.loc[i,'Gene']=g
        if (tiecorrect(rankdata(np.concatenate((np.asarray(X[:,i]),np.asarray(Y[:,i])))))==0) :
            if debug :
                print('P-value not calculable for {}'.format(g))
            p.loc[i,'pval']=np.nan
        else :
            _,p.loc[i,'pval']=mannwhitneyu(X[:,i],Y[:,i]) # continuity correction is True
    print('\n... mwu computed in {:.2f}-s\n'.format(time.time() - start))
    if True :
        # ignore NaNs, since can't do a comparison on these (change numbers for correction)
        p_corrected = p.loc[p['pval'].notna(),:]
        if p['pval'].isna().any():
            if verbose:
                print('Following genes had NA p-val:')
                for gene in p['Gene'][p['pval'].isna()]:
                    print('  %s' % gene)
    else :
        p_corrected = p
    new_pvals = multipletests(p_corrected['pval'],method='fdr_bh')
    p_corrected['pval_corrected'] = new_pvals[1]
    return p_corrected

def log2aveFC(X,Y,gene_names,AnnData=None) :
    '''not sensitivity to directionality due to subtraction

    X and Y full arrays, subsetting performed here

    `gene_names` (list): reduced list of genes to calc

    `adata` (sc.AnnData): to calculate reduced list. NOTE: assumes X,Y drawn from adata.var_names
    '''
    if not AnnData is None :
        g_idx = [i for i,g in enumerate(AnnData.var_names) if g in gene_names]
        fc=pd.DataFrame({'Gene':AnnData.var_names[g_idx],
                         'log2FC':np.log2(X[:,g_idx].mean(axis=0)) - np.log2(Y[:,g_idx].mean(axis=0))}) # returns NaN if negative value
    else :
        fc=pd.DataFrame({'Gene':gene_names,
                         'log2FC':np.log2(X.mean(axis=0)) - np.log2(Y.mean(axis=0))})
    return fc
