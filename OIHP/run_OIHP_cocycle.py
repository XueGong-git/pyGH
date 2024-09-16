from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import numpy as np
from numpy import matlib
import gudhi as gd
import networkx as nx
import math
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from GeneralisedFormanRicci.frc import gen_graph
from scipy.sparse import *
from scipy import *
from scipy.io import savemat
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
import matplotlib.transforms as transforms
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
from pyGH.GH import uGH
import multiprocessing as mp 
import os 
import sys 
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
import seaborn as sns

simplefilter("ignore", ClusterWarning)
#import umap.umap_ as umap

def convertpdb(filename):
    f=open(filename, "r")
    if f.mode == 'r':
        contents = f.readlines()
    
    #recordname = []

    #atomNum = []
    atomName = []
    #altLoc = []
    #resName = []

    #chainID = []
    #resNum = []
    X = []
    Y = []
    Z = []

    #occupancy = []
    #betaFactor = []
    element = []
    #charge = []
    
    
    for i in range(len(contents)):
        thisLine = contents[i]

        if thisLine[0:4]=='ATOM' or thisLine[0:6]=='HETATM':
            #recordname = np.append(recordname,thisLine[:6].strip())
            #atomNum = np.append(atomNum, float(thisLine[6:11]))
            atomName = np.append(atomName, thisLine[12:16])
            #altLoc = np.append(altLoc,thisLine[16])
            #resName = np.append(resName, thisLine[17:20].strip())
            #chainID = np.append(chainID, thisLine[21])
            #resNum = np.append(resNum, float(thisLine[23:26]))
            X = np.append(X, float(thisLine[30:38]))
            Y = np.append(Y, float(thisLine[38:46]))
            Z = np.append(Z, float(thisLine[46:54]))
            #occupancy = np.append(occupancy, float(thisLine[55:60]))
            #betaFactor = np.append(betaFactor, float(thisLine[61:66]))
            element = np.append(element,thisLine[12:14])

    #print(atomName)
    a = {'PRO': [{'atom': atomName, 'typ': element, 'pos': np.transpose([X,Y,Z])}]}
    np.savez(filename[:-4]+".npz", **a)

def faces(simplices):
    faceset = set()
    for simplex in simplices:
        numnodes = len(simplex)
        for r in range(numnodes, 0, -1):
            for face in combinations(simplex, r):
                faceset.add(tuple(sorted(face)))
    return faceset

def n_faces(face_set, n):
    return filter(lambda face: len(face)==n+1, face_set)

def boundary_operator(face_set, i):
    source_simplices = list(n_faces(face_set, i))
    target_simplices = list(n_faces(face_set, i-1))
    #print(source_simplices, target_simplices)

    if len(target_simplices)==0:
        S = dok_matrix((1, len(source_simplices)), dtype=np.float64)
        S[0, 0:len(source_simplices)] = 1
    else:
        source_simplices_dict = {source_simplices[j]: j for j in range(len(source_simplices))}
        target_simplices_dict = {target_simplices[i]: i for i in range(len(target_simplices))}

        S = dok_matrix((len(target_simplices), len(source_simplices)), dtype=np.float64)
        for source_simplex in source_simplices:
            for a in range(len(source_simplex)):
                target_simplex = source_simplex[:a]+source_simplex[(a+1):]
                i = target_simplices_dict[target_simplex]
                j = source_simplices_dict[source_simplex]
                S[i, j] = -1 if a % 2==1 else 1
    
    return S

def GHM(all_eigval, all_eigvec):
    # Input: A persistent array of eigenvalues and eigenvectors from filtration process
    # Output: An array of Gromov-Norm Matrix from filtration process
    
    clean_eigvec = [] 
    for f in range(len(all_eigvec)):
        eigvec = all_eigvec[f]

        # eigenvectors are adjusted to ensure that the first element of each eigenvector is positive
        col_idx = (eigvec[0, :] < 0)   # find columns where the first elemet is zero
        eigvec[:, col_idx] = -  eigvec[:, col_idx] 

        #eigvec[np.abs(eigvec)<1e-3] = 0 # Zero the entries due to precision 
        clean_eigvec.append(eigvec)

    gnm = []
    for f in range(len(all_eigval)):
        ll = list(np.where(all_eigval[f]<1e-3)[0]) #list(range(len(all_eigval[f])))#
        v1 = clean_eigvec[f][:, ll] 
        print(str(len(ll)) + " independent cycles")

        dx = np.zeros((len(ll), len(ll))) # Harmonic Norm matrix for structure at f
        if len(v1) > 0:
            for i in range(len(dx)):
                for j in range(0, i):
                    x1, x2 = v1[:, i], v1[:, j]
                    dx[i, j] = np.abs(np.linalg.norm(x1, ord=1)-np.linalg.norm(x2, ord=1))
            dx += np.transpose(np.tril(dx))
        gnm.append(dx)
    return gnm


def _uGH(dist_mat, i, j):
    op = (i,j, uGH(dist_mat[i], dist_mat[j]))
    print(op)#, np.shape(dist_mat[i]), np.shape(dist_mat[j]))
    return op


def calDis(f, shape):
    flist = glob.glob('./data/'+shape+'/*f9[0-9][0-9].txt')
    flist = sorted(flist)
    for ll in range(len(flist)):
        print(flist[ll])
        file = open(flist[ll])
        contents = file.readlines()
        for i in range(len(contents)):
            contents[i] = contents[i].rstrip("\n").split(",")
            contents[i] = [float(s) for s in contents[i]]
        
        # Removing redundant rows
        points_unique = []
        for point in contents:
            if point not in points_unique:
                points_unique.append(point)
        contents = points_unique
        
        all_eigval, all_eigvec = [], []
        
        #all_eigval, all_eigvec = [], []
        #rc = gd.AlphaComplex(coords)
        #simplex_tree = rc.create_simplex_tree()
        #val = list(simplex_tree.get_filtration())
        #print(val)
        alpha = gd.AlphaComplex(contents)
        st = alpha.create_simplex_tree()
        
        #rips_complex = gd.RipsComplex(points=contents, max_edge_length=4.3)
        #st = rips_complex.create_simplex_tree(max_dimension=2)
        
        val = list(st.get_filtration())
        
        # Extract only the simplices (without filtration values) if you need just the simplices

        
        #for f in [3.5]:#np.arange(3, 10, 1):
            #print(flist[ll], f)
        simplices = set()
        for v in val:
            if v[1] <= f:
                simplices.add(tuple(v[0]))
         
        laplacian = np.matmul(boundary_operator(simplices, 2).toarray(), np.transpose(boundary_operator(simplices, 2).toarray()))+np.matmul(np.transpose(boundary_operator(simplices, 1).toarray()), boundary_operator(simplices, 1).toarray())
        eigval, eigvec = np.linalg.eigh(laplacian)
        all_eigval.append(eigval)
        all_eigvec.append(eigvec)
        gnm = GHM(all_eigval, all_eigvec)
        np.save("./data/processed/" + flist[ll][7:-4]+"_gnm_cocycle.npy", gnm)
        #np.save("./data/processed/" + flist[ll][7:-4]+"_cleanvec.npy", v1)

def cal_uGH_matrix(f, shape):

    flist = glob.glob('./data/processed/'+shape+'/*f9[0-9][0-9]_gnm_cocycle.npy')
    flist = sorted(flist)
    
    dist_mat = []
    for ll in range(len(flist)):
        #print(flist[ll])
        data = np.load(flist[ll], allow_pickle=True)
        #print(np.shape(data))
        dist_mat.append(data[0])
    
    mat = np.zeros((len(dist_mat), len(dist_mat)))

    
    pairs = []
    for i in range(len(mat)):
        for j in range(0, i):
            if len(dist_mat[i])>0 and len(dist_mat[j])>0 and np.array_equal(dist_mat[i], dist_mat[j])==False:
                pairs.append((dist_mat,i,j))
    
    
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    vals = p.starmap(_uGH, pairs)
    p.close()
    p.join()
    
    
    for v in vals:
        mat[v[0], v[1]] = v[2] 
        #print(v[0], v[1], v[2])
    
    mat += np.transpose(np.tril(mat))
    #print(mat)
    
    np.save("./results/GH_OIHP_"+ shape +"_cocycle_fil_"+ str(f) +".npy", mat)
    
    
def cluster_l1(ncluster, f, shape):
    
    if f == 'multi':
        #mat_3 = np.load("./results/GH_OIHP_"+ shape +"_l1norm_fil_3.npy", allow_pickle=True)[:300, :300]
        mat_3_5 = np.load("./results/GH_OIHP_"+ shape +"_cocycle_fil_3.5.npy", allow_pickle=True)
        mat_4 = np.load("./results/GH_OIHP_"+ shape +"_cocycle_fil_4.npy", allow_pickle=True)
        mat_5 = np.load("./results/GH_OIHP_"+ shape +"_cocycle_fil_5.npy", allow_pickle=True)
        mat_6 = np.load("./results/GH_OIHP_"+ shape +"_cocycle_fil_6.npy", allow_pickle=True)
        mat = np.concatenate(( mat_3_5, mat_4, mat_5, mat_6), axis=1)

    else:
        mat = np.load("./results/GH_OIHP_"+shape+"_cocycle_fil_"+ str(f) +".npy", allow_pickle=True)

        
    # plot mat as heat map and save image
    plt.imshow(mat, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to show the scale
    plt.savefig("./results/uGH_cocycle_fil_"+ str(f) + ".png", dpi=300, bbox_inches='tight')  # Save the figure with high resolution
    plt.show()  # Display the plot
    
   
    
    
    feat = mat    
    
    ndata = 300
    frd = 10
    frs = ndata//ncluster + 10
    values = TSNE(n_components=2, verbose=2).fit_transform(feat)
    
    #values = umap.UMAP(random_state=42).fit_transform(feat)
    plt.figure(figsize=(5,5), dpi=200)
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    
    if ncluster == 9:
            
        plt.scatter(values[:(frs-frd),0], values[:(frs-frd),1], marker='.', color='tab:blue', alpha=0.75, linewidth=.5, s=20, label="Br-Cubic")
        plt.scatter(values[(frs-frd):2*(frs-frd),0], values[(frs-frd):2*(frs-frd),1], marker='.', color='tab:orange', alpha=0.75,  linewidth=0.5, s=20, label="Br-Ortho")
        plt.scatter(values[2*(frs-frd):3*(frs-frd),0], values[2*(frs-frd):3*(frs-frd),1], marker='.', color='tab:green', alpha=0.75,  linewidth=0.5, s=20, label="Br-Tetra")
        
        plt.scatter(values[3*(frs-frd):4*(frs-frd),0], values[3*(frs-frd):4*(frs-frd),1], marker='.', color='tab:red', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Cubic")
        plt.scatter(values[4*(frs-frd):5*(frs-frd),0], values[4*(frs-frd):5*(frs-frd),1], marker='.', color='tab:purple', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Ortho")
        plt.scatter(values[5*(frs-frd):6*(frs-frd),0], values[5*(frs-frd):6*(frs-frd),1], marker='.', color='tab:brown', alpha=0.75,  linewidth=0.5, s=20, label="Cl-Tetra")
        
        plt.scatter(values[6*(frs-frd):7*(frs-frd),0], values[6*(frs-frd):7*(frs-frd),1],  marker='.',color='tab:pink', alpha=0.75,  linewidth=0.5, s=20, label="I-Cubic")
        plt.scatter(values[7*(frs-frd):8*(frs-frd),0], values[7*(frs-frd):8*(frs-frd),1],  marker='.',color='tab:gray', alpha=0.75,  linewidth=0.5, s=20, label="I-Ortho")
        plt.scatter(values[8*(frs-frd):9*(frs-frd),0], values[8*(frs-frd):9*(frs-frd),1],  marker='.',color='tab:olive', alpha=0.75,  linewidth=0.5, s=20, label="I-Tetra")

    if ncluster == 3:
        plt.scatter(values[:(frs-frd),0], values[:(frs-frd),1], marker='.', color='tab:blue', alpha=0.75, linewidth=.5, s=20, label="Br")
        plt.scatter(values[(frs-frd):2*(frs-frd),0], values[(frs-frd):2*(frs-frd),1], marker='.', color='tab:orange', alpha=0.75,  linewidth=0.5, s=20, label="Cl")
        plt.scatter(values[2*(frs-frd):3*(frs-frd),0], values[2*(frs-frd):3*(frs-frd),1], marker='.', color='tab:green', alpha=0.75,  linewidth=0.5, s=20, label="I")
        
    
    plt.legend(ncol = 3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"./results/tsne_cocycle_{ncluster}_clusters_{shape}_fil_"+ str(f) +".png", dpi=200)
    plt.show()
    plt.close()


if __name__ == '__main__':
    for f in [3, 3.5, 4, 5, 6]:
        for shape in ['cubic', 'orthohombic', 'tetragonal']: #[, 'orthohombic']:

    #for f in ['multi']:
            print("Start running cocycle for filtration = " + str(f))
            calDis(f, shape)  # calculate distance matrix for each structure and save data
            cal_uGH_matrix(f, shape) # calculate pairwise uGH between structures and save the matrix
            #cluster_cocycle(3, f, shape) # cluster data according to uGH matrix
            #cluster_cocycle(9, f, shape) # cluster data according to uGH matrix
            print("Finish running cocycle for filtration = " + str(f))

    
    