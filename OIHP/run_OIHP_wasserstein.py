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
import ot # Python Optimal Transport Package
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
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

def WM(all_eigval, all_eigvec, all_M):
    # Input: A persistent array of eigenvalues and eigenvectors from filtration process
    #        M is a square matrix consisting of cost entries for transport between simplex i and simplex j
    # Output: An array of Gromov-Norm Matrix from filtration process
    
    clean_eigvec = [] 
    for f in range(len(all_eigvec)):
        eigvec = all_eigvec[f]
        eigvec[np.abs(eigvec)<1e-3] = 0 # Zero the entries due to precision 
        clean_eigvec.append(eigvec)

    wm = []
    for f in range(len(all_eigval)):
        print(f, len(all_eigval[f]))
        ll = list(np.where(all_eigval[f]<1e-3)[0]) #list(range(len(all_eigval[f])))#
        v1 = clean_eigvec[f][:, ll] 

        dx = np.zeros((len(ll), len(ll))) # Harmonic Norm matrix for structure at f
        if len(v1) > 0:
            for i in range(len(dx)):
                for j in range(0, i):
                    x1, x2 = v1[:, i]**2, v1[:, j]**2
                    #print(x1, x2, np.linalg.norm(np.abs(x1)-np.abs(x2)))
                    if np.sum(x1) < 1:
                        x1[np.argmax(x1)] += 1-np.sum(x1)
                        
                    if np.sum(x2) < 1: 
                        x2[np.argmax(x2)] += 1-np.sum(x2)
                    dx[i, j] = ot.emd2(x1, x2, all_M[f])
                    #print(i, j, dx[i, j])
            dx += np.transpose(np.tril(dx))
        wm.append(dx)
    return wm

def _uGH(i, j):
    op = (i,j, uGH(dist_mat[i], dist_mat[j]))
    print(op)#, np.shape(dist_mat[i]), np.shape(dist_mat[j]))
    return op

def build_wm(flist):
    #print(flist[ll])
    file = open(flist)
    contents = file.readlines()
    for i in range(len(contents)):
        contents[i] = contents[i].rstrip("\n").split(",")
        contents[i] = [float(s) for s in contents[i]]
        
    all_eigval, all_eigvec, all_graphs = [], [], []
    all_ex, all_vx = [], []
    all_M = []
    #all_eigval, all_eigvec = [], []
    #rc = gd.AlphaComplex(coords)
    #simplex_tree = rc.create_simplex_tree()
    #val = list(simplex_tree.get_filtration())
    #print(val)
    alpha = gd.AlphaComplex(contents)
    st = alpha.create_simplex_tree()
    val = list(st.get_filtration())
    for f in [3.5]:#np.arange(3, 10, 1):
        print(flist, f)
        simplices = set()
        for v in val:
            if np.sqrt(v[1])*2 <= f:
                simplices.add(tuple(v[0]))

        #edge_idx = list(n_faces(simplices,1))
        #vert_idx = list(n_faces(simplices,0))
        edges = list(n_faces(simplices,1))
        pts = np.array(contents)
        M = np.zeros((len(edges), len(edges)))
        for i in range(len(M)):
            for j in range(i):
                e1 = edges[i]; e2 = edges[j]
                if e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]:
                    M[i,j] = 0
                else:
                    tmp = [np.linalg.norm(pts[e1[0]]-pts[e2[0]]), np.linalg.norm(pts[e1[1]]-pts[e2[0]]), np.linalg.norm(pts[e1[0]]-pts[e2[1]]), np.linalg.norm(pts[e1[1]]-pts[e2[1]])]
                    M[i,j] = np.min(tmp)

        M += np.transpose(np.tril(M))
        #all_ex.append(edge_idx)
        #all_vx.append(vert_idx)
        #G = nx.Graph()
        #for i in range(len(vert_idx)):
            #G.add_node((i))
        #for (x,y) in edge_idx:
            #G.add_edge(x,y)
        #all_graphs.append(G)
        #print(edge_idx, G.edges())
        #nx.draw(G, with_labels=True)
        #laplacian = np.matmul(boundary_operator(simplices, 1).toarray(), np.transpose(boundary_operator(simplices, 1).toarray()))
        laplacian = np.matmul(boundary_operator(simplices, 2).toarray(), np.transpose(boundary_operator(simplices, 2).toarray()))+np.matmul(np.transpose(boundary_operator(simplices, 1).toarray()), boundary_operator(simplices, 1).toarray())
        #laplacian = np.matmul(boundary_operator(simplices, 3).toarray(), np.transpose(boundary_operator(simplices, 3).toarray()))+np.matmul(np.transpose(boundary_operator(simplices, 2).toarray()), boundary_operator(simplices, 2).toarray())
        eigval, eigvec = np.linalg.eigh(laplacian)
        #u, s, vh = np.linalg.svd(laplacian)
        #eigval = s*s
        #eigvec = np.transpose(vh)
        #print(eigval)
        all_eigval.append(eigval)
        all_eigvec.append(eigvec)
        all_M.append(M)
    all_sx = [all_vx, all_ex]
    #h1 = nx.cycle_basis(G)
    wm = WM(all_eigval, all_eigvec, all_M)
    np.save("./data/processed/" + flist[7:-4]+"_wm.npy", wm)




### Load coordinates and build Wasserstein Distance matrix with multiprocessing

def build_wm_multiprocessing():
    # Create a multiprocessing pool

    flist = glob.glob('./data/*f9[6-9][0-9].txt')
    flist = sorted(flist)
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    vals = p.map(build_wm, flist)
    #vals = p.map(build_wm, ['./data/MAPbI3_Tetragonal_CNXPb_atmlist_L5_f997.txt', './data/MAPbI3_Tetragonal_CNXPb_atmlist_L5_f998.txt'])
    p.close()
    p.join()
    

### generate pairs of structures for calculation of pariwise uGH

def build_uGH():
    
    flist = glob.glob('./data/processed/*f9[6-9][0-9]_wm.npy')
    flist = sorted(flist)
    
    dist_mat = []
    for ll in range(len(flist)):
        print(flist[ll])
        data = np.load(flist[ll], allow_pickle=True)
        #print(np.shape(data))
        dist_mat.append(data[0])
    
    mat = np.zeros((len(dist_mat), len(dist_mat)))
    
    pairs = []
    for i in range(len(mat)):
        for j in range(0, i):
            if len(dist_mat[i])>0 and len(dist_mat[j])>0 and np.array_equal(dist_mat[i], dist_mat[j])==False:
                pairs.append((i,j))

    # Create a multiprocessing pool

    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    vals = p.starmap(_uGH, pairs)
    p.close()
    p.join()
    
    mat = np.zeros((len(dist_mat), len(dist_mat)))

    for v in vals:
        mat[v[0], v[1]] = v[2] 
        #print(v[0], v[1], v[2])
    
    mat += np.transpose(np.tril(mat))
    #print(mat)
    
    np.save("GH_OIHP_all_wm.npy", mat)
    

def cluster_wm(ncluster = None):
    mat = np.load("GH_OIHP_all_wm.npy", allow_pickle=True)
    
    
    # plot mat as heat map and save mimage
    plt.imshow(mat, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to show the scale
    plt.savefig('uGH_wm.png', dpi=300, bbox_inches='tight')  # Save the figure with high resolution
    plt.show()  # Display the plot

    feat = mat
    

    
    #feat2 = []
    #for i in range(len(feat)):
    #    tmp = []
    #    for j in range(0, 360, 40):
    #        tmp.append(np.min(feat[i][j:j+40]))
    #        tmp.append(np.max(feat[i][j:j+40]))
    #        tmp.append(np.mean(feat[i][j:j+40]))
    #        tmp.append(np.std(feat[i][j:j+40]))
    #    for j in range(0, 360, 120):
    #        tmp.append(np.min(feat[i][j:j+120]))
    #        tmp.append(np.max(feat[i][j:j+120]))
    #        tmp.append(np.mean(feat[i][j:j+120]))
    #        tmp.append(np.std(feat[i][j:j+120]))
    #    feat2.append(tmp)
    
    #feat = np.array(feat2)
    #print(type(feat[0]))
    
    
    frd = 10
    frs = 360//ncluster + 10
    
    #values = PCA(n_components=2).fit_transform(feat)
    #print(values.explained_variance_ratio_)
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
        
        #plt.ylim(np.min(values[:, 1])-10, np.max(values[:,1])+50)
        #plt.xlim(-100, 100)
        #plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.savefig("tsne_stats_40_9types_wm.png", dpi=200)
        #plt.show()

if __name__ == '__main__':
    #build_wm_multiprocessing()
    #build_uGH()
    cluster_wm(ncluster = 9)