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
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import os




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
                    dx[i, j] = np.sum(np.abs(x1-x2)) # L1 distance
            dx += np.transpose(np.tril(dx))
        gnm.append(dx)
    return [gnm, v1]

def _uGH(dist_mat, i, j):
    op = (i,j, uGH(dist_mat[i], dist_mat[j]))
    print(op)#, np.shape(dist_mat[i]), np.shape(dist_mat[j]))
    return op

### Compute simplicial complices and eigenvalues, save eigenvalues and eigen vectors to npy file
def buildSC():
    flist = glob.glob('./data/*f9[6-9][0-9].txt')
    flist = sorted(flist)
    for ll in range(len(flist)):
        #print(flist[ll])
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
        
        all_eigval, all_eigvec, all_graphs = [], [], []
        all_ex, all_vx = [], []
        
        #all_eigval, all_eigvec = [], []
        #rc = gd.AlphaComplex(coords)
        #simplex_tree = rc.create_simplex_tree()
        #val = list(simplex_tree.get_filtration())
        #print(val)
        #alpha = gd.AlphaComplex(contents)
        #st = alpha.create_simplex_tree()
        
        rips_complex = gd.RipsComplex(points=contents, max_edge_length=4.3)
        st = rips_complex.create_simplex_tree(max_dimension=2)
        val = list(st.get_filtration())
        
        # Extract only the simplices (without filtration values) if you need just the simplices

        
        #for f in [3.5]:#np.arange(3, 10, 1):
            #print(flist[ll], f)
        simplices = set()
        for v in val:
            #    if np.sqrt(v[1])*2 <= f:
            simplices.add(tuple(v[0]))
         
        laplacian = np.matmul(boundary_operator(simplices, 2).toarray(), np.transpose(boundary_operator(simplices, 2).toarray()))+np.matmul(np.transpose(boundary_operator(simplices, 1).toarray()), boundary_operator(simplices, 1).toarray())
        eigval, eigvec = np.linalg.eigh(laplacian)
        all_eigval.append(eigval)
        all_eigvec.append(eigvec)
        #h1 = nx.cycle_basis(G)
        np.save("./data/processed/" + flist[ll][7:-4]+"_eigval.npy", eigval)
        np.save("./data/processed/" + flist[ll][7:-4]+"_eigvec.npy", eigvec)

def calDis():
     flist = glob.glob('./data/*f9[6-9][0-9].txt')
     flist = sorted(flist)
     for ll in range(len(flist)):
         all_eigval = np.load("./data/processed/" + flist[ll][7:-4]+"_eigval.npy")
         all_eigvec = np.load("./data/processed/" + flist[ll][7:-4]+"_eigvec.npy")        
         gnm, v1 = GHM(all_eigval, all_eigvec)
         np.save("./data/processed/" + flist[ll][7:-4]+"_gnm_l1norms2.npy", gnm)
         np.save("./data/processed/" + flist[ll][7:-4]+"_cleanvec.npy", v1)

def cal_uGH_matrix():

    flist = glob.glob('./data/processed/*f9[6-9][0-9]_gnm_l1norms2.npy')
    flist = sorted(flist)
    
    dist_mat = []
    for ll in range(len(flist)):
    #for ll in range(6):
        #print(flist[ll])
        data = np.load(flist[ll], allow_pickle=True)
        #print(np.shape(data))
        dist_mat.append(data[0])
    
    mat = np.zeros((len(dist_mat), len(dist_mat)))

    pairs = []
    for i in range(len(mat)):
        for j in range(0, i):
            if len(dist_mat[i])>0 and len(dist_mat[j])>0 and np.array_equal(dist_mat[i], dist_mat[j])==False:
                pairs.append((dist_mat, i,j))
    
    
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
    
    np.save("GH_OIHP_all_l1norms2.npy", mat)
    
    
def cluster_l1(ncluster):
    
    mat = np.load("GH_OIHP_all_l1norms2.npy", allow_pickle=True)
    
    # plot mat as heat map and save mimage
    plt.imshow(mat, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to show the scale
    plt.savefig('uGH_l1.png', dpi=300, bbox_inches='tight')  # Save the figure with high resolution
    plt.show()  # Display the plot
    
   
    
    
    feat = mat
    
    
    """
    # Perform hierarchical clustering using the distance matrix
    Z = linkage(feat, method='average')
    
    # Plot the dendrogram
    plt.figure(figsize=(8, 4))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()
    
    # Get cluster labels (e.g., 2 clusters)
    cluster_labels = fcluster(Z, t=3, criterion='maxclust')
    print("Cluster labels:", cluster_labels)
    """
    
    """
    #### Spectral clustering ######
    sigma = 0.5
    affinity_matrix = np.exp(-feat ** 2 / (2. * sigma ** 2))
    spectral = SpectralClustering(n_clusters=3, affinity='precomputed', random_state=42)
    cluster_labels = spectral.fit_predict(affinity_matrix)
    
    print("Cluster labels:", cluster_labels)
    """
    
    
    #feat2 = []
    #for i in range(len(feat)):
    #    tmp = []
    #    if ncluster == 9:
    #        for j in range(0, 360, 40):
    #            tmp.append(np.min(feat[i][j:j+40]))
                #tmp.append(np.max(feat[i][j:j+40]))
                #tmp.append(np.mean(feat[i][j:j+40]))
                #tmp.append(np.std(feat[i][j:j+40]))
    #    elif ncluster == 4:
    #        for j in range(0, 360, 120):
    #            tmp.append(np.min(feat[i][j:j+120]))
    #            tmp.append(np.max(feat[i][j:j+120]))
    #            tmp.append(np.mean(feat[i][j:j+120]))
    #            tmp.append(np.std(feat[i][j:j+120]))
    #    feat2.append(tmp)
    
    #feat = np.array(feat2)
    #print(type(feat[0]))
    
    frd = 10
    frs = 360//ncluster + 10
    
    #values = PCA(n_components=2).fit_transform(feat)
    #print(values.explained_variance_ratio_)
    #values = TSNE(n_components=2, verbose=2, perplexity= 20, random_state=40,  metric='jaccard').fit_transform(feat)
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
    
    elif ncluster == 3:
        plt.scatter(values[:(frs-frd),0], values[:(frs-frd),1], marker='.', color='tab:blue', alpha=0.75, linewidth=.5, s=20, label="Br")
        plt.scatter(values[(frs-frd):2*(frs-frd),0], values[(frs-frd):2*(frs-frd),1], marker='.', color='tab:orange', alpha=0.75,  linewidth=0.5, s=20, label="Cl")
        plt.scatter(values[2*(frs-frd):3*(frs-frd),0], values[2*(frs-frd):3*(frs-frd),1], marker='.', color='tab:green', alpha=0.75,  linewidth=0.5, s=20, label="I")
   
    #plt.ylim(np.min(values[:, 1])-10, np.max(values[:,1])+50)
    #plt.xlim(-100, 100)
    plt.legend(ncol=3, loc='upper left', handlelength=.5, borderpad=.25, fontsize=10)
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"tsne_l1norms_{ncluster}_clusters.png", dpi=200)
    plt.show()


def visualize_data(ll):
    # load coordinates
    flist = glob.glob('./data/*f9[6-9][0-9].txt')
    flist = sorted(flist)
    
    # open raw location file
    coorfile = flist[ll]
    file = open(coorfile)
    points = file.readlines()
    for i in range(len(points)):
        points[i] = points[i].rstrip("\n").split(",")
        points[i] = [float(s) for s in points[i]]    
    
    # Removing redundant rows
    points_unique = []
    for point in points:
        if point not in points_unique:
            points_unique.append(point)
    
    points = points_unique
    # plot coordinates as 3D
    
    # Extracting x, y, z coordinates from the list
    x_coords = [coord[0] for coord in points]
    y_coords = [coord[1] for coord in points]
    z_coords = [coord[2] for coord in points]
    
    
    # construct simplicial complex
    # Example 3D point cloud
    
    # Create a Rips complex from the points, with a max edge length of 1.5
    rips_complex = gd.RipsComplex(points=points, max_edge_length=4.3)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    edges = [simplex for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
    triangles = [simplex for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]
    print(str(len(edges)) + " edges and " + str(len(triangles)) + " triangles")
    
    # Alpha complex
    #alpha_complex = gd.AlphaComplex(points=points)
    #simplex_tree = alpha_complex.create_simplex_tree()
    #filtered_simplices = [ simplex for simplex in simplex_tree.get_filtration() if np.sqrt(simplex[1])*2 <= 3.5]
    # Extract edges (1-simplices) and triangles
    #edges = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 2]
    #triangles = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 3]

    
    
    # visualize the simplicial complex
    

    # Create scatter plot for points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(x_coords, y_coords, z_coords, color='black', s=5, label='Points')
    
    # Plot the edges
    for edge in edges:
        ax.plot([x_coords[edge[0]], x_coords[edge[1]]], 
                [y_coords[edge[0]], y_coords[edge[1]]], 
                [z_coords[edge[0]], z_coords[edge[1]]], 'b-', linewidth=2)
    
    
    if len(x_coords) >= 3:

        for triangle in triangles:
            tri_points = np.array([points[triangle[0]], points[triangle[1]], points[triangle[2]]])
            ax.plot_trisurf(tri_points[:, 0], tri_points[:, 1], tri_points[:, 2], color='orange', alpha=0.5)



    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set a title
    ax.set_title(flist[ll][7:-4])
    
    # Show the plot
    #plt.savefig(".figure/" + flist[ll][7:-4]+"_alpha_simplex.png")
    output_dir = 'figure'
    output_path = os.path.join(output_dir, flist[ll][7:-4]+ ".png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


if __name__ == '__main__':
    buildSC() # build simplicial complex; compute eigenvalues and eigenvectors
    # calDis()  # calculate distance matrix for each structure and save data, takes ~ 3 min
    # cal_uGH_matrix() # calculate pairwise uGH between structures and save the matrix
    # cluster_l1(ncluster = 3) # cluster data according to uGH matrix
    
    ll = 50
    
    flist = glob.glob('./data/*f9[6-9][0-9].txt')
    flist = sorted(flist)
    
    #for ll in range(len(flist)):
    #for ll in [4]:
    #    print(ll)
    visualize_data(ll)


    # load gnm matrix
    filename = "./data/processed/" + flist[ll][7:-4]+"_gnm_l1norms2.npy"
    #print(filename)
    # Load the .npy file
    gnm = np.load(filename)
    
    eigval = np.load("./data/processed/" + flist[ll][7:-4]+"_eigval.npy")
    eigvec = np.load("./data/processed/" + flist[ll][7:-4]+"_eigvec.npy")
    cleanvec =  np.load("./data/processed/" + flist[ll][7:-4]+"_cleanvec.npy")
    print(str(cleanvec.shape[1])+" independent cycles")
    print(np.sum(np.abs(cleanvec[:, 1]-cleanvec[:, 2])))

    # Plot the heatmap
    #plt.imshow(gnm[0], cmap='hot', interpolation='nearest')
    #plt.colorbar()  # Add a colorbar to the heatmap
    #plt.title('Heatmap of the Data')
    #plt.show()
    
    
    
    
    