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


def visualize_data(ll, f):
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
    #rips_complex = gd.RipsComplex(points=points, max_edge_length=4.3)
    #simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    #edges = [simplex for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
    #triangles = [simplex for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]
    
    # Alpha complex
    alpha_complex = gd.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()
    
    
    filtered_simplices = [ simplex for simplex in simplex_tree.get_filtration() if simplex[1] <= f]
    # Extract edges (1-simplices) and triangles
    edges = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 2]
    edges = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 2]
    triangles = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 3]
    print(str(len(points))+ " nodes, " + str(len(edges)) + " edges, and " + str(len(triangles)) + " triangles")

    
    
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
    ax.set_title(flist[ll][7:-4]+"_filtration="+str(r))
    plt.figtext(0.5, 0.01, str(len(points))+ " nodes, " + str(len(edges)) + " edges, and " + str(len(triangles)) + " triangles", ha="center", fontsize=10)

    # Show the plot
    #plt.savefig(".figure/" + flist[ll][7:-4]+"_alpha_simplex.png")
    output_dir = 'figure'
    output_path = os.path.join(output_dir, flist[ll][7:-4] + "_" + str(r) + ".png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


if __name__ == '__main__':

    
    ll = 1
    f = 2
    
    flist = glob.glob('./data/*f9[6-9][0-9].txt')
    flist = sorted(flist)
    
    #for ll in range(len(flist)):
    #for f in range(4, 360, 40):
    #    print(ll)

     #   for r in range(1, 22, 5):
     #       visualize_data(ll, f)


    # load gnm matrix
    filename1 = "./data/processed/" + flist[ll][7:-4]+"_gnm_l1norms2_"+ str(f) +".npy"
    #filename2 = "./data/processed/" + flist[ll+80][7:-4]+"_gnm_l1norms2_"+ str(f) +".npy"
    #print(filename1)
    #print(filename2)

    # Load the .npy file
    gnm1 = np.load(filename1)
    #gnm2 = np.load(filename2)

    eigval = np.load("./data/processed/" + flist[ll][7:-4]+"_eigval_"+ str(f) +".npy")
    eigvec = np.load("./data/processed/" + flist[ll][7:-4]+"_eigvec_"+ str(f) +".npy")
    
    #cleanvec =  np.load("./data/processed/" + flist[ll][7:-4]+"_cleanvec.npy")
    #print(str(cleanvec.shape[1])+" independent cycles")
    #print(np.sum(np.abs(cleanvec[:, 0]-cleanvec[:, 1])))
    #print(str(len(gnm1[0]))+ " cycles in structure 1; ", str(len(gnm2[0]))+ " cycles in structure 2. ", )
    #print(uGH(gnm1[0], gnm2[0]))
    
    
    # Plot the heatmap
    #plt.imshow(gnm[0], cmap='hot', interpolation='nearest')
    #plt.colorbar()  # Add a colorbar to the heatmap
    #plt.title('Heatmap of the Data')
    #plt.show()
    
    # check final uGH matrix
    #uGH = np.load("GH_OIHP_all_l1norms2_"+ str(f) +".npy")

    
    
    
    