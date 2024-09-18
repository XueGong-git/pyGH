import glob
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import os 
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
import time
from matplotlib import cm  # Import colormap



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

def visualize_data(ll, f, sc = 'alpha', persistence = False, barcode = False):
    # load coordinates
    flist = glob.glob('./data/archive/*f9[6-9][0-9].txt')
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

    
    if sc == 'vr':
    # Create a Rips complex from the points, with a max edge length of 1.5
        rips_complex = gd.RipsComplex(points=points, max_edge_length=f)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        edges = [simplex for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        triangles = [simplex for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]
    
    elif sc == 'alpha':
    # Alpha complex
        alpha_complex = gd.AlphaComplex(points = points)
        simplex_tree = alpha_complex.create_simplex_tree()
        filtered_simplices = [ simplex for simplex in simplex_tree.get_filtration() if simplex[1] <= f]
        val = list(simplex_tree.get_filtration())
        simplices = set()
        for v in val:
            if v[1] <= f:
                simplices.add(tuple(v[0]))
        edges = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 2]
        triangles = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 3]

    laplacian = np.matmul(boundary_operator(simplices, 2).toarray(), np.transpose(boundary_operator(simplices, 2).toarray()))+np.matmul(np.transpose(boundary_operator(simplices, 1).toarray()), boundary_operator(simplices, 1).toarray())
    eigval, eigvec = np.linalg.eigh(laplacian)
    
    
    if persistence == True:

        # Compute persistence
        persistence = simplex_tree.persistence()    
        # Plot persistence diagram
        gd.plot_persistence_diagram(persistence)

        plt.title(flist[ll][7:-4])
        output_dir = 'figure'
        output_path = os.path.join(output_dir, flist[ll][7:-4]+ "_persistence.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.show()
        plt.close()
    
    if barcode == True:
    
        # Plot persistence diagram
        gd.plot_persistence_barcode(persistence)
        plt.xlim(0, 15)  # Set the limits for the x-axis (e.g., 0 to 2)

        plt.title(flist[ll][7:-4])
        output_dir = 'figure'
        output_path = os.path.join(output_dir, flist[ll][7:-4]+ "_barcode.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.show()
        plt.close()
    
    # visualize the simplicial complex
    

    # Create scatter plot for points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points
    ax.scatter(x_coords, y_coords, z_coords, color='black', s=5, label='Points')
    
    value_vector = abs(eigvec[2])
    # Normalize the values to use with the colormap
    norm = plt.Normalize(value_vector.min(), value_vector.max())
    cmap = cm.seismic  # Choose a colormap, e.g., 'viridis'
    
        
    # Add a color bar to the figure
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(value_vector)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    
    if len(x_coords) >= 3:
        for triangle in triangles:
            tri_points = np.array([points[triangle[0]], points[triangle[1]], points[triangle[2]]])
            ax.plot_trisurf(tri_points[:, 0], tri_points[:, 1], tri_points[:, 2], color='lightgrey', alpha=0.5)


    # Plot the edges
    for i, edge in enumerate(edges):
        color = cmap(norm(value_vector[i])) 
        ax.plot([x_coords[edge[0]], x_coords[edge[1]]], 
                [y_coords[edge[0]], y_coords[edge[1]]], 
                [z_coords[edge[0]], z_coords[edge[1]]], color=color,  alpha=0.5, linewidth=2)
        

    ax.grid(False)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    text_str = str(len(x_coords) ) + " nodes," + str(len(edges)) + " edges, " + str(len(triangles)) + " triangles"
    print(text_str)
    # Set a title
    ax.set_title(flist[ll][7:-4]+ "_fil_" + str(f))
    ax.grid(False)
    ax.set_axis_off()

    ax.text(x=-20, y=-15, z=31, s=text_str, fontsize=12, color='black')  # Place text at coordinates (2, 5, 8)
    # Remove the grey background by setting pane face colors to be transparent
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Set the x-pane to be transparent
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Set the y-pane to be transparent
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))  # Set the z-pane to be transparent
    

    # Show the plot
    output_dir = 'figure'
    output_path = os.path.join(output_dir, flist[ll][7:-4]+ "_fil_" + str(f) +".png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()




if __name__ == '__main__':
    for f in [3, 3.5, 4, 5, 6]:
        for ll in range(4, 360, 40):
            print(ll)
            visualize_data(ll, f, sc = 'alpha', persistence = False, barcode = False)
    
            
    

    
    
    
    