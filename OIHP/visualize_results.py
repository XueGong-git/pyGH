import glob
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import os 




def visualize_data(ll, f, sc = 'alpha', persistence = False, barcode = False):
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
    
    if sc == 'vr':
    # Create a Rips complex from the points, with a max edge length of 1.5
        rips_complex = gd.RipsComplex(points=points, max_edge_length=f)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        edges = [simplex for simplex, _ in simplex_tree.get_skeleton(1) if len(simplex) == 2]
        triangles = [simplex for simplex, _ in simplex_tree.get_skeleton(2) if len(simplex) == 3]
    
    elif sc == 'alpha':
    # Alpha complex
        alpha_complex = gd.AlphaComplex(points=points)
        simplex_tree = alpha_complex.create_simplex_tree()
        filtered_simplices = [ simplex for simplex in simplex_tree.get_filtration() if simplex[1] <= f]
        # Extract edges (1-simplices) and triangles
        edges = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 2]
        triangles = [simplex[0] for simplex in filtered_simplices if len(simplex[0]) == 3]

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
    
    text_str = str(len(x_coords) ) + " nodes," + str(len(edges)) + " edges, " + str(len(triangles)) + " triangles"
    print(text_str)
    # Set a title
    ax.set_title(flist[ll][7:-4]+ "_fil_" + str(f))

    ax.text(x=-20, y=-15, z=31, s=text_str, fontsize=12, color='black')  # Place text at coordinates (2, 5, 8)

    # Show the plot
    output_dir = 'figure'
    output_path = os.path.join(output_dir, flist[ll][7:-4]+ "_fil_" + str(f) +".png")
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()




if __name__ == '__main__':
    f = 3    
    for ll in range(4, 360, 40):
        visualize_data(ll, f, sc = 'alpha', persistence = False, barcode = False)

        
    

    
    
    
    