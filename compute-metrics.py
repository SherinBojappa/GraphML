import pandas as pd
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("network_file", help="network file to be used")
args = parser.parse_args()
print(args.network_file)

edge_df = pd.read_csv(args.network_file, header=None, sep = ' ')
print(edge_df.head())

# columns in the dataframe now have the nodes, make sure they are counted once
# get all nodes from the 2 columns of the data frame
nodes_from_edge_list = edge_df[[0,1]].values.ravel()

# remove duplicates
unique_nodes = pd.unique(nodes_from_edge_list)
print(unique_nodes)
nodes_sorted = np.sort(unique_nodes)
print(nodes_sorted)

# count the number of nodes
num_nodes = unique_nodes.shape[0]
print("number of nodes are " + str(num_nodes))

# create nodes which are 0 based instead of 1 based; easier to access in
# adjacency matrix
# in case node is represented by a string then this code needs to be adjusted
# accordingly
# mapping between node in network and 0 based ids. e.x 1 in nw is rep as 0
nodes = {}
nodes_to_netowrk_id = {}
for node_idx, node_val in enumerate(nodes_sorted):
    nodes[node_val] = node_idx
    nodes_to_netowrk_id[node_idx] = node_val

print(nodes)

# create adjacency matrix, from the data in dataframe
# if two nodes are connected weight = 1, else weight = 1000
weight_mat = np.full((num_nodes, num_nodes), 1000)
np.fill_diagonal(weight_mat, 0)
adj_mat = np.zeros((num_nodes, num_nodes))

print(adj_mat.shape)

# populate the adjacency matrix based on the edge data frame
for index, row in edge_df.iterrows():
    # undirected graph
    row_val = nodes[row[0]]
    col_val = nodes[row[1]]
    adj_mat[row_val, col_val] = 1
    adj_mat[col_val, row_val] = 1
    weight_mat[row_val, col_val] = 1
    weight_mat[col_val, row_val] = 1

print(adj_mat)
print(adj_mat.shape)

# compute density

# count edges only once
num_edges = np.count_nonzero(adj_mat == 1)/2
print("The number of edges are " + str(num_edges))

graph_density = round((2*num_edges)/(num_nodes*(num_nodes-1)), 5)
print("The graph density is " + str(graph_density))

# write this value to file output.txt
output_file = open("output.txt", "w")
output_file.write(str(graph_density)+ '\n')


# compute the diameter of the graph - maximum distance between two nodes





# compute the maximum node degree
# list to store the node degree
node_degree = []
for node_val in adj_mat:
    val = np.count_nonzero(node_val == 1)
    node_degree.append(val)

max_node_degree = round(max(node_degree), 5)
output_file.write(str(max_node_degree)+ '\n')


def dfs(node_idx, visited_nodes, connected_components):
    visited_nodes[node_idx] = 1
    connected_components.append(node_idx)
    for child_node_idx, child_node in enumerate(adj_mat[node_idx]):
        if child_node == 1 and visited_nodes[child_node_idx] != 1:
            dfs(child_node_idx, visited_nodes, connected_components)

# finding the number of connected components in the graph and listing it
visited_nodes = [0]*num_nodes
num_connected_comp = 0
connected_components_list = []


for node_idx, node in enumerate(adj_mat):
    connected_components = []
    if visited_nodes[node_idx] != 1:
        dfs(node_idx, visited_nodes, connected_components)
        cc = [nodes_to_netowrk_id[item] for item in connected_components]
        connected_components_list.append(np.sort(cc))
        num_connected_comp += 1


output_file.write(str(num_connected_comp)+ '\n')

for comp in connected_components_list:
    for node_id in comp:
        output_file.write(str(node_id) + ' ')
    output_file.write('\n')

print("The number of connected components are " + str(num_connected_comp))
print("The connected components are " + str(connected_components_list))

"""
Degree centrality of a node refers to the number of edges attached to the node. 
In order to know the standardized score, you need to divide each score by n-1 
(n = the number of nodes). 
"""
for node in adj_mat:
    num_edges = np.count_nonzero(node == 1)
    node_degree_centrality = round((num_edges/(num_nodes-1)), 5)
    output_file.write(str(node_degree_centrality) + ' ')

def find_smallest_dist_vertex(start_vertex, dist_mat, adj_mat):
    smallest_dist = 1000
    for vertex, connection in enumerate(adj_mat[start_vertex]):
        if connection == 1 and dist_mat[vertex] < smallest_dist and vertex not in visited_nodes:
            smallest_dist = dist_mat[vertex]
            smallest_vertex = vertex

    return smallest_vertex



def find_neighbors(current_node):
    neighbor_list = []
    for neighbor, weight in enumerate(adj_mat[current_node]):
        if weight == 1:
            neighbor_list.append(neighbor)

    return neighbor_list


def update_weights(shortest_paths, current_node, curr_neighbor):
    #new_dist = shortest_paths[current_node] + weight_mat[current_node][curr_neighbor]
    new_dist = shortest_paths[current_node] + weight_mat[current_node][curr_neighbor]
    if new_dist < shortest_paths[curr_neighbor]:
        shortest_paths[curr_neighbor] = new_dist
        previous_vertex[curr_neighbor] = current_node

    return shortest_paths, previous_vertex



def dijkstra(start_vertex, visited_vertices, unvisited_vertices, shortest_paths, previous_vertex):
    current_node = start_vertex
    while len(unvisited_vertices) != 0:

        neighbors = np.array(find_neighbors(current_node))
        dist_neighbors = [shortest_paths[neighbor] for neighbor in neighbors]
        sorted_neighbors = list(neighbors[np.argsort(dist_neighbors)])
        sorted_neighbors = [neighbor for neighbor in sorted_neighbors
                            if neighbor not in visited_vertices]
        num_neighbors = len(sorted_neighbors)
        for idx in range(num_neighbors):
            curr_neighbor = sorted_neighbors[idx]
            shortest_paths, previous_vertex = update_weights(shortest_paths,
                                                             current_node,
                                                             curr_neighbor)

        #print(current_node)
        visited_vertices.append(current_node)
        unvisited_vertices.pop(unvisited_vertices.index(current_node))
        #print(unvisited_vertices)
        #print("sorted neighbors")
        #print(sorted_neighbors)
        if len(sorted_neighbors)>0:
            current_node = sorted_neighbors[0]
        else:
            if len(unvisited_vertices)>0:
                current_node = unvisited_vertices[0]

    return shortest_paths, previous_vertex

# calculate the distance matrix computing the shortest distance from each node
# to every other node
dist_mat = np.full((num_nodes, num_nodes), 1000)
visited_vertices = []
# initially all vertices are unvisited
unvisited_vertices = [vertex for vertex in range(num_nodes)]

# use dijkstras shortest path algorithm on all nodes to know the shortest path
# between two nodes
short_paths = []
prev_node = []
for vertex in range(num_nodes):
    start_vertex = vertex
    dist_mat[start_vertex][start_vertex] = 0
    shortest_paths = [1000]*num_nodes
    previous_vertex = [0]*num_nodes
    visited_vertices = []
    # initially all vertices are unvisited
    unvisited_vertices = [vertex for vertex in range(num_nodes)]
    # distance of vertex with itself is 0
    #weight_mat[start_vertex][start_vertex] = 0
    # shortest path of the vertex with itself is 0
    shortest_paths[start_vertex] = 0
    shortest_paths, previous_vertex = dijkstra(start_vertex, visited_vertices, unvisited_vertices, shortest_paths, previous_vertex)
    print("source node is " + str(vertex) )
    print(shortest_paths)
    print(previous_vertex)
    short_paths.append(shortest_paths)
    prev_node.append(previous_vertex)

output_file.close()


