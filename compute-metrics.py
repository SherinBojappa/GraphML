import pandas as pd
import argparse
import numpy as np


class Graph:
    def __init__(self):
        self.num_nodes = 0
        self.nodes_sorted = None
        self.nodes_to_network_id = {}
        self.adj_mat = None
        self.weight_mat = None
        self.edge_df = None
        self.nodes = {}

    def read_nodes(self, network_file):
        self.edge_df = pd.read_csv(network_file, header=None, sep=' ')
        # print(self.edge_df.head())

        # columns in the dataframe now have the nodes, make sure they are
        # counted once get all nodes from the 2 columns of the data frame
        nodes_from_edge_list = self.edge_df[[0, 1]].values.ravel()

        # remove duplicates
        unique_nodes = pd.unique(nodes_from_edge_list)
        # print(unique_nodes)
        self.nodes_sorted = np.sort(unique_nodes)
        # print(self.nodes_sorted)

        # count the number of nodes
        self.num_nodes = unique_nodes.shape[0]
        # print("number of nodes are " + str(self.num_nodes))

    def map_nodes_to_network_id(self):
        # create nodes which are 0 based instead of 1 based; easier to access in
        # adjacency matrix
        # in case node is represented by a string then this code needs to be
        # adjusted accordingly
        # mapping between node in network and 0 based ids. e.x 1 in nw is rep
        # as 0
        for node_idx, node_val in enumerate(self.nodes_sorted):
            self.nodes[node_val] = node_idx
            self.nodes_to_network_id[node_idx] = node_val
        # print(self.nodes)

    def populate_adj_weight_matrix(self):
        # create adjacency matrix, from the data in dataframe
        # if two nodes are connected weight = 1, else weight = 1000
        self.weight_mat = np.full((self.num_nodes, self.num_nodes), 1000)
        np.fill_diagonal(self.weight_mat, 0)
        self.adj_mat = np.zeros((self.num_nodes, self.num_nodes))

        # print(self.adj_mat.shape)

        # populate the adjacency matrix based on the edge data frame
        for index, row in self.edge_df.iterrows():
            # undirected graph
            row_val = self.nodes[row[0]]
            col_val = self.nodes[row[1]]
            self.adj_mat[row_val, col_val] = 1
            self.adj_mat[col_val, row_val] = 1
            self.weight_mat[row_val, col_val] = 1
            self.weight_mat[col_val, row_val] = 1

        # print(self.adj_mat)
        # print(self.adj_mat.shape)


# compute density
def compute_density(G):
    # count edges only once
    num_edges = np.count_nonzero(G.adj_mat == 1) / 2
    # print("The number of edges are " + str(num_edges))

    graph_density = round((2 * num_edges) / (G.num_nodes * (G.num_nodes - 1)),
                          5)
    # print("The graph density is " + str(graph_density))
    return graph_density


def compute_node_degree(G):
    # compute the maximum node degree
    # list to store the node degree
    node_degree = []
    for node_val in G.adj_mat:
        val = np.count_nonzero(node_val == 1)
        node_degree.append(val)

    max_node_degree = round(max(node_degree), 5)
    return max_node_degree


def dfs(G, node_idx, visited_nodes, connected_components):
    visited_nodes[node_idx] = 1
    connected_components.append(node_idx)
    for child_node_idx, child_node in enumerate(G.adj_mat[node_idx]):
        if child_node == 1 and visited_nodes[child_node_idx] != 1:
            dfs(G, child_node_idx, visited_nodes, connected_components)


def compute_connected_components(G):
    # finding the number of connected components in the graph and listing it
    visited_nodes = [0] * G.num_nodes
    num_connected_comp = 0
    connected_components_list = []

    for node_idx, node in enumerate(G.adj_mat):
        connected_components = []
        if visited_nodes[node_idx] != 1:
            dfs(G, node_idx, visited_nodes, connected_components)
            cc = [G.nodes_to_network_id[item] for item in connected_components]
            connected_components_list.append(np.sort(cc))
            num_connected_comp += 1
    return connected_components_list, num_connected_comp


def find_neighbors(G, current_node):
    neighbor_list = []
    for neighbor, weight in enumerate(G.adj_mat[current_node]):
        if weight == 1:
            neighbor_list.append(neighbor)

    return neighbor_list


def update_weights(G, previous_vertex, shortest_paths, current_node,
                   curr_neighbor):
    new_dist = shortest_paths[current_node] + G.weight_mat[current_node][
        curr_neighbor]
    if new_dist < shortest_paths[curr_neighbor]:
        shortest_paths[curr_neighbor] = new_dist
        previous_vertex[curr_neighbor] = current_node

    return shortest_paths, previous_vertex


def dijkstra(G, start_vertex, visited_vertices, unvisited_vertices,
             shortest_paths, previous_vertex):
    current_node = start_vertex
    while len(unvisited_vertices) != 0:

        neighbors = np.array(find_neighbors(G, current_node))
        dist_neighbors = [shortest_paths[neighbor] for neighbor in neighbors]
        sorted_neighbors = list(neighbors[np.argsort(dist_neighbors)])
        sorted_neighbors = [neighbor for neighbor in sorted_neighbors
                            if neighbor not in visited_vertices]
        num_neighbors = len(sorted_neighbors)
        for idx in range(num_neighbors):
            curr_neighbor = sorted_neighbors[idx]
            shortest_paths, previous_vertex = update_weights(G, previous_vertex,
                                                             shortest_paths,
                                                             current_node,
                                                             curr_neighbor)

        visited_vertices.append(current_node)
        unvisited_vertices.pop(unvisited_vertices.index(current_node))

        if len(sorted_neighbors) > 0:
            current_node = sorted_neighbors[0]
        else:
            if len(unvisited_vertices) > 0:
                current_node = unvisited_vertices[0]

    return shortest_paths, previous_vertex


def compute_shortest_path(G):
    # calculate the distance matrix computing the shortest distance from each
    # node to every other node
    dist_mat = np.full((G.num_nodes, G.num_nodes), 1000)
    visited_vertices = []
    # initially all vertices are unvisited
    unvisited_vertices = [vertex for vertex in range(G.num_nodes)]

    # use Dijkstra's shortest path algorithm on all nodes to know the shortest
    # path between two nodes
    short_paths = []
    prev_node = []
    for vertex in range(G.num_nodes):
        start_vertex = vertex
        dist_mat[start_vertex][start_vertex] = 0
        shortest_paths = [1000] * G.num_nodes
        previous_vertex = [0] * G.num_nodes
        visited_vertices = []
        # initially all vertices are unvisited
        unvisited_vertices = [vertex for vertex in range(G.num_nodes)]
        # distance of vertex with itself is 0
        # shortest path of the vertex with itself is 0
        shortest_paths[start_vertex] = 0
        shortest_paths, previous_vertex = dijkstra(G, start_vertex,
                                                   visited_vertices,
                                                   unvisited_vertices,
                                                   shortest_paths,
                                                   previous_vertex)
        # print("source node is " + str(vertex))
        # print(shortest_paths)
        # print(previous_vertex)
        short_paths.append(shortest_paths)
        prev_node.append(previous_vertex)

    return short_paths


def compute_diameter(short_paths):
    # compute diameter of the graph
    diameter = 0
    for vertex_paths in short_paths:
        for distance in vertex_paths:
            if diameter < distance < 1000:
                diameter = distance
    return diameter


"""
Degree centrality of a node refers to the number of edges attached to the node. 
In order to know the standardized score, you need to divide each score by n-1 
(n = the number of nodes). 
"""


def compute_and_populate_node_closeness_degree_centrality(G, output_file,
                                                          short_paths):
    for node_idx, node in enumerate(G.adj_mat):
        num_edges = np.count_nonzero(node == 1)
        node_degree_centrality = round((num_edges / (G.num_nodes - 1)), 5)
        output_file.write(str(node_degree_centrality) + ' ')
        node_closeness_centrality = round(
            ((G.num_nodes - 1) / sum(short_paths[node_idx])), 5)
        output_file.write(str(node_closeness_centrality) + '\n')

    return node_closeness_centrality


def list_populate_connected_components(connected_components_list, output_file):
    for comp in connected_components_list:
        for node_id in comp:
            output_file.write(str(node_id) + ' ')
        output_file.write('\n')

    output_file.close()


def populate_out_file(graph_density, max_node_degree, diameter,
                      num_connected_comp):
    output_file = open("output.txt", "w")
    output_file.write(str(graph_density) + '\n')
    output_file.write(str(diameter) + '\n')
    output_file.write(str(num_connected_comp) + '\n')
    output_file.write(str(max_node_degree) + '\n')

    return output_file


def main(args):
    G = Graph()
    G.read_nodes(args.network_file)
    G.map_nodes_to_network_id()
    G.populate_adj_weight_matrix()

    graph_density = compute_density(G)
    max_node_degree = compute_node_degree(G)
    short_paths = compute_shortest_path(G)

    diameter = compute_diameter(short_paths)
    connected_components_list, num_connected_comp = \
        compute_connected_components(G)

    output_file = populate_out_file(graph_density, max_node_degree, diameter,
                                    num_connected_comp)

    compute_and_populate_node_closeness_degree_centrality(G, output_file,
                                                          short_paths)

    list_populate_connected_components(connected_components_list, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network_file", help="network file to be used")
    args = parser.parse_args()
    # print(args.network_file)

    main(args)
