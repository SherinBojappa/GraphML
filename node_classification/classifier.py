import argparse
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import scipy


def plot_graph(G, num_nodes):
    """
    :param G:
    :param num_nodes:
    :return:
    """
    # plot graph


def main(args):
    # read the nodes
    G = nx.read_edgelist(args.network_file, nodetype=int)
    print(nx.info(G))

    # convert the network file into a pandas dataframe
    hyperlinks = pd.read_csv(args.network_file, sep=' ',
                             names=["source", "target"])
    #print(hyperlinks.head())

    classes = pd.read_csv(args.categories, sep=' ',
                          names=["class_label", "category"])
    #print(classes.head())

    # create the training dataset

    # plot sample graph
    # plot_graph(G, num_nodes)
    # categories = pd.read_csv("args.categories")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("network_file", help="network file to be used")
    parser.add_argument("categories", help="category of nodes")
    parser.add_argument("titles", help="titles of nodes in the graph")
    parser.add_argument("train", help="training nodes with category")
    parser.add_argument("val", help="validation nodes with category")
    parser.add_argument("test", help="test nodes")
    args = parser.parse_args()

    main(args)

