import argparse
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import scipy


def plot_graph(G):
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
    #TODO remap the nodes to ones starting from 0
    # the nodes are already in sorted order 0 - something
    print(sorted(G.nodes()))

    # convert the network file into a pandas dataframe
    hyperlinks = pd.read_csv(args.network_file, sep=' ',
                             names=["source", "target"])
    print("Hyperlinks")
    print(hyperlinks.head())

    class_labels = pd.read_csv(args.categories, sep=' ',
                          names=["class_label_id", "category"])
    print(class_labels.head())
    print("class labels")

    # read the training, validation, test data
    train_data = pd.read_csv(args.train, sep=' ', names=["node", "class_label"])
    print("train data")
    print(train_data.head())

    val_data = pd.read_csv(args.val, sep=' ', names=["node", "class_label"])
    print("val data")
    print(val_data.head())

    test_data = pd.read_csv(args.test, names=["node"])
    print("test data")
    print(test_data.head())

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

