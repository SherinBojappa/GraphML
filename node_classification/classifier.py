import argparse
import networkx as nx
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy
import numpy as np


def plot_graph(G):
    """
    :param G:
    :param num_nodes:
    :return:
    """
    # plot graph


def extract_features():
    print("binary features for each node")

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
    #print("Hyperlinks")
    #print(hyperlinks.head())

    class_labels = pd.read_csv(args.categories, sep=' ',
                          names=["class_label_id", "category"])
    #print(class_labels.head())
    #print("class labels")

    # read the training, validation, test data
    train_data = pd.read_csv(args.train, sep=' ', names=["node", "class_label"])
    print("train data")
    print(train_data.head())

    val_data = pd.read_csv(args.val, sep=' ', names=["node", "class_label"])
    #print("val data")
    #print(val_data.head())

    test_data = pd.read_csv(args.test, names=["node"])
    #print("test data")
    #print(test_data.head())

    # plot sample graph; shows one giant component and the remaining ones.
    #plt.figure(figsize=(10,10))
    #wiki_graph = nx.from_pandas_edgelist(hyperlinks.sample(n=5000))
    #nx.draw_spring(wiki_graph)
    #plt.show()

    # ignore graph data and use data only from title to get the baseline
    # accuracy
    with open(args.titles) as f:
        Lines = f.readlines()

    # dictionary having keys = nodes and values = titles
    #TODO look into reading this in pandas - issues in getting 2 columns because
    # of whitespace and special strings in tiles
    node_to_title_train = {}
    for line in Lines:
        line = line.replace('\n', '')
        line = line.split(' ', 1)
        node_to_title_train[line[0]] = line[1]

    print("Done processing dict")

    title_unique = []
    repeated_words = []

    for key, val in node_to_title_train.items():
        dict_words = val.split(sep=" ")
        for word in dict_words:
            if word not in title_unique:
                title_unique.append(word)
            else:
                repeated_words.append(word)

    # title unique has the list of all unique words in the titles.
    # for each word, we will then have features which are binary indicating the
    # presence or absence of each word in the current nodes title
    print("Unique titles {}".format(len(title_unique)))

    node_to_vec = {}
    for line in Lines:
        line = line.replace('\n', '')
        line = line.split(' ', 1)
        #print(line)
        # form the vector for each node
        node_feature_vector = [0]*len(title_unique)
        for word in line[1].split():
            #if word not in title_unique:
            #    print(word)
            #    exit()
            node_feature_vector[title_unique.index(word)] = 1

        node_to_vec[int(line[0])] = node_feature_vector

    print("Done with extracting features for all nodes")

    # concatenate features for train data
    training_nodes = train_data["node"].to_list()
    node_vectors = [node_to_vec[node] for node in training_nodes]
    train_feature_vectors = np.array(node_vectors)

    val_nodes = val_data["node"].to_list()
    node_vectors = [node_to_vec[node] for node in val_nodes]
    val_feature_vectors = np.array(node_vectors)

    test_nodes = test_data["node"].to_list()
    node_vectors = [node_to_vec[node] for node in test_nodes]
    test_feature_vectors = np.array(node_vectors)


    if args.model == "MLP":
        hidden_units = 32
        dropout_rate = 0.2
        lr = 0.001
        num_epochs = 300
        batch_size = 256
        feature_dim = len(node_to_vec[0])
        num_classes = len(class_labels["class_label_id"].to_list())
        input_features = keras.Input(shape=(feature_dim, ))
        mlp = Sequential()
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.BatchNormalization())
        mlp.add(keras.layers.Dropout(dropout_rate))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.BatchNormalization())
        mlp.add(keras.layers.Dropout(dropout_rate))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.BatchNormalization())
        mlp.add(keras.layers.Dropout(dropout_rate))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.BatchNormalization())
        mlp.add(keras.layers.Dropout(dropout_rate))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        mlp.add(keras.layers.BatchNormalization())
        mlp.add(keras.layers.Dropout(dropout_rate))
        mlp.add(keras.layers.Dense(hidden_units, activation='gelu'))
        # softmax
        mlp.add(keras.layers.Dense(num_classes, name='logits'))
        logits = mlp(input_features)

        model = keras.Model(input_features, logits)

        model.summary()

        model.compile(
            optimizer=Adam(lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
            )

        # Create an early stopping callback.
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_acc", patience=30, restore_best_weights=True
        )

        x_train = train_feature_vectors
        y_train = train_data["class_label"].to_numpy()


        # Fit the model.
        history = model.fit(
            x=x_train,
            y=y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            #validation_data=(val_feature_vectors,
            #                 val_data["class_label"].to_numpy()),
            validation_split=0.2,
            callbacks=[early_stopping],
        )

        #logits = baseline_model.predict(new_instances)
        #probabilities = keras.activations.softmax(
        #    tf.convert_to_tensor(logits)).numpy()
        #display_class_probabilities(probabilities)

    print("End point")








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("network_file", help="network file to be used")
    parser.add_argument("categories", help="category of nodes")
    parser.add_argument("titles", help="titles of nodes in the graph")
    parser.add_argument("train", help="training nodes with category")
    parser.add_argument("val", help="validation nodes with category")
    parser.add_argument("test", help="test nodes")
    parser.add_argument("model", default="GNN", help="MLP or GNN")
    args = parser.parse_args()

    main(args)

