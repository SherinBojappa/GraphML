import argparse
import networkx as nx
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import numpy as np
import gensim
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import gensim.downloader
import os
import random
import sklearn

def extract_graph_features(G, node_to_vec):
    # (2, numedges) - (source, target)
    edges = np.array(G.edges).T
    # Create an edge weights array of ones.
    edge_weights = tf.ones(shape=edges.shape[1])
    # Create a node features array of shape [num_nodes, num_features].
    node_features = tf.cast(np.vstack(list(node_to_vec.values())),
                            dtype=tf.dtypes.float32)
    # Create graph info tuple with node_features, edges, and edge_weights.
    graph_info = (node_features, edges, edge_weights)

    return graph_info

def run_experiment_gnn(args, model, x_src_train, x_tgt_train, y_train, x_src_val, x_tgt_val, y_val):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=[x_src_train, x_tgt_train],
        y=y_train,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=([x_src_val, x_tgt_val], y_val),
        callbacks=[early_stopping],
    )

    return history


def run_experiment(args, model, x_train, y_train, x_val, y_val):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
    )

    return history

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)

class GraphConvLayer(layers.Layer):
    def __init__(
            self,
            hidden_units,
            dropout_rate=0.2,
            aggregation_type="mean",
            combination_type="concat",
            normalize=False,
            *args,
            **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize
        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        num_nodes = tf.math.reduce_max(node_indices) + 1
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(
                f"Invalid aggregation type: {self.aggregation_type}.")
        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(
                f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.
        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """
        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations,
                                             neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations,
                                          edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(node_indices, neighbour_messages)
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)


class GNNNodeClassifier(tf.keras.Model):
    def __init__(
            self,
            graph_info,
            num_classes,
            hidden_units,
            aggregation_type="sum",
            combination_type="concat",
            dropout_rate=0.2,
            normalize=True,
            *args,
            **kwargs,
    ):
        super(GNNNodeClassifier, self).__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(
            self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate,
                                     name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate,
                                      name="postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")
        self.hidden_units = hidden_units

    def call(self, input_node_indices_src, input_node_indices_tgt):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)

        #print("before gather")
        #print(x.shape)
        # Fetch node embeddings for the input node_indices.
        dd_src = tf.gather(x, input_node_indices_src)
        dd_tgt = tf.gather(x, input_node_indices_tgt)
        #print("before squeeze")
        #print(dd.shape)
        # node_embeddings = tf.squeeze(dd)
        dd = tf.concat(dd_src, dd_tgt)
        node_embeddings = tf.reshape(dd, [-1, self.hidden_units[-1]])
        #print("after squeeze")
        #print(node_embeddings.shape)
        # Compute logits
        logits = self.compute_logits(node_embeddings)
        #print("logits shape:")
        #print(logits.shape)
        return logits

def create_baseline_model(num_features, hidden_units, num_classes, dropout_rate=0.2):
    inputs = layers.Input(shape=(num_features*2,), name="input_features")
    x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
    for block_idx in range(4):
        # Create an FFN block.
        x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
        # Add skip connection.
        x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
    # Compute logits.
    logits = layers.Dense(num_classes, name="logits")(x)
    # Create the model.
    return keras.Model(inputs=inputs, outputs=logits, name="baseline")

def create_train_val_dataset(num_features, G, node_features_df, citation_train_val):
    print("creating train and test dataset")

    # positive samples
    x_train_pos = np.zeros([len(citation_train_val["source"]), num_features*2])
    y_train_pos = np.zeros(len(citation_train_val["source"]))
    for ind in citation_train_val.index:
        #print(citation_train_val["source"][ind], citation_train_val["target"][ind])
        src = citation_train_val["source"][ind]
        tgt = citation_train_val["target"][ind]
        feat = np.hstack([node_features_df['features'][src], node_features_df['features'][tgt]])
        x_train_pos[ind] = feat
        y_train_pos[ind] = 1

    # negative samples
    x_train_neg = np.zeros([len(citation_train_val["source"]), num_features*2])
    y_train_neg = np.zeros(len(citation_train_val["source"]))

    max_node_id = len(node_features_df["node"]) - 1
    for ind in range(len(citation_train_val["source"])):
        while(1):
            src = random.randint(0, max_node_id)
            tgt = random.randint(0, max_node_id)

            # check if this src and target exists
            if(G.has_edge(src, tgt)):
                continue
            else:
                break

        feat = np.hstack([node_features_df['features'][src], node_features_df['features'][tgt]])
        x_train_neg[ind] = feat
        y_train_neg[ind] = 0

    x = np.vstack([x_train_pos, x_train_neg])
    y = np.hstack([y_train_pos, y_train_neg])


    # shuffle the positive and negative samples
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, random_state = 2, test_size=0.2)

    return x_train, y_train, x_val, y_val

def create_train_val_dataset_gnn(num_features, G, node_features_df, citation_train_val):
    print("creating train and test dataset")

    # positive samples
    node_id_src_pos = np.zeros(len(citation_train_val["source"]))
    node_id_tgt_pos = np.zeros(len(citation_train_val["target"]))
    y_train_pos = np.zeros(len(citation_train_val["source"]))
    for ind in citation_train_val.index:
        #print(citation_train_val["source"][ind], citation_train_val["target"][ind])
        src = citation_train_val["source"][ind]
        tgt = citation_train_val["target"][ind]
        #feat_src = node_features_df['features'][src]
        #feat_tgt = node_features_df['features'][tgt]
        #x_train_pos[ind] = feat
        node_id_src_pos[ind] = src
        node_id_tgt_pos[ind] = tgt
        y_train_pos[ind] = 1

    # negative samples
    #x_train_neg = np.zeros([len(citation_train_val["source"]), num_features*2])
    y_train_neg = np.zeros(len(citation_train_val["source"]))
    node_id_src_neg = np.zeros(len(citation_train_val["source"]))
    node_id_tgt_neg = np.zeros(len(citation_train_val["target"]))

    max_node_id = len(node_features_df["node"]) - 1
    for ind in range(len(citation_train_val["source"])):
        while(1):
            src = random.randint(0, max_node_id)
            tgt = random.randint(0, max_node_id)

            # check if this src and target exists
            if(G.has_edge(src, tgt)):
                continue
            else:
                break

        #feat = np.hstack([node_features_df['features'][src], node_features_df['features'][tgt]])
        #x_train_neg[ind] = feat
        node_id_src_neg[ind] = src
        node_id_tgt_neg[ind] = tgt
        y_train_neg[ind] = 0

    x_src = np.vstack([node_id_src_pos, node_id_src_neg])
    x_tgt = np.vstack([node_id_tgt_pos, node_id_tgt_neg])
    y = np.hstack([y_train_pos, y_train_neg])


    # shuffle the positive and negative samples
    (x_src_train, x_src_val, x_tgt_train, x_tgt_val, y_train, y_val) = train_test_split(x_src, x_tgt, y, random_state = 2, test_size=0.2)

    return x_src_train, x_src_val, x_tgt_train, x_tgt_val, y_train, y_val

def main(args):
    train_path = os.path.join(os.getcwd(), "data", "train.txt")
    G = nx.read_edgelist(train_path, delimiter=',', nodetype=int)
    print(nx.info(G))

    # convert the network file into a pandas dataframe
    citation_train_val = pd.read_csv(train_path, sep=',',
                             names=["source", "target"])

    test_path = os.path.join(os.getcwd(), "data", "test.txt")
    citation_test = pd.read_csv(test_path, sep=',',
                             names=["source", "target"])

    #print(citation_train_val.head())
    #print("test data")
    #print(citation_test.head())

    features_path = os.path.join(os.getcwd(), "data", "node-feat.txt")
    node_features_df = pd.read_csv(features_path, sep='\t', names=["node", "features"])
    print(node_features_df.head())

    # convert features into a floating value
    node_features_df['features'] = node_features_df['features'].apply(lambda a: np.fromstring(a, dtype=float, sep=','))

    hidden_units = [32, 32]
    num_classes = 1

    # create data to be fed into the MLP
    #x_train, y_train, x_test, y_test = create_train_test_dataset(node_features_df, citation_train_val)
    num_features = len(node_features_df["features"][0])
    x_train, y_train, x_val, y_val = create_train_val_dataset(num_features, G, node_features_df, citation_train_val)

    if(args.model == 'MLP'):
        print("MLP")
        baseline_model = create_baseline_model(num_features, hidden_units, num_classes,
                                               args.dropout_rate)
        baseline_model.summary()
        history = run_experiment(args, baseline_model, x_train, y_train, x_val, y_val)

        #_, test_accuracy = baseline_model.evaluate(x=x_test, y=y_test,
        #                                           verbose=0)
        #print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

    elif(args.model == 'GNN'):
        print("GNN")

        train_val_nodes = []
        for ind in range(len(citation_train_val["source"])):

            citation_tuple = (citation_train_val["source"][ind], citation_train_val["target"][ind])
            train_val_nodes.append(citation_tuple)

        print("done getting the node indices")

        hidden_units = [32, 32]

        node_to_vec = {}

        for ind in range(len(node_features_df)):
            node_to_vec[node_features_df["node"][ind]] = node_features_df["features"][ind]

        graph_info = extract_graph_features(G, node_to_vec)

        gnn_model = GNNNodeClassifier(
            graph_info=graph_info,
            num_classes=num_classes,
            hidden_units=hidden_units,
            dropout_rate=args.dropout_rate,
            name="gnn_model",
        )

        x_src_train, x_src_val, x_tgt_train, x_tgt_val, y_train, y_val = \
            create_train_val_dataset_gnn(num_features, G, node_features_df, citation_train_val)

        history = run_experiment_gnn(gnn_model,
                                 np.array(x_src_train), np.array(x_tgt_train), y_train,
                                 np.array(x_src_val), np.array(x_tgt_val), y_val)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='MLP', help="GNN or MLP")
    parser.add_argument("--learning_rate", default='0.01', type=float, help="GNN or MLP")
    parser.add_argument("--dropout_rate", default=0.5, type=float, help="dropout")
    parser.add_argument("--num_epochs", default=50, type=int,
                        help="dropout")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="dropout")


    args = parser.parse_args()
    print(args)
    main(args)