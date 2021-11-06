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
from gensim.models.doc2vec import Doc2Vec
from collections import namedtuple


def extract_graph_features(G, node_to_vec):
    # (2, numedges) - (source, target)
    edges = np.array(G.edges).T
    # Create an edge weights array of ones.
    edge_weights = tf.ones(shape=edges.shape[1])
    # Create a node features array of shape [num_nodes, num_features].
    node_features = tf.cast(np.array(list(node_to_vec.values())), dtype=tf.dtypes.float32)
    # Create graph info tuple with node_features, edges, and edge_weights.
    graph_info = (node_features, edges, edge_weights)

    return graph_info

def run_experiment(model, x_train, y_train, x_val, y_val, class_weight):
    # Compile the model.

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.9

    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        #optimizer=keras.optimizers.SGD(args.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=1000, restore_best_weights=True
    )
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        #class_weight = class_weight,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        #validation_split=0.3,
        callbacks=[callback, early_stopping],
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
            #aggregation_type="sum",
            combination_type="concat",
            #combination_type="add",
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
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
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
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")
        self.hidden_units = hidden_units

    def call(self, input_node_indices):
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
        dd = tf.gather(x, input_node_indices)
        #print("before squeeze")
        #print(dd.shape)
        # node_embeddings = tf.squeeze(dd)
        node_embeddings = tf.reshape(dd, [-1, self.hidden_units[-1]])
        #print("after squeeze")
        #print(node_embeddings.shape)
        # Compute logits
        logits = self.compute_logits(node_embeddings)
        #print("logits shape:")
        #print(logits.shape)
        return logits


def main(args):
    # read the nodes
    G = nx.read_edgelist(args.network_file, nodetype=int)
    print(nx.info(G))
    # the nodes are already from 0
    #print(sorted(G.nodes()))

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
    if(args.doc2vec == True):
        print("doc2vec is chosen")

        node_to_vec = {}

        # Train model (set min_count = 1, if you want the model to work with the provided example data set)

        #model = Doc2Vec(docs, vector_size=100, min_count=1)
        model = Doc2Vec(corpus_file=args.titles, vector_size=100, min_count=1)

        # Get the vectors
        print(model.docvecs[0])
        print(model.docvecs[1])

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
            node_feature_vector[title_unique.index(word)] = 1
        if (args.doc2vec == True):
            node_to_vec[int(line[0])] = model.docvecs[int(line[0])]
        else:
            node_to_vec[int(line[0])] = node_feature_vector

    #print("Done with extracting features for all nodes")

    # concatenate features for train data
    training_nodes = train_data["node"].to_list()
    val_nodes = val_data["node"].to_list()
    test_nodes = test_data["node"].to_list()

    if (args.doc2vec == True):
        node_vectors_train = [model.docvecs[node] for node in training_nodes]
        node_vectors_val = [model.docvecs[node] for node in val_nodes]
        node_vectors_test = [model.docvecs[node] for node in test_nodes]

    else:
        node_vectors_train = [node_to_vec[node] for node in training_nodes]
        node_vectors_val = [node_to_vec[node] for node in val_nodes]
        node_vectors_test = [node_to_vec[node] for node in test_nodes]


    train_feature_vectors = np.array(node_vectors_train)
    val_feature_vectors = np.array(node_vectors_val)
    test_feature_vectors = np.array(node_vectors_test)

    feature_dim = len(node_to_vec[0])
    num_classes = len(class_labels["class_label_id"].to_list())

    x_train = train_feature_vectors
    y_train = train_data["class_label"].to_numpy()

    from imblearn.over_sampling import RandomOverSampler

    oversample = RandomOverSampler()
    # fit and apply the transform
    training_nodes, y_train = oversample.fit_resample(np.array(training_nodes).reshape(-1,1), y_train)






    (unique, counts) = np.unique(y_train, return_counts=True)

    """
    
    class_weights = (1.0 / counts)
    class_weights = class_weights / np.min(class_weights)
    class_weight = {}
    for idx in range(num_classes):
        if idx in unique:
            class_weight[idx] = class_weights[list(unique).index(idx)]
        else:
            class_weight[idx] = 100.0
    #class_weight = {key: value for (key, value) in zip(unique, class_weights)}
    """

    import sklearn
    weights = sklearn.utils.class_weight.compute_class_weight('balanced',
                                                              unique,
                                                              y_train)
    print(weights)

    class_weight = {}
    for idx in range(num_classes):
        if idx in unique:
            class_weight[idx] = weights[list(unique).index(idx)]
        else:
            class_weight[idx] = np.max(weights)+1.0

    x_val = val_feature_vectors
    y_val = val_data["class_label"].to_numpy()

    x_test = test_feature_vectors

    if args.model == "MLP":
        hidden_units = 32
        dropout_rate = args.dropout_rate
        lr = args.learning_rate
        num_epochs = args.num_epochs
        batch_size = args.batch_size

        input_features = keras.Input(shape=(feature_dim, ))

        x = keras.layers.Dense(hidden_units, activation='gelu')(input_features)
        x = keras.layers.BatchNormalization()(x)
        x1 = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([x, x1])
        x = keras.layers.Dense(hidden_units, activation='gelu')(input_features)
        x = keras.layers.BatchNormalization()(x)
        x1 = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([x, x1])
        x = keras.layers.Dense(hidden_units, activation='gelu')(input_features)
        x = keras.layers.BatchNormalization()(x)
        x1 = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Add()([x, x1])
        logits = keras.layers.Dense(num_classes, name='logits')(x)

        model = keras.Model(input_features, logits)

        model.summary()

        model.compile(
            optimizer=Adam(lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
            )

        # Create an early stopping callback.
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_acc", patience=100, restore_best_weights=True
        )


        # Fit the model.
        history = model.fit(
            x=x_train,
            y=y_train,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            #validation_split=0.2,
            callbacks=[early_stopping],
        )

        #logits = model.predict(tf.convert_to_tensor(test_nodes))
        logits = model.predict(x_test)
        probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy().squeeze()
        preds = np.argmax(probabilities, axis=1)
        print(preds[0])

        output_file = open("predictions.txt", "w")
        for iter, node in enumerate(test_nodes):
            output_file.write(str(node) + ' ' + str(preds[iter])+"\n")

        #logits = baseline_model.predict(new_instances)
        #probabilities = keras.activations.softmax(
        #    tf.convert_to_tensor(logits)).numpy()
        #display_class_probabilities(probabilities)

    if(args.model == "GNN"):

        hidden_units = [32, 32]

        graph_info = extract_graph_features(G, node_to_vec)

        gnn_model = GNNNodeClassifier(
            graph_info=graph_info,
            num_classes=num_classes,
            hidden_units=hidden_units,
            dropout_rate=args.dropout_rate,
            name="gnn_model",
        )

        #print("GNN output shape:", gnn_model([1, 5]))

        #gnn_model.summary()

        history = run_experiment(gnn_model,
                                 np.array(training_nodes), y_train,
                                 np.array(val_nodes), y_val, class_weight)

        logits = gnn_model.predict(tf.convert_to_tensor(test_nodes))
        probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy().squeeze()
        preds = np.argmax(probabilities, axis=1)
        #print(preds[0])

        output_file = open("predictions.txt", "w")
        for iter, node in enumerate(test_nodes):
            output_file.write(str(node) + ' ' + str(preds[iter])+"\n")


        #TODO comment out
        output_file = open("predictions_readable.txt", "w")
        for iter, node in enumerate(test_nodes):
            output_file.write(str(node_to_title_train[str(node)]) + ' ' + str(class_labels["category"].loc[preds[iter]])+"\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("network_file", help="network file to be used")
    parser.add_argument("categories", help="category of nodes")
    parser.add_argument("titles", help="titles of nodes in the graph")
    parser.add_argument("train", help="training nodes with category")
    parser.add_argument("val", help="validation nodes with category")
    parser.add_argument("test", help="test nodes")
    parser.add_argument("model", default="GNN", help="MLP or GNN")
    parser.add_argument("num_epochs", default=300, type=int, help="number of epochs")
    parser.add_argument("batch_size", default=128, type=int, help="samples in a batch")
    parser.add_argument("learning_rate", default=0.01, type=float, help="learning rate")
    parser.add_argument("dropout_rate", default=0.2, type=float, help="dropout_rate")
    parser.add_argument("doc2vec", default=True, type=bool, help="whether to convert titles to distributed vectors")

    args = parser.parse_args()

    main(args)

