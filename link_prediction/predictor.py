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

def run_experiment(args, model, x_train, y_train, x_val, y_val):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
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

def create_baseline_model(num_features, hidden_units, num_classes, dropout_rate=0.2):
    inputs = layers.Input(shape=(num_features,), name="input_features")
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
    x_train_pos = np.zeros([len(citation_train_val["source"]), num_features])
    y_train_pos = np.zeros(len(citation_train_val["source"]))
    for ind in citation_train_val.index:
        #print(citation_train_val["source"][ind], citation_train_val["target"][ind])
        src = citation_train_val["source"][ind]
        tgt = citation_train_val["target"][ind]
        feat = np.hstack([node_features_df['features'][src], node_features_df['features'][tgt]])
        np.append(x_train_pos, feat)
        np.append(y_train_pos, 1)

    # negative samples
    x_train_neg = np.zeros([len(citation_train_val["source"]), num_features])
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
        np.append(x_train_neg, feat)
        np.append(y_train_neg, 0)

    x = np.vstack([x_train_pos, x_train_neg])
    y = np.vstack([[y_train_pos, y_train_neg]])


    # shuffle the positive and negative samples
    (x_train, x_val, y_train, y_val) = train_test_split(x, y, random_state = 2, test_size=0.2)

    return x_train, y_train, x_val, y_val

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='MLP', help="GNN or MLP")
    parser.add_argument("--learning_rate", default='0.01', type=float, help="GNN or MLP")
    parser.add_argument("--dropout_rate", default=0.5, type=float, help="dropout")
    parser.add_argument("--num_epochs", default=2, type=int,
                        help="dropout")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="dropout")


    args = parser.parse_args()
    print(args)
    main(args)