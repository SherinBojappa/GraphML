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
import scipy
import numpy as np
import gensim
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import gensim.downloader
import os

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
    node_features = pd.read_csv(features_path, sep='\t', names=["node", "features"])
    print(node_features.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='MLP', help="GNN or MLP")

    args = parser.parse_args()
    print(args)
    main(args)