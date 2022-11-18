## Requirements:
numpy, pandas, and argparse

## Files and instructions to run
1. compute_metrics.py - This can be used to compute graph metrics such as density, diameter, number of connected components
                        "python compute_metrics.py <net-sample.txt>"
                        
2. node classification - Classify a node in a graph into a particular category.
Setup:
=======

Install the following python packages:
Argparse 1.1
Networkx - 2.6.3
Pandas - 1.3.2
Tensorflow - 2.6.0
Keras 2.6.0
Nltk - 3.6.5
Numpy 1.21.4
Gensim - 4.0.1
imblearn - 0.8.1

Running:
=========
Run the following command
python classifier.py "network.txt" "categories.txt" "titles.txt" "train.txt" "val.txt" "test.txt"

The file predictions.txt will have data in the form node, category; the nodes
are from test.txt file

Approach:
=========
The following steps were followed:

Word embeddings/features
========================
For the pre-trained word embedding I tried:
1. word2vec-google-news-300 which is trained on 100B tokens from google news and uses a pre-trained word2vec model.
2. glove-wiki-gigaword-200 which is trained on  6B words from wikipedia and more related to our problem.
3. For out of vocab words I just use 0s, this seemed to give better performance than random vectors.
4. Once word embeddings are obtained, I iterate over the words in the tile, get their word embeddings and average them
 to get one embedding per title.
6. Word2vec embeddings performs better than glove embeddings and word presence vectors.
7. Concatenating graph properties such as clustering coeff of a node, degree centrality of a node along with the title
embedding, hurt the performance by around 2% on validation dataset and hence these features were not used.
8. The word presence binary vectors were obtained by finding the unique words in all the tiltes; vector for each node
 was obtained by indicating the presence or absence of the unique words in its title.

Data extraction
===============
1. The data from the train, val, and test files were extracted to be used later as pandas data frames.
2. The word presence binary vector was created for the entire network by looking up the title of each node in the network file.

GNN
===
1. The graph neural network used is adapted from the ipython notebook shared during lecture
(tutorial on node classification with gnns)
2. The edges of the nodes are extracted from networkx and all edges are weighted as 1.
3. Graph neural network then uses these extracted edges to find neighbors of a node.
4. The word-embeddings are passed into a feed-forward network and this is used as initial features for the graph neural network.
5. 2 graph convolutional layers are then used with skipped connections.
6. The graph convolutional layers use mean aggregation for neighbours with concatenation of previous layer representations
to get the final node features.
7. The node representation is then passed to a multi-class classifier using an MLP and logits are obtained.
8. These logits are converted into probabilities by passing it through a softmax layer,
the class with the highest probability value is then picked as the prediction for that particular node.
12. Training is done using the nodes listed in train.txt, validation is done using the nodes in val.txt.
13. The best model based on validation accuracy is saved under best_model and this is loaded for predictions.
13. For testing we predict the labels on the nodes listed in test.txt and write the predictions in a file predictions.txt
which is of the form ni, ci -> ni = node; ci = category.
14. The train.txt contains nodes from 25 classes; examples from node 3 are missing.
15. The node categories in train.txt are imbalanced and to solve this issue we tried sklreans's compute_class_weight
and also tried penalizing the dominant class manually but both did not give good performance.
16. To solve the problem of class imbalance we oversample the least represented classes using imbalanced-learn oversampler,
each batch now has balanced data accross classifiers.
17. By following the above steps we consistently get validation accuracy of around 60%.

3. link_prediction/predictor.py - predicting whether a link exists between 2 nodes.
Readme
=======================

Setup:
=======

Install the following python packages:
Argparse 1.1
Networkx - 2.6.3
Pandas - 1.3.2
Tensorflow - 2.6.0
Keras 2.6.0
Nltk - 3.6.5
Numpy 1.21.4
Gensim - 4.0.1
imblearn - 0.8.1
sklearn - 0.24.2

Running:
=========
Run the following command
python predictor.py data/

The file predictions.txt will have data in the form source node, target node,
and probability that there is a node between source and target.

Approach:
=========
The following steps were followed:

1. node features were used from node-feat.txt
2. The nodes in train.txt was used for training and validation (20%)
3. train.txt lists edges, to create a balanced dataset negative samples were
created by randomly picking 2 nodes which do not have an edge between them.

MLP
===

1. For MLP the features of each node in the node pair was concatenated and presented
to the Multi-layer Perceptron with skipped connections.
2. Adam was used as the optimizer and binary crossentropy loss was minimized.
3. This provided a validation accuracy of around 80%


GNN
===

1.  The graph neural network is adapted from the ipython notebook shared during lecture
(tutorial on node classification with gnns) and modified for the link prediction problem.
2. The edges of the nodes are extracted from train.txt and all edges are weighted as 1.
3. Graph neural network then uses these extracted edges to find neighbors of a node.
4. The graph convolutional layers use mean aggregation for neighbours with concatenation
of previous layer representations to get the final node features.
5. The node representation for the target and source node is then combined, it was observed
that element wise multiplication of source and target embeddings provides better performance
than concatenation of source and target embeddings; this is then passed to a classifier using an MLP and logits are obtained.
6. These logits are converted into probabilities by passing it through a sigmoid layer
and this is used as prediction of link between source and target nodes.
7. The best model based on validation accuracy is saved under best_model and this is loaded for predictions.
8. For testing we predict the existance of link on the nodes listed in test.txt and write the predictions
in a file predictions.txt which is of the form ni,nj,pij -> ni = source node; nj = target node, and
pij = probability of existance of a link between source and target nodes.
9. We use learning rate scheduler to decrease learning rate after 10 epochs.
10. By following the above steps we get an accuracy of around 88%.
