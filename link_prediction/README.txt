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

