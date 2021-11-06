Readme
=======================

Setup:
=======

Install argparse, networkx, pandas, tensorflow, keras, matplotlib, scipy, numpy (latest version '1.21.4'), gensim
and imbalanced-learn.

Running:
=========
Run the following command
python classifier.py "network.txt" "categories.txt" "titles.txt" "train.txt" "val.txt" "test.txt"

The file predictions.txt will have data in the form node, category


Approach:
=========
The following steps were followed:

1. The data from the train, val, and test files were extracted to be used later as pandas data frames.
2. The word presence binary vector was created for the entire network by looking up the title of each node in the network file.
3. The word presence binary vectors were obtained by counting the number of unique words in all the tiltes; vector for each node
 was obtained by indicating the presence or absence of the unique words in its title.
4. The graph neural network is used adapted from the ipython notebook shared during lecture (tutorial on node classification with gnns)
5. The edges of the nodes are extracted from networkx and all edges are weighted as 1.
6. Graph neural network then uses these extracted edges to find neighbors of a node.
7. The word-presence feature vectors are passed into a feed-forward network and this is used as initial features for the graph neural network.
8. 2 graph convolutional layers are then used with skipped connections.
9. The graph convolutional layers use mean aggregation for neighbours with concatenation of previous layer representations to get the final node features.
10. The node representation is then passed to a multi-class classifier using an MLP and logits are obtained.
11. These logits are converted into probabilities by passing it through a softmax layer.
12. Training is done using the nodes listed in train.txt, validation is done using the nodes in val.txt
13. For testing we predict the labels on the nodes listed in test.txt and write the predictions in a file predictions.txt
which is of the form ni, ci -> ni = node; ci = category.
14. The train.txt contains nodes from 25 classes; examples from node 3 are missing.
15. The categories are imbalanced and to solve this issue we tried sklreans's compute_class_weight and also tried penalizing the dominant class manually
but both did not give good performance.
16. To solve the problem of class imbalance we oversample the least represented classes using imbalanced-learn oversampler, each batch now has balanced
 data accross classifiers. This gave an accuracy of around 45% on the validation set.
17. To check if the binary word presence vector features were causing bad performance, we tried using doc2vec from
gensim which associates each node title with a distributed vector based on the paper "Distributed Representations of Sentences and Documents"
But even this has validation accuracy of ~45%

