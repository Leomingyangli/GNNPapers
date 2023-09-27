GNN: graph neural network

These reading it to implementing scene mapping as a graph using pointcloud or RGBD images.

# Basic

1. Semi-supervised classification with graph convolutional networks ICLR 2016. [paper](https://arxiv.org/pdf/1609.02907.pdf) 
   1. Propose GCN
   2. Transductive learning, needs whole graph. Semi-supervised learning
2. Inductive Representation Learning on Large Graphs ICLR 2017. [paper](https://arxiv.org/pdf/1706.02216.pdf) 
   1. Instead of train the whole graph(all neighbors), SAGE aggregate node with sampled fixed number of  kth-order neighbors. 
   2. Inductive learning that generalize to dynamic nodes or graph.
   3. Best result when using LSTM aggregator. However, it assumes the a consistent sequential node ordering across neighborhoods
3. Graph Attention Networks ICLR 2018. [paper](https://arxiv.org/pdf/1710.10903.pdf)
   1. Weight calculation from node connectivity to node features
   2. Time complexity of a single GAT attention: O(|V|FF+|E|F)
   3. Not depend on upfront access to the whole graph; no need to know the graph structure upfront
   4. Undirected edge not required (no edge, no computing)
   5. Inductive learning. No fixed-size neighborhood
   6. Predict article class by keywords as nodes and citations as edges.

# Localization

1. SEM-GAT: Explainable Semantic Pose Estimation using Learned Graph Attention [paper](https://arxiv.org/pdf/2308.03718.pdf) 
   1. Given two sequential pointclouds from lidar (we use pixels from RGB), each node contains
      1. 3d coordinate
      2. semantic(e.g., people,sidewalk,building) - (we have semantic category)
      3. feature(origin, centroid, corner, surface) by calculating curvature, -(we try hierarchical to get low level feature)
      4. Instance id
then use GCN to encode graph.
   2. Connect nodes(centroid, origin) between two graphs with same semantic under predefined distance threshold(3meter), edge weight is calculated by Graph Attention Network, serving as confidence score. For each node in current, select edge with maximal score.
   3. Generating cross-covariance matrix for nodes pair.
   4. Use weighted Singular Value Decomposition(SVD) to get Rotation and Translation from matrix.
