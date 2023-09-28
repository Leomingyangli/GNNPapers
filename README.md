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
  
# Edge

1. Edge-Labeling Graph Neural Network for Few-shot Learning. CVPR 2019. [paper](https://arxiv.org/pdf/1905.01436.pdf)
   1. Few-shot Learning. Node as tasks with (input, label). Edges is 2d(intra, inter) indicate if two nodes share same label
   2. The goal is to predict edge between support set S(with label) and query set Q(without label)
   3. ![567d1f17b245e97242e55b1152fb1370](https://github.com/Leomingyangli/GNNPapers/assets/39786611/5f9078bf-c50a-494a-93af-2f08ccd8df30)


# Sequential RNN

1. GGS_NN: Gated Graph Sequence Neural Networks. ICLR2016. [paper](https://arxiv.org/pdf/1511.05493.pdf)  [zhihu](https://zhuanlan.zhihu.com/p/28170197)
   1. Given Graph with sequential input, using GRU arhictecure to transmit information 
      
      ![d99919bb50ede0e9241d4f241622d284](https://github.com/Leomingyangli/GNNPapers/assets/39786611/4fc2fd57-9b3b-4bdd-ac6d-ab8c12d7c1b8)

      
      ![8cce063cc3e1d85125f806c570c50682](https://github.com/Leomingyangli/GNNPapers/assets/39786611/6ef98bc2-a5b6-4c93-b7a5-7db0edc56358)

   2. BABI tasks.
      
      ![ef9bcfaf01ca4df5bab2a101a5ee6772](https://github.com/Leomingyangli/GNNPapers/assets/39786611/68fff72f-8a6e-4448-b868-a34f36a8db9c)


2. GGT_NN Learning Graphical State Transitions. ICLR2017. [paper](https://openreview.net/pdf?id=HJ0NvFzxl)
   1. A GGS-NN based framework. BABI tasks
   2. Node has
      1. annotation of type belif value, vector sum to 1, 
      2. strength of existence,scalar. 
      3. hidden state
   3. Edge is belif if two nodes contain edge type, could be multiple types or no edge.
      
      ![979e40bde4a1cb8508fa7506ab7dc212](https://github.com/Leomingyangli/GNNPapers/assets/39786611/1862e887-659a-4294-94c2-1cc6b89c75b1)

   4. Could either output single words or sequential words, based on last layer
      
      ![462ecdf9a613097b5a8a36f3c4e5e7e8](https://github.com/Leomingyangli/GNNPapers/assets/39786611/8885773c-07f4-4b94-8700-f86fc9d5d059)

   5. The most valuable part is how to add new nodes by GRU(G, input vector) -> Graph with new nodes(annotation, strength, state)
      
      ![fef8f7996ff11257238ccf1af2e55c42](https://github.com/Leomingyangli/GNNPapers/assets/39786611/d2ea4472-a331-40c1-8d52-5f12a52288dd)

