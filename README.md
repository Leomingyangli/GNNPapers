GNN: graph neural network

These reading it to implementing scene mapping as a graph using pointcloud or RGBD images.

## [Content](#content)
- <a href="#basic">1. Basic</a>
- <a href="#survey">2. Survey</a>
- <a href="#map">3. Map</a>
- <a href="#Localization">4. Localization</a>
- <a href="#Edge">5. Edge</a>
- <a href="#sequential-RNN">6. RNN</a>
- <a href="#graph-construction">7.Graph Construction</a>
- <a href="#3D PointCloud">8. 3D PointCloud</a>


## [Basic](#content)
 
1. **Semi-supervised classification with graph convolutional networks** ICLR 2016. [paper](https://arxiv.org/pdf/1609.02907.pdf) 
   1. Propose GCN
   2. Transductive learning, needs whole graph. Semi-supervised learning
2. **Inductive Representation Learning on Large Graphs** ICLR 2017. [paper](https://arxiv.org/pdf/1706.02216.pdf) 
   1. Instead of train the whole graph(all neighbors), SAGE aggregate node with sampled fixed number of  kth-order neighbors. 
   2. Inductive learning that generalize to dynamic nodes or graph.
   3. Best result when using LSTM aggregator. However, it assumes the a consistent sequential node ordering across neighborhoods
3. **Graph Attention Networks** ICLR 2018. [paper](https://arxiv.org/pdf/1710.10903.pdf)
   1. Weight calculation from node connectivity to node features
   2. Time complexity of a single GAT attention: O(|V|FF+|E|F)
   3. Not depend on upfront access to the whole graph; no need to know the graph structure upfront
   4. Undirected edge not required (no edge, no computing)
   5. Inductive learning. No fixed-size neighborhood
   6. Predict article class by keywords as nodes and citations as edges.

## [Survey](#content)


1. **Deep Learning for 3D Point Clouds: A Survey** TPAMI 2020. [paper](https://arxiv.org/pdf/1912.12033.pdf))
<img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/97fda8a0-a6ed-4504-b4e9-0d717964f06f" alt="Image" style="width: 50%;" />

2. **Foundations and modelling of dynamic networks using Dynamic Graph Neural Networks: A survey** 2021 [paper](https://arxiv.org/pdf/2005.07496.pdf)
   1. Define dynamic graph networks.
      
## [Map](#content)


1.**NICE-SLAM: Neural Implicit Scalable Encoding for SLAM** CVPR 2022. [paper](https://arxiv.org/pdf/2112.12130.pdf)
   1. Input RGBD images and pose, predict the pixel depth and color.
   2. Given pose, sampled points along the ray, the pixel depth is weighted sum of <u>occupancy probability times sampled depth</u> along the points. Color has color values.
   3. The grid decoder is fixed. The gird feautre and color weight $\theta,\omega$ are parameters to be optimized
   4. The Mapping is to optimize $\theta,\omega, R, t$ of K selected keyframes
   5. The tracking is to optimize  $R, t$ of current frame
   <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/ed59cc64-9246-4d6b-8734-3b7b0183ae66" alt="Image" style="width: 50%;" />
      
2.**iMAP: Implicit Mapping and Positioning in Real-Time** CVPR 2021 [paper](https://arxiv.org/pdf/2103.12352.pdf)
   <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/39394061-22db-44f0-99b7-ecce9c408dac" alt="Image" style="width: 50%;" />

3.**Neural Implicit Dense Semantic SLAM** CVPR 2023. [paper](https://arxiv.org/pdf/2304.14560.pdf)

   <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/defcd9fb-61ec-46b3-8390-96f6103c3176" alt="Image" style="width: 50%;" />

   
## [Localization](#content)


1. **SEM-GAT: Explainable Semantic Pose Estimation using Learned Graph Attention** [paper](https://arxiv.org/pdf/2308.03718.pdf)
   <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/94439d5c-ddf6-4f40-8607-7ba10aef7f4e" alt="Image" style="width: 50%;" />

   1. Given two sequential pointclouds from lidar (we use pixels from RGB), each node contains
      1. 3d coordinate
      2. semantic(e.g., people,sidewalk,building) - (we have semantic category)
      3. feature(origin, centroid, corner, surface) by calculating curvature, -(we try hierarchical to get low level feature)
      4. Instance id
then use GCN to encode graph.
   2. Connect nodes(centroid, origin) between two graphs with same semantic under predefined distance threshold(3meter), edge weight is calculated by Graph Attention Network, serving as confidence score. For each node in current, select edge with maximal score.
   3. Generating cross-covariance matrix for nodes pair.
   4. Use weighted Singular Value Decomposition(SVD) to get Rotation and Translation from matrix.
  
## [Edge](#content)


1. **Edge-Labeling Graph Neural Network for Few-shot Learning.** CVPR 2019. [paper](https://arxiv.org/pdf/1905.01436.pdf)
   1. Few-shot Learning. Node as tasks with (input, label). Edges is 2d(intra, inter) indicate if two nodes share same label
   2. The goal is to predict edge between support set S(with label) and query set Q(without label)
   3. <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/5f9078bf-c50a-494a-93af-2f08ccd8df30" alt="Image" style="width: 50%;" />


## [Sequential RNN](#content)


1. **GGS_NN: Gated Graph Sequence Neural Networks.** ICLR2016. [paper](https://arxiv.org/pdf/1511.05493.pdf)  [zhihu](https://zhuanlan.zhihu.com/p/28170197)
   1. Given Graph with sequential input, using GRU arhictecure to transmit information.
   2. All the decription of the task is in the adjacent matrix. For question, add annotation [1,0] as start node and [0,1] as target for initialization.
      <p float="left">
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/4fc2fd57-9b3b-4bdd-ac6d-ab8c12d7c1b8" alt="Image" style="width: 40%;" />
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/6ef98bc2-a5b6-4c93-b7a5-7db0edc56358" alt="Image" style="width: 40%;" />
      </p>
   4. The GRU are processed multiple times, meaning hidden state of each node update several times.
      <p float="left">
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/a1fed10f-6aee-4813-9c21-25cc1e2d10e1" alt="Image" style="width: 40%;" />
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/d56bc899-4785-4aee-a4d8-ec14c7917a85" alt="Image" style="width: 40%;" />
      </p>
   6. BABI tasks
  
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/68fff72f-8a6e-4448-b868-a34f36a8db9c" alt="Image" style="width: 30%;" />

## [Graph Construction](#content)


1. **GGT_NN Learning Graphical State Transitions.** ICLR2017. [paper](https://openreview.net/pdf?id=HJ0NvFzxl)
   1. A GGS-NN based framework. BABI tasks
   2. Node has
      1. annotation of type belif value, vector sum to 1, 
      2. strength of existence,scalar. 
      3. hidden state
   3. Edge is belif if two nodes contain edge type, could be multiple types or no edge.
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/1862e887-659a-4294-94c2-1cc6b89c75b1" alt="Image" style="width: 50%;" />

   4. Could either output single words or sequential words, based on last layer
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/8885773c-07f4-4b94-8700-f86fc9d5d059" alt="Image" style="width: 50%;" />
      
   5. The most valuable part is how to add new nodes from GRU(G, input vector) to Graph with new nodes(annotation, strength, state)
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/d2ea4472-a331-40c1-8d52-5f12a52288dd" alt="Image" style="width: 50%;" />


## [3D PointCloud](#content)


1. **Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images** CVPR 2018. [paper](https://arxiv.org/pdf/1804.01654.pdf)
   1. Use GCN to produce 3D shape in triangular mesh from a single color image
      <img src="https://github.com/Leomingyangli/GNNPapers/assets/39786611/0a873dbd-90ae-4383-9995-f8a930479561" alt="Image" style="width: 50%;" />
