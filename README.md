# Tensor Mutual Information K-means
The codes for the paper 'Optimal Clusters Determination in Disease Diagnosis using Tensor K-Means Variants' are published here, which includes the contrast experiment on diverse medical datasets between our method and other clustering algorithms.

**Remark:** 'ClusteringIndex.py' and 'CPrankupperbound.py' gives the experiment results on CP rank and NMI indicator; meanwhile, 'Med-2cls-NMI&Score.ipynb' has three parts: One for the experiment comparing with 5 different clustering algorithms; one for the experiment comparing with 6 different measures on 10 public datasets; one for the experiment conducting on private and some public datasets with k-fold cross-validation.

## Clustering Algorithms
- **Tensor Mutual Information K-means(Ours)**
- K-means
- K-means with TSNE
- AgglomerativeClustering
- DBSCAN
## Requirements

### Environment

1. Python 3.10.12
2. Pandas 1.5.3
3. Numpy 1.23.5
4. Sklearn 1.2.2
5. Matplotlib 3.7.1

### Dataset
The public datasets have already uploaded.

## Citing
