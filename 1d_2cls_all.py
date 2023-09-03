# coding: utf-8
import os
import io

import random
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score,pair_confusion_matrix
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

# Normalization
# Mainly used the 'DMM' one
def DMM(data):
    scaler = MinMaxScaler()
    data_standard = scaler.fit_transform(data)
    return data_standard

def DSD(data):
    scaler = StandardScaler()
    data_standard = scaler.fit_transform(data)
    return data_standard

# Evaluation criterion
def accuracy(labels_true, labels_pred):
  clusters = np.unique(labels_pred)
  labels_true = np.reshape(labels_true, (-1, 1))
  labels_pred = np.reshape(labels_pred, (-1, 1))
  count = []
  for c in clusters:
    idx = np.where(labels_pred == c)[0]
    labels_tmp = labels_true[idx, :].reshape(-1)
    count.append(np.bincount(labels_tmp).max())
  return np.sum(count) / labels_true.shape[0]

def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
  (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
  ri = (tp + tn) / (tp + tn + fp + fn)
  p, r = tp / (tp + fp), tp / (tp + fn)
  f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
  return ri, p, r, f_beta

def H(X):
    ''' calculate mutual entropy from observation of ramdom variable
    '''
    return calculate_ent_from_observation(X)

def H2(X, Y):
    '''
        calculat Joint entropy of two ramdom variables
    '''
    return calculate_ent_from_observation(list(zip(X, Y)))

def calculate_ent_from_observation(observation):
    ''' input [a, a, b, c], return 1.5 '''

    c = Counter(observation)  # Counting
    l = len(observation)
    distribution = []
    for v in c.values():
        distribution.append(v / l)

    if abs(sum(distribution) - 1) > 0.0000001:
        raise RuntimeError('sum of a distribution is not 1!')
    return calculate_ent(distribution)

# Information Entropy H(x)=-∑p(x)log p(x)
def calculate_ent(distribution):
    ''' example: input [0.5, 0.25, 0.25], return 1.5 '''

    if abs(sum(distribution) - 1) > 0.0000001:
        raise RuntimeError('sum of a distribution is not 1!')
    ent = 0
    for p in distribution:
        ent -= p * m.log(p, 2)
    return ent

# Clustering through mutual information
def cal_mi(data, clu, k):
    """
    Calculate the distance between centroids and sample points.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Distance matrix between centroids and sample points
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(2*(H(data[i])+H(clu[j])-H2(data[i],clu[j]))/(H(data[i])+H(clu[j])))
            #dis[i].append(normalized_mutual_info_score(data[i], clu[j]))
    return np.asarray(dis)

def divide(data, dis):
    """
    Group the sample points.
    :param data: Set of sample points
    :param dis: Distance from the centroids to all samples
    :param k: Number of classes
    :return: Set of Samples grouped
    """
    clusterRes = [0] * len(data) # An array of '0' elements of length 'len(data)'
    for i in range(len(data)):
        seq = np.argsort(dis[i]) # In ascending order of distance
        clusterRes[i] = seq[-1] # The last element in 'seq' joins 'clusterRes'

    return np.asarray(clusterRes)

def center(data, clusterRes, k):
    """
    Calculate the centroids.
    :param group: Set of samples grouped
    :param k: Number of classes
    :return: Centroids calculated
    """
    clunew = []
    for i in range(k):
        # Calculate the new centroid for each group
        idx = np.where(clusterRes == i)
        sum = data[idx].sum(axis=0)
        avg_sum = sum/len(data[idx])
        clunew.append(avg_sum)
    clunew = np.array(clunew)
    return clunew[:, 0:]

def classfy_mi(data, clu, k):
    """
    Update centroids iteratively.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Error and new centroids
    """
    clulist = cal_mi(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes

#2.Distance
def cal_dis(data, clu, k):
    """
    Calculate the distance between centroids and sample points.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Distance matrix between centroids and sample points
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(np.sqrt(np.sum((data[i]-clu[j])**2)))
    return np.asarray(dis)

def classfy_dis(data, clu, k):
    """
    Update centroids iteratively.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Error and new centroids
    """
    clulist = cal_dis(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes

#3.Manhattan Distance
def cal_Manhattan(data, clu, k):
    """
    Calculate the distance between centroids and sample points.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Distance matrix between centroids and sample points
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(np.sum(abs(data[i]-clu[j])))
    return np.asarray(dis)

def classfy_Manhattan(data, clu, k):
    """
    Update centroids iteratively.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Error and new centroids
    """
    clulist = cal_Manhattan(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes

#4.q=3 Minkowski Distance
def cal_Minkowski_3(data, clu, k):
    """
    Calculate the distance between centroids and sample points.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Distance matrix between centroids and sample points
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(pow(np.sum((data[i]-clu[j])**3),1/3))
    return np.asarray(dis)

def classfy_Minkowski_3(data, clu, k):
    """
    Update centroids iteratively.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Error and new centroids
    """
    clulist = cal_Minkowski_3(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes

#5.q=4 Minkowski Distance
def cal_Minkowski_4(data, clu, k):
    """
    Calculate the distance between centroids and sample points.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Distance matrix between centroids and sample points
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(pow(np.sum((data[i]-clu[j])**4),1/4))
    return np.asarray(dis)

def classfy_Minkowski_4(data, clu, k):
    """
    Update centroids iteratively.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Error and new centroids
    """
    clulist = cal_Minkowski_4(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes

#6.q=5 Minkowski Distance
def cal_Minkowski_5(data, clu, k):
    """
    Calculate the distance between centroids and sample points.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Distance matrix between centroids and sample points
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(pow(np.sum((data[i]-clu[j])**5),1/5))
    return np.asarray(dis)

def classfy_Minkowski_5(data, clu, k):
    """
    Update centroids iteratively.
    :param data: Set of sample points
    :param clu: Set of centroids
    :param k: Number of classes
    :return: Error and new centroids
    """
    clulist = cal_Minkowski_5(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes

# load data
hcvda = pd.read_csv('/original data/PROCEDUREEVENTS_MV.csv')
hcvda.replace('?', np.nan, inplace=True)
column_means = hcvda.apply(pd.to_numeric, errors='coerce').mean()
hcvda.fillna(column_means, inplace=True)
hcvda.head()
Target = np.array(hcvda.iloc[:,-1].values,dtype=float)
data = np.array(hcvda.iloc[:,:11].values,dtype=float)
data = DMM(data)

#count_0 = hcvda['status'].value_counts()[0]
#count_1 = hcvda['status'].value_counts()[1]

#sampled_0 = hcvda[hcvda['status'] == 0].sample(n=count_1, random_state=42)
#sampled_1 = hcvda[hcvda['status'] == 1].sample(n=count_1, random_state=42)

#sampled_df = pd.concat([sampled_0, sampled_1])

#Target = np.array(sampled_df.iloc[:,-1].values,dtype=float)
#data = np.array(sampled_df.iloc[:,:6].values,dtype=float)
#data = DMM(data)

#Kmeans

k = 2 # num_centroids
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k)
kmeans.fit(data)

cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

mi = []
meanmi = [0] * k
counter = [0] * k
for i in range(len(data)):
  j = labels[i]
  mi.append(2*(H(data[i])+H(cluster_centers[j])-H2(data[i],cluster_centers[j]))/(H(data[i])+H(cluster_centers[j])))
  meanmi[j] += mi[i]
  counter[j] += 1

for i in range(k):
  if counter[i] == 0:
    meanmi[i] = 0
  else:
    meanmi[i] = meanmi[i] / (counter[i])

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,labels,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# TSNE+Kmeans
from sklearn.manifold import TSNE

tsne = TSNE(n_components=3)
lowdata = tsne.fit_transform(data)

km = kmeans.fit(lowdata)

cluster_centers = km.cluster_centers_
labels = km.labels_

mi = []
meanmi = [0] * k
counter = [0] * k
for i in range(len(lowdata)):
  j = labels[i]
  mi.append(2*(H(lowdata[i])+H(cluster_centers[j])-H2(lowdata[i],cluster_centers[j]))/(H(lowdata[i])+H(cluster_centers[j])))
  meanmi[j] += mi[i]
  counter[j] += 1

for i in range(k):
  if counter[i] == 0:
    meanmi[i] = 0
  else:
    meanmi[i] = meanmi[i] / (counter[i])

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,labels,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=k)

labels = agg.fit_predict(data)

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,labels,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(min_samples=100)

labels = dbscan.fit_predict(data)

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,labels,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# Rank characteristics from small to large according to mutual information indicators
mi = []
for i in range(len(data)):
    mi.append([])
    for j in range(len(data)):
        mi[i].append(normalized_mutual_info_score(data[i], data[j]))

minmi = np.sum(mi,axis=0)
seq = np.argsort(minmi)

# The effect of mutual information index clustering
k = 2 # num_centroids

clu = data[seq[0:k]]
clu = np.asarray(clu)
err, clunew, k, clusterRes = classfy_mi(data, clu, k)
iter = 0
while np.any(abs(err) > 0):
  err, clunew, k, clusterRes = classfy_mi(data, clunew, k)
  iter += 1
  if np.any(abs(err) < 0.01):
    break

milist = cal_mi(data, clunew, k)
clusterResult = divide(data, milist)

#print('Based on mutual information index, the cluster result of %d class is as follows.' % (k))
#print(clusterResult,'\n')

print('The cluster center of %d class is as follows.' % (k))
print(clunew,'\n')

mi = []
meanmi = [0] * k
counter = [0] * k
for i in range(len(data)):
  j = clusterResult[i]
  mi.append(2*(H(data[i])+H(clunew[j])-H2(data[i],clunew[j]))/(H(data[i])+H(clunew[j])))
  meanmi[j] += mi[i]
  counter[j] += 1

for i in range(k):
  if counter[i] == 0:
    meanmi[i] = 0
  else:
    meanmi[i] = meanmi[i] / (counter[i])

print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
print(np.mean(meanmi),'\n')

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,clusterResult,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# Euclidean distance
clu = data[seq[0:k]]
clu = np.asarray(clu)
err, clunew, k, clusterRes = classfy_dis(data, clu, k)
iter = 0
while np.any(abs(err) > 0):
  err, clunew, k, clusterRes = classfy_dis(data, clunew, k)
  iter += 1
  if np.any(abs(err) < 0.01):
    break

clulist = cal_dis(data, clunew, k)
clusterResult = divide(data, clulist)

print('The cluster center of %d class is as follows.' % (k))
print(clunew,'\n')

mi = []
meanmi = [0] * k
counter = [0] * k
for i in range(len(data)):
  j = clusterResult[i]
  mi.append(2*(H(data[i])+H(clunew[j])-H2(data[i],clunew[j]))/(H(data[i])+H(clunew[j])))
  meanmi[j] += mi[i]
  counter[j] += 1

for i in range(k):
  if counter[i] == 0:
    meanmi[i] = 0
  else:
    meanmi[i] = meanmi[i] / (counter[i])

print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
print(np.mean(meanmi),'\n')

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,clusterResult,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# Manhattan distance
clu = data[seq[0:k]]
clu = np.asarray(clu)
err, clunew, k, clusterRes = classfy_Manhattan(data, clu, k)
iter = 0
while np.any(abs(err) > 0):
  err, clunew, k, clusterRes = classfy_Manhattan(data, clunew, k)
  iter += 1
  if np.any(abs(err) < 0.01):
    break

clulist = cal_Manhattan(data, clunew, k)
clusterResult = divide(data, clulist)

print('The cluster center of %d class is as follows.' % (k))
print(clunew,'\n')

mi = []
meanmi = [0] * k
counter = [1] * k
for i in range(len(data)):
  j = clusterResult[i]
  mi.append(2*(H(data[i])+H(clunew[j])-H2(data[i],clunew[j]))/(H(data[i])+H(clunew[j])))
  meanmi[j] += mi[i]
  counter[j] += 1

for i in range(k):
  if counter[i] == 0:
    meanmi[i] = 0
  else:
    meanmi[i] = meanmi[i] / (counter[i])

print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
print(np.mean(meanmi),'\n')

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,clusterResult,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# Minkowski distance(q=3)
clu = data[seq[0:k]]
clu = np.asarray(clu)
err, clunew, k, clusterRes = classfy_Minkowski_3(data, clu, k)
iter = 0
while np.any(abs(err) > 0):
  err, clunew, k, clusterRes = classfy_Minkowski_3(data, clunew, k)
  iter += 1
  if np.any(abs(err) < 0.01):
    break

clulist = cal_Minkowski_3(data, clunew, k)
clusterResult = divide(data, clulist)

print('The cluster center of %d class is as follows.' % (k))
print(clunew,'\n')

mi = []
meanmi = [0] * k
counter = [0] * k
for i in range(len(data)):
  j = clusterResult[i]
  mi.append(2*(H(data[i])+H(clunew[j])-H2(data[i],clunew[j]))/(H(data[i])+H(clunew[j])))
  meanmi[j] += mi[i]
  counter[j] += 1

for i in range(k):
  if counter[i] == 0:
    meanmi[i] = 0
  else:
    meanmi[i] = meanmi[i] / (counter[i])

print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
print(np.mean(meanmi),'\n')

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,clusterResult,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# Minkowski distance(q=4)
clu = data[seq[0:k]]
clu = np.asarray(clu)
err, clunew, k, clusterRes = classfy_Minkowski_4(data, clu, k)
iter = 0
while np.any(abs(err) > 0):
  err, clunew, k, clusterRes = classfy_Minkowski_4(data, clunew, k)
  iter += 1
  if np.any(abs(err) < 0.01):
    break

clulist = cal_Minkowski_4(data, clunew, k)
clusterResult = divide(data, clulist)

print('The cluster center of %d class is as follows.' % (k))
print(clunew,'\n')

mi = []
meanmi = [0] * k
counter = [1] * k
for i in range(len(data)):
  j = clusterResult[i]
  mi.append(2*(H(data[i])+H(clunew[j])-H2(data[i],clunew[j]))/(H(data[i])+H(clunew[j])))
  meanmi[j] += mi[i]
  counter[j] += 1

for i in range(k):
  if counter[i] == 0:
    meanmi[i] = 0
  else:
    meanmi[i] = meanmi[i] / (counter[i])

print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
print(np.mean(meanmi),'\n')

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,clusterResult,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")

# Minkowski distance(q=5)
clu = data[seq[0:k]]
clu = np.asarray(clu)
err, clunew, k, clusterRes = classfy_Minkowski_5(data, clu, k)
iter = 0
while np.any(abs(err) > 0):
  err, clunew, k, clusterRes = classfy_Minkowski_5(data, clunew, k)
  iter += 1
  if np.any(abs(err) < 0.01):
    break

clulist = cal_Minkowski_5(data, clunew, k)
clusterResult = divide(data, clulist)

print('The cluster center of %d class is as follows.' % (k))
print(clunew,'\n')

mi = []
meanmi = [0] * k
counter = [0] * k
for i in range(len(data)):
  j = clusterResult[i]
  mi.append(2*(H(data[i])+H(clunew[j])-H2(data[i],clunew[j]))/(H(data[i])+H(clunew[j])))
  meanmi[j] += mi[i]
  counter[j] += 1

for i in range(k):
  if counter[i] == 0:
    meanmi[i] = 0
  else:
    meanmi[i] = meanmi[i] / (counter[i])

print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
print(np.mean(meanmi),'\n')

ri, p, r, f_beta = get_rand_index_and_f_measure(Target,clusterResult,beta=1.)
print(f"\nri:{ri}\np:{p}\nr:{r}\nf_measure:{f_beta}")