import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
from collections import *
from numpy.linalg import svd
import pandas as pd
from scipy.io import arff
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.metrics import calinski_harabaz_score
from sklearn.cluster import KMeans

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

    c = Counter(observation)  
    l = len(observation)
    distribution = []
    for v in c.values():
        distribution.append(v / l)

    if abs(sum(distribution) - 1) > 0.0000001:
        raise RuntimeError('sum of a distribution is not 0!')
    return calculate_ent(distribution)

# Information Entropy H(x)=-∑p(x)log p(x)
def calculate_ent(distribution):
    ''' example: input [0.5, 0.25, 0.25], return 1.5 '''

    if abs(sum(distribution) - 1) > 0.0000001:
        raise RuntimeError('sum of a distribution is not 0!')
    ent = 0
    for p in distribution:
        ent -= p * m.log(p, 2)
    return ent

# load data
def load_data(data_path):
    points = np.loadtxt(data_path, delimiter='\t')
    return points

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
    return np.asarray(dis)

# Euclidean distance
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

# Minkowski distance(q=3)
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

# Minkowski distance(q=4)
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

# Minkowski distance(q=5)
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


# Manhattan distance
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


def divide(data, dis):
    """
    Group the sample points.
    :param data: Set of sample points
    :param dis: Distance from the centroids to all samples
    :param k: Number of classes
    :return: Set of Samples grouped
    """
    clusterRes = [0] * len(data)
    for i in range(len(data)):
        seq = np.argsort(dis[i])
        clusterRes[i] = seq[-1]

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
    clunew = np.asarray(clunew)
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


file_name='Relation Network KEGG.txt'
df = pd.read_csv(file_name)
df=pd.DataFrame(df)
data = df.values.T
data = np.delete(data, (0,10,11,15), axis = 0)

# Rank characteristics from small to large according to mutual information indicators
mi = []
for i in range(len(data)):
    mi.append([])
    for j in range(len(data)):
        mi[i].append(2*(H(data[i])+H(data[j])-H2(data[i],data[j]))/(H(data[i])+H(data[j])))

minmi = np.sum(mi,axis=0)
seq = np.argsort(minmi)

K = [2, 3, 4, 5, 6, 7, 8]
# The effect of mutual information index clustering
for k in K:
    clu = data[seq[0:k]]  # num_centroids
    clu = np.asarray(clu)
    err, clunew, k, clusterRes = classfy_mi(data, clu, k)
    iter = 0
    while np.any(abs(err) > 0):
        err, clunew, k, clusterRes = classfy_mi(data, clunew, k)
        iter += 1
        if iter == 200:
            break

    clulist = cal_mi(data, clunew, k)
    clusterResult = divide(data, clulist)

    print('Based on mutual information index, the cluster result of %d class is as follows.' % (k))
    print(clusterResult,'\n')
    print('The clustering centroids of %d class is as follows.' % (k))
    print(clunew,'\n')

    mi = []
    meanmi = [0] * k
    counter = [0] * k
    for i in range(len(data)):
        j = clusterResult[i]
        mi.append(2 * (H(data[i]) + H(clunew[j]) - H2(data[i], clunew[j])) / (H(data[i]) + H(clunew[j])))
        meanmi[j] += mi[i]
        counter[j] += 1

    for i in range(k):
        meanmi[i] = meanmi[i] / counter[i]

    print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
    print(np.mean(meanmi),'\n')

    S = [0] * len(data)
    for i in range(len(data)):
        Sum = [0] * k
        num = [0] * k
        a = 0
        b = 0
        for j in range(len(data)):
            Sum[clusterResult[j]] += 2 * (H(data[i]) + H(data[j]) - H2(data[i], data[j])) / (H(data[i]) + H(data[j]))
            num[clusterResult[j]] += 1
        for l in range(k):
            Sum[l] = Sum[l] / num[l]
        a = Sum[clusterResult[i]]
        Sum = np.delete(Sum, clusterResult[i], axis=0)
        b = min(Sum)
        S[i] = (a - b) / max(a, b)

    print('The clustering Contour factor of %d class is as follows.' % (k))
    print(S,'\n')

# Euclidean distance
for k in K:
    clu = data[seq[0:k]]  
    clu = np.asarray(clu)
    err, clunew, k, clusterRes = classfy_dis(data, clu, k)
    iter = 0
    while np.any(abs(err) > 0):
        err, clunew, k, clusterRes = classfy_dis(data, clunew, k)
        iter += 1
        if iter == 200:
            break

    clulist = cal_dis(data, clunew, k)
    clusterResult = divide(data, clulist)

    print('the cluster result of %d class is as follows.' % (k))
    print(clusterResult,'\n')
    print('The clustering centroids of %d class is as follows.' % (k))
    print(clunew,'\n')

    mi = []
    meanmi = [0] * k
    counter = [0] * k
    for i in range(len(data)):
        j = clusterResult[i]
        mi.append(2 * (H(data[i]) + H(clunew[j]) - H2(data[i], clunew[j])) / (H(data[i]) + H(clunew[j])))
        meanmi[j] += mi[i]
        counter[j] += 1

    for i in range(k):
        meanmi[i] = meanmi[i] / counter[i]

    print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
    print(np.mean(meanmi),'\n')

    S = [0] * len(data)
    for i in range(len(data)):
        Sum = [0] * k
        num = [0] * k
        a = 0
        b = 0
        for j in range(len(data)):
            Sum[clusterResult[j]] += 2 * (H(data[i]) + H(data[j]) - H2(data[i], data[j])) / (H(data[i]) + H(data[j]))
            num[clusterResult[j]] += 1
        for l in range(k):
            Sum[l] = Sum[l] / num[l]
        a = Sum[clusterResult[i]]
        Sum = np.delete(Sum, clusterResult[i], axis=0)
        b = min(Sum)
        S[i] = (a - b) / max(a, b)

    print('The clustering Contour factor of %d class is as follows.' % (k))
    print(S,'\n')

# Minkowski distance(q=3)
for k in K:
    clu = data[seq[0:k]]  
    clu = np.asarray(clu)
    err, clunew, k, clusterRes = classfy_Minkowski_3(data, clu, k)
    iter = 0
    while np.any(abs(err) > 0):
        err, clunew, k, clusterRes = classfy_Minkowski_3(data, clunew, k)
        iter += 1
        if iter == 200:
            break

    clulist = cal_Minkowski_3(data, clunew, k)
    clusterResult = divide(data, clulist)

    print('the cluster result of %d class is as follows.' % (k))
    print(clusterResult,'\n')
    print('The clustering centroids of %d class is as follows.' % (k))
    print(clunew,'\n')

    mi = []
    meanmi = [0] * k
    counter = [0] * k
    for i in range(len(data)):
        j = clusterResult[i]
        mi.append(2 * (H(data[i]) + H(clunew[j]) - H2(data[i], clunew[j])) / (H(data[i]) + H(clunew[j])))
        meanmi[j] += mi[i]
        counter[j] += 1

    for i in range(k):
        meanmi[i] = meanmi[i] / counter[i]

    print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
    print(np.mean(meanmi),'\n')

    S = [0] * len(data)
    for i in range(len(data)):
        Sum = [0] * k
        num = [0] * k
        a = 0
        b = 0
        for j in range(len(data)):
            Sum[clusterResult[j]] += 2 * (H(data[i]) + H(data[j]) - H2(data[i], data[j])) / (H(data[i]) + H(data[j]))
            num[clusterResult[j]] += 1
        for l in range(k):
            Sum[l] = Sum[l] / num[l]
        a = Sum[clusterResult[i]]
        Sum = np.delete(Sum, clusterResult[i], axis=0)
        b = min(Sum)
        S[i] = (a - b) / max(a, b)

    print('The clustering Contour factor of %d class is as follows.' % (k))
    print(S,'\n')

# Minkowski distance(q=4)
for k in K:
    clu = data[seq[0:k]]  
    clu = np.asarray(clu)
    err, clunew, k, clusterRes = classfy_Minkowski_4(data, clu, k)
    iter = 0
    while np.any(abs(err) > 0):
        err, clunew, k, clusterRes = classfy_Minkowski_4(data, clunew, k)
        iter += 1
        if iter == 200:
            break

    clulist = cal_Minkowski_4(data, clunew, k)
    clusterResult = divide(data, clulist)

    print('the cluster result of %d class is as follows.' % (k))
    print(clusterResult,'\n')
    print('The clustering centroids of %d class is as follows.' % (k))
    print(clunew,'\n')

    mi = []
    meanmi = [0] * k
    counter = [0] * k
    for i in range(len(data)):
        j = clusterResult[i]
        mi.append(2 * (H(data[i]) + H(clunew[j]) - H2(data[i], clunew[j])) / (H(data[i]) + H(clunew[j])))
        meanmi[j] += mi[i]
        counter[j] += 1

    for i in range(k):
        meanmi[i] = meanmi[i] / counter[i]

    print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
    print(np.mean(meanmi),'\n')

    S = [0] * len(data)
    for i in range(len(data)):
        Sum = [0] * k
        num = [0] * k
        a = 0
        b = 0
        for j in range(len(data)):
            Sum[clusterResult[j]] += 2 * (H(data[i]) + H(data[j]) - H2(data[i], data[j])) / (H(data[i]) + H(data[j]))
            num[clusterResult[j]] += 1
        for l in range(k):
            Sum[l] = Sum[l] / num[l]
        a = Sum[clusterResult[i]]
        Sum = np.delete(Sum, clusterResult[i], axis=0)
        b = min(Sum)
        S[i] = (a - b) / max(a, b)

    print('The clustering Contour factor of %d class is as follows.' % (k))
    print(S,'\n')

# Minkowski distance(q=5)
for k in K:
    clu = data[seq[0:k]]  
    clu = np.asarray(clu)
    err, clunew, k, clusterRes = classfy_Minkowski_5(data, clu, k)
    iter = 0
    while np.any(abs(err) > 0):
        err, clunew, k, clusterRes = classfy_Minkowski_5(data, clunew, k)
        iter += 1
        if iter == 200:
            break

    clulist = cal_Minkowski_5(data, clunew, k)
    clusterResult = divide(data, clulist)

    print('the cluster result of %d class is as follows.' % (k))
    print(clusterResult,'\n')
    print('The clustering centroids of %d class is as follows.' % (k))
    print(clunew,'\n')

    mi = []
    meanmi = [0] * k
    counter = [0] * k
    for i in range(len(data)):
        j = clusterResult[i]
        mi.append(2 * (H(data[i]) + H(clunew[j]) - H2(data[i], clunew[j])) / (H(data[i]) + H(clunew[j])))
        meanmi[j] += mi[i]
        counter[j] += 1

    for i in range(k):
        meanmi[i] = meanmi[i] / counter[i]

    print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
    print(np.mean(meanmi),'\n')

    S = [0] * len(data)
    for i in range(len(data)):
        Sum = [0] * k
        num = [0] * k
        a = 0
        b = 0
        for j in range(len(data)):
            Sum[clusterResult[j]] += 2 * (H(data[i]) + H(data[j]) - H2(data[i], data[j])) / (H(data[i]) + H(data[j]))
            num[clusterResult[j]] += 1
        for l in range(k):
            Sum[l] = Sum[l] / num[l]
        a = Sum[clusterResult[i]]
        Sum = np.delete(Sum, clusterResult[i], axis=0)
        b = min(Sum)
        S[i] = (a - b) / max(a, b)

    print('The clustering Contour factor of %d class is as follows.' % (k))
    print(S,'\n')

# Manhattan distance
for k in K:
    clu = data[seq[0:k]]  
    clu = np.asarray(clu)
    err, clunew, k, clusterRes = classfy_Manhattan(data, clu, k)
    iter = 0
    while np.any(abs(err) > 0):
        err, clunew, k, clusterRes = classfy_Manhattan(data, clunew, k)
        iter += 1
        if iter == 200:
            break

    clulist = cal_Manhattan(data, clunew, k)
    clusterResult = divide(data, clulist)

    print('the cluster result of %d class is as follows.' % (k))
    print(clusterResult,'\n')
    print('The clustering centroids of %d class is as follows.' % (k))
    print(clunew,'\n')

    mi = []
    meanmi = [0] * k
    counter = [0] * k
    for i in range(len(data)):
        j = clusterResult[i]
        mi.append(2 * (H(data[i]) + H(clunew[j]) - H2(data[i], clunew[j])) / (H(data[i]) + H(clunew[j])))
        meanmi[j] += mi[i]
        counter[j] += 1

    for i in range(k):
        meanmi[i] = meanmi[i] / counter[i]

    print('The clustering mutual information evaluation index of %d class is as follows.' % (k))
    print(np.mean(meanmi),'\n')

    S = [0] * len(data)
    for i in range(len(data)):
        Sum = [0] * k
        num = [0] * k
        a = 0
        b = 0
        for j in range(len(data)):
            Sum[clusterResult[j]] += 2 * (H(data[i]) + H(data[j]) - H2(data[i], data[j])) / (H(data[i]) + H(data[j]))
            num[clusterResult[j]] += 1
        for l in range(k):
            Sum[l] = Sum[l] / num[l]
        a = Sum[clusterResult[i]]
        Sum = np.delete(Sum, clusterResult[i], axis=0)
        b = min(Sum)
        S[i] = (a - b) / max(a, b)

    print('The clustering Contour factor of %d class is as follows.' % (k))
    print(S,'\n')