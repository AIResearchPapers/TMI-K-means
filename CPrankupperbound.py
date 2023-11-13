import numpy as np
from collections import *
import math
from numpy.linalg import svd
import pandas as pd
from scipy.io import arff
import tensorly as tl
from tensorly.decomposition import parafac


def construct_mi_tensor(X, Y, Z, expand=False):
    mi_tensor = np.zeros((len(X), len(Y), len(Z)))
    print(mi_tensor)
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(len(Z)):
                if expand:
                    mi_tensor[i, j, k] = calculate_mi_of_ramdom_variables_expanded(X[i], Y[j], Z[k])
                else:
                    mi_tensor[i, j, k] = calculate_mi_of_ramdom_variables(X[i], Y[j], Z[k])
                    print(mi_tensor[i, j, k])
    return mi_tensor


def calculate_mi_of_ramdom_variables_expanded(X, Y, Z):
    return H(X) + H(Y) + H(Z) - H3(X, Y, Z)


def calculate_mi_of_ramdom_variables(X, Y, Z):
    return H(X) + H(Y) + H(Z) - H2(X, Y) - H2(Y, Z) - H2(X, Z) + H3(X, Y, Z)


def H(X):
    ''' calculate mutual entropy from observation of ramdom variable
    '''
    return calculate_ent_from_observation(X)


def H2(X, Y):
    '''
        calculat Joint entropy of two ramdom variables
    '''
    return calculate_ent_from_observation(list(zip(X, Y)))


def H3(X, Y, Z):
    '''
        calculat Joint entropy of two ramdom variables
    '''
    return calculate_ent_from_observation(list(zip(X, Y, Z)))


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


def calculate_ent(distribution):
    ''' example: input [0.5, 0.25, 0.25], return 1.5 '''

    if abs(sum(distribution) - 1) > 0.0000001:
        raise RuntimeError('sum of a distribution is not 0!')
    ent = 0
    for p in distribution:
        ent -= p * math.log(p, 2)
    return ent



file_name='Relation Network.txt'
df = pd.read_csv(file_name)
df=pd.DataFrame(df)
data = df.values.T
data = np.delete(data, (0,10,11,15), axis = 0) 

# Generate mutual information tensors
l,_ = np.shape(data) # Use l to get the number of data features 
div = [7,14] # Set the data splitting point
X = data[0:div[0]]
Y = data[div[0]:div[1]]
Z = data[div[1]:l]

mi_tensor = np.zeros((len(X), len(Y), len(Z)))
for i in range(len(X)):
    for j in range(len(Y)):
        for k in range(len(Z)):
            mi_tensor[i,j,k] = calculate_mi_of_ramdom_variables(X[i], Y[j], Z[k])

# Calculate the upper bound of the CP rank
i,j,k = np.shape(mi_tensor)

X1 = np.empty((k))
for n in range(i):
    for m in range(j):
        X1 = np.vstack((X1,mi_tensor[n,m,:]))
X1 = np.delete(X1, 0, axis = 0)

X2 = np.empty((j))
for n in range(i):
    for m in range(k):
        X2 = np.vstack((X2,mi_tensor[n,:,m]))
X2 = np.delete(X2, 0, axis = 0)

X3 = np.empty((i))
for n in range(j):
    for m in range(k):
        X3 = np.vstack((X3,mi_tensor[:,n,m]))
X3 = np.delete(X3, 0, axis = 0)

X1 = X1.T
X2 = X2.T
X3 = X3.T

X_list = [X1,X2,X3]
Sum = []
for X in X_list:
    U,S,V = svd(X)
    r = np.linalg.matrix_rank(X)
    temp = 0
    for i in range(r):
        e = np.zeros(U.shape[1])
        e[i] = 1
        M = np.dot(U,e).T
        M = np.dot(M,X)
        temp += np.linalg.matrix_rank(M)
    Sum.append(temp)
CPrank = min(Sum)

print('Mutual information tensor norms:',np.sqrt(np.sum(mi_tensor**2)),'Threshold:',np.sqrt(np.sum(mi_tensor**2))/1000)

for i in range(1,CPrank+2):
    factors = parafac(mi_tensor, rank=i)
    full_tensor = tl.kruskal_to_tensor(factors)
    Y = mi_tensor-full_tensor
    print('rank = ',i, 'error=', np.sum(Y*Y))
