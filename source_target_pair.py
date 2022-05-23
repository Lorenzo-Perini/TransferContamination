import numpy as np
import random
import math
from sklearn.cluster import KMeans

def split_biased_domains(X, y, n_clust = 5, source_prop = 0.8, seed = 331):
    
    '''
    Our goal is to obtain two domains with different distribution and contamination factor from a single given dataset 
    in order to apply our method. We treat separately the normal and the anomalous class. 

    - Normal class. First we cluster the data with 5 clusters by kMeans, where 5 is small enough to get different distributions. 
    Second, we assign the square of a random integer weight in [1,100] to each cluster, i.e. examples in the same cluster 
    get the same weight. Finally, we sample according to the weights 80% of the examples for the source domain, and the 
    remaining 20% goes to the target domain. 

    - Anomaly class. Let's call r = #A_s / #A_t, the ratio between the number of anomalies in the source and the target domain. 
    First, we draw a random number between 0 and 1 from a Beta(2,6) random variable (mean = 0.25). Second, we flip a coin 
    and assign this value to r or to 1/r based on the result. This allows us to equally get more anomalies in either the 
    domains. 
    Finally, we derive the number of anomalies for each domain and randomly (uniformly) assign them. 
    '''
    
    np.random.seed(seed)
    ys = 0
    yt = 0
    norm_idx = np.where(y ==0)[0]
    n = np.shape(X)[0]
    source_size = int(np.shape(norm_idx)[0]*source_prop)
    target_size = int(np.shape(norm_idx)[0]*(1-source_prop))

    while ys <= 5 or ys >= source_size or yt <= 5 or yt >= target_size:
        rate = np.random.beta(2,6,1)[0]
        if np.random.binomial(1,0.5) == 0: #rate source/target
            ys = int((rate*sum(y))/(1+rate))
            yt = sum(y) - ys
        else:
            yt = int((rate*sum(y))/(1+rate))
            ys = sum(y) - yt

    source_idx = np.random.choice(np.where(y == 1)[0], size=ys, replace=False)
    biased_weights = {}
    cluster_labels = KMeans(n_clusters = n_clust, random_state=seed).fit(X[norm_idx]).labels_ +1
    for w in range(1,n_clust+1):
        biased_weights[w] = np.random.randint(1,100)**2
    weights = np.asarray([biased_weights[w] for w in cluster_labels])
    weights = weights/sum(weights)
    # draw subsample
    source_idx = np.concatenate((np.random.choice(norm_idx, size=source_size, p=weights, replace=False), source_idx))

    X_source = X[source_idx, :]
    y_source = y[source_idx]
    X_target = X[[i for i in np.arange(n) if i not in source_idx],:]
    y_target = y[[i for i in np.arange(n) if i not in source_idx]]

    return X_source, y_source, X_target, y_target