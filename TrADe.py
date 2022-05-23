import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from scipy.stats.kde import gaussian_kde
from sklearn.metrics import f1_score
from scipy.optimize import differential_evolution
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
import scipy.integrate as integrate
import math


def TrADe(s_scores, t_scores, s_lambda, seed = 331):
    
    ''' Function TrADe that returns the transferred contamination factor given the source and target anomaly scores
    
    Inputs:
            - s_scores: n x 1 array containing the normalized SOURCE anomaly scores (in [0,1]);
            - t_scores: m x 1 array containing the normalized TARGET anomaly scores (in [0,1]);
            - s_lambda: float corresponding to the SOURCE predictive threshold;
    
    Outputs:
            - gamma_Tm: float corresponding to the transferred contamination factor (in (0,0.5)).
    '''
    
    np.random.seed(seed)
    
    #normalize the normal scores
    s_nor = s_scores[np.where(s_scores<=s_lambda)]
    s_nor = (s_nor - min(s_nor))/(max(s_nor) - min(s_nor))
    
    #set up the optimization problem
    args = (t_scores, s_nor, s_lambda)
    inz = np.array([[h] for h in np.arange(0.1,0.71,0.02)]) #0.91
    res = differential_evolution(minimize_KL, bounds=[(.01,.99)], args = args, maxiter=100, mutation = .4,
                         recombination=0.2, seed=seed, updating='immediate', init = inz)

    v = 0
    
    #if no solution is obtained, enlarge the search space.
    while res.fun == 10000.0 and v <= 10:
        v +=1 
        res = differential_evolution(minimize_KL, bounds=[(.01,.99)], args = args, maxiter=200, init = inz,
                                     mutation = .4 + 0.05*v, recombination=0.2+0.02+v, seed=seed, updating='immediate')
        
    #if still the solution is not obtained, then return the same threshold as the source domain
    #note that this option is never realized in our experiments.
    if res.fun == 10000.0:
        lambda_Tm = s_lambda
    else:
        lambda_Tm = res.x[0]
        
    #derive the contamination factor from the threshold by computing the proportion of examples above the threshold.
    gamma_Tm = sum(map(lambda x : x > lambda_Tm, t_scores))/len(t_scores)
    
    return gamma_Tm

def TrADe_ensemble(X_source, X_target, s_gamma, models, seed = 333, noise = False):
    
    ''' Function TrADe_ensemble that returns the transferred contamination factor for each variant: the ensemble variant, and 
        the variant where each detector is used individually.
    
    Inputs:
            - X_source: n x d array containing the SOURCE domain;
            - X_target: m x d array containing the TARGET domain;
            - s_gamma: float corresponding to the SOURCE contamination factor;
            - models: list of objects containing the detectors in the ensemble (not fitted, just hyperparameters set);
            - seed: to make experiments reproducible;
            - noise: True if you want to add some noise to the scores;
    
    Outputs:
            - ensemble_gamma: float corresponding to the transferred ENSEMBLE contamination factor (in (0,0.5));
            - gamma_Tm: list of contamination factors, one for each single variant used (in (0,0.5)).
    '''
        
    nmodels = len(models)
    
    #target_model_list = models.copy()
    gamma_Tm = np.zeros(nmodels, np.float)
    kl_val = np.zeros(nmodels, np.float)
    
    for idx_ad, ad in enumerate(models):
        #pick one detector, set the contamination factor and fit the model on the source domain
        s_ad = take_detector(ad, seed)
        #s_ad = models[idx_ad]
        s_ad.contamination = s_gamma
        s_ad.fit(X_source)
        #pick the same detector, fit the model on the target domain
        #t_ad = target_model_list[idx_ad]
        t_ad = take_detector(ad, seed)
        t_ad.fit(X_target)
        #from the 2 detectors extract the anomaly scores and normalize them (minmax normalization to [0,1])
        s_scores, t_scores, s_lambda = normalizeScoresLambda(s_ad, t_ad, s_gamma, seed = seed, noise = noise)

        #transfer the contamination factor to the target domain
        gamma_Tm[idx_ad]= TrADe(s_scores, t_scores, s_lambda, seed = seed)
        #compute the KL divergence between the scores distributions
        kl_val[idx_ad] = max(KL2(s_scores,t_scores), 0.01) #  KL_div_estimator
            
        print("Done AD number",idx_ad, "with transfer gamma =", gamma_Tm[idx_ad],"source gamma =", s_gamma, "and KL =",
              kl_val[idx_ad])
    sumkl = sum(kl_val)
    #we need more models, otherwise you can use the TrADe function.
    if nmodels == 1:
        #print('Only 1 model passed as input')
        return gamma_Tm[0], gamma_Tm[0], kl_val[0]
    #derive the final ensemble contamination factor by weighting the single detectors' estimates through the KL divergence
    weights = np.array(list(map(lambda x : (1 - x/sumkl)/(nmodels - 1), kl_val)))
    ensemble_gamma = np.dot(gamma_Tm, weights)
    
    return ensemble_gamma, gamma_Tm, kl_val


def normalizeScoresLambda(s_clf, t_clf, gamma_s, seed = 331, noise = False):
    
    np.random.seed(seed)
    s_scores = s_clf.decision_scores_ 
    t_scores = t_clf.decision_scores_ 
    
    s_scores = (s_scores - min(s_scores)) / (max(s_scores) - min(s_scores))
    t_scores = (t_scores - min(t_scores)) / (max(t_scores) - min(t_scores))
    
    if noise:
        s_scores += np.random.normal(0,0.0005,len(s_scores))
        t_scores += np.random.normal(0,0.0005,len(t_scores))
        s_scores = (s_scores - min(s_scores)) / (max(s_scores) - min(s_scores))
        t_scores = (t_scores - min(t_scores)) / (max(t_scores) - min(t_scores))
    
    s_anomalies = int(gamma_s*len(s_scores))
    idx = np.argpartition(s_scores, -s_anomalies)[-s_anomalies:]  # Indices not sorted
    s_lambda = s_scores[idx[np.argmin(s_scores[idx])]]
    
    return s_scores, t_scores, s_lambda


def getf1(X_t, y_t, detector, gamma):
    
    if gamma == 0:
        gamma = 0.000001
    elif gamma >= 0.5:
        gamma = 0.49999
        
    detector.contamination = gamma
    detector.fit(X_t)
    prediction = detector.labels_
    f1score = f1_score(y_t, prediction)

    return f1score

    
def minimize_KL(x, *scores):
    #objective function that is minimized by TrADe to obtain the best target threshold
    t_scores, s_nor, s_lambda = scores
    klval = 10000.0
    if sum(map(lambda val : val > x, t_scores))/len(t_scores) > .25: #here we check if the cont_fact is > .25
        return klval
    else:
        t_nor = t_scores[np.where(t_scores <= x)]
        if np.shape(np.unique(t_nor))[0] == 1: #if there is only one score, then we cannot compute KL
            #print("Thr =",x, "KL =", klval)
            return klval
        else:
            t_nor = (t_nor - min(t_nor))/(max(t_nor) - min(t_nor))
            klval = KL(s_nor,t_nor)
            return klval
    return klval


def KL(s_scores,t_scores):

    source_density = gaussian_kde(s_scores)
    target_density = gaussian_kde(t_scores)
    div = integrate.quad(lambda x: source_density(x)*log_dens(source_density,target_density, x) , 0, 1)[0] 
    #np.log(source_density(x)/target_density(x))
    if math.isnan(div):
        return 0.1
    return div

def log_dens(s,t,x):
    if s(x) == 0:
        return 0
    elif t(x) == 0:
        return 0
    return min(max(np.log(s(x)/t(x)),0.001),10)


