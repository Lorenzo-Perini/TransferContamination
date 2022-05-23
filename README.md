# TransferContamination

`TransferContamination` (Transferring the contamination factor) is a GitHub repository containing the **TrADe** [1] algorithm.
It refers to the paper titled *Transferring the Contamination Factor between Anomaly Detection Domains by Shape Similarity*.

Read the pdf here: [[pdf](https://people.cs.kuleuven.be/~lorenzo.perini/files/TrADePaper.pdf)].

## Abstract

Anomaly detection attempts to find examples in a dataset that do not conform to the expected behaviour. Algorithms for this task assign an anomaly score to each example representing its degree of anomalousness. Setting a threshold on the anomaly scores enables converting these scores into a discrete prediction for each example. Setting an appropriate threshold is challenging in practice since anomaly detection is often treated as an unsupervised problem. A common approach is to set the threshold based on the dataset's contamination factor, i.e., the proportion of anomalous examples in the data. While the contamination factor may be known based on domain knowledge, it is often necessary to estimate it by labeling data. However, many anomaly detection problems involve monitoring multiple related, yet slightly different entities (e.g., a fleet of machines). Then, estimating the contamination factor for each dataset separately by labeling data would be extremely time-consuming. Therefore, this paper introduces a method for *transferring the known contamination factor* from one dataset (the source domain) to a related dataset where it is unknown (the target domain). Our approach does not require labeled target data and is based on modelling the shape of the distribution of the anomaly scores in both domains. We theoretically analyze how our method behaves when the (biased) target domain anomaly score distribution converges to its true one. Empirically, we demonstrate the effectiveness of our method on real-world datasets showing that it outperforms several baselines.


## Contents and usage

The repository contains:
- TrADe.py, a function that allows to get the estimate of the target contamination factor gamma_Tm;
- Notebook.ipynb, a notebook showing how to use TrADe on an artificial dataset;
- source_target_pair.py, a function to create two similar yet with different distribution datasets from a single dataset in order to apply TrADe meaningfully;
- Appendix.pdf, a pdf with the supplementary material used for the paper.

To use TrADe, import the github repository or simply download the files.


## TRansferring the contamination factor between Anomaly DEtection domains by shape similarity (TrADe)

Given a dataset with attributes **X**, an unsupervised anomaly detector assigns to each example an anomaly score, representing its degree of anomalousness. Here, we assume to have two related datasets, i.e. datasets with the same feature space and related field (e.g., fleets of assets). TrADe's key assumption is that if the distributions over the anomaly scores of the normal examples computed by a given anomaly detection algorithm, are similar in shape in both the source and target domain, the target anomaly score threshold can be derived from the (known) source threshold. TrADe operates as follows:

- First, it uses the known source contamination factor to construct a proper distribution over the normal examples in the source domain.

- Then, it finds a threshold on the target domain anomaly scores that makes the distribution over the anomaly scores of the resulting (normal) target examples as similar as possible to the earlier-derived source distribution. This is constructed as an optimization problem.

- Finally, it uses the resulting threshold to infer the target domain's contamination factor gamma_Tm.

Given a source dataset **X_s** with the contamination factor **s_gamma**, and a target dataset **X_t**, the algorithm is applied as follows:

```python
from pyod.models.knn import KNN
from TrADe import *

# Fit the same anomaly detection model on the source and target datasets (separately)
s_ad = KNN(n_neighbors = 25, contamination = s_gamma).fit(X_s)
t_ad = KNN(n_neighbors = 25).fit(X_t)

# Normalize the scores and get the source predictive threshold s_lambda:
s_scores, t_scores, s_lambda = normalizeScoresLambda(s_ad, t_ad, s_gamma, seed = 331, noise = False)

# Apply TrADe and get the target contamination:
gamma_Tm = TrADe(s_scores, t_scores, s_lambda, seed = 331)

```

In case you want to simulate a pair of source-target datasets, you can create such a structure using the function in source_target_pair.py as follows. Given a single dataset **X** with labels **y** (only used to modify the Y|X distribution):

```
#Source dataset X_s, source labels y_s, target dataset X_t, target labels y_t
X_s, y_s, X_t, y_t = split_biased_domains(X, y, n_clust = 5, source_prop = .8, seed = 331)

```


## Dependencies

The `TrADe` function requires the following python packages to be used:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Pandas](https://pandas.pydata.org/)


## Contact

Contact the author of the paper: [lorenzo.perini@kuleuven.be](mailto:lorenzo.perini@kuleuven.be).


## References

[1] Perini, L., Vercruyssen, V., Davis, J.: *Transferring the Contamination Factor between Anomaly Detection Domains by Shape Similarity* In: Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI-22).