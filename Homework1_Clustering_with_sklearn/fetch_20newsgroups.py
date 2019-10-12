# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause
import warnings

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

warnings.filterwarnings("ignore")  # 消除警告


def load():
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")

    print(__doc__)
    op.print_help()

    def is_interactive():
        return not hasattr(sys.modules['__main__'], '__file__')

    # work-around for Jupyter notebook and IPython console
    argv = [] if is_interactive() else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)
    # #############################################################################
    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    # Uncomment the following to do the analysis on all the categories
    # categories = None

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)
    labels = dataset.target
    true_k = np.unique(labels).shape[0]

    t0 = time()
    if opts.use_hashing:
        if opts.use_idf:
            # Perform an IDF normalization on the output of HashingVectorizer
            hasher = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english', alternate_sign=False,
                                       norm=None, binary=False)
            vectorizer = make_pipeline(hasher, TfidfTransformer())
        else:
            vectorizer = HashingVectorizer(n_features=opts.n_features,
                                           stop_words='english',
                                           alternate_sign=False, norm='l2',
                                           binary=False)
    else:
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                     min_df=2, stop_words='english',
                                     use_idf=opts.use_idf)
    X = vectorizer.fit_transform(dataset.data)
    if opts.n_components:
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        explained_variance = svd.explained_variance_ratio_.sum()
    return X, opts, labels


def fetch_20newsgroups_kmeans(data, opts, labels):
    print(50 * '*')
    print('Kmeans')
    t0 = time()
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1,
                    verbose=opts.verbose).fit(data)
    print('K-means accuracy:{:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, kmeans.labels_),
                                                    time() - t0))
    # K-means accuracy:0.493264  time：6.465519
    t0 = time()
    kmeans = KMeans(n_clusters=4, init='random', max_iter=100, n_init=1,
                    verbose=opts.verbose).fit(data)
    print('random accuracy: {:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, kmeans.labels_),
                                                    time() - t0))
    # random accuracy: 0.565869  time：10.802336


# AffinityPropagation
def fetch_20newsgroups_AffinityPropagation(data, opts, labels):
    print(50 * '*')
    print("AffinityPropagation status")
    t0 = time()
    affinity_propagation = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=False,
                                               affinity="euclidean", verbose=opts.verbose).fit(data)
    print('AffinityPropagation accuracy:{:f}  time：{:f}'.format(
        metrics.normalized_mutual_info_score(labels, affinity_propagation.labels_), time() - t0))
    # AffinityPropagation accuracy:0.410770  time：12.577235


# mean_shift
def fetch_20newsgroups_mean_shift(data, labels):
    print(50 * '*')
    print("mean_shift status")
    t0 = time()
    mean_shift = MeanShift(bandwidth=0.65, bin_seeding=True).fit(data.toarray())
    print('MeanShift accuracy: {:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, mean_shift.labels_),
                                                       time() - t0))
    # MeanShift accuracy: 0.172390  time：86.779402


# SpectralClustering
def fetch_20newsgroups_SpectralClustering(data, labels):
    print(50 * '*')
    print("SpectralClustering status")
    t0 = time()
    spectral_clustering = SpectralClustering(n_clusters=4, eigen_solver='arpack',
                                             affinity="nearest_neighbors").fit(data)
    print('SpectralClustering accuracy: {:f}  time：{:f}'.format(
        metrics.normalized_mutual_info_score(labels, spectral_clustering.labels_), time() - t0))
    # SpectralClustering accuracy: 0.666457  time：2.707817


# AgglomerativeClustering
def fetch_20newsgroups_AgglomerativeClustering(data, labels):
    print(50 * '*')
    print("AgglomerativeClustering status")
    t0 = time()
    agglomerative_clustering = AgglomerativeClustering(n_clusters=4).fit(data.toarray())
    print('AgglomerativeClustering accuracy: {:f}  time：{:f}'.format(
        metrics.normalized_mutual_info_score(labels, agglomerative_clustering.labels_), time() - t0))
    # AgglomerativeClustering accuracy: 0.557379  time：60.692479


# DBSCAN
def fetch_20newsgroups_DBSCAN(data, labels):
    print(50 * '*')
    print("DBSCAN status")
    t0 = time()
    # min_samples为1时
    dbscan = DBSCAN(eps=0.3, min_samples=1).fit(data)
    print('DBSCAN accuracy: {:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, dbscan.labels_),
                                                    time() - t0))
    # DBSCAN accuracy: 0.410842  time：0.448800
    # min_samples为默认的2时
    dbscan = DBSCAN(eps=0.5, min_samples=2).fit(data)
    print('DBSCAN accuracy: {:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, dbscan.labels_),
                                                    time() - t0))
    # DBSCAN accuracy: 0.091996  time：0.892634


# GaussianMixture
def fetch_20newsgroups_GaussianMixture(data, labels):
    print(50 * '*')
    print("GaussianMixture status")
    t0 = time()
    gaussian_mixture = GaussianMixture(n_components=4, tol=0.1, max_iter=20, warm_start=True).fit_predict(
        data.toarray())
    print('GaussianMixture accuracy: {:f}  time：{:f}'.format(
        metrics.normalized_mutual_info_score(labels, gaussian_mixture), time() - t0))
    # GaussianMixture accuracy: 0.522177  time：342.461635


if __name__ == '__main__':
    X, opts, labels = load()  # 加载数据集
    fetch_20newsgroups_kmeans(X, opts, labels)
    fetch_20newsgroups_AffinityPropagation(X, opts, labels)
    fetch_20newsgroups_mean_shift(X, labels)
    fetch_20newsgroups_SpectralClustering(X, labels)
    fetch_20newsgroups_AgglomerativeClustering(X, labels)
    fetch_20newsgroups_DBSCAN(X, labels)
    fetch_20newsgroups_GaussianMixture(X, labels)
