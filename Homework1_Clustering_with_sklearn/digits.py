import json
from time import time

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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings("ignore")


def Digits():
    digits = load_digits()  # 加载数据集
    data = scale(digits.data)  # 获取data和labels
    labels = digits.target
    digits_kmeans(data, labels)
    digits_AffinityPropagation(data, labels)
    digits_mean_shift(data, labels)
    digits_SpectralClustering(data, labels)
    digits_AgglomerativeClustering(data, labels)
    digits_DBSCAN(data, labels)
    digits_GaussianMixture(data, labels)


# kmeans
def out(name, data, labels, estimator, t0):

    print('%-9s\t\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t\t%.3f\t\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_,
                                                average_method='arithmetic'),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=10), metrics.normalized_mutual_info_score(labels, estimator.labels_)))


def digits_kmeans(data, labels):
    print(50 * '*')
    print("kmeans status")
    #print('init\t\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\t\tsilhouette\t\tNMI')
    # 采用k-means++初始化进行聚类
    t0 = time()
    kmeans = KMeans(n_clusters=10, max_iter=50, n_init=10, init='k-means++').fit(data)
    print('K-means accuracy: {:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, kmeans.labels_),
                                                     time() - t0))
    # K-means accuracy: 0.664382  time：0.211425
    # out('kmeans', data, labels, kmeans, t0)

    # 采用random初始化进行聚类
    t0 = time()
    kmeans = KMeans(n_clusters=10, max_iter=50, n_init=10, init='random').fit(data)
    print('random accuracy:{:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, kmeans.labels_),
                                                   time() - t0))
    # random accuracy:0.664879  time：0.186764
    # 采用PCA降维后
    #out('random', data, labels, kmeans, t0)

    t0 = time()
    pca = PCA(n_components=10).fit(data)
    kmeans = KMeans(n_clusters=10, max_iter=50, n_init=10, init=pca.components_).fit(data)
    print('PCA accuracy:{:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, kmeans.labels_),
                                                time() - t0))
    # PCA accuracy:0.684580  time：0.069791
    #out('PCA', data, labels, kmeans, t0)


# AffinityPropagation
def digits_AffinityPropagation(data, labels):
    print(50 * '*')
    print("AffinityPropagation status")
    t0 = time()
    affinity_propagation = AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=False,
                                               affinity="euclidean").fit(data)
    # result_affinity_propagation = affinity_propagation.fit_predict(X)
    print('AffinityPropagation accuracy:{:f}  time：{:f}'.format(
        metrics.normalized_mutual_info_score(labels, affinity_propagation.labels_), time() - t0))
    # AffinityPropagation accuracy:0.654879  time：4.711166


# mean_shift
def digits_mean_shift(data, labels):
    print(50 * '*')
    print("mean_shift status")
    t0 = time()
    mean_shift = MeanShift(bandwidth=0.65, bin_seeding=True).fit(data)
    print('MeanShift accuracy: {:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, mean_shift.labels_),
                                                       time() - t0))
    # MeanShift accuracy: 0.554300  time：1.738129


# SpectralClustering
def digits_SpectralClustering(data, labels):
    print(50 * '*')
    print("SpectralClustering status")
    t0 = time()
    spectral_clustering = SpectralClustering(n_clusters=10, eigen_solver='arpack',
                                             affinity="nearest_neighbors").fit(data)
    print('SpectralClustering accuracy: {:f}  time：{:f}'.format(
        metrics.normalized_mutual_info_score(labels, spectral_clustering.labels_), time() - t0))
    # SpectralClustering accuracy: 0.828585  time：0.408024


# AgglomerativeClustering
def digits_AgglomerativeClustering(data, labels):
    print(50 * '*')
    print("AgglomerativeClustering status")
    t0 = time()
    agglomerative_clustering = AgglomerativeClustering(n_clusters=10).fit(data)
    print('AgglomerativeClustering accuracy: {:f}  time：{:f}'.format(
        metrics.normalized_mutual_info_score(labels, agglomerative_clustering.labels_), time() - t0))
    # AgglomerativeClustering accuracy: 0.796541  time：0.178536


# DBSCAN
def digits_DBSCAN(data, labels):
    print(50 * '*')
    print("DBSCAN status")
    t0 = time()
    # min_samples为1时
    dbscan = DBSCAN(eps=3, min_samples=1).fit(data)
    print('DBSCAN accuracy: {:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, dbscan.labels_),
                                                    time() - t0))
    # DBSCAN accuracy: 0.577005  time：0.476783
    # min_samples为默认的2时
    dbscan = DBSCAN(eps=3, min_samples=2).fit(data)
    print('DBSCAN accuracy: {:f}  time：{:f}'.format(metrics.normalized_mutual_info_score(labels, dbscan.labels_),
                                                    time() - t0))
    # DBSCAN accuracy: 0.332550  time：0.899813


# GaussianMixture
def digits_GaussianMixture(data, labels):
    print(50 * '*')
    print("GaussianMixture status")
    t0 = time()
    gaussian_mixture = GaussianMixture(n_components=10).fit_predict(data)
    print('GaussianMixture accuracy: {:f}  time：{:f}'.format(
        metrics.normalized_mutual_info_score(labels, gaussian_mixture), time() - t0))
    # GaussianMixture accuracy: 0.616632  time：0.857220


if __name__ == '__main__':
    Digits()
