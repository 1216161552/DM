# DM
### DataMining Homework1——2019/10/15
#### Datasets——sklearn.datasets.load_digits
| Method name | Accuracy rate | Time(单位秒) |
| :-----:| :----: | :----: |
| K-means_k-means++ | 0.664382 | 0.211 |
| K-means_random | 0.664879 | 0.187 |
| K-means_PCA| 0.684580 | 0.069 |
| AffinityPropagation| 0.654879 |4.711 |
| Mean_shift| 0.554300 | 1.738 |
| SpectralClustering| 0.828585 | 0.408 |
| AgglomerativeClustering| 0.796541 | 0.179 |
| DBSCAN_min_samples=1| 0.577005 | 0.477 |
| DBSCAN_min_samples=2| 0.332550 | 0.899 |
| GaussianMixture| 0.616632| 0.857|


### Datasets——sklearn.datasets.fetch_20newsgroups
| Method name | Accuracy rate | Time(单位秒) |
| :-----:| :----: | :----: |
| K-means_k-means++ | 0.493264| 6.466|
| K-means_random | 0.565969| 10.802|
| AffinityPropagation| 0.410770|12.577|
| Mean_shift| 0.172390| 86.779|
| SpectralClustering| 0.666457| 2.708|
| AgglomerativeClustering| 0.557379| 60.692 |
| DBSCAN_min_samples=1| 0.410842| 0.449|
| DBSCAN_min_samples=2| 0.091996| 0.892|
| GaussianMixture| 0.522177| 342.461|
