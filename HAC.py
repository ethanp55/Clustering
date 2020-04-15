import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.distance_matrix = self._calc_distance_matrix(X)
        self.clusters = set()

        while True:
            if len(self.clusters) == 0:
                for i in range(np.shape(X)[0]):
                    self.clusters.add(frozenset([i]))

            else:
                next_indices_to_merge = self._find_next_to_merge()
                i, j = next_indices_to_merge

                two_clusters_to_merge = []

                for cluster in self.clusters:
                    if i in cluster or j in cluster:
                        two_clusters_to_merge.append(cluster)

                self.clusters.add(two_clusters_to_merge[0].union(two_clusters_to_merge[1]))
                self.clusters.remove(two_clusters_to_merge[0])
                self.clusters.remove(two_clusters_to_merge[1])

            if len(self.clusters) == self.k:
                break

        self._calc_centroid_and_sse(X)

        return self

    def _calc_centroid_and_sse(self, X):
        self.final_data = []
        self.total_sse = 0

        for cluster in self.clusters:
            centroid = np.zeros(shape=(1, np.shape(X)[1]))

            for i in cluster:
                centroid += X[i, :].reshape(1, -1)

            centroid /= len(cluster)

            sse = 0

            for i in cluster:
                sse += np.sum((centroid - X[i, :].reshape(1, -1)) ** 2, axis=1)[0]

            self.total_sse += sse

            self.final_data.append((cluster, centroid[0], sse))

    def _find_next_to_merge(self):
        min_distance = float('inf')

        for cluster in self.clusters:
            for other_cluster in self.clusters:
                if other_cluster == cluster:
                    continue

                best_cluster_distance = float('inf') if self.link_type == 'single' else 0

                for i in cluster:
                    for j in other_cluster:
                        if self.link_type == 'single':
                            if self.distance_matrix[i][j] < best_cluster_distance:
                                best_cluster_distance = self.distance_matrix[i][j]
                                curr_indices = (i, j)

                        else:
                            if self.distance_matrix[i][j] > best_cluster_distance:
                                best_cluster_distance = self.distance_matrix[i][j]
                                curr_indices = (i, j)

                if best_cluster_distance < min_distance:
                    min_distance = best_cluster_distance
                    next_indices_to_merge = curr_indices

        return next_indices_to_merge


    def _calc_distance_matrix(self, X):
        distance_matrix = []

        for i in range(np.shape(X)[0]):
            distance_matrix.append([])

            for j in range(np.shape(X)[0]):
                if j == i:
                    distance = [0]

                else:
                    distance = np.sum((X[i, :].reshape(1, -1) - X[j, :].reshape(1, -1)) ** 2, axis=1) ** 0.5

                distance_matrix[i].append(distance[0])

        return distance_matrix

    def save_clusters(self,filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
        f = open(filename, "w+")

        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(self.total_sse))

        for cluster, centroid, sse in self.final_data:
            f.write(np.array2string(centroid, precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(cluster)))
            f.write("{:.4f}\n\n".format(sse))

        f.close()
