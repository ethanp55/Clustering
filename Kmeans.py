import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,debug=False): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.n_features = np.shape(X)[1]
        centroids = self._get_initial_centroids(X)

        changed = True

        while changed:
            self.clusters = [0] * self.k

            for i in range(np.shape(X)[0]):
                # Initialize cluster so it is out of bounds for the cluster array
                cluster = self.k + 1
                min_distance = float('inf')

                for j in range(len(centroids)):
                    distance = np.sum((X[i, :].reshape(1, -1) - centroids[j].reshape(1, -1)) ** 2, axis=1) ** 0.5

                    if distance[0] < min_distance:
                        min_distance = distance[0]
                        cluster = j

                if self.clusters[cluster] == 0:
                    self.clusters[cluster] = [i]

                else:
                    self.clusters[cluster].append(i)

            new_centroids = self._calc_new_centroids(X)

            equal_centroids = [np.all(np.equal(new_centroids[i], centroids[i])) for i in range(len(new_centroids))]

            if False not in set(equal_centroids):
                changed = False

            else:
                centroids = new_centroids

        self._calc_centroid_and_sse(X, centroids)

        return self

    def _calc_centroid_and_sse(self, X, centroids):
        self.final_data = []
        self.total_sse = 0

        for i in range(len(self.clusters)):
            sse = 0

            for j in self.clusters[i]:
                sse += np.sum((centroids[i] - X[j, :].reshape(1, -1)) ** 2, axis=1)[0]

            self.total_sse += sse

            self.final_data.append((self.clusters[i], centroids[i], sse))

    def _calc_new_centroids(self, X):
        new_centroids = [0] * self.k

        for i in range(len(self.clusters)):
            centroid = np.zeros(shape=(1, self.n_features))

            for j in self.clusters[i]:
                centroid += X[j, :].reshape(1, -1)

            centroid /= len(self.clusters[i])

            new_centroids[i] = centroid[0]

        return new_centroids

    def _get_initial_centroids(self, X):
        centroids = []

        if self.debug:
            for i in range(self.k):
                centroids.append(X[i, :].reshape(1, -1))

        else:
            indices = np.arange(0, np.shape(X)[0])
            random_indices = np.random.choice(indices, size=self.k, replace=False)

            for i in random_indices:
                centroids.append(X[i, :].reshape(1, -1))

        return centroids

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
