from arff import *
from HAC import *
from Kmeans import *
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples

def empty_clusters(clusters):
    for cluster in clusters:
        if len(cluster) == 0:
            return True

    return False

# Part 1
# DEBUGGING DATASET RESULTS
mat = Arff("datasets/abalone.arff",label_count=0) ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
data = raw_data

# Normalize the data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

# KMEANS
KMEANS = KMEANSClustering(k=5,debug=True)
KMEANS.fit(norm_data)
KMEANS.save_clusters("debug_kmeans.txt")

# HAC SINGLE LINK
HAC_single = HACClustering(k=5,link_type='single')
HAC_single.fit(norm_data)
HAC_single.save_clusters("debug_hac_single.txt")

# HAC COMPLETE LINK
HAC_complete = HACClustering(k=5,link_type='complete')
HAC_complete.fit(norm_data)
HAC_complete.save_clusters("debug_hac_complete.txt")

# EVALUATION DATASET RESULTS
mat = Arff("datasets/seismic-bumps_train.arff",label_count=0) ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
data = raw_data

# Normalize the data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

# KMEANS
KMEANS = KMEANSClustering(k=5,debug=True)
KMEANS.fit(norm_data)
KMEANS.save_clusters("evaluation_kmeans.txt")

# HAC SINGLE LINK
HAC_single = HACClustering(k=5,link_type='single')
HAC_single.fit(norm_data)
HAC_single.save_clusters("evaluation_hac_single.txt")

# HAC COMPLETE LINK
HAC_complete = HACClustering(k=5,link_type='complete')
HAC_complete.fit(norm_data)
HAC_complete.save_clusters("evaluation_hac_complete.txt")


# Part 2
# WITHOUT LABEL AS A FEATURE
mat = Arff("datasets/iris.arff",label_count=0) ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
data = raw_data[:, 0:-1]

# Normalize the data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

k_vals = [2, 3, 4, 5, 6, 7]
k_means_sses = []
hac_single_sses = []
hac_complete_sses = []

for k in k_vals:
    print(k)

    # KMEANS
    KMEANS = KMEANSClustering(k=k, debug=False)
    KMEANS.fit(norm_data)

    while empty_clusters(KMEANS.clusters):
        KMEANS = KMEANSClustering(k=k, debug=False)
        KMEANS.fit(norm_data)

    k_means_sses.append(KMEANS.total_sse)

    # HAC SINGLE LINK
    HAC_single = HACClustering(k=k, link_type='single')
    HAC_single.fit(norm_data)
    hac_single_sses.append(HAC_single.total_sse)

    # HAC COMPLETE LINK
    HAC_complete = HACClustering(k=k, link_type='complete')
    HAC_complete.fit(norm_data)
    hac_complete_sses.append(HAC_complete.total_sse)

# KMEANS Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, k_means_sses)
ax.set_ylabel('Total SSE')
ax.set_title('K Means Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HAC Single Link Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, hac_single_sses)
ax.set_ylabel('Total SSE')
ax.set_title('HAC Single Link Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HAC Complete Link Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, hac_complete_sses)
ax.set_ylabel('Total SSE')
ax.set_title('HAC Complete Link Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# WITH LABEL AS A FEATURE
mat = Arff("datasets/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
data = raw_data

# Normalize the data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

k_vals = [2, 3, 4, 5, 6, 7]
k_means_sses = []
hac_single_sses = []
hac_complete_sses = []

for k in k_vals:
    print(k)

    # KMEANS
    KMEANS = KMEANSClustering(k=k, debug=False)
    KMEANS.fit(norm_data)

    while empty_clusters(KMEANS.clusters):
        KMEANS = KMEANSClustering(k=k, debug=False)
        KMEANS.fit(norm_data)

    k_means_sses.append(KMEANS.total_sse)

    # HAC SINGLE LINK
    HAC_single = HACClustering(k=k, link_type='single')
    HAC_single.fit(norm_data)
    hac_single_sses.append(HAC_single.total_sse)

    # HAC COMPLETE LINK
    HAC_complete = HACClustering(k=k, link_type='complete')
    HAC_complete.fit(norm_data)
    hac_complete_sses.append(HAC_complete.total_sse)

# KMEANS Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, k_means_sses)
ax.set_ylabel('Total SSE')
ax.set_title('K Means Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HAC Single Link Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, hac_single_sses)
ax.set_ylabel('Total SSE')
ax.set_title('HAC Single Link Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HAC Complete Link Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, hac_complete_sses)
ax.set_ylabel('Total SSE')
ax.set_title('HAC Complete Link Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

kmeans = []

for _ in range(5):
    # KMEANS
    KMEANS = KMEANSClustering(k=4, debug=False)
    KMEANS.fit(norm_data)
    kmeans.append(KMEANS.total_sse)

print(kmeans)


# Part 3
# WITHOUT LABEL AS A FEATURE
mat = Arff("datasets/iris.arff",label_count=0) ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
data = raw_data[:, 0:-1]

# Normalize the data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

k_vals = [2, 3, 4, 5, 6, 7]
k_means_sses = []
hac_single_sses = []
hac_complete_sses = []

for k in k_vals:
    print(k)

    # KMEANS
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(norm_data)
    k_means_sses.append(kmeans.inertia_)

    # HAC SINGLE LINK
    hac_single = AgglomerativeClustering(n_clusters=k, linkage='single')
    preds = hac_single.fit_predict(norm_data)

    # Get centroids
    pred_tuples = [(idx, label) for idx, label in enumerate(preds)]
    pred_tuples.sort(key=lambda tup: tup[1])
    centroids = []
    curr_centroid = np.zeros(shape=(1, np.shape(norm_data)[1]))
    prev_cluster = pred_tuples[0][1]
    cluster_count = 0

    for i, label in pred_tuples:
        if label != prev_cluster:
            centroids.append(curr_centroid / cluster_count)
            prev_cluster = label
            cluster_count = 0
            curr_centroid = np.zeros(shape=(1, np.shape(norm_data)[1]))

        curr_centroid += norm_data[i, :].reshape(1, -1)
        cluster_count += 1

    centroids.append(curr_centroid / cluster_count)

    # Get total sse
    curr_sses = []
    curr_sse = 0
    prev_cluster = pred_tuples[0][1]
    j = 0

    for i, label in pred_tuples:
        if label != prev_cluster:
            curr_sses.append(curr_sse)
            curr_sse = 0
            prev_cluster = label
            j += 1

        curr_sse += np.sum((centroids[j] - norm_data[i, :].reshape(1, -1)) ** 2, axis=1)[0]

    curr_sses.append(curr_sse)

    hac_single_sses.append(sum(curr_sses))

    # HAC COMPLETE LINK
    hac_complete = AgglomerativeClustering(n_clusters=k, linkage='complete')
    preds = hac_complete.fit_predict(norm_data)

    # Get centroids
    pred_tuples = [(idx, label) for idx, label in enumerate(preds)]
    pred_tuples.sort(key=lambda tup: tup[1])
    centroids = []
    curr_centroid = np.zeros(shape=(1, np.shape(norm_data)[1]))
    prev_cluster = pred_tuples[0][1]
    cluster_count = 0

    for i, label in pred_tuples:
        if label != prev_cluster:
            centroids.append(curr_centroid / cluster_count)
            prev_cluster = label
            cluster_count = 0
            curr_centroid = np.zeros(shape=(1, np.shape(norm_data)[1]))

        curr_centroid += norm_data[i, :].reshape(1, -1)
        cluster_count += 1

    centroids.append(curr_centroid / cluster_count)

    # Get total sse
    curr_sses = []
    curr_sse = 0
    prev_cluster = pred_tuples[0][1]
    j = 0

    for i, label in pred_tuples:
        if label != prev_cluster:
            curr_sses.append(curr_sse)
            curr_sse = 0
            prev_cluster = label
            j += 1

        curr_sse += np.sum((centroids[j] - norm_data[i, :].reshape(1, -1)) ** 2, axis=1)[0]

    curr_sses.append(curr_sse)

    hac_complete_sses.append(sum(curr_sses))

# KMEANS Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, k_means_sses)
ax.set_ylabel('Total SSE')
ax.set_title('K Means Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HAC Single Link Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, hac_single_sses)
ax.set_ylabel('Total SSE')
ax.set_title('HAC Single Link Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HAC Complete Link Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, hac_complete_sses)
ax.set_ylabel('Total SSE')
ax.set_title('HAC Complete Link Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# WITH LABEL AS A FEATURE
mat = Arff("datasets/iris.arff", label_count=0)  ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
data = raw_data

# Normalize the data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

k_vals = [2, 3, 4, 5, 6, 7]
k_means_sses = []
hac_single_sses = []
hac_complete_sses = []

for k in k_vals:
    print(k)

    # KMEANS
    kmeans = KMeans(n_clusters=k, init='random')
    kmeans.fit(norm_data)
    k_means_sses.append(kmeans.inertia_)

    # HAC SINGLE LINK
    hac_single = AgglomerativeClustering(n_clusters=k, linkage='single')
    preds = hac_single.fit_predict(norm_data)

    # Get centroids
    pred_tuples = [(idx, label) for idx, label in enumerate(preds)]
    pred_tuples.sort(key=lambda tup: tup[1])
    centroids = []
    curr_centroid = np.zeros(shape=(1, np.shape(norm_data)[1]))
    prev_cluster = pred_tuples[0][1]
    cluster_count = 0

    for i, label in pred_tuples:
        if label != prev_cluster:
            centroids.append(curr_centroid / cluster_count)
            prev_cluster = label
            cluster_count = 0
            curr_centroid = np.zeros(shape=(1, np.shape(norm_data)[1]))

        curr_centroid += norm_data[i, :].reshape(1, -1)
        cluster_count += 1

    centroids.append(curr_centroid / cluster_count)

    # Get total sse
    curr_sses = []
    curr_sse = 0
    prev_cluster = pred_tuples[0][1]
    j = 0

    for i, label in pred_tuples:
        if label != prev_cluster:
            curr_sses.append(curr_sse)
            curr_sse = 0
            prev_cluster = label
            j += 1

        curr_sse += np.sum((centroids[j] - norm_data[i, :].reshape(1, -1)) ** 2, axis=1)[0]

    curr_sses.append(curr_sse)

    hac_single_sses.append(sum(curr_sses))

    # HAC COMPLETE LINK
    hac_complete = AgglomerativeClustering(n_clusters=k, linkage='complete')
    preds = hac_complete.fit_predict(norm_data)

    # Get centroids
    pred_tuples = [(idx, label) for idx, label in enumerate(preds)]
    pred_tuples.sort(key=lambda tup: tup[1])
    centroids = []
    curr_centroid = np.zeros(shape=(1, np.shape(norm_data)[1]))
    prev_cluster = pred_tuples[0][1]
    cluster_count = 0

    for i, label in pred_tuples:
        if label != prev_cluster:
            centroids.append(curr_centroid / cluster_count)
            prev_cluster = label
            cluster_count = 0
            curr_centroid = np.zeros(shape=(1, np.shape(norm_data)[1]))

        curr_centroid += norm_data[i, :].reshape(1, -1)
        cluster_count += 1

    centroids.append(curr_centroid / cluster_count)

    # Get total sse
    curr_sses = []
    curr_sse = 0
    prev_cluster = pred_tuples[0][1]
    j = 0

    for i, label in pred_tuples:
        if label != prev_cluster:
            curr_sses.append(curr_sse)
            curr_sse = 0
            prev_cluster = label
            j += 1

        curr_sse += np.sum((centroids[j] - norm_data[i, :].reshape(1, -1)) ** 2, axis=1)[0]

    curr_sses.append(curr_sse)

    hac_complete_sses.append(sum(curr_sses))

# KMEANS Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, k_means_sses)
ax.set_ylabel('Total SSE')
ax.set_title('K Means Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HAC Single Link Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, hac_single_sses)
ax.set_ylabel('Total SSE')
ax.set_title('HAC Single Link Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# HAC Complete Link Plot
x = np.arange(len(k_vals))
fig, ax = plt.subplots()
rects = ax.bar(x, hac_complete_sses)
ax.set_ylabel('Total SSE')
ax.set_title('HAC Complete Link Total SSE vs. K')
ax.set_xticks(x)
ax.set_xticklabels(k_vals)
plt.xlabel('K')
plt.show()

# CUSTOM DATASET
mat = Arff("datasets/pasture.arff",label_count=0) ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
data = raw_data

# Normalize the data
scaler = MinMaxScaler()
scaler.fit(data)
norm_data = scaler.transform(data)

# Try different clusterings
k_vals = [2, 3, 4, 5, 6, 7]

for k in k_vals:
    print(k)

    # KMEANS
    fig, ax = plt.subplots()
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(norm_data) + (k + 1) * 10])
    kmeans = KMeans(n_clusters=k, init='random')
    preds = kmeans.fit_predict(norm_data)
    print('K means SSE: ' + str(kmeans.inertia_))
    print('K means silhouette score: ' + str(silhouette_score(norm_data, preds)))
    sample_silhouette_values = silhouette_samples(norm_data, preds)
    y_lower = 10

    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[preds == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                          edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("Silhouette Plot")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_score(norm_data, preds), color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # HAC SINGLE LINK
    hac_complete = AgglomerativeClustering(n_clusters=k, linkage='single')
    preds = hac_complete.fit_predict(norm_data)
    print('HAC single silhouette score: ' + str(silhouette_score(norm_data, preds)))

    # HAC COMPLETE LINK
    hac_complete = AgglomerativeClustering(n_clusters=k, linkage='complete')
    preds = hac_complete.fit_predict(norm_data)
    print('HAC complete silhouette score: ' + str(silhouette_score(norm_data, preds)) + '\n')

plt.show()
