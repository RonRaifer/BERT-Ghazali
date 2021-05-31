import json
import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def produce_heatmap(big_size):
    utils.heat_map_plot, ax = plt.subplots(figsize=(11, 9), dpi=100)
    # color map
    cmap = sns.light_palette("seagreen", as_cmap=True)
    sns.heatmap(utils.heat_map, annot=True, cmap=cmap, fmt=".4f",
                linewidth=0.3, cbar_kws={"shrink": .8}, annot_kws={"size": 12 if big_size else 5}, ax=ax)


def produce_kmeans():
    avgdArr = np.average(utils.heat_map, axis=0)
    kmeans = KMeans(
        init='k-means++',
        n_clusters=2,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    res2 = kmeans.fit(
        avgdArr.reshape(-1, 1))

    utils.labels = np.zeros((len(res2.labels_),), dtype=int)

    centroids = res2.cluster_centers_
    anchorGhazaliLabel = res2.labels_[0]
    anchorPseudoGhazaliLabel = res2.labels_[8]
    for i, lbl in enumerate(res2.labels_):
        if lbl == anchorPseudoGhazaliLabel:
            utils.labels[i] = 1

    utils.kmeans_plot = plt.figure(figsize=(6, 5), dpi=100)
    ax = plt.axes()
    x = np.linspace(-1, 11, 100)
    ax.plot(x, x * 0.00000000000001 + centroids[0][0])
    ax.plot(x, x * 0.00000000000001 + centroids[1][0])
    plt.scatter(range(0, len(avgdArr)), avgdArr, c=utils.labels, s=50, cmap='viridis')

    silVal = silhouette_score(avgdArr.reshape(-1, 1), utils.labels)
    utils.silhouette_calc = silVal
    silhouetteDemandSatisfied = silVal > utils.params['SILHOUETTE_THRESHOLD']
    anchorsDemandSatisfied = anchorGhazaliLabel != anchorPseudoGhazaliLabel
    if not silhouetteDemandSatisfied or not anchorsDemandSatisfied:
        print("the given configurations yield unstable classification values.")
        if not silhouetteDemandSatisfied:
            print("\tsilhouette threshold is: " + str(
                utils.params['SILHOUETTE_THRESHOLD']) + ", actual silhouette value: " + str(silVal))
        if not anchorsDemandSatisfied:
            print("\tanchors belong to the same cluster")
    else:
        print("succesfully classified, the labels are: " + str(utils.labels))


def show_results():
    if utils.heat_map is None:
        utils.heat_map = np.load(os.getcwd() + r"\Data\PreviousRuns\\" + utils.params['Name'] + ".npy")
    produce_kmeans()
    produce_heatmap(big_size=False)


def read_json():
    data_base = []
    with open('Data/PreviousRuns/PreviousRuns.json', 'r') as json_file:
        try:
            data_base = json.load(json_file)
        except Exception as e:
            print("got %s on json.load()" % e)
    return data_base


def save_results():
    data_base = read_json()
    if data_base is None:
        data_base = [utils.params]
    else:
        data_base.append(utils.params)
    with open('Data/PreviousRuns/PreviousRuns.json', 'w') as f:
        json.dump(data_base, f, indent=4)
    with open(os.getcwd() + r"\Data\PreviousRuns\\" + utils.params['Name'] + ".npy", 'wb') as m:
        np.save(m, utils.heat_map)
    with open(os.getcwd() + r"\Data\PreviousRuns\\" + utils.params['Name'] + ".txt", 'w') as log_file:
        log_file.write(utils.log_content)
