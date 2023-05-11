from sklearn.neighbors import NearestNeighbors


def two_NN(X):
    neigh = NearestNeighbors(n_neighbors=2).fit(X)

    NN = neigh.kneighbors(return_distance=True)

    NN_dist = NN[0]

    NN_dist = NN_dist[(NN_dist[:, 0] != 0) & (NN_dist[:, 1] != 0)]

    MU = NN_dist[:, 1] / NN_dist[:, 0]

    d = (X.shape[0]) / ((np.log(MU)).sum())

    return d
