from config import *

def clustering(point_cloud, cfg):
    if 'DBSCAN' in cfg.CLUSTER:
        cluster = DBSCAN(eps=0.3, min_samples=10)
    elif 'OPTICS' in cfg.CLUSTER:
        cluster = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
    else:
        raise Exception('No valid clustering algorithem selected')

    new_point_cloud = np.asarray([point[:3] for point in point_cloud if point[3] is 1])


    cluster.fit(new_point_cloud)

    cluster_index = cluster.labels_

    new_point_cloud = np.concatenate([new_point_cloud, cluster_index], axis=1)

    clustered_point_cloud = [[] for _ in range(max(cluster_index))]

    for point in new_point_cloud:
        clustered_point_cloud[point[3]].append(point)

    return np.asarray(clustered_point_cloud)



if __name__ == '__main__':
    cfg = config()



'''
 cluster = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

            new_point_cloud = np.zeros((int(sum(point_cloud[:, 3])), 3))
            i = 0
            for point in point_cloud:
                if point[3] > 0:
                    new_point_cloud[i] = point[:3]
                    i -= -1
            clusters_index = cluster.fit(new_point_cloud)
            clusters_index = clusters_index.labels_

            for i in range(len(clusters_index)):
                clusters_index[i] += 1

            clustered_point_cloud = np.concatenate([new_point_cloud, clusters_index.reshape((-1, 1))], axis=1)

            print(type(clustered_point_cloud))

'''