from config import *


class dataPreProsessor:
    def __init__(self, path_data, path_label, cfg):
        self.path_data = path_data
        self.path_label = path_label
        self.pc_data = pcl.PointCloud.PointXYZRGBA()
        self.pc_label = pcl.PointCloud.PointXYZRGBL()

        self.pc_normal = None
        self.dim = cfg.NUMBER_OF_ATTRIBUTES
        self.labels = None
        self.cfg = cfg

        self._load_PC_data()
        self._load_PC_label()

    def _load_PC_data(self):
        reader = pcl.io.PCDReader()
        reader.read(self.path_data, self.pc_data)

    def _load_PC_label(self):
        reader = pcl.io.PCDReader()
        reader.read(self.path_label, self.pc_label)
        self.labels = np.zeros((self.pc_label.size(), self.cfg.NUMBER_OF_CLASSES), dtype=np.float32)
        index = np.asarray(self.pc_label.label, dtype=int)
        for i, ind in enumerate(index):
            self.labels[i][ind] = 1

    def getKNN(self):
        k = self.cfg.GRAPH_K
        kd = pcl.kdtree.KdTreeFLANN.PointXYZRGBA()

        if isinstance(self.pc_data, np.ndarray):
            xyz = self.pc_data[:, :3]
            rgb = np.asarray(self.pc_data[:, 3:], dtype=int)
            self.pc_data = pcl.PointCloud.PointXYZRGBA().from_array(xyz, rgb)

        kd.setInputCloud(self.pc_data)
        k_i = pclpy.pcl.vectors.Int()
        k_s = pclpy.pcl.vectors.Float()
        u = 0
        edg_b = np.ones((self.pc_data.size() * k), dtype=int)
        edg_n = np.ones((self.pc_data.size() * k), dtype=int)
        for i, p in enumerate(self.pc_data.points):
            kd.nearestKSearch(p, k, k_i, k_s)
            for index in k_i:
                edg_b[u] = i
                edg_n[u] = index
                u += 1
        return torch.tensor(edg_n), torch.tensor(edg_b)

    def applyVoxelGrid_XYZRGB(self):
        leaf_size = self.cfg.LEAF_SIZE

        vg = pcl.filters.VoxelGrid.PointXYZRGB()
        vg.setInputCloud(self.pc_data)
        vg.setLeafSize(leaf_size, leaf_size, leaf_size)
        vg.filter(self.pc_data)

    def applyPassThroughFilter(self):
        bound = self.cfg.Z_REFFERSNCE_PASSTHROUGH_FILTER
        ptf = pcl.filters.PassThrough.PointXYZRGB()
        ptf.setInputCloud(self.pc_data)
        ptf.setFilterFieldName("z")
        ptf.setFilterLimits(bound[0], bound[1])
        ptf.filter(self.pc_data)

    def applyRandomReductionFilter(self):
        X, Y = self._pcl_to_numpy(), self.labels.reshape([-1, self.cfg.NUMBER_OF_CLASSES])
        if len(X) > len(Y):
            X = X[:len(Y)]
        elif len(Y) > len(X):
            Y = Y[:len(X)]

        data = np.concatenate((X, Y), axis=1)
        np.random.shuffle(data)

        if self.pc_data.size() > self.cfg.TARGET_GRAPH_SIZE:
            fac = int(np.floor(self.pc_data.size() / self.cfg.TARGET_GRAPH_SIZE))
            X, Y = data[::fac, :X.shape[1]], data[::fac, X.shape[1]:]

            self.pc_data = X
            self.labels = Y
        else:
            print("Point cloud (" + str(self.pc_data.size()) + ") is smaller than target value (" + str(
                self.cfg.TARGET_GRAPH_SIZE) + ")")
            X, Y = data[:, :X.shape[1]], data[:, X.shape[1]:]
            self.pc_data = X
            self.labels = Y

    def addKNN(self, points=None):

        pc = pcl.PointCloud.PointXYZ()
        if pc is None:
            xyz = self.pc_data[:, :3]
        else:
            xyz = points
        pc = pc.from_array(xyz)
        kd = pcl.kdtree.KdTreeFLANN.PointXYZ()
        kd.setInputCloud(pc)
        k_i = pclpy.pcl.vectors.Int()
        k_s = pclpy.pcl.vectors.Float()
        knn = np.zeros((pc.size(), self.cfg.k_n))
        for i, p in enumerate(pc.points):
            kd.nearestKSearch(p, self.cfg.k_n, k_i, k_s)
            knn[i] = np.asarray(k_i, dtype=int)

        return torch.tensor(knn, dtype=torch.int64)

    def pcl_to_pytorch(self):
        Y = self.labels
        X = self.pc_data

        X = torch.tensor(X, dtype=self.cfg.DATA_TYPE)
        Y = torch.tensor(Y, dtype=self.cfg.DATA_TYPE)
        return X, Y

    def pcl_to_pytorch_octett(self):
        Y = self.labels
        X = self.pc_data
        ind = self.getTrainIDs()

        train = np.zeros(len(Y))

        train[ind] = 1


        data = np.concatenate((X, Y, train.reshape(-1,1)), axis=1)

        print(data.shape)

        x = self._split_nummpy([data], 0)
        xy = self._split_nummpy(x, 1)
        xyz = self._split_nummpy(xy, 2)

        attributes = self.cfg.NUMBER_OF_ATTRIBUTES
        classes = self.cfg.NUMBER_OF_CLASSES

        return_data = []

        for split_data in xyz:
            data_x = split_data[:, :attributes]
            data_y = split_data[:, attributes:attributes+classes]
            ids = split_data[:, attributes+classes]
            knn = self.addKNN(data_x[:,:3])
            return_data.append([data_x, data_y, ids, knn])

        return return_data

    def _split_nummpy(self, array, index):

        splitted_data = []

        for data in array:

            median = np.median(data[:, index])

            arr1, arr2 = [], []

            for row in data:
                if row[index] > median:
                    arr1.append(row)
                else:
                    arr2.append(row)

            splitted_data.append(np.asarray(arr1))
            splitted_data.append(np.asarray(arr2))
        return splitted_data

    def _pcl_to_numpy(self):
        if self.pc_normal is None and self.cfg.NUMBER_OF_ATTRIBUTES == 9:
            self.computeNormalVectors()

        n = np.zeros((self.pc_data.size(), self.dim))
        g = 0
        if self.dim == 9:
            for i, p in enumerate(self.pc_data.points):
                n[i][0] = p.x
                n[i][1] = p.y
                n[i][2] = p.z
                n[i][3] = p.r
                n[i][4] = p.g
                n[i][5] = p.b

                if np.isnan(self.pc_normal[i][0]):
                    g += 1
                    n[i][6] = 0
                else:
                    n[i][6] = self.pc_normal[i][0]

                if np.isnan(self.pc_normal[i][1]):
                    g += 1
                    n[i][7] = 0
                else:
                    n[i][7] = self.pc_normal[i][1]

                if np.isnan(self.pc_normal[i][2]):
                    g += 1
                    n[i][8] = 0
                else:
                    n[i][8] = self.pc_normal[i][2]
        elif self.dim == 6:
            for i, p in enumerate(self.pc_data.points):
                n[i][0] = p.x
                n[i][1] = p.y
                n[i][2] = p.z
                n[i][3] = p.r
                n[i][4] = p.g
                n[i][5] = p.b
        else:
            Exception("NOT SUPORTED OPPERATION!!")
        return n

    def computeNormalVectors(self):
        self.pc_normal = pclpy.compute_normals(self.pc_data, k=self.cfg.NORMAL_VECTOR_K)
        self.pc_normal = self.pc_normal.normals



    def printNormalVsPoint(self):
        i = random.randint(0, self.getSize() - 1)
        if self.pc_normal is None or not len(self.pc_normal) == self.getSize():
            self.computeNormalVectors()
        print(self.pc_normal[i])
        print(np.sqrt(self.pc_normal[i][0] ** 2 +
                      self.pc_normal[i][1] ** 2 +
                      self.pc_normal[i][2] ** 2))
        print(self.pc_data.xyz[i])

    def getTrainIDs(self):
        id_label_00 = [x for x in range(len(self.labels)) if self.labels[x][0] == 1]
        id_label_01 = [x for x in range(len(self.labels)) if self.labels[x][1] == 1]

        np.random.shuffle(id_label_00)
        np.random.shuffle(id_label_01)

        id_label_00 = np.random.choice(id_label_00, len(id_label_01), replace=False)

        n = np.append(id_label_01, id_label_00)
        np.random.shuffle(n)

        return n

    def getSize(self):
        return self.pc.size()

    def getPC(self):
        return self.pc
