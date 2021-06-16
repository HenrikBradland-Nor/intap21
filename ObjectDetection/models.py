from config import *
from ObjectDetection.NN_suport import *

'''
class DGCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv1 = dglnn.EdgeConv(in_feat=cfg.X_FEATS, out_feat=cfg.DGCNN_H1_FEATS, batch_norm=True)
        self.conv2 = dglnn.EdgeConv(in_feat=cfg.DGCNN_H1_FEATS, out_feat=cfg.DGCNN_H2_FEATS, batch_norm=True)
        self.conv3 = dglnn.EdgeConv(in_feat=cfg.DGCNN_H2_FEATS, out_feat=cfg.DGCNN_H3_FEATS, batch_norm=True)
        self.conv4 = dglnn.EdgeConv(in_feat=cfg.DGCNN_H3_FEATS, out_feat=cfg.DGCNN_H4_FEATS, batch_norm=True)
        self.L1 = nn.Linear(in_features=cfg.DGCNN_H4_FEATS + cfg.DGCNN_H3_FEATS + cfg.DGCNN_H2_FEATS
                                        + cfg.DGCNN_H1_FEATS, out_features=cfg.DGCNN_L1_FEATS)
        self.L2 = nn.Linear(in_features=cfg.DGCNN_L1_FEATS, out_features=cfg.DGCNN_L2_FEATS)
        self.L3 = nn.Linear(in_features=cfg.DGCNN_L2_FEATS, out_features=cfg.NUMBER_OF_CLASSES)
        self.SoftMax = nn.Softmax(dim=1)
        self.kg = KNNGraph(cfg.GRAPH_K)

    def forward(self, inputs, graph=None):
        if graph is None:
            graph = self.kg(inputs[:, :3]).to(self.cfg.DEVICE)
        h1 = self.conv1(graph, inputs)
        h1 = F.leaky_relu(h1)
        # AvgPool/MaxPool ???

        g2 = self.kg(h1).to(self.cfg.DEVICE)
        h2 = F.leaky_relu(self.conv2(g2, h1))
        # AvgPool/MaxPool ???

        g3 = self.kg(h2).to(self.cfg.DEVICE)
        h3 = F.leaky_relu(self.conv3(g3, h2))
        # AvgPool/MaxPool ???

        g4 = self.kg(h3).to(self.cfg.DEVICE)
        h4 = F.leaky_relu(self.conv4(g4, h3))
        # AvgPool/MaxPool ???

        h = torch.cat((h1, h2, h3, h4), dim=1)

        h = h.view(h.size(0), -1)

        h = self.L1(h)
        h = F.relu(h)
        h = self.L2(h)
        h = F.relu(h)
        h = self.L3(h)
        out = self.SoftMax(h)
        return out
'''


class DGCNN(nn.Module):
    def __init__(self, cfg):
        super(DGCNN, self).__init__()
        self.cfg = cfg
        self.k = cfg.GRAPH_K

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(cfg.DGCNN_L1_FEATS)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, cfg.DGCNN_L1_FEATS, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=cfg.DROPOUT)
        self.conv9 = nn.Conv1d(256, cfg.NUMBER_OF_CLASSES, kernel_size=1, bias=False)

        #self.softmax = nn.Sigmoid()# nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(1, 6, -1)
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k)#, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return F.softmax(x.view(-1, self.cfg.NUMBER_OF_CLASSES), dim=1 )

class DGCNN_cls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv1 = dglnn.EdgeConv(in_feat=cfg.X_FEATS, out_feat=cfg.DGCNN_H1_FEATS, batch_norm=True)
        self.conv2 = dglnn.EdgeConv(in_feat=cfg.DGCNN_H1_FEATS, out_feat=cfg.DGCNN_H2_FEATS, batch_norm=True)
        self.conv3 = dglnn.EdgeConv(in_feat=cfg.DGCNN_H2_FEATS, out_feat=cfg.DGCNN_H3_FEATS, batch_norm=True)
        self.conv4 = dglnn.EdgeConv(in_feat=cfg.DGCNN_H3_FEATS, out_feat=cfg.DGCNN_H4_FEATS, batch_norm=True)

        self.mlpEmb = nn.Linear(in_features=cfg.DGCNN_H3_FEATS + cfg.DGCNN_H2_FEATS + cfg.DGCNN_H1_FEATS,
                                out_features=cfg.DGCNN_L1_FEATS)

        self.L1 = nn.Linear(in_features=cfg.DGCNN_H4_FEATS + cfg.DGCNN_H3_FEATS + cfg.DGCNN_H2_FEATS
                                        + cfg.DGCNN_H1_FEATS, out_features=cfg.DGCNN_L1_FEATS)
        self.L2 = nn.Linear(in_features=cfg.DGCNN_L1_FEATS, out_features=cfg.DGCNN_L2_FEATS)
        self.L3 = nn.Linear(in_features=cfg.DGCNN_L2_FEATS, out_features=cfg.NUMBER_OF_CLASSES)

        self.SoftMax = nn.Softmax(dim=1) # nn.Tanh()# nn.Sigmoid() # nn.Softmax(dim=1)

        self.kg = KNNGraph(cfg.GRAPH_K)
        self.dp1 = nn.Dropout(p=cfg.DROPOUT)
        self.dp2 = nn.Dropout(p=cfg.DROPOUT)

    def forward(self, inputs, graph=None):
        if graph is None:
            graph = self.kg(inputs[:, :3]).to(self.cfg.DEVICE)
        h1 = F.leaky_relu(self.conv1(graph, inputs))
        h1 = F.leaky_relu(self.conv2(graph, h1))
        # AvgPool/MaxPool ???

        g2 = self.kg(h1).to(self.cfg.DEVICE)
        h2 = F.leaky_relu(self.conv2(g2, h1))
        h2 = F.leaky_relu(self.conv2(g2, h2))
        # AvgPool/MaxPool ???

        g3 = self.kg(h2).to(self.cfg.DEVICE)
        h3 = F.leaky_relu(self.conv3(g3, h2))
        # AvgPool/MaxPool ???

        h = torch.cat((h1, h2, h3), dim=1)

        h = self.mlpEmb(h)
        h = h.max(dim=0, keepdim=True)[0]

        h = h.repeat(inputs.size(0), 1)

        h = h.view(h1.size(0), -1)
        h = torch.cat((h, h1, h2, h3), dim=1)

        h = h.view(inputs.size(0), -1)

        h = self.dp1(F.leaky_relu(self.L1(h)))
        h = self.dp2(F.leaky_relu(self.L2(h)))
        h = self.SoftMax(self.L3(h))

        return h


class VoxelNet(nn.Module):
    def __init__(self, cfg):
        super(VoxelNet, self).__init__()
        self.cfg = cfg
        self.svfe = SVFE(self.cfg)
        self.cml = ConvoMidLayer(self.cfg)
        self.mlp = MLP(self.cfg)
        self.tli = trilinear_interpolation(self.cfg)

    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]
        if self.cfg.SUPORTS_CUDA:
            dense_feature = Variable(torch.zeros(dim, 1, self.cfg.voxelX, self.cfg.voxelY,
                                                 self.cfg.voxelZ).to(self.cfg.DEVICE))  # dense_feature = Variable(torch.zeros(dim, 1, self.cfg.voxelX, self.cfg.voxelY, self.cfg.voxelZ).cuda())
        else:
            dense_feature = Variable(torch.zeros(dim, 1, self.cfg.voxelX, self.cfg.voxelY,
                                                 self.cfg.voxelZ))  # dense_feature = Variable(torch.zeros(dim, 1, self.cfg.voxelX, self.cfg.voxelY, self.cfg.voxelZ))

        dense_feature[:, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = sparse_features.transpose(1,0)  # , coords[:, 3]

        return dense_feature.transpose(0, 1)

    def forward(self, point_cloud):
        voxel_features, voxel_coords = voxelize_inset(point_cloud, self.cfg)
        vwfs = self.svfe(voxel_features)
        vwfs = self.voxel_indexing(vwfs, voxel_coords)
        cml = self.cml(vwfs)

        point_voxel_index, point_voxel_dist = get_voxel_nn(self.cfg, point_cloud)


        point_features = cml[point_voxel_index[:,:,0],
                             point_voxel_index[:,:,1],
                             point_voxel_index[:,:,2]]


        point_features = torch.mul(point_features, point_voxel_dist)

        point_features = torch.cat([point_features.view(-1, 64*8),
                                    point_cloud[:, :3]], dim=1)


        #print("hei \n"*20)

        point_classes = self.mlp(point_features)

        return point_classes


class PointGNN(nn.Module):
    def __init__(self, cfg):
        super(PointGNN, self).__init__()
        self.cfg = cfg
        self.cls_fn = multi_layer_fc_fn(Ks=[9, 1384], num_layers=2, num_classes=cfg.NUMBER_OF_CLASSES,
                                        is_logits=True)  # Ks=[300, 64]

    def forward(self, features):
        logits = self.cls_fn(features)
        logits = logits.view(-1, self.cfg.NUMBER_OF_CLASSES)
        return F.softmax(logits, dim=1)


class PointNet(nn.Module):
    def __init__(self, cfg, feature_transform=True):
        super(PointNet, self).__init__()
        self.k = cfg.NUMBER_OF_CLASSES
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.round = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(1, x.shape[1], x.shape[0])
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = self.round(x.view(-1, self.k))
        # x = x.view(batchsize, n_pts, self.k)
        return x  # ,trans, trans_feat


class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )
        self.device = device

        #self = self.to(device)

    def forward(self, input):
        r"""
            Forward pass
            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points
            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        N = input.size(1)
        d = self.decimation

        coords = input[...,:3].clone()#.cpu()
        x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)
        x = self.bn_start(x) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        x = x[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]


        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:,:N//decimation_ratio],#.cpu().contiguous(), # original set
                coords[:,:d*N//decimation_ratio],#.cpu().contiguous(), # upsampled set
                1
            ) # shape (B, N, 1)
            print(neighbors.get_device())
#           if neighbors.get_device() < 0:
            neighbors = neighbors.to(self.device).contiguous()

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            print("x", x.size())
            print("extended_neighbors", extended_neighbors.size())

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x), dim=1) #x_stack.pop()

            print("HEI")
            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:,:,torch.argsort(permutation)]

        scores = self.fc_end(x)

        return scores.squeeze(-1)


class RandomForest:
    def __init__(self, cfg):
        self.model = RandomForestClassifier(n_estimators=cfg.number_of_trees,
                                            n_jobs=cfg.number_of_jobs_in_parallel,
                                            warm_start=True)

    def fit(self, atributes, labels):
        self.model.fit(atributes[:, 3:], labels)

    def pred(self, atributes):
        return self.model.predict(atributes[:, 3:])


class SVM:
    def __init__(self, cfg):
        self.model = svm.SVC(C=cfg.svm_C,
                             kernel=cfg.svm_kernel)

    def fit(self, atributes, labels):
        self.model.fit(atributes[:, 3:], labels)

    def pred(self, atributes):
        return self.model.predict(atributes[:, 3:])
