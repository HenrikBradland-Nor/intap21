from config import *


# DGCNN

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


# PointNet


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.view(-1, 3, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]  # [0]
        x = x.view(-1, 1024)
        x = self.fc1(x)
        # x = self.bn4(x)
        x = F.relu(x)
        # x = self.bn5(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=6)  # STN3d()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


# VoxelNet

# conv2d + bn + relu
class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


# conv3d + bn + relu
class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


# Fully Connected Network
class FCN(nn.Module):

    def __init__(self, cin, cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk * t, -1))
        x = F.relu(self.bn(x))
        return x.view(kk, t, -1)


# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self, cin, cout, cfg):
        super(VFE, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FCN(cin, self.units)
        self.cfg = cfg

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, min(self.cfg.maxPointsVoxel, len(pwf[0])), 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2)  # pwcf = torch.cat((pwf, laf), dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)  # mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf


# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self, cfg):
        super(SVFE, self).__init__()
        self.cfg = cfg
        self.vfe_1 = VFE(6, 32, cfg=self.cfg)  # self.vfe_1 = VFE(7, 32, cfg=self.cfg)
        self.vfe_2 = VFE(32, 128, cfg=self.cfg)
        self.fcn = FCN(128, 128)

    def forward(self, x):
        mask = torch.ne(torch.max(x, 2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x, 1)[0]
        return x


class ConvoMidLayer(nn.Module):
    def __init__(self, cfg):
        super(ConvoMidLayer, self).__init__()
        self.cfg = cfg
        self.conv3d_1 = Conv3d(128, 64, 3, s=(1, 1, 1), p=(1, 1, 1), batch_norm=False)
        self.conv3d_2 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(1, 1, 1), batch_norm=False)
        self.conv3d_3 = Conv3d(64, 64, 3, s=(1, 1, 1), p=(1, 1, 1), batch_norm=False)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x.view(self.cfg.voxelX, self.cfg.voxelY, self.cfg.voxelZ, 64)


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(64 * 8 + 3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, self.cfg.NUMBER_OF_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=0)

        return x


class trilinear_interpolation():
    def __init__(self, cfg):
        self.cfg = cfg

    def forward(self, point, voxel_grid):

        # new_points = []
        # for point in point_cloud:
        voxel_indexes = self.point_to_voxel_index(point)

        intermediat_points_x = []
        for vi in range(0, len(voxel_indexes), 2):
            intermediat_points_x.append(self.linear_interpolation(voxel_indexes[vi],
                                                                  voxel_indexes[vi + 1],
                                                                  point,
                                                                  axis=0,
                                                                  voxel_grid=voxel_grid,
                                                                  from_voxel=True))

        intermediat_points_y = []
        for ind in range(0, len(intermediat_points_x), 2):
            intermediat_points_y.append(self.linear_interpolation(intermediat_points_x[ind],
                                                                  intermediat_points_x[ind + 1],
                                                                  point,
                                                                  axis=1,
                                                                  voxel_grid=None,
                                                                  from_voxel=False))

        return self.linear_interpolation(intermediat_points_y[0],
                                         intermediat_points_y[1],
                                         point,
                                         axis=2,
                                         voxel_grid=None,
                                         from_voxel=False)

        # new_points = torch.as_tensor(new_points[0]) #torch.from_numpy(np.asarray(new_points[0], dtype=np.float32))
        # print(type(new_points))
        # print(new_points)
        # return new_points

    def linear_interpolation(self, a, b, point_x, axis=1, voxel_grid=None, from_voxel=True):
        if axis is 0:
            ref = self.cfg.pcRangeX[0]
        elif axis is 1:
            ref = self.cfg.pcRangeY[0]
        elif axis is 2:
            ref = self.cfg.pcRangeZ[0]
        else:
            raise Exception('Invalide axis')

        point = torch.zeros(3 + 64)

        if from_voxel:
            da = abs(ref + a[axis] * self.cfg.voxelDIm[axis] - point_x[axis])
            db = abs(ref + b[axis] * self.cfg.voxelDIm[axis] - point_x[axis])

            wa = da / (da + db)
            wb = db / (da + db)

            atr = wa * voxel_grid[a[0], a[1], a[2], :] + \
                  wb * voxel_grid[b[0], b[1], b[2], :]

            for ind in range(3):
                point[ind] = (wa * (ref + a[ind] * self.cfg.voxelDIm[ind]) +
                              wb * (ref + b[ind] * self.cfg.voxelDIm[ind]))
            point[3:] = atr

            return point

        else:
            da = abs(a[axis] - point_x[axis])
            db = abs(b[axis] - point_x[axis])

            wa = da / (da + db)
            wb = db / (da + db)

            atr = wa * a[3] + wb * b[3]

            for ind in range(3):
                point[ind] = (wa * a[ind] + wb * b[ind])

            point[3:] = atr

            return point

    def point_to_voxel_index(self, point):
        ind_x = (point[0] - self.cfg.pcRangeX[0]) / self.cfg.voxelDIm[0]
        ind_y = (point[1] - self.cfg.pcRangeY[0]) / self.cfg.voxelDIm[1]
        ind_z = (point[2] - self.cfg.pcRangeZ[0]) / self.cfg.voxelDIm[2]

        return [[math.ceil(ind_x), math.ceil(ind_y), math.ceil(ind_z)],
                [math.floor(ind_x), math.ceil(ind_y), math.ceil(ind_z)],

                [math.ceil(ind_x), math.floor(ind_y), math.ceil(ind_z)],
                [math.floor(ind_x), math.floor(ind_y), math.ceil(ind_z)],

                [math.ceil(ind_x), math.ceil(ind_y), math.floor(ind_z)],
                [math.floor(ind_x), math.ceil(ind_y), math.floor(ind_z)],

                [math.ceil(ind_x), math.floor(ind_y), math.floor(ind_z)],
                [math.floor(ind_x), math.floor(ind_y), math.floor(ind_z)]]


def get_voxel_nn(cfg, point_cloud):
    pcRangeX = [min(point_cloud[:, 0]), max(point_cloud[:, 0])]
    pcRangeY = [min(point_cloud[:, 1]), max(point_cloud[:, 1])]
    pcRangeZ = [min(point_cloud[:, 2]), max(point_cloud[:, 2])]

    x = torch.div(torch.add(point_cloud[:, 0], -pcRangeX[0]), cfg.voxelDIm[0]).view(-1, 1).to(cfg.DEVICE)
    y = torch.div(torch.add(point_cloud[:, 1], -pcRangeY[0]), cfg.voxelDIm[1]).view(-1, 1).to(cfg.DEVICE)
    z = torch.div(torch.add(point_cloud[:, 2], -pcRangeZ[0]), cfg.voxelDIm[2]).view(-1, 1).to(cfg.DEVICE)

    v = torch.cat((torch.cat([torch.ceil(x), torch.ceil(y), torch.ceil(z)], dim=1).view(-1, 1, 3),
                   torch.cat([torch.floor(x), torch.ceil(y), torch.ceil(z)], dim=1).view(-1, 1, 3),
                   torch.cat([torch.ceil(x), torch.floor(y), torch.ceil(z)], dim=1).view(-1, 1, 3),
                   torch.cat([torch.floor(x), torch.floor(y), torch.ceil(z)], dim=1).view(-1, 1, 3),
                   torch.cat([torch.ceil(x), torch.ceil(y), torch.floor(z)], dim=1).view(-1, 1, 3),
                   torch.cat([torch.floor(x), torch.ceil(y), torch.floor(z)], dim=1).view(-1, 1, 3),
                   torch.cat([torch.ceil(x), torch.floor(y), torch.floor(z)], dim=1).view(-1, 1, 3),
                   torch.cat([torch.floor(x), torch.floor(y), torch.floor(z)], dim=1).view(-1, 1, 3)
                   ), dim=1).to(cfg.DEVICE)

    xyz = torch.repeat_interleave(torch.cat([x, y, z], dim=1), 8, dim=0)
    #print(xyz.view(-1, 8, 3).size(), v.size())
    d = torch.add(v, -xyz.view(-1, 8, 3))
    d = torch.pow(d[:,:,0]+d[:,:,1]+d[:,:,2], 1)
    d = torch.pow(d, -1)

    d = torch.repeat_interleave(d.view(-1, 8, 1), 64, 2)

    return v.long(), d


def voxelize_inset(lidar, cfg):  # preprocessing
    lidar = lidar.cpu()
    lidar = lidar.numpy()

    cfg.pcRangeX = [min(lidar[:, 0]), max(lidar[:, 0])]
    cfg.pcRangeY = [min(lidar[:, 1]), max(lidar[:, 1])]
    cfg.pcRangeZ = [min(lidar[:, 2]), max(lidar[:, 2])]

    voxelX = math.ceil((cfg.pcRangeX[1] - cfg.pcRangeX[0]) / cfg.voxelDIm[0])
    voxelY = math.ceil((cfg.pcRangeY[1] - cfg.pcRangeY[0]) / cfg.voxelDIm[1])
    voxelZ = math.ceil((cfg.pcRangeZ[1] - cfg.pcRangeZ[0]) / cfg.voxelDIm[2])

    np.random.shuffle(lidar)
    voxel_coords_initial = ((lidar[:, :3] - np.array([cfg.pcRangeX[0], cfg.pcRangeY[0], cfg.pcRangeZ[0]])) / (
        voxelX, voxelY, voxelZ))
    # convert to  (D, H, W)
    voxel_coords = voxel_coords_initial[:, [2, 1, 0]].astype(np.int32)
    # unique voxel coordinates, index in original array of each element in unique array
    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)
    voxel_features = list()

    for i in range(len(voxel_coords)):
        voxel = np.zeros((cfg.maxPointsVoxel, cfg.NUMBER_OF_ATTRIBUTES),
                         dtype=np.float32)  # voxel = np.zeros((cfg.maxPointsVoxel, 7), dtype=np.float32)
        # all pts belong to that voxel
        pts = lidar[inv_ind == i]
        if voxel_counts[i] > cfg.maxPointsVoxel:
            pts = pts[:cfg.maxPointsVoxel, :]
            voxel_counts[i] = cfg.maxPointsVoxel
        # normalize all pts in this voxel and append it colomnwise behind voxel coords
        voxel[:pts.shape[0], :] = pts[:, :cfg.NUMBER_OF_ATTRIBUTES] - np.mean(pts[:, :cfg.NUMBER_OF_ATTRIBUTES],
                                                                              0)  # np.concatenate((pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
        voxel_features.append(voxel)
    voxel_features = np.array(voxel_features)

    return torch.tensor(voxel_features).to(cfg.DEVICE), \
           torch.tensor(voxel_coords, dtype=torch.long).to(cfg.DEVICE)


def vn_knn(ref, query, k):
    ref = ref.view(1, 3, -1).to("cuda:0")
    query = query.view(1, 3, -1).to("cuda:0")
    ref_t = torch.transpose(ref, 1, 2)
    ref_t = ref_t.float().to("cuda:0")
    ref_t2 = torch.sum(ref_t * ref_t, dim=2, keepdim=True)

    query_t = torch.transpose(query, 1, 2)
    query2 = torch.sum(query * query, dim=1, keepdim=True)

    m = torch.bmm(ref_t, query)

    dist = ref_t2 - 2 * m + query2
    top_ind = torch.topk(dist, k, largest=False, dim=2)[1].long().to("cuda:0")

    top_dist = torch.gather(dist, 2, top_ind).to("cuda:0")

    return top_ind, top_dist


# Point-GNN
def multi_layer_fc_fn(Ks=[300, 64, 32, 64], num_classes=4, is_logits=False, num_layers=4):
    assert len(Ks) == num_layers
    linears = []
    for i in range(1, len(Ks)):
        linears += [
            nn.Linear(Ks[i - 1], Ks[i]),
            nn.ReLU(),
            nn.BatchNorm1d(Ks[i])
        ]
    if is_logits:
        linears += [
            nn.Linear(Ks[-1], num_classes)]
    else:
        linears += [
            nn.Linear(Ks[-1], num_classes),
            nn.ReLU(),
            nn.BatchNorm1d(num_classes)
        ]
    return nn.Sequential(*linears)


# RandLA-Net
class SharedMLP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            transpose=False,
            padding_mode='zeros',
            bn=False,
            activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network
            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass
            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple
            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)
        # if USE_CUDA:
        # neighbors = neighbors.to("cuda:0")
        # extended_coords = extended_coords.to("cuda:0")
        # dist = dist.to("cuda:0")

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass
            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out // 2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass
            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud
            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn(coords,  # .cpu().contiguous(),
                         coords,  # .cpu().contiguous(),
                         self.num_neighbors)  # cpu(). .

        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))


## LOSS FUNCTIONS

from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        # target = target[:,1]#.view(-1,1)
        # print(target.size())
        # print(input.size())
        logpt = F.log_softmax(input, dim=1)
        # print(logpt)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = (1 - pt) ** self.gamma * F.binary_cross_entropy(input, target.type_as(input))  ##logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
