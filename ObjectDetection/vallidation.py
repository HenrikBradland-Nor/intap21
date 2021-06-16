from ObjectDetection.preProsessing import dataPreProsessor
from ObjectDetection.graph import graphMaster
from config import *
from ObjectDetection.models import DGCNN, DGCNN_cls, PointGNN, VoxelNet, PointNet, RandomForest, SVM


def val(cfg):
    print("\n Loading model \n")
    if cfg.ALGORITHEM is 'DCGNN_1':
        model = DGCNN(cfg=cfg)
    elif cfg.ALGORITHEM is 'DCGNN_2':
        model = DGCNN_cls(cfg=cfg)
    elif cfg.ALGORITHEM is 'PointGNN':
        model = PointGNN(cfg=cfg)
    elif cfg.ALGORITHEM is 'VoxelNet':
        model = VoxelNet(cfg=cfg)
    elif cfg.ALGORITHEM is 'PointNet':
        model = PointNet(cfg=cfg)
    elif cfg.ALGORITHEM is 'RandomForest':
        model = RandomForest(cfg=cfg)
    elif cfg.ALGORITHEM is 'SVM':
        model = SVM(cfg=cfg)
    elif cfg.ALGORITHEM is 'LightGBM':
        print(' ')
    else:
        raise Exception("No model is chosen")

    if cfg.ALGORITHEM in ['RandomForest', 'SVM']:
        model = joblib.load(cfg.MODEL_WEIGHTS_PATH)
    elif cfg.ALGORITHEM in ['LightGBM']:
        model = lgb.Booster(model_file=cfg.MODEL_WEIGHTS_PATH)
    else:
        model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH))

    if cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'PointGNN', 'VoxelNet', 'PointNet']:
        model.to(cfg.DEVICE)

    val_data = []

    print("\n Loading val data \n")

    for i, file_name in tqdm(enumerate(os.listdir(cfg.DATA_PATH + '/val/pc'))):
        path_label = cfg.DATA_PATH + '/val/label/' + file_name
        path_data = cfg.DATA_PATH + '/val/pc/' + file_name

        dpp = dataPreProsessor(path_data=path_data, path_label=path_label, cfg=cfg)
        dpp.computeNormalVectors()
        dpp.applyRandomReductionFilter()

        if cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'VoxelNet', 'PointNet', 'PointGNN']:
            val_data.append(dpp.pcl_to_pytorch_octett())
        else:
            x, label = dpp.pcl_to_pytorch()
            ids = dpp.getTrainIDs()

            print("Loading file:", file_name, "of size:", x.shape[0])

            if cfg.ALGORITHEM in ['RandomForest', 'SVM']:
                val_data.append([x[:, 3:], label, ids])
            elif cfg.ALGORITHEM in ['LightGBM']:
                x, label = x.numpy(), label.numpy()
                val_data.append([x, label[:, 1], ids])
        if 'debug' in cfg.STATES:
            break
        if cfg.SMAL_DATA_SET and i == 3:
            break

    lab = np.zeros((1))
    pred = np.zeros((1))
    p = []

    if cfg.ALGORITHEM in ['RandomForest', 'SVM']:
        for x, y, ids in tqdm(val_data):
            x, y = x.numpy(), y.numpy()
            t = time.time()
            y_hat = model.pred(x[ids])
            delta_t = time.time() - t

            y = y[ids, 1]

            lab = np.append(lab, y, axis=0)
            pred = np.append(pred, y_hat, axis=0)

            correct = 0
            for i in range(len(y_hat)):
                if y_hat[i] == y[i]:
                    correct += 1

            accuracy = correct / len(y_hat)

            if 'wandb' in cfg.STATES:
                wandb.log({"Forward time": delta_t})
                wandb.log({"acc": accuracy})
                point_cloud = np.zeros((len(y_hat), 4))
                for i, xyz in enumerate(x[ids]):
                    point_cloud[i] = np.array([xyz[0], xyz[1], xyz[2], y_hat[i]])
                wandb.log({"point_cloud": wandb.Object3D(point_cloud)})

    elif cfg.ALGORITHEM in ['LightGBM']:
        for data, lab, ids in val_data:
            t = time.time()
            ypred = model.predict(data[:, 3:])
            delta_t = time.time() - t
            pred = np.around(np.asarray(ypred))

            correct = 0
            for i in range(len(pred)):
                if i in ids:
                    if pred[i] == lab[i]:
                        correct += 1

            accuracy = correct / len(pred[ids])


            print(np.average(pred))

            data = np.asarray(data)
            point_cloud = np.concatenate((data[:, :3],
                                          pred.reshape(-1, 1)), axis=1)

            print(point_cloud.shape)
            wandb.log({"point_cloud": wandb.Object3D(point_cloud)})
            wandb.log({"Forward time": delta_t})
            wandb.log({"acc": accuracy})

    elif cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'VoxelNet', 'PointNet', 'PointGNN']:

        pred = np.zeros((1,2))
        lab = np.zeros((1,2))

        for octett in val_data:
            local_pred = np.zeros((1, 2))
            local_lab = np.zeros((1, 2))

            point_cloud = np.zeros((1, 4))
            XYZ = np.zeros((1,3))
            C = np.zeros((1,1))
            t = time.time()
            '''
            if cfg.ALGORITHEM is 'PointNet':
                XYZ = np.zeros((1, 3))
                X = torch.zeros((1, cfg.NUMBER_OF_ATTRIBUTES), dtype=cfg.DATA_TYPE)
                Y = torch.zeros((1, cfg.NUMBER_OF_CLASSES), dtype=cfg.DATA_TYPE)
                IDS = torch.zeros((1), dtype=torch.int64)
                for x, y, index, knn in octett:
                    xyz = x[:, :3]
                    y = torch.tensor(y, dtype=cfg.DATA_TYPE)

                    ids = torch.tensor([i for i in range(y.size()[0]) if index[i] > 0], dtype=torch.int64)

                    # Load the graph to the GPU
                    x = torch.tensor(normalize(x, axis=0), dtype=cfg.DATA_TYPE)
                    y = torch.index_select(y, 0, ids)

                    X = torch.cat([X, x], dim=0)
                    Y = torch.cat([Y, y], dim=0)
                    IDS = torch.cat([IDS, ids], dim=0)

                    XYZ = np.concatenate((XYZ, xyz), axis=0)

                X = X[1:].to(cfg.DEVICE)
                Y = Y[1:].to(cfg.DEVICE)
                IDS = IDS[1:].to(cfg.DEVICE)

                predictions = model(X)
                delta_t = time.time() - t

                predictions = predictions.cpu().detach().numpy()

                pred_class = np.around(predictions)
                C = pred_class[:, 1].reshape((-1, 1))
                XYZ = XYZ[1:]

                predictions = torch.index_select(torch.tensor(predictions).to(cfg.DEVICE), 0, IDS)

                lab = Y.cpu().detach().numpy()
                pred = predictions.cpu().detach().numpy()
                wandb.log({"Forward time": delta_t})
            '''

            for x, y, index, knn in octett:
                xyz = x[:, :3]

                x = torch.tensor(x, dtype=cfg.DATA_TYPE).to(cfg.DEVICE) # normalize(x, axis=0)
                y = torch.tensor(y, dtype=cfg.DATA_TYPE)

                # Load the graph to the GPU
                ids = torch.tensor([i for i in range(y.size()[0]) if index[i] > 0], dtype=torch.int64)

                y = torch.index_select(y, 0, ids)

                x = x.to(cfg.DEVICE)
                # Forwar
                predictions = model(x)
                #print(predictions[:10])
                delta_t = time.time() - t

                ids = ids.to(cfg.DEVICE)


                predictions = predictions.cpu().detach().numpy()


                pred_class = np.around(predictions)
                p.append(pred_class)

                C = np.concatenate((C, pred_class[:,1].reshape((-1,1))), axis=0)
                XYZ = np.concatenate((XYZ, xyz), axis=0)

                predictions = torch.index_select(torch.tensor(predictions).to(cfg.DEVICE), 0, ids)

                lab = np.concatenate([lab, y.cpu().detach().numpy()], axis=0)
                pred = np.concatenate([pred, predictions.cpu().detach().numpy()], axis=0)

                local_lab = np.concatenate([local_lab, y.cpu().detach().numpy()], axis=0)
                local_pred = np.concatenate([local_pred, predictions.cpu().detach().numpy()], axis=0)

                del predictions

                wandb.log({"Forward time": delta_t})

            # wandb.log({"Clusters": wandb.Object3D(clustered_point_cloud)})
            point_cloud = np.concatenate([XYZ, C], axis=1)


            accuracy = (np.around(local_pred[:, 1]) == np.around(local_lab[:, 1], decimals=0)).sum()/len(local_lab)
            wandb.log({"point_cloud": wandb.Object3D(point_cloud[1:])})
            wandb.log({"accuracy": accuracy})
            print("\n Point cloud saved \n")
        pred = pred[:,1]
        lab = lab[:,1]

    confMat = confusion_matrix(lab[1:], np.around(pred[1:]))
    tot_accuracy = (confMat[0,0]+ confMat[1,1])/(confMat[0,0]+ confMat[0,1]+confMat[1,0]+ confMat[1,1])
    print("\n", confMat, "\n")
    print("Total accuracy:", tot_accuracy)
    if 'wandb' in cfg.STATES:
        wandb.log({"Confusion_matrix": confMat})
        wandb.log({"acc": tot_accuracy})

