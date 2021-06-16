from ObjectDetection.preProsessing import dataPreProsessor
from ObjectDetection.graph import graphMaster
from config import *
from ObjectDetection.models import *

from ObjectDetection.NN_suport import FocalLoss


def train(cfg):
    print("\n Loading model \n")
    if cfg.ALGORITHEM is 'DCGNN_1':
        model = DGCNN(cfg=cfg)
        model.to(cfg.DEVICE)
        if 'load' in cfg.STATES:
            model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH))
        opt = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
        schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.T_0)
        loss_func = nn.MSELoss() #
        loss_func_2 = nn.BCELoss() #

    elif cfg.ALGORITHEM is 'DCGNN_2':
        model = DGCNN_cls(cfg=cfg)
        model.to(cfg.DEVICE)
        if 'load' in cfg.STATES:
            model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH))
        #opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        opt = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
        schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.T_0)
        loss_func = nn.MSELoss() # nn.BCELoss() #  nn.BCEWithLogitsLoss()  #     nn.NLLLoss() #

    elif cfg.ALGORITHEM is 'PointGNN':
        model = PointGNN(cfg=cfg)
        model.to(cfg.DEVICE)
        if 'load' in cfg.STATES:
            model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.T_0)
        loss_func = nn.MSELoss() # nn.BCELoss()

    elif cfg.ALGORITHEM is 'VoxelNet':
        model = VoxelNet(cfg=cfg)
        model.to(cfg.DEVICE)
        if 'load' in cfg.STATES:
            model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH))
        #opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        opt = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
        schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.T_0)
        #schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=[150], gamma=0.1)

        loss_func = nn.MSELoss() # nn.BCELoss()  #

    elif cfg.ALGORITHEM is 'PointNet':
        model = PointNet(cfg=cfg)
        model.to(cfg.DEVICE)
        if 'load' in cfg.STATES:
            model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        #opt = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
        #schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.T_0)
        schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=range(0, cfg.NUM_EPOCHS, 30), gamma=0.9)
        loss_func = nn.MSELoss() #nn.BCELoss()  # FocalLoss(-1) #    nn.NLLLoss() #

    elif cfg.ALGORITHEM is 'RandLA':
        model = RandLANet(cfg.NUMBER_OF_ATTRIBUTES, cfg.NUMBER_OF_CLASSES, device=cfg.DEVICE)
        model.to(cfg.DEVICE)
        model.train()
        if 'load' in cfg.STATES:
            model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS_PATH))
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.T_0)
        loss_func = nn.BCELoss()  # nn.MSELoss()

    elif cfg.ALGORITHEM is 'LightGBM':
        print(' ')

    elif cfg.ALGORITHEM is 'RandomForest':
        model = RandomForest(cfg=cfg)
        if 'load' in cfg.STATES:
            model = joblib.load(cfg.MODEL_WEIGHTS_PATH)

    elif cfg.ALGORITHEM is 'SVM':
        model = SVM(cfg=cfg)
        if 'load' in cfg.STATES:
            model = joblib.load(cfg.MODEL_WEIGHTS_PATH)
    else:
        raise Exception("No model is chosen")


    if 'wandb' in cfg.STATES:
        if cfg.ALGORITHEM in ['PointGNN', 'VoxelNet', 'PointNet', 'RandLA']:
            wandb.watch(model, loss_func, log="all")


    train_data = []

    print("\n Loading train data \n")
    for i, file_name in tqdm(enumerate(os.listdir(cfg.DATA_PATH + '/train/pc'))):
        path_label = cfg.DATA_PATH + '/train/label/' + file_name
        path_data = cfg.DATA_PATH + '/train/pc/' + file_name

        dpp = dataPreProsessor(path_data=path_data, path_label=path_label, cfg=cfg)
        dpp.computeNormalVectors()
        dpp.applyRandomReductionFilter()

        if cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'PointNet', 'RandLA', 'VoxelNet']:
            train_data.append(dpp.pcl_to_pytorch_octett())
        else:
            x, label = dpp.pcl_to_pytorch()
            ids = dpp.getTrainIDs()

            if cfg.ALGORITHEM in ['RandomForest', 'SVM']:
                train_data.append([x[:,3:], label, ids])
            elif cfg.ALGORITHEM in ['LightGBM']:
                x, label = x.numpy(), label.numpy()
                if len(train_data) < 10:
                    train_data = [x[ids, 3:], label[ids, 1]]
                else:
                    train_data = [np.concatenate((train_data[0], x[ids, 3:]), axis=0),
                                  np.concatenate((train_data[1], label[ids, 1]), axis=0)]

        if 'debug' in cfg.STATES:
            break
        if cfg.SMAL_DATA_SET and i == 13:
            break

    print("Number of training sets:", len(train_data))
    if 'test' in cfg.STATES:
        val_data = []
        print("\n Loading test data \n")

        for i, file_name in tqdm(enumerate(os.listdir(cfg.DATA_PATH + '/val/pc'))):
            path_label = cfg.DATA_PATH + '/val/label/' + file_name
            path_data = cfg.DATA_PATH + '/val/pc/' + file_name

            dpp = dataPreProsessor(path_data=path_data, path_label=path_label, cfg=cfg)
            dpp.computeNormalVectors()
            dpp.applyRandomReductionFilter()

            if cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'VoxelNet', 'PointNet', 'RandLA']:
                val_data.append(dpp.pcl_to_pytorch_octett())
            else:
                x, label = dpp.pcl_to_pytorch()
                ids = dpp.getTrainIDs()

                print("Loading file:", file_name, "of size:", x.shape[0])
                if cfg.ALGORITHEM in ['RandomForest', 'SVM']:
                    val_data.append([x[:,3:], label, ids])
                elif cfg.ALGORITHEM in ['LightGBM']:
                    x, label = x.numpy(), label.numpy()
                    if len(val_data) < 10:
                        val_data = [x[ids, 3:], label[ids, 1]]
                    else:
                        val_data = [np.concatenate((val_data[0], x[ids, 3:]), axis=0),
                                    np.concatenate((val_data[1], label[ids, 1]), axis=0)]
            if 'debug' in cfg.STATES:
                break
            if cfg.SMAL_DATA_SET and i == 3:
                break
        print("Number of validation sets:", len(val_data))





    print("\n Training model \n")
    if cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'VoxelNet', 'PointNet', 'RandLA']:
        for epoch in tqdm(cfg.EPOCHS):
            all_loss_train = []
            all_loss_val = []
            random.shuffle(train_data)

            if 'test' in cfg.STATES:
                random.shuffle(val_data)

            # Training

            if cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'PointGNN', 'PointNet', 'RandLA', 'VoxelNet']:
                for octett in train_data:
                    '''
                    if cfg.ALGORITHEM is "DCGNN_1":
                        X = torch.zeros((1, cfg.NUMBER_OF_ATTRIBUTES), dtype=cfg.DATA_TYPE)
                        Y = torch.zeros((1, cfg.NUMBER_OF_CLASSES), dtype=cfg.DATA_TYPE)
                        IDS = torch.zeros((1), dtype=torch.int64)
                        for x, y, index, knn in octett:

                            y = torch.tensor(y, dtype=cfg.DATA_TYPE)

                            ids = torch.tensor([i for i in range(y.size()[0]) if index[i] > 0], dtype=torch.int64)

                            # Load the graph to the GPU
                            x = torch.tensor(x, dtype=cfg.DATA_TYPE) #normalize()
                            y = torch.index_select(y, 0, ids)

                            X = torch.cat([X, x], dim=0)
                            Y = torch.cat([Y, y], dim=0)
                            IDS = torch.cat([IDS, ids], dim=0)

                        X = X[1:].to(cfg.DEVICE)
                        Y = Y[1:].to(cfg.DEVICE)
                        IDS = IDS[1:].to(cfg.DEVICE)


                        predictions = torch.index_select(model(X), 0, IDS)
                        loss = loss_func(predictions, Y)

                        opt.zero_grad()
                        # print("Making gradient")
                        loss.backward()
                        # print("Gradient generated")
                        opt.step()
                        # Logging
                        all_loss_train.append(loss.cpu().detach().numpy())
                    '''
                    for x, y, index, knn in octett:

                        y = torch.tensor(y, dtype=cfg.DATA_TYPE)

                        ids = torch.tensor([i for i in range(y.size()[0]) if index[i] > 0], dtype=torch.int64)

                        # Load the graph to the GPU
                        if cfg.ALGORITHEM in ['DCGNN_2']:
                            x = torch.tensor(x, dtype=cfg.DATA_TYPE).to(cfg.DEVICE)  # normalize(x, axis=0)
                        else:
                            x = torch.tensor(normalize(x, axis=0), dtype=cfg.DATA_TYPE).to(cfg.DEVICE)  #

                        y = torch.index_select(y, 0, ids).to(cfg.DEVICE)
                        # Forward
                        ids = ids.to(cfg.DEVICE)

                        if cfg.ALGORITHEM is "RandLA":
                            predictions = torch.index_select(model(x.view(1, -1, cfg.NUMBER_OF_ATTRIBUTES)), 0, ids)
                        else:
                            predictions = torch.index_select(model(x), 0, ids)

                        if cfg.ALGORITHEM in ["DCGNN_1", "DCGNN_2", 'VoxelNet']:
                            if cfg.DUBLE_LOSS and epoch > cfg.LOSS_SWAP:
                                loss = loss_func_2(predictions, y)
                            else:
                                loss = loss_func(predictions, y)


                            opt.zero_grad()
                            #print("Making gradient")
                            loss.backward()
                            #print(predictions[0], y[0], loss)
                            #print("Gradient generated")
                            opt.step()
                            all_loss_train.append(loss.cpu().detach().numpy())

                        elif cfg.ALGORITHEM in ["PointNet"]:
                            end = int(len(y) / cfg.BATCH_SIZE) * cfg.BATCH_SIZE
                            if end == 0:
                                end = len(y)
                            for i in range(0, end, cfg.BATCH_SIZE):
                                predictions = torch.index_select(model(x), 0, ids)
                                indexes = torch.tensor(range(i, i + cfg.BATCH_SIZE)).to(cfg.DEVICE)
                                loss = loss_func(torch.index_select(predictions, 0, indexes),
                                                 torch.index_select(y, 0, indexes))

                                opt.zero_grad()
                                loss.backward()
                                #print("Gradient generated")
                                opt.step()
                                all_loss_train.append(loss.cpu().detach().numpy())

                        else:
                            indexes = torch.tensor(random.sample(range(len(y)), cfg.BATCH_SIZE)).to(cfg.DEVICE)

                            # Loss
                            loss = loss_func(torch.index_select(predictions, 0, indexes),
                                             torch.index_select(y, 0, indexes))
                            #print("loss", loss)

                            #loss = loss_func(predictions, y)

                            # Backward
                            opt.zero_grad()
                            #print("Making gradient")
                            loss.backward()
                            #print("Gradient generated")
                            opt.step()

                            #print("TEST")


                        #print("Next step")

                        # del predictions
                        # Logging

            # Validation
            if 'test' in cfg.STATES:
                if cfg.ALGORITHEM in ['RandLA']:
                    for graph, ids in val_data:
                        # Load the graph to the GPU
                        graph = graph.to(cfg.DEVICE)

                        # Forward
                        predictions = model(graph, graph.ndata['feat'])

                        # Loss
                        loss = loss_func(predictions[ids], graph.ndata['label'][ids])

                        # Logging
                        all_loss_val.append(loss.cpu().detach().numpy())

                        del predictions

                elif cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'PointGNN', 'VoxelNet', 'PointNet']:
                    for octett in val_data:
                        for x, y, index, knn in octett:

                            if cfg.ALGORITHEM in ['DCGNN_2']:
                                x = torch.tensor(x, dtype=cfg.DATA_TYPE).to(cfg.DEVICE) # normalize(x, axis=0)
                            else:
                                x = torch.tensor(normalize(x, axis=0), dtype=cfg.DATA_TYPE).to(cfg.DEVICE)  #

                            y = torch.tensor(y, dtype=cfg.DATA_TYPE)

                            ids = torch.tensor([i for i in range(y.size(0)) if index[i] > 0], dtype=torch.int64)
                            # Load the graph to the GPU
                            y = torch.index_select(y, 0, ids).to(cfg.DEVICE)
                            #y = y.to(cfg.DEVICE)
                            # Forward
                            ids = ids.to(cfg.DEVICE)
                            predictions = torch.index_select(model(x), 0, ids)
                            # Loss
                            loss = loss_func(predictions, y)

                            all_loss_val.append(loss.cpu().detach().numpy())

                            del predictions
                            del loss
                elif cfg.ALGORITHEM in ['None']:
                    for octett in val_data:
                        X = torch.zeros((1, cfg.NUMBER_OF_ATTRIBUTES), dtype=cfg.DATA_TYPE)
                        Y = torch.zeros((1, cfg.NUMBER_OF_CLASSES), dtype=cfg.DATA_TYPE)
                        IDS = torch.zeros((1), dtype=torch.int64)
                        for x, y, index, knn in octett:

                            y = torch.tensor(y, dtype=cfg.DATA_TYPE)

                            ids = torch.tensor([i for i in range(y.size()[0]) if index[i] > 0], dtype=torch.int64)

                            # Load the graph to the GPU
                            x = torch.tensor(normalize(x[:6], axis=0), dtype=cfg.DATA_TYPE)
                            y = torch.index_select(y, 0, ids)

                            X = torch.cat([X, x], dim=0)
                            Y = torch.cat([Y, y], dim=0)
                            IDS = torch.cat([IDS, ids], dim=0)

                        X = X[1:].to(cfg.DEVICE)
                        Y = Y[1:].to(cfg.DEVICE)
                        IDS = IDS[1:].to(cfg.DEVICE)


                        predictions = torch.index_select(model(X), 0, IDS)
                        loss = loss_func(predictions, Y)
                        all_loss_val.append(loss.cpu().detach().numpy())

                        del predictions
                        del loss
                        del X
                        del Y
                        del IDS

                else:
                    for x, y, ids in train_data:
                        # Load the graph to the GPU
                        x = x.to(cfg.DEVICE)
                        #y = y.to(cfg.DEVICE)

                        # Forward
                        predictions = model(x)

                        # Loss
                        loss = loss_func(predictions[ids], y[ids])

                        # Logging
                        all_loss_val.append(loss.cpu().detach().numpy())

                        del predictions
            if len(all_loss_val) is 0:
                all_loss_val.append(0)
            # Logging
            if 'wandb' in cfg.STATES and cfg.ALGORITHEM in ['DCGNN_1', 'DCGNN_2', 'PointGNN', 'VoxelNet', 'PointNet']:
                #print("Train loss", sum(all_loss_train) / len(all_loss_train))
                #print("Val loss", sum(all_loss_val) / len(all_loss_val))
                wandb.log({"Train loss": sum(all_loss_train) / len(all_loss_train), "Epoch": epoch,
                           "Val loss": sum(all_loss_val) / len(all_loss_val),
                           "Learning rate": schedular.get_last_lr().__getitem__(0)})


            if cfg.SCHEDUALER:
                schedular.step()




    elif cfg.ALGORITHEM in ['RandomForest', 'SVM']:
        random.shuffle(train_data)

        X = np.zeros((1, cfg.NUMBER_OF_ATTRIBUTES-3))
        Y = np.zeros((1))

        for x, y, ids in train_data:
            x, y = x.numpy(), y.numpy()

            X = np.append(X, x[ids], axis=0)
            Y = np.append(Y, y[ids, 1], axis=0)

        if Y.shape[0] > cfg.svm_pc_size:
            i = int(np.floor((Y.shape[0]/cfg.svm_pc_size)))
            X, Y = X[::i], Y[::i]

        print("Size of X and Y", X.shape, Y.shape)
        print("Avg label", np.average(Y.reshape((-1))))

        model.fit(X, Y)

    elif cfg.ALGORITHEM in ['LightGBM']:
        dats_set = lgb.Dataset(train_data[0], label=train_data[1].flatten())
        if 'test' in cfg.STATES:
            #print(train_data[0].shape)
            #print(train_data[1].shape)
            #print(val_data[0].shape)
            #print(val_data[1].shape)

            val_set = lgb.Dataset(val_data[0], label=val_data[1].flatten())
            bst = lgb.train(cfg.LGBM_param,
                            train_set=dats_set,
                            valid_sets=val_set,
                            early_stopping_rounds=10)
        else:
            bst = lgb.train(cfg.LGBM_param,
                            train_set=dats_set,
                            early_stopping_rounds=10)

    print("\n Model trained \n")

    if 'save' in cfg.STATES:
        if cfg.ALGORITHEM in ['RandomForest', 'SVM']:
            joblib.dump(model, cfg.MODEL_WEIGHTS_PATH)
        elif cfg.ALGORITHEM in ['LightGBM']:
            bst.save_model(cfg.MODEL_WEIGHTS_PATH, num_iteration=bst.best_iteration)
        else:
            torch.save(model.state_dict(), cfg.MODEL_WEIGHTS_PATH)
        print("Model saved !")


if __name__ == '__main__':
    cfg = config(5)
    train(cfg=cfg)
