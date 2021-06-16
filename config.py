# LIBs
import dgl
import dgl.nn as dglnn
import pclpy
from time import time
import networkx as nx
from pclpy import pcl
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import matplotlib.pyplot as plt
import random
import wandb
import time
import math
from typing import List, Tuple
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn import svm
from dgl.nn.pytorch.factory import KNNGraph
from sklearn.cluster import *
import lightgbm as lgb



class config():
    def __init__(self, clasificationb_algorithem=0, clustering_algorithem=0):
        # Global
        self.STATES = ['val',
                       'wandb',
                       'train',
                       'test',
                       #'load',
                       #'debug',
                       'save'
                       ]

        self.SMAL_DATA_SET = False

        self.DATA_TYPE = torch.float32

        self.LIST_OF_ALGORITHEMS = ('DCGNN_1',     #0
                                    'DCGNN_2',     #1
                                    'PointGNN',    #2
                                    'RandLA',      #3
                                    'VoxelNet',    #4
                                    'PointNet',    #5
                                    'RandomForest',#6
                                    'SVM',         #7
                                    'LightGBM')    #8
        self.ALGORITHEM = self.LIST_OF_ALGORITHEMS[clasificationb_algorithem]
        self.NAME = self.ALGORITHEM

        # Training
        self.DATA_SPLIT_RATIO = 0.8

        self.BATCH_SIZE = 256
        self.LEARNING_RATE = 0.1
        self.T_0 = 10
        self.MOMENTUM = 0.9
        self.DROPOUT = 0.5
        self.SCHEDUALER = True

        self.NUM_EPOCHS = 200
        self.EPOCHS = range(self.NUM_EPOCHS)

        self.DUBLE_LOSS = False
        self.LOSS_SWAP = 120


        # Data
        self.DATA_PATH = 'data/batteriSmalAndLarge'
        self.DATA_ATTRIBUTES = ['x', 'y', 'z', 'r', 'g', 'b'] #, 'x_n', 'y_n', 'z_n'
        self.NUMBER_OF_ATTRIBUTES = len(self.DATA_ATTRIBUTES)
        self.DATA_LABELS = ['background', 'cable']
        self.NUMBER_OF_CLASSES = len(self.DATA_LABELS)
        self.DIM_XYZRGBN = 6

        if self.ALGORITHEM in ['RandomForest', 'SVM']:
            self.FILE_EXTENTION = '.joblib'
        elif self.ALGORITHEM in ['LightGBM']:
            self.FILE_EXTENTION = '.txt'
        else:
            self.FILE_EXTENTION = '.pth'

        self.MODEL_WEIGHTS_PATH = 'ObjectDetection/MODEL_WEIGHTS/' + str(self.ALGORITHEM) + '/'+ self.NAME + str(
            self.FILE_EXTENTION)

        # HardWare
        self.SUPORTS_CUDA = None
        self.DEVICE = None

        # Pre prosessing
        self.LEAF_SIZE = 2
        self.NORMAL_VECTOR_K = 100
        self.Z_REFFERSNCE_PASSTHROUGH_FILTER = [0, 1900]

        # Clustering
        self.CLUSTER_ALGORITHEMS = ('DBSCAN', 'OPTICS')
        self.CLUSTER = self.CLUSTER_ALGORITHEMS[clustering_algorithem]
        self.MIN_CLUSTER_SIZE = 50

        # Graph
        self.GRAPH_K = 20
        self.TARGET_GRAPH_SIZE = 40_000

        # DGCNN
        self.X_FEATS = self.DIM_XYZRGBN
        self.DGCNN_H1_FEATS = 64
        self.DGCNN_H2_FEATS = 64
        self.DGCNN_H3_FEATS = 64
        self.DGCNN_H4_FEATS = 1024
        self.DGCNN_L1_FEATS = 1024
        self.DGCNN_L2_FEATS = 256


        # VoxelNet
        self.voxelDIm = (0.05, 0.05, 0.05)
        self.pcRangeX = [-1, 1]
        self.pcRangeY = [-1, 1]
        self.pcRangeZ = [-1, 1]

        self.maxPointsVoxel = 35

        self.voxelX = math.ceil((self.pcRangeX[1] - self.pcRangeX[0]) / self.voxelDIm[0])
        self.voxelY = math.ceil((self.pcRangeY[1] - self.pcRangeY[0]) / self.voxelDIm[1])
        self.voxelZ = math.ceil((self.pcRangeZ[1] - self.pcRangeZ[0]) / self.voxelDIm[2])

        # RandLA
        self.k_n = 16

        # RandomForest
        self.number_of_trees = 400
        self.number_of_jobs_in_parallel = -1

        # SVM
        self.svm_C = 1.0
        self.svm_kernel = 'linear' # 'linear', 'poly', 'rbf', ('sigmoid')
        self.svm_pc_size = 100000

        # LightGBM
        self.LGBM_param = {'num_leaves': 32,
                           'objective': 'binary',
                           'device_type': 'gpu',
                           'num_iterations': 1000}


