import warnings
from typing import Optional, Sequence
import numpy as np
import os
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torchmetrics.functional import pairwise_cosine_similarity
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid

import lightly
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss

from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.models import FeatureExtractorBackbone
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
)
from avalanche.logging import InteractiveLogger

from datasets.coil100 import Coil100Dataset
from self_supervision.ssl import BarlowTwins
from clp.clp import CLP


def main():
    
    # root_dir = 'C:/Users/ehajizad/lava-nc/neuromorphic-continual-learning'
    # dataset_dir = 'C:/Users/ehajizad/Projects/data/coil-100'
    
    root_dir = '/home/ehajizad/ss_learning/neuromorphic-continual-learning'
    dataset_dir = '/home/ehajizad/ss_learning/ssl_tests/datasets/coil-100'
    
    dataset = Coil100Dataset(root_dir=dataset_dir, transform=None)

    train_ds = Coil100Dataset(root_dir=dataset_dir,
                              transform=transforms.ToTensor(), size=64,
                              train=True)
    test_ds = Coil100Dataset(root_dir=dataset_dir,
                             transform=transforms.ToTensor(), size=64,
                             train=False)

    test_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - test_size

    coil100_nc_bm = nc_benchmark(
            train_ds, test_ds, n_experiences=100, shuffle=True, seed=1234,
            task_labels=False
    )

    train_stream = coil100_nc_bm.test_stream

    device = "cuda" if torch.cuda.is_available() else "cpu"

    resnet = lightly.models.ResNetGenerator('resnet-9')
    backbone = nn.Sequential(*list(resnet.children())[:-1],
                             nn.AdaptiveAvgPool2d(2))

    # backbone = CoilCNNBackbone(in_channels = 3)

    # simpleCNN = SimpleCNN()
    # backbone = nn.Sequential(*list(simpleCNN.children())[:-1][0][:-2],
    # nn.AdaptiveAvgPool2d(2))

    model = BarlowTwins(backbone)
    model = model.to(device)

    model.load_state_dict(torch.load(
            root_dir+"/models/coil100_barlow_twins_modified_resnet9.pth",
            map_location=device))
    model = model.backbone[0:-1]
    model.eval()

    # embeddings = np.load(
    #         root_dir+"/embeddings/coil100_bt_resnet_embeddings.npz")
    # X = torch.from_numpy(embeddings["X"])
    # y = torch.from_numpy(embeddings["y"])
    
    # Generate embedding
    
    feat_ext_dl = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=4)
    
    embeddings = []
    with torch.no_grad():
        for batch in feat_ext_dl:
            image, label = batch 
            # print("Before data:", torch.cuda.memory_allocated(device)/1e9)
            image, label = image.to(device), label.to(device)
            emb = model(image).flatten(start_dim=1)
            embeddings.append(emb)
        
    embeddings = torch.cat(embeddings, 0)
    
    
    eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            loggers=[InteractiveLogger()],
    )

    criterion = torch.nn.CrossEntropyLoss()

    # CREATE THE STRATEGY INSTANCE

    clvq = CLP(clvq_model=model,
               n_protos=2000,
               bmu_metric="dot_product",
               criterion=criterion,
               alpha_start=0.1,
               tau_alpha_decay=18,
               tau_alpha_growth=18,
               max_allowed_mistakes=1,
               input_size=embeddings.shape[1],
               num_classes=100,
               eval_mb_size=2,
               train_mb_size=2,
               train_epochs=1,
               device=device,
               evaluator=eval_plugin)

    clvq.init_prototypes_from_data(embeddings)

    # TRAINING LOOP
    print("Starting experiment...")
    for i, exp in enumerate(coil100_nc_bm.train_stream):
        # fit SLDA model to batch (one sample at a time)
        clvq.train(exp)

        # evaluate model on test data
        clvq.eval(coil100_nc_bm.test_stream)


if __name__ == "__main__":
    main()