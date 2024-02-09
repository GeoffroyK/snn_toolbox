import os
import gc
import json
import torch
import random
import argparse
import psutil

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import gc

from tqdm import tqdm
from torch.optim import SGD, Adam
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import neuron, encoding, learning, functional, layer
from utils.dataset import CustomImageDataset, ProcessImage

def getRAMstate():
    ram_info = psutil.virtual_memory()
    print("================ RESOURCES ===============")
    print(f"Total: {ram_info.total / 1024 / 1024 / 1024:.2f} GB")
    print(f"Available: {ram_info.available / 1024 / 1024 / 1024:.2f} GB")
    print(f"Used: {ram_info.used / 1024 / 1024 / 1024:.2f} GB")
    print(f"Percentage usage: {ram_info.percent}%")
    print("==========================================")


def f_weight(x):
    return torch.clamp(x, 0, 1)

input = 58 * 58
feature = input * 2
output = 500
T = 4000

latency = 20

net= nn.Sequential(
    layer.Linear(input, feature, bias=False), #TODO Try to fuse the two layer in one # N_in, N_features
    neuron.IFNode(),
    layer.Linear(feature, output, bias=False), #N_features, N_out
    neuron.IFNode()  
)

instances_stdp = (nn.Linear,)
# Define STDP Learners
stdp_learners = []
for i in range(net.__len__()):
    if isinstance(net[i], instances_stdp):
        stdp_learners.append(
            learning.STDPLearner(
                step_mode='s',
                synapse=net[i],
                sn=net[i+1],
                tau_pre=0.2,
                tau_post=0.2,
                f_pre=f_weight,
                f_post=f_weight
            )
        )

params_stdp = []
for m in net.modules():
    if isinstance(m, instances_stdp):
        for p in m.parameters():
            params_stdp.append(p)
optimizer_stdp = SGD(params_stdp, lr=0.01, momentum=0.)
optimizer_stdp.zero_grad() #Clear the gradient

# Define pixel value -> Spike Encoder
encoder = encoding.LatencyEncoder(T=latency)

# Determine the maximum sample for the DataLoader
image_transform = transforms.Compose([
    ProcessImage([58,58], 15)
])
max_samples = 500
dataset_file = '/media/geoffroy/T7/VPRTempo/vprtempo/dataset/nordland' + '.csv'
data_dir = '/media/geoffroy/T7/VPRTempo/vprtempo/dataset/'
database_dirs = 'spring, fall'
database_dirs = [dir.strip() for dir in database_dirs.split(',')]
filter = 1
img_range = [0,499]
device = "cpu"

train_dataset = CustomImageDataset(annotations_file=dataset_file, 
            base_dir=data_dir,
            img_dirs=database_dirs,
            transform=image_transform,
            skip=filter,
            test=False,
            img_range=img_range,
            max_samples=max_samples)

# Initialize the data loader
train_loader = DataLoader(train_dataset, 
                        batch_size=1, 
                        shuffle=True,
                        num_workers=8,
)
model_name = "baphomet"


for epoch in range(4):
    pbar = tqdm(total=T,
            desc=f"Training of {model_name} in features layer",
            position=0)
    for spikes, labels in train_loader:
        spikes, labels = spikes.to(device), labels.to(device)
        for _ in range(latency):
            spikes = encoder(spikes)
            net(spikes)
            stdp_learners[0].step(on_grad=False)
        
        pbar.update(1)


for epoch in range(4):
    pbar = tqdm(total=T,
        desc=f"Training of {model_name} in output layer",
        position=0)
    for spikes, labels in train_loader:
        spikes, labels = spikes.to(device), labels.to(device)
        for _ in range(latency):
            spikes = encoder(spikes)
            net(spikes)
            stdp_learners[1].step(on_grad=False)
        
        pbar.update(1)


""" Save the trained model to models output folder."""
model_out = "trained_snn.pth"
state_dicts = {}
state_dicts[f'model_{i}'] = net.state_dict()
torch.save(state_dicts, model_out)
