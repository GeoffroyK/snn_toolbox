'''
Re-implementation of VPRTempo (https://github.com/QVPR/VPRTempo) with the SpikingJelly Library
'''
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

from tqdm import tqdm
from torch.optim import SGD, Adam
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import neuron, encoding, learning, functional
from utils.dataset import CustomImageDataset, ProcessImage

def f_weight(x):
    return torch.clamp(x, 0, 1)

def getRAMstate():
    ram_info = psutil.virtual_memory()
    print("================ RESOURCES ===============")
    print(f"Total: {ram_info.total / 1024 / 1024 / 1024:.2f} GB")
    print(f"Available: {ram_info.available / 1024 / 1024 / 1024:.2f} GB")
    print(f"Used: {ram_info.used / 1024 / 1024 / 1024:.2f} GB")
    print(f"Percentage usage: {ram_info.percent}%")
    print("==========================================")

class TempoJelly(nn.Module):
    def __init__(self, args, num_modules, output_folder, out_dim, out_dim_remainder):
        super(TempoJelly, self).__init__()

        # Set the device
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # Set the args
        if args is not None:
            self.args = args
            for arg in vars(args):
                setattr(self, arg, getattr(args, arg))

        # Set the dataset file
        self.dataset_file = os.path.join('/media/geoffroy/T7/VPRTempo/vprtempo/dataset', self.dataset + '.csv')
        self.database_dirs = [dir.strip() for dir in self.database_dirs.split(',')]
        self.query_dir = [dir.strip() for dir in self.query_dir.split(',')]

        # Define the number of parameters in the architecture
        self.input = int(self.dims[0]*self.dims[1])
        self.feature = int(self.input * 2)

        # Output dimension changes for final module if not an even distribution of places.
        if not out_dim_remainder is None:
            self.output = out_dim_remainder
        else:
            self.output = out_dim

        # Model informations
        self.num_modules = num_modules
        self.output_folder = output_folder

        # Define network architecture 
        """ TODO Maybe train only the first layer N times and then procede with the second layer training
            -> Create a dict to add a name to layers
            -> Create a way to add layers dynamically like in the original implementation
        """
        self.layer = nn.Sequential(
            nn.Linear(self.input, self.feature, bias=False), #TODO Try to fuse the two layer in one # N_in, N_features
            neuron.IFNode(),
            nn.Linear(self.feature, self.output, bias=False), #N_features, N_out
            neuron.IFNode()  
        )

        # Training parameters
        if self.training:
            # Set STDP instances
            instances_stdp = (nn.Linear,)

            # Define STDP Learners
            self.stdp_learners = []
            for i in range(self.layer.__len__()):
                if isinstance(self.layer[i], instances_stdp):
                    self.stdp_learners.append(
                        learning.STDPLearner(
                            step_mode='s',
                            synapse=self.layer[i],
                            sn=self.layer[i+1],
                            tau_pre=self.tau_pre,
                            tau_post=self.tau_post,
                            f_pre=f_weight,
                            f_post=f_weight
                        )
                    )
            # Set Optimizer for STDP
            params_stdp = []
            for m in self.layer.modules():
                if isinstance(m, instances_stdp):
                    for p in m.parameters():
                        params_stdp.append(p)
            self.optimizer_stdp = SGD(params_stdp, lr=self.learning_rate, momentum=0.)
            self.optimizer_stdp.zero_grad() #Clear the gradient

            # Define pixel value -> Spike Encoder
            self.encoder = encoding.LatencyEncoder(self.latency)

        self.location_repeat = len(self.database_dirs) # Number of times to repeat the locations
        self.T = int(self.max_module * self.location_repeat * self.epoch)

    def forward(self, x):
        """Forward pass of the network, if the training flag is on, a step of STDP is done.

        Args:
            x (Tensor): Binary Tensor representing each spikes at a specific timestep

        Returns:
           y (Tensor): Output of the forward pass of the network
        """
        if self.training:
            x = self.encoder(x)

            s_post = self.layer(x)
            s_post = torch.unsqueeze(s_post, dim=0) # shape = [B, N_in]
            self.s_post = s_post 

            s_pre = x
            s_pre = torch.unsqueeze(s_pre, dim=0) # shape = [B, N_features]
            self.s_pre = s_pre

            # STDP on the first Layer
            self.stdp_learners[0].step(on_grad=True)
            self.optimizer_stdp.step()
            
            # STDP on the second layer
            #self.stdp_learners[1].step()

        return self.layer(x)
    def evaluate(self):
        pass

    def save_model(self, models, model_out):
        """ Save the trained model to models output folder."""
        state_dicts = {}
        for i, model in enumerate(models):
            state_dicts[f'model_{i}'] = model.state_dict()
        torch.save(state_dicts, model_out)


    def train_model(self, train_loader, model_name):
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.T,
                    desc=f"Training of {model_name}",
                    position=0)
        for _ in range(self.epoch):
            for spikes, labels in train_loader:
                # Intensity of the pixel in latency divided in T bins
                for _ in range(self.latency):
                    spikes, labels = spikes.to(self.device), labels.to(self.device)
                    self.forward(spikes)
               
                # Clear memory to avoid memory overflow
                print("Cleaning da shit m8")
                functional.reset_net(self.layer)
                functional.detach_net(self.layer)
                for i in range(self.stdp_learners.__len__()):
                    self.stdp_learners[i].reset()
                
                # Update the tqsm progress bar
                pbar.update(1)

        # Close the tqdm progress bar
        pbar.close()

        # Free up memory
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            gc.collect()

def train_new_model(model, model_name):
    """Setup the training data and train a new model.

    Args:
        model (nn.Module): Spiking Neural Network to train.
    """
    #TODO Generate model name
    # Initialize the image transforms and datasets
    image_transform = transforms.Compose([
        ProcessImage(model.dims, model.patches)
    ])

    # Automatically generate user_input_ranges
    user_input_ranges = []
    start_idx = 0
    for _ in range(model.num_modules):
        range_temp = [start_idx, start_idx+((model.max_module-1)*model.filter)]
        user_input_ranges.append(range_temp)
        start_idx = range_temp[1] + model.filter

    #TODO If we want to separate the learning of the two SNN layers we should add a for loop here
    """TODO
    For now there is only one model for testing purpose, we should have plenty expert models for the traversal subsets
    of the given dataset, so another loop here for all the desired models (need to change the argument too!)
    """
    model.train()
    model.to(model.device) # Move module to whether CPU or GPU if available for training

    # Determine the maximum sample for the DataLoader
    if model.database_places < model.max_module:
        max_samples = model.database_places
    elif model.output < model.max_module:
        max_samples = model.output
    else:
        max_samples = model.max_module

    # Initialize new dataset with unique range for each module
    img_range = user_input_ranges[0]
    train_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                base_dir=model.data_dir,
                img_dirs=model.database_dirs,
                transform=image_transform,
                skip=model.filter,
                test=False,
                img_range=img_range,
                max_samples=max_samples)

    # Initialize the data loader
    train_loader = DataLoader(train_dataset, 
                            batch_size=1, 
                            shuffle=True,
                            num_workers=8,
    )

    model.train_model(train_loader, model_name)
    model.to(torch.device("cpu")) # Move module to CPU for storage after training

def init_model(args, dims):
    # Determine number of modules to generate based on user input
    places = args.database_places 
    # Calculate number of modules
    num_modules = 1
    while places > args.max_module:
        places -= args.max_module
        num_modules += 1

    # TODO Generate a name for the network following the same procedure as in the original implementation
    model_name = "dummy_name.pth"

    # Define output folder
    output_folder = "/output"

    # If the final module has less than max_module, reduce the dim of the output layer
    remainder = args.database_places % args.max_module
    if remainder != 0: # There are remainders, adjust output neuron count in final module
        out_dim = int((args.database_places - remainder) / (num_modules - 1))
        final_out_dim = remainder
    else: # No remainders, all modules are even
        out_dim = int(args.database_places / num_modules)
        final_out_dim = out_dim

    """TODO Init N times the object for each modules (subsets experts) here for testing purpose, there is only one"""
    model = TempoJelly(args,
                       num_modules,
                       output_folder,
                       out_dim,
                       out_dim_remainder=final_out_dim
                       )
    model.eval()
    model.to(torch.device('cpu')) # Move module to CPU for storage (necessary for large models)


    train_new_model(model, model_name)


def parse_network():
    """Parse arguments by default or provided by the user"""
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # Define the Network hyperparameters
    parser.add_argument('--tau_pre', type=float, default=0.2,
                            help="Leaky parameter of trace pre spikes")
    parser.add_argument('--tau_post', type=float, default=0.2,
                        help="Leaky parameter of trace post spikes")
    parser.add_argument('--latency', type=int, default=20,
                        help="Number of bins for generating intensity to latency spikes")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                    help="Learning rate for the STDP")  
    
    # Define the dataset arguments
    parser.add_argument('--dataset', type=str, default='nordland',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='/media/geoffroy/T7/VPRTempo/vprtempo/dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--database_places', type=int, default=500,
                            help="Number of places to use for training")
    parser.add_argument('--query_places', type=int, default=500,
                            help="Number of places to use for inferencing")
    parser.add_argument('--max_module', type=int, default=500,
                            help="Maximum number of images per module")
    parser.add_argument('--database_dirs', type=str, default='spring, fall',
                            help="Directories to use for training")
    parser.add_argument('--query_dir', type=str, default='summer',
                            help="Directories to use for testing")
    parser.add_argument('--shuffle', action='store_true',
                            help="Shuffle input images during query")
    
    # Define training arguments
    parser.add_argument('--filter', type=int, default=1,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch', type=int, default=4,
                            help="Number of epochs to train the model")
    parser.add_argument('--training', type=bool, default=True,
                            help="Make the network learn or inference only")

    # Define image transformations parameters
    parser.add_argument('--patches', type=int, default=15,
                            help="Number of patches to generate for patch normalization image into")
    parser.add_argument('--dims', type=str, default="56,56",
                        help="Dimensions to resize the image to")
    args = parser.parse_args()
    return args
    
#TODO Add a check if trained networks already exists and add a way to cleanly make inferences with the network 
if __name__ == "__main__":
    args = parse_network()
    dims = [int(x) for x in args.dims.split(",")]
    args.dims = dims
    init_model(args, dims)