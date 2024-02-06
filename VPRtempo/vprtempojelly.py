'''
Re-implementation of VPRTempo (https://github.com/QVPR/VPRTempo) with the SpikingJelly Library
'''
import os
import gc
import json
import torch
import random

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import neuron, encoding, learning
#from utils.plasticity import STDPLearner
from utils.dataset import CustomImageDataset, ProcessImage


def f_weight(x):
    return torch.clamp(x, 0, 1)

class SNNLayer(nn.Module):
    def __init__(self, dims=[0,0,0], tau_pre=0.2, tau_post=0.2, lr=0.001, f_weight=f_weight,device=None,training=False,args=None):
        super(SNNLayer, self).__init__()
        """
        dims: [input, features, output] dimensions of the layer
        thr_range: [min, max] range of thresholds
        fire_rate: [min, max] range of firing rates
        ip_rate: learning rate for input threshold plasticity
        stdp_rate: learning rate for stdp
        const_inp: [min, max] range of constant input
        p: [min, max] range of connection probabilities
        spk_force: boolean to force spikes
        """
        # Device
        self.device = device

        self.training = training

        self.layer = nn.Sequential(
            nn.Linear(dims[0], dims[1], bias=False), #TODO Try to fuse the two layer in one # N_in, N_features
            neuron.IFNode(),
            nn.Linear(dims[1], dims[2], bias=False), #N_features, N_out
            neuron.IFNode()
        )


        # Training parameters
        if self.training:
            # Set training flag
            # Set STDP instances
            instances_stdp = (nn.Linear,)
            # Define STDP Learners
            self.stdp_learners = []
            for i in range(self.layer.__len__()):
                if isinstance(self.layer[i], instances_stdp):
                    self.stdp_learners.append(
                        learning.STDPLearner(step_mode="s", synapse=self.layer[i], sn=self.layer[i+1], tau_pre=tau_pre, tau_post=tau_post, f_post=f_weight, f_pre=f_weight) #TODO Change to classical SpikingJelly Learner ?
                    ) 
            # Define Pixel Value -> Spike Encoder
            self.encoder = encoding.LatencyEncoder(T=20) #TODO Change Hyperparameter T to something more interesting and add in arguments

    # Forward pass of the SNN layer, if the training flag is on, a step of STDP is done
    def forward(self, x):
        print(self.training)

        if self.training:
            x = self.encoder(x)

            s_post = self.layer(x)
            s_post = torch.unsqueeze(s_post, dim=0) # shape = [B, N_in]
            self.s_post = s_post 

            s_pre = x
            s_pre = torch.unsqueeze(s_pre, dim=0) # shape = [B, N_features]
            self.s_pre = s_pre

            # STDP on the first Layer
            self.stdp_learners[0].step()

            # STDP on the second layer
            self.stdp_learners[1].step()
        return self.layer(x)

class VPRTempo(nn.Module):
    def __init__(self, args, dims, logger, num_modules, output_folder, out_dim, out_dim_remainder=None):
        super(VPRTempo, self).__init__()

        # Set the args
        if args is not None:
            self.args = args
            for arg in vars(args):
                setattr(self, arg, getattr(args, arg))
        setattr(self, 'dims', dims)
        
        # Set the device
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        
        # Set input args
        self.logger = logger
        self.num_modules = num_modules
        self.output_folder = output_folder

        # Set the dataset file
        self.dataset_file = os.path.join('/media/geoffroy/T7/VPRTempo/vprtempo/dataset', self.dataset + '.csv')
        self.query_dir = [dir.strip() for dir in self.query_dir.split(',')]

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0
        self.database_dirs = [dir.strip() for dir in self.database_dirs.split(',')]
        self.location_repeat = len(self.database_dirs) # Number of times to repeat the locations
        if not out_dim_remainder is None:
            self.T = int(out_dim_remainder * self.location_repeat * self.epoch)
        else:
            self.T = int(self.max_module * self.location_repeat * self.epoch)


        # Define layer architecture
        self.input = int(self.dims[0]*self.dims[1])
        self.feature = int(self.input * 2)

        # Output dimension changes for final module if not an even distribution of places
        if not out_dim_remainder is None:
            self.output = out_dim_remainder
        else:
            self.output = out_dim

        # Define trainable layers
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature, self.output],
            device=self.device,
            training=False,
            tau_pre=0.2,
            tau_post=0.4,
            lr=0.001,
        )

        # self.add_layer(
        #     'output_layer',
        #     dims=[self.feature, self.output],
        #     device=self.device,
        #     training=True,
        #     tau_pre=0.2,
        #     tau_post=0.4,
        #     lr=0.001,
        # )

    def add_layer(self, name, **kwargs):
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        
        # Add a new SNN
        setattr(self, name, SNNLayer(**kwargs))

        # Add layer name and index to the layer_dict
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1
        
    def train_model(self, train_loader, layer, model, model_num, prev_layers=None):
        """_summary_

        Args:
            train_loader (_type_): _description_
            layer (_type_): _description_
            model (_type_): _description_
            model_num (_type_): _description_
            prev_layers (_type_, optional): _description_. Defaults to None.
        """
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.T,
                    desc=f"Module {model_num+1}",
                    position=0)
        for _ in range(self.epoch):
            for spikes, labels in train_loader:
                # Intensity of the pixel in latency divided in T bins
                for _ in range(20): #TODO Change hardcoded 20 to an attribute of the SNNLayer Object
                    spikes, labels = spikes.to(self.device), labels.to(self.device)
                    model(spikes, layer)
                pbar.update(1)
        # Close the tqdm progress bar
        pbar.close()

        # Free up memory
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            gc.collect()

    def forward(self, in_spikes, layer):
        """Compute the forward pass of the corresponding layer."""
        out_spikes = layer(in_spikes)
        return out_spikes

    def save_model(self, models, model_out):
        """ Save the trained model to models output folder."""
        state_dicts = {}
        for i, model in enumerate(models):
            state_dicts[f'model_{i}'] = model.state_dict()
        torch.save(state_dicts, model_out)

def train_new_model(models, model_name):
    """ Train a new model

    Args:
        models: Model to train
        model_name: Name of the model to save after training
    """
    # Initialize the image transforms and datasets
    image_transform = transforms.Compose([
        ProcessImage(models[0].dims, models[0].patches)
    ])

    # Automatically generate user_input_ranges
    user_input_ranges = []
    start_idx = 0
    for _ in range(models[0].num_modules):
        range_temp = [start_idx, start_idx+((models[0].max_module-1)*models[0].filter)]
        user_input_ranges.append(range_temp)
        start_idx = range_temp[1] + models[0].filter

    # Keep track of trained layers
    trained_layers = []
    # Training each layer
    for layer_name, _ in sorted(models[0].layer_dict.items(), key=lambda item: item[1]):
        print(f"Training layer {layer_name}")
        # Retrieve the layer object
        for i, model in enumerate(models):
            model.train()
            model.to(torch.device(model.device))
            layer = (getattr(model, layer_name))
            # Determine the maximum sample for the DataLoader
            if model.database_places < model.max_module:
                max_samples = model.database_places
            elif model.output < model.max_module:
                max_samples = model.output
            else:
                max_samples = model.max_module

            # Initialize new dataset with unique range for each module
            img_range=user_input_ranges[i]
            train_dataset = CustomImageDataset(annotations_file=models[0].dataset_file, 
                        base_dir=models[0].data_dir,
                        img_dirs=models[0].database_dirs,
                        transform=image_transform,
                        skip=models[0].filter,
                        test=False,
                        img_range=img_range,
                        max_samples=max_samples)
            # Initialize the data loader
            train_loader = DataLoader(train_dataset, 
                                    batch_size=1, 
                                    shuffle=True,
                                    num_workers=8,
                                    persistent_workers=True)
   
            # Train the layers
            model.train_model(train_loader, layer, model, i, prev_layers=trained_layers)
            model.to(torch.device("cpu"))


def init_model(args, dims):
    # Determine number of modules to generate based on user input
    places = args.database_places 
    # Caclulate number of modules
    num_modules = 1
    while places > args.max_module:
        places -= args.max_module
        num_modules += 1
    
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

    models = []
    final_out = None
    for mod in tqdm(range(num_modules), desc="Initializing modules"):
        model = VPRTempo(
            args,
            dims,
            None,
            num_modules,
            output_folder,
            out_dim,
            out_dim_remainder=final_out
            ) 
        model.eval()
        model.to(torch.device('cpu')) # Move module to CPU for storage (necessary for large models)
        models.append(model) # Create module list
        if mod == num_modules - 2:
            final_out = final_out_dim
        model_name = "michel"
        print(f"Model name: {model_name}")
    return models, model_name

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nordland")
    parser.add_argument("--database_dirs", type=str, default="spring, fall")
    parser.add_argument("--query_dir", type=str, default="summer")
    parser.add_argument("--max_module", type=int, default=40)
    
    # Define the dataset argument
    parser.add_argument('--data_dir', type=str, default='/media/geoffroy/T7/VPRTempo/vprtempo/dataset/',
                            help="Directory where dataset files are stored")

    parser.add_argument('--database_places', type=int, default=40,
                        help="Number of places to use for training")


    # Define training parameters
    parser.add_argument('--filter', type=int, default=1,
                            help="Images to skip for training and/or inferencing")
    parser.add_argument('--epoch', type=int, default=4,
                            help="Number of epochs to train the model")

    # Define image transformation parameters
    parser.add_argument('--patches', type=int, default=15,
                            help="Number of patches to generate for patch normalization image into")
    parser.add_argument('--dims', type=str, default="56,56",
                        help="Dimensions to resize the image to")
    
    args = parser.parse_args()
    #tempo = VPRTempo(args,[10,10],1,1,1,1)
    #models = [tempo]
    #train_new_model(models, "michel.pth")
    dims = [int(x) for x in args.dims.split(",")]
    models, model_name = init_model(args, dims)
    train_new_model(models, model_name)    


    #tempo.train_model(None,None,None,model_num=0)
