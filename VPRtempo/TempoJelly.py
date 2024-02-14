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
import gc

from tqdm import tqdm
from torch.optim import SGD, Adam
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Subset
from spikingjelly.activation_based import neuron, encoding, learning, functional, layer
from utils.dataset import CustomImageDataset, ProcessImage
from utils.metrics import recallAtK, createPR

from memory_profiler import profile

def f_weight(x):
    return torch.clamp(x, 0, 1)

def generate_model_name(model,quant=False):
    """
    Generate the model name based on its parameters.
    """
    if quant:
        model_name = (''.join(model.database_dirs)+"_"+
                "VPRTempoQuant_" +
                "IN"+str(model.input)+"_" +
                "FN"+str(model.feature)+"_" + 
                "DB"+str(model.database_places) +
                ".pth")
    else:
        model_name = (''.join(model.database_dirs)+"_"+
                "VPRTempo_" +
                "IN"+str(model.input)+"_" +
                "FN"+str(model.feature)+"_" + 
                "DB"+str(model.database_places) +
                ".pth")
    return model_name

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
            layer.Linear(self.input, self.feature, bias=False), #TODO Try to fuse the two layer in one # N_in, N_features
            neuron.IFNode(v_threshold=1.),
            layer.Linear(self.feature, self.output, bias=False), #N_features, N_out
            neuron.IFNode()  
        )
        nn.init.normal_(self.layer[0].weight, mean=0.4)

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
        """Forward pass of the network, if the training flag is on, a step of STDP is done."""
        return self.layer(x)
    
    def evaluate(self, test_loader):
        """ Run inferences and calculate models accuracy

        Args:
            test_loader: Testing data loader
        """
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.query_places,
                    desc="Running the test network",
                    position=0)
        
        # Initialize the output spikes variable
        out = []
        labels = []

        # TODO remove with latency after 

        # Run inferences for the specified number of timesteps
        s_list = []
        for spikes, label in test_loader:
            # Set device 
            spikes = spikes.to(self.device)
            for _ in range(self.latency):
                spikes = self.encoder(spikes)
                # Forward pass
                spikes = self.forward(spikes)

            labels.append(label.detach().cpu().item())
            # Add output spikes to list
            out.append(spikes.detach().cpu().tolist())
            pbar.update(1)
            # Reset network
            functional.reset_net(self)
            functional.detach_net(self)
 
        # Close the progress bar
        pbar.close()
        # Reshape output spikes into a similarity matrix
        out = np.reshape(np.array(out), (self.query_places, self.database_places))

        # Create GT Matrix 
        GT = np.zeros((self.query_places, self.database_places), dtype=int)
        for n, ndx in enumerate(labels):
            if self.filter != 1:
                ndx = ndx//self.filter
            GT[n][ndx] = 1

        # Create GT soft matrix
        if self.GT_tolerance > 0:
            GTsoft = np.zeros((self.query_places,self.database_places), dtype=int)
            for n, ndx in enumerate(labels):
                if self.filter !=1:
                    ndx = ndx//self.filter
                GTsoft[n, ndx] = 1
                # Apply tolerance
                for i in range(max(0, n - self.GT_tolerance), min(self.query_places, n + self.GT_tolerance + 1)):
                    GTsoft[i, ndx] = 1
        else:
            GTsoft = None

        # If user specified, generate a PR curve
        if self.PR_curve:
            # Create PR curve
            P, R = createPR(out, GThard=GT, GTsoft=GTsoft, matching='single', n_thresh=100)
            # Combine P and R into a list of lists
            PR_data = {
                    "Precision": P,
                    "Recall": R
                }
            output_file = "PR_curve_data.json"
            # Construct the full path
            full_path = f".{self.output_folder}/{output_file}"
            # Write the data to a JSON file
            with open(full_path, 'w') as file:
                json.dump(PR_data, file) 
            # Plot PR curve
            plt.plot(R,P)    
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.show()
            
        # Recall@N
        N = [1,5,10,15,20,25] # N values to calculate
        R = [] # Recall@N values
        # Calculate Recall@N
        for n in N:
            R.append(round(recallAtK(out,GThard=GT,GTsoft=GTsoft,K=n),2))
        # Print the results
        table = PrettyTable()
        table.field_names = ["N", "1", "5", "10", "15", "20", "25"]
        table.add_row(["Recall", R[0], R[1], R[2], R[3], R[4], R[5]])
        print(table)


    def save_model(self, model_name):
        """ Save the trained model to models output folder."""       
        state_dicts = {}
        state_dicts[f'model'] = self.state_dict()
        torch.save(state_dicts, model_name)

    def train_model(self, train_loader, model_name):
        # Initialize the tqdm progress bar
        pbar = tqdm(total=self.T,
                    desc=f"Training of {model_name} in features layer",
                    position=0)
        # Training in feature layer
        
        self.optimizer_stdp.zero_grad()

        for _ in range(self.epoch):
            for spikes, labels in train_loader:
                # Intensity of the pixel in latency divided in T bins
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                for _ in range(self.latency):
                    spikes = self.encoder(spikes)
                    self.forward(spikes)
                    self.stdp_learners[0].step(on_grad=True)
                    self.optimizer_stdp.step()
                self.layer[0].weight.data = torch.clamp(self.layer[0].weight.data, 0, 1)
                # Reset membrane potential between images
                functional.reset_net(self)
                functional.detach_net(self)
                for i in range(self.stdp_learners.__len__()):
                    self.stdp_learners[i].reset()                
                # Clear gradient 
                self.zero_grad()
                pbar.update(1)
        
        # Clear memory to avoid memory overflow
        functional.reset_net(self)
        functional.detach_net(self)
        for i in range(self.stdp_learners.__len__()):
            self.stdp_learners[i].reset()

        pbar = tqdm(total=self.T,
            desc=f"Training of {model_name} in output layer",
            position=0)
        # Training in output layer
        for _ in range(self.epoch):
            for spikes, labels in train_loader:
                # Intensity of the pixel in latency divided in T bins
                spikes, labels = spikes.to(self.device), labels.to(self.device)
                for _ in range(self.latency):
                    spikes = self.encoder(spikes)
                    self.forward(spikes)
                self.stdp_learners[1].step(on_grad=True)
                self.optimizer_stdp.step()
                # Clear gradient 
                self.zero_grad()
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

    print(len(train_dataset))
    # Initialize the data loader
    train_loader = DataLoader(train_dataset, 
                            batch_size=1, 
                            shuffle=True,
                            num_workers=8,
    )

    model.train_model(train_loader, model_name)
    model.to(torch.device("cpu")) # Move module to CPU for storage after training
    return model

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
    models = []
    # Create the modules
    for mod in tqdm(range(num_modules), desc="Initializing modules"):
        model = TempoJelly(args,
                        num_modules,
                        output_folder,
                        out_dim,
                        out_dim_remainder=final_out_dim
                        )
        model.to(torch.device('cpu')) # Move module to CPU for storage (necessary for large models)
        models.append(model) # Create module list
        if model_name == num_modules - 2:
            final_out = final_out_dim

    # Generate the model name
    model_name = generate_model_name(model)
    print(f"Model name: {model_name}")

    return model, model_name
    #train_new_model(model, model_name)


def run_inferences(model):
    image_transform = ProcessImage(model.dims, model.patches)
    # Determines if querying a subset of the database or the entire database
    if model.query_places == model.database_places:
        subset = False # Entire database
    elif model.query_places < model.database_places:  
        subset = True # Subset of the database
    else:
        raise ValueError("The number of query places must be less than or equal to the number of database places.")
    
    # Initialize the test dataset
    test_dataset = CustomImageDataset(annotations_file=model.dataset_file, 
                                      base_dir=model.data_dir,
                                      img_dirs=model.query_dir,
                                      transform=image_transform,
                                      max_samples=model.database_places,
                                      skip=model.filter)
    
    # If using a subset of the database
    if subset:
        if model.shuffle: # For a randomized selection of database places
             test_dataset = Subset(test_dataset, random.sample(range(len(test_dataset)), model.query_places))
        else: # For a sequential selection of database places
            indices = [i for i in range(model.database_places) if i % model.filter == 0]
            # Limit to the desired number of queries
            indices = indices[:model.query_places]
            # Create the subset
            test_dataset = Subset(test_dataset, indices)


    # Initialize the data loader
    test_loader = DataLoader(test_dataset, 
                             batch_size=1, 
                             shuffle=model.shuffle,
                             num_workers=8,
                
                             persistent_workers=True)
    model.eval() 
    with torch.no_grad():
        model.evaluate(test_loader)

def load_model(model, path):
    if model.device == "cpu":
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
    # Load without specifying map_location storage
    else:    
        model.load_state_dict(torch.load(path))
    model.eval()
    return model

def parse_network():
    """Parse arguments by default or provided by the user"""
    parser = argparse.ArgumentParser(description="Args for base configuration file")

    # Define the Network hyperparameters
    parser.add_argument('--tau_pre', type=float, default=0.2,
                            help="Leaky parameter of trace pre spikes")
    parser.add_argument('--tau_post', type=float, default=0.2,
                        help="Leaky parameter of trace post spikes")
    parser.add_argument('--latency', type=int, default=5,
                        help="Number of bins for generating intensity to latency spikes")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                    help="Learning rate for the STDP")  
    
    # Define the dataset arguments
    parser.add_argument('--dataset', type=str, default='nordland',
                            help="Dataset to use for training and/or inferencing")
    parser.add_argument('--data_dir', type=str, default='/media/geoffroy/T7/VPRTempo/vprtempo/dataset/',
                            help="Directory where dataset files are stored")
    parser.add_argument('--database_places', type=int, default=5,
                            help="Number of places to use for training")
    parser.add_argument('--query_places', type=int, default=5,
                            help="Number of places to use for inferencing")
    parser.add_argument('--max_module', type=int, default=5,
                            help="Maximum number of images per module")
    parser.add_argument('--database_dirs', type=str, default='spring, fall',
                            help="Directories to use for training")
    parser.add_argument('--query_dir', type=str, default='summer',
                            help="Directories to use for testing")
    parser.add_argument('--shuffle', action='store_true',
                            help="Shuffle input images during query")
    parser.add_argument('--GT_tolerance', type=int, default=1,
                        help="Ground truth tolerance for matching")
    
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
    
    # Define metrics functionnality
    parser.add_argument('--PR_curve', action='store_true', default=True,
                        help="Flag to generate a Precision-Recall curve")
    parser.add_argument('--sim_mat', action='store_true', default=True,
                            help="Flag to plot the similarity matrix, GT, and GTsoft")
    
    args = parser.parse_args()

    return args
    
#TODO Add a check if trained networks already exists and add a way to cleanly make inferences with the network 
if __name__ == "__main__":
    args = parse_network()
    dims = [int(x) for x in args.dims.split(",")]
    args.dims = dims

    # Training 
    model, model_name = init_model(args, dims)
    # model = train_new_model(model, model_name)
    # model.save_model(f"model_latency_{args.latency}_{args.database_places}_places.pth")
    model = load_model(model, 'model_latency_5_5_places.pth')
    run_inferences(model)
    # model.save_model("model.pth")

    # Testing
    # model, model_name = init_model(args, dims)
    # model = load_model(model, '/media/geoffroy/T7/model.pth')
    # run_inferences(model)
