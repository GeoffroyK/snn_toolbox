import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from spikingjelly.activation_based import encoding, functional

class VPRModel():
    def __init__(self, snn_model, training_data, *args, **kwargs) -> None:
        self.net = snn_model
        self.training_data = training_data

        # Set the args as object's attribute
        if args is not None:
            self.args = args
            for arg in vars(args):
                setattr(self, arg, getattr(args, arg))

    @property
    def f_weight(self, x) -> torch.Tensor:
        return torch.clamp(x, 0, 1)
    
    def save_model(self, model_name):
        """ Save the trained model with the specified name

        Args:
            model_name (_type_): _description_
        """
        state_dicts = {}
        state_dicts[f'model'] = self.net.state_dict()
        torch.save(state_dicts, model_name)

    def stdp_training(self, encoder, optimizer, learner) -> nn.Module:
        """ STDP Training with grayscaled, patches images of the nordland dataset

        Args:
            encoder (_type_): _description_
            optimizer (_type_): _description_
            learner (_type_): _description_

        Returns:
            nn.Module: _description_
        """
        
        # Check if the provided encoder is Latency based encoder
        assert type(encoder) == encoding.LatencyEncoder, "The encoder need to be a LatencyEncoder (see. https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.4/clock_driven_en/2_encoding.html)"

        # Check and define the training device (CUDA GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init tqdm bar for training progress
        pbar = tqdm(total=self.epochs,
            desc=f"Training of model with STDP on {device}",
            position=0)

        # Initial weights of the network on the Linear layers
        for layer in self.net.children():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight.data, mean=self.w_mean) 

        # Define dictionnary for inhibition
        selective_neuron = {}

        # Begin STDP training
        for _ in range(self.epochs):
            self.net.train()

            for label, img in enumerate(self.training_data):
                self.optimizer.zero_grad()
                img.to(device)
                label.to(device)

                winner_index = 0
                inhibited_neurons = []

                # Discretization of the input image in spike train using a pixel intensity to latency encoder
                for _ in range(encoder.T):
                    encoded_img = encoder(img)
                    out_spike = self.net(encoded_img).detach()

                    # Create inhibition to make a Winner-take-All inhibiton mechanism
                    if 1. in out_spike: # If at least one spike has occured
                        winner_index = torch.argmax(out_spike)
                    if label not in selective_neuron and winner_index not in selective_neuron.values():
                        selective_neuron[label] = winner_index
                    if selective_neuron.__len__() > 0 and label in selective_neuron:
                        for idx in range(len(self.net.layer[-1].v[0].detach())): # Inhib all non-winning spiking neurons of the output layer.
                            if idx != selective_neuron[label]:
                                inhibited_neurons.append(idx)

                # Prevent non-winning neurons from spiking with a negative fixed potential
                for neuron in inhibited_neurons:
                    self.net.layer[-1].v[0][neuron] = -10 #TODO change this fixed parameter to a variable !

                # Clamp the weights of the network between 0 and 1 to avoid huge values to appear
                self.net.layer[1].weight.data = torch.clamp(self.net.layer[1].weight.data, 0, 1)

                # Calculate the delta of weights (STDP step)
                learner.step(on_grad=True)
                optimizer.step()
            
                # Due to the stateful nature of the encoder, we have to reset it between each image presentation
                encoder.reset()
                
                # Reset network state between each images to have a fair learning basis for each images
                functional.reset_net(self.net)
                # Reset the stdp learner to avoid memory overflow
                learner.reset()

            pbar.update(1) # Update tqdm bar
        pbar.close() # Close tqdm bar
        return self.net
    
    def plot_spiking_timing(self, img, title):
        """_summary_

        Args:
            img (_type_): _description_
            title (_type_): _description_
        """
        functional.reset_net(self.net)
        self.net.eval()

        encoder = encoding.LatencyEncoder(T=self.T)

        plt.figure(title)

        v_list = []
        s_list = []

        with torch.no_grad():
            for _ in range(encoder.T):
                encoded_img = encoder(img)
                s_list.append(self.net(encoded_img).numpy().squeeze(0))
                v_list.append(self.net.layer[-1].v.numpy().squeeze(0))
                    
        s_list = np.array(s_list)
        v_list = np.array(v_list)

        # Plot spikes    
        for neuron in range(len(s_list)):
            plt.subplot(len(s_list)+1, 1, neuron+1)
            spike_times = [i for i, x in enumerate(s_list[:,neuron]) if x == 1]
            plt.eventplot(spike_times,lineoffsets=0, linewidths=1., colors='r', label=f"neuron {neuron}")
            plt.yticks([])
            plt.xlim(-0.5, self.T-1)
            plt.text(0.5,0.5,f"Fire rate : {np.mean(s_list[:,0]):.4f}")
            plt.title(f"Neuron {neuron+1} activity")

        # Plot membrane potential
        plt.subplot(len(s_list)+1, 1, len(s_list)+1)
        for pot in range(v_list.shape[1]):
            plt.plot(v_list[:,pot], label=f"Neuron {pot}")
        plt.plot([0., self.T-1], [self.threshold, self.threshold], "k--", label="threshold")
        plt.xlim(-0.5, self.T-1)
        plt.title("Membrane potential of neurons")

        plt.legend()
        plt.tight_layout()

    
